"""Detecção das mãos com MediaPipe e desenho do esqueleto."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Self

import cv2
import numpy as np

from kara_sygna import config
from kara_sygna.visao.caracteristicas import (
    NOMES_LADO,
    lado_de_cada_indice,
    slot_do_rotulo,
)

logger = logging.getLogger(__name__)


def _cor_bgr(valor: Any, padrao: tuple[int, int, int]) -> tuple[int, int, int]:
    """Converte uma cor hexadecimal para BGR, formato usado pelo OpenCV."""
    if not isinstance(valor, str):
        return padrao

    texto = valor.lstrip("#")
    if len(texto) != 6:
        return padrao

    try:
        vermelho = int(texto[0:2], 16)
        verde = int(texto[2:4], 16)
        azul = int(texto[4:6], 16)
        return azul, verde, vermelho
    except ValueError:
        return padrao


#: Detectores do rtmlib (OpenMMLab, ONNX no processador), em teste desde
#: 24/10/2026. Com o ONNX Runtime usando os 28 núcleos (o padrão), o RTMPose
#: levava 128 ms por quadro; com 8, 26 ms (o MediaPipe Hands, ~22 ms).
RTMLIB = {
    "rtmpose": ("RTMPose (mão)", "Hand", "lightweight"),
    # O modo "lightweight" do rtmlib é o RTMW-m (rápido, mas erra a mão dobrada);
    # o "balanced", o RTMW-x, que acertou a mão dobrada melhor que o RTMPose.
    "rtmw_m": ("RTMW-m (corpo inteiro, leve)", "Wholebody", "lightweight"),
    "rtmw": ("RTMW-x (corpo inteiro)", "Wholebody", "balanced"),
}
#: Mão com nota média menor que isto é descartada (antes: 0,3, e a mão sumia
#: em 23% dos quadros do vídeo do defeito).
NOTA_MINIMA_MAO = 0.15
#: No corpo inteiro (RTMW), o modelo sempre devolve as duas mãos, mesmo fora da
#: imagem (com nota baixa): exige mais nota e a mão dentro da imagem, senão a
#: malha ficava na tela depois de a mão sair (24/10/2026).
NOTA_MINIMA_MAO_CORPO = 0.45
DENTRO_DA_IMAGEM = 0.8
#: Rótulo do MediaPipe de cada mão do corpo inteiro (a chave do filtro).
_LADOS_CORPO = {"dir": "Right", "esq": "Left"}
_LADO_OPOSTO = {"Left": "Right", "Right": "Left"}


def _rtmw_x_rapido(rtmlib) -> type:
    """RTMW-x (dedos) com o detector de pessoa pequeno (YOLOX-tiny).

    O detector grande (YOLOX-m) só acha você na imagem; trocá-lo pelo pequeno
    levou o RTMW-x de 105 para 56 ms por quadro, com os mesmos dedos.
    """
    modos = rtmlib.Wholebody.MODE

    class RTMWxRapido(rtmlib.Wholebody):
        def __init__(self, **kwargs) -> None:
            super().__init__(
                det=modos["lightweight"]["det"],
                det_input_size=modos["lightweight"]["det_input_size"],
                pose=modos["balanced"]["pose"],
                pose_input_size=modos["balanced"]["pose_input_size"],
                backend=kwargs.get("backend", "onnxruntime"),
                device=kwargs.get("device", "cpu"),
            )

    return RTMWxRapido


@functools.cache
def usa_placa_video() -> bool:
    """Se o rtmlib roda na placa de vídeo (``RTMLIB_PLACA_VIDEO`` e DirectML)."""
    if not config.RTMLIB_PLACA_VIDEO:
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "DmlExecutionProvider" in ort.get_available_providers()


def _modelo_rtmlib(tipo: str) -> Any:
    """Um modelo do rtmlib, com poucos núcleos por chamada (``RTMLIB_NUCLEOS``).

    Com o padrão do ONNX (todos os 28 núcleos), as chamadas brigavam pelo
    processador e o app e o treino da IA ficavam travados. Com o rtmlib o app
    usa um detector de mãos só, numa thread própria (veja a janela principal).
    """
    try:
        import onnxruntime as ort
        import rtmlib
    except ImportError as erro:
        raise RuntimeError(
            "Este detector precisa do rtmlib (pip install rtmlib onnxruntime)."
        ) from erro
    _, classe, modo = RTMLIB[tipo]
    original = ort.InferenceSession
    placa = usa_placa_video()

    def sessao(*args, **kwargs):
        opcoes = ort.SessionOptions()
        if placa:
            # O DirectML pede estas duas opções.
            opcoes.enable_mem_pattern = False
            opcoes.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            kwargs["providers"] = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            opcoes.intra_op_num_threads = config.RTMLIB_NUCLEOS
            opcoes.inter_op_num_threads = 1
        kwargs["sess_options"] = opcoes
        return original(*args, **kwargs)

    logger.info("%s na %s.", RTMLIB[tipo][0], "placa de vídeo" if placa else "CPU")

    # O rtmlib cria as sessões do ONNX sem opções: troca só durante a criação.
    ort.InferenceSession = sessao
    try:
        # PoseTracker: procura a mão do zero só a cada alguns quadros e, no
        # meio, segue a mão pela posição anterior (como o MediaPipe faz).
        return rtmlib.PoseTracker(
            _rtmw_x_rapido(rtmlib) if tipo == "rtmw" else getattr(rtmlib, classe),
            det_frequency=config.RTMLIB_DETECCAO_A_CADA,
            tracking=False,
            mode=modo,
            backend="onnxruntime",
            device="cpu",
        )
    finally:
        ort.InferenceSession = original


class _OneEuro:
    """Filtro One-Euro (Casiez, 2012) para pontos, em pixels.

    Com os pontos quase parados suaviza muito (tira a tremida); rápidos, quase
    nada.
    """

    def __init__(self, corte: float | None = None, beta: float | None = None) -> None:
        self.corte = config.RTMLIB_EURO_CORTE if corte is None else corte
        self.beta = config.RTMLIB_EURO_BETA if beta is None else beta
        self._x = self._dx = self._t = None

    @staticmethod
    def _alfa(corte, dt: float):
        r = 2 * np.pi * corte * dt
        return r / (r + 1)

    def __call__(self, pontos: np.ndarray, agora: float) -> np.ndarray:
        if self._x is None:
            self._x, self._dx, self._t = pontos, np.zeros_like(pontos), agora
            return pontos
        dt = max(1e-3, agora - self._t)
        self._t = agora
        self._dx = self._dx + self._alfa(1.0, dt) * ((pontos - self._x) / dt - self._dx)
        corte = self.corte + self.beta * np.abs(self._dx)
        self._x = self._x + self._alfa(corte, dt) * (pontos - self._x)
        return self._x


class _FiltroMao:
    """Filtro dos 21 pontos de uma mão.

    Na placa de vídeo: o pulso com um filtro leve (a mão acompanha o movimento
    rápido) e a forma dos dedos (cada dedo menos o pulso) com um forte (sem
    tremer). Na CPU, um filtro só, leve, porque o modelo roda a cada 2 quadros.
    """

    def __init__(self) -> None:
        if usa_placa_video():
            self._pulso = _OneEuro(*config.RTMLIB_EURO_PLACA_PULSO)
            self._dedos = _OneEuro(*config.RTMLIB_EURO_PLACA_DEDOS)
        else:
            self._pulso = None
            self._dedos = _OneEuro()

    def __call__(self, mao: np.ndarray, agora: float) -> np.ndarray:
        if self._pulso is None:
            return self._dedos(mao, agora)
        pulso = self._pulso(mao[:1], agora)
        return pulso + self._dedos(mao - mao[:1], agora)


def _sem_copia(
    maos: list[tuple[str, np.ndarray, np.ndarray]], corpo: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Tira a cópia de uma mão no corpo inteiro (RTMW).

    Às vezes o RTMW põe as duas mãos no mesmo lugar (nas letras feitas só com
    a mão direita, a "esquerda" aparecia em cima da direita). Com as duas mãos
    boas e os pulsos a menos de uma palma, a cópia é a que está mais perto do
    pulso do OUTRO braço (pulso 9 do corpo para a mão direita, bloco 91; 10
    para a esquerda). Mãos de verdade encostadas ficam: cada uma continua
    mais perto do próprio braço.

    Args:
        maos: ``[("dir", pontos, notas), ("esq", pontos, notas)]``.
        corpo: Os 133 pontos da pessoa, em pixels.
    """
    boas = [m for m in maos if float(np.mean(m[2])) >= NOTA_MINIMA_MAO_CORPO]
    if len(boas) != 2:
        return maos
    (_, direita, _), (_, esquerda, _) = boas
    palma = max(
        float(np.linalg.norm(direita[9] - direita[0])),
        float(np.linalg.norm(esquerda[9] - esquerda[0])),
        1.0,
    )
    if np.linalg.norm(direita[0] - esquerda[0]) > palma:
        return maos
    def no_proprio_braco(mao: np.ndarray, proprio: int, outro: int) -> bool:
        return bool(
            np.linalg.norm(mao[0] - corpo[proprio])
            <= np.linalg.norm(mao[0] - corpo[outro])
        )

    direita_certa = no_proprio_braco(direita, 9, 10)
    esquerda_certa = no_proprio_braco(esquerda, 10, 9)
    if direita_certa and not esquerda_certa:
        return [boas[0]]
    if esquerda_certa and not direita_certa:
        return [boas[1]]
    return maos


class DetectorMaos:
    """Encapsula o detector de mãos do MediaPipe.

    Args:
        tipo: ``"hands"`` (MediaPipe Hands, o de sempre), ``"landmarker"``
            (Hand Landmarker, do MediaPipe Tasks) ou um do rtmlib (``RTMLIB``:
            ``"rtmpose"``, ``"rtmw_m"``, ``"rtmw"``), estes em teste. Padrão:
            ``config.DETECTOR_MAOS``. Os dois devolvem o resultado no mesmo
            formato (``multi_hand_landmarks``...), então o resto do app não muda.
    """

    def __init__(
        self,
        max_maos: int = config.MAX_MAOS,
        confianca_deteccao: float = config.CONFIANCA_DETECCAO,
        confianca_rastreio: float = config.CONFIANCA_RASTREIO,
        tipo: str | None = None,
    ) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Esta versão do MediaPipe não tem 'mp.solutions'. "
                "Instale mediapipe==0.10.21."
            )

        self._hands_mp = mp.solutions.hands
        self._desenho = mp.solutions.drawing_utils
        self.tipo = config.DETECTOR_MAOS if tipo is None else tipo
        self._hands = None
        self._landmarker = None
        self._rtmlib = None
        if self.tipo in RTMLIB:
            self._rtmlib = _modelo_rtmlib(self.tipo)
            # Filtro e último pulso de cada mão (pela chave da mão, não pelo lado).
            self._filtros: dict[str, _FiltroMao] = {}
            self._pulsos: dict[str, np.ndarray] = {}
            self._proxima_chave = 0
            self._quadro = 0
            self._ultimo = None
            logger.info("%s (rtmlib) inicializado.", RTMLIB[self.tipo][0])
            return
        if self.tipo == "landmarker":
            self._criar_landmarker(max_maos, confianca_deteccao, confianca_rastreio)
            logger.info("Hand Landmarker (MediaPipe Tasks) inicializado.")
            return
        self._hands = self._hands_mp.Hands(
            max_num_hands=max_maos,
            min_detection_confidence=confianca_deteccao,
            min_tracking_confidence=confianca_rastreio,
        )
        logger.info("MediaPipe Hands inicializado.")

    def _criar_landmarker(self, max_maos, confianca_deteccao, confianca_rastreio):
        from mediapipe.tasks.python import BaseOptions, vision

        caminho = config.MODELO_HAND_LANDMARKER
        if not caminho.exists():
            raise RuntimeError(
                f"Modelo do Hand Landmarker não encontrado em {caminho}. Baixe "
                "hand_landmarker.task do MediaPipe (Google) para essa pasta."
            )
        opcoes = vision.HandLandmarkerOptions(
            # Pelo conteúdo: no Windows, o caminho do arquivo dá erro no MediaPipe.
            base_options=BaseOptions(model_asset_buffer=caminho.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_maos,
            min_hand_detection_confidence=confianca_deteccao,
            min_hand_presence_confidence=confianca_deteccao,
            min_tracking_confidence=confianca_rastreio,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opcoes)
        self._ultimo_ms = 0

    def detectar(self, quadro_bgr: np.ndarray) -> Any:
        """Detecta mãos sem alterar o quadro recebido."""
        if self._rtmlib is not None:
            # Na CPU o RTMW é pesado: roda a cada RTMLIB_A_CADA quadros e, no
            # meio, repete as últimas mãos. Na placa de vídeo, todo quadro.
            self._quadro += 1
            a_cada = 1 if usa_placa_video() else config.RTMLIB_A_CADA
            if self._ultimo is None or self._quadro % a_cada == 0:
                self._ultimo = self._detectar_rtmlib(quadro_bgr)
            return self._ultimo
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        if self._landmarker is not None:
            return self._detectar_landmarker(rgb)
        return self._hands.process(rgb)

    def _associar(self, maos: list) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """RTMPose (só mãos): cada mão recebe a chave da mão anterior mais perto.

        Pelo pulso: a mão de agora é a do quadro anterior cujo pulso está mais
        perto. Mão nova (ou longe de todas) ganha uma chave nova.
        """
        livres = dict(self._pulsos)
        saida = []
        for mao, nota in sorted(maos, key=lambda m: -float(np.mean(m[1]))):
            pulso = np.asarray(mao[0], dtype=np.float64)
            palma = float(np.linalg.norm(np.asarray(mao[9]) - pulso)) or 1.0
            perto = min(
                livres, key=lambda c: np.linalg.norm(livres[c] - pulso), default=None
            )
            if perto is not None and np.linalg.norm(livres[perto] - pulso) < 2 * palma:
                chave = perto
                del livres[perto]
            else:
                self._proxima_chave += 1
                chave = f"mao{self._proxima_chave}"
            self._pulsos[chave] = pulso
            saida.append((chave, mao, nota))
        return saida

    def _detectar_rtmlib(self, quadro_bgr: np.ndarray) -> Any:
        """RTMPose/RTMW (rtmlib), no formato do MediaPipe Hands (só 2D).

        Sem pontos do mundo (3D): a forma da mão sai dos pontos da imagem.
        """
        from types import SimpleNamespace

        from mediapipe.framework.formats import classification_pb2, landmark_pb2

        altura, largura = quadro_bgr.shape[:2]
        usados: set[str] = set()
        pontos, notas = self._rtmlib(quadro_bgr)
        pontos, notas = np.asarray(pontos), np.asarray(notas)
        # Cada mão com a sua chave, para o filtro nunca misturar as duas mãos.
        # (Antes a chave era a metade da imagem: a mão que cruzava o meio
        # trocava de filtro e a malha era puxada para a outra mão.)
        maos: list[tuple[str, np.ndarray, np.ndarray]] = []
        if self.tipo == "rtmpose":  # só mãos, sem saber qual é qual
            maos = self._associar(list(zip(pontos, notas, strict=True)))
        elif len(pontos):  # corpo inteiro: o modelo sabe qual mão é qual
            pessoa = int(np.argmax(notas.mean(axis=1)))
            # A imagem chega espelhada, então a mão "esquerda" do modelo
            # (pontos 91-111, no pulso 9 do corpo) é a mão DIREITA de verdade.
            # No vídeo das letras feitas só com a mão direita (26/10/2026), ela
            # veio no bloco 91 em 149 de 164 quadros. Antes o lado saía da
            # metade da tela, e a mão que passava do meio trocava de lado.
            maos = [
                (chave, pontos[pessoa, a : a + 21], notas[pessoa, a : a + 21])
                for chave, a in (("dir", 91), ("esq", 112))
            ]
            maos = _sem_copia(maos, pontos[pessoa])
        listas, lados, slots = [], [], []
        for chave, mao, nota in maos:
            minima = (
                NOTA_MINIMA_MAO if self.tipo == "rtmpose" else NOTA_MINIMA_MAO_CORPO
            )
            if len(mao) != 21 or float(np.mean(nota)) < minima:
                continue
            dentro = (mao[:, 0] >= 0) & (mao[:, 0] < largura)
            dentro &= (mao[:, 1] >= 0) & (mao[:, 1] < altura)
            if dentro.mean() < DENTRO_DA_IMAGEM:  # mão fora da tela
                continue
            mao = np.array(mao, dtype=np.float64)
            # Filtro One-Euro em cada ponto (em pixels): tira a tremida do
            # RTMPose sem atrasar o movimento rápido. No vídeo do defeito, com
            # o PoseTracker, os pulos de dedo caíram de 21 para 3.
            filtro = self._filtros.setdefault(chave, _FiltroMao())
            mao = filtro(mao, time.monotonic())
            usados.add(chave)
            if chave in _LADOS_CORPO:
                rotulo = _LADOS_CORPO[chave]
                slots.append(slot_do_rotulo(rotulo))
            else:  # RTMPose (só a mão): pela metade da imagem
                rotulo = "Right" if float(np.mean(mao[:, 0])) > largura / 2 else "Left"
                slots.append(None)
            listas.append(
                landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        landmark_pb2.NormalizedLandmark(
                            x=float(x) / largura, y=float(y) / altura, z=0.0
                        )
                        for x, y in mao
                    ]
                )
            )
            lados.append(
                classification_pb2.ClassificationList(
                    classification=[
                        classification_pb2.Classification(
                            index=int(rotulo == "Right"),
                            score=float(np.mean(nota)),
                            label=rotulo,
                        )
                    ]
                )
            )
        for chave in set(self._filtros) - usados:  # mão que sumiu: recomeça
            del self._filtros[chave]
            self._pulsos.pop(chave, None)
        resultado = SimpleNamespace(
            multi_hand_landmarks=listas or None,
            multi_handedness=lados or None,
            multi_hand_world_landmarks=None,
        )
        if slots and None not in slots:
            # O lado de cada mão (como o dos braços, em conferir_maos), para a
            # posição na tela não decidir o lado.
            resultado.lados = slots
        return resultado

    def _detectar_landmarker(self, rgb: np.ndarray) -> Any:
        """Hand Landmarker, devolvido no mesmo formato do MediaPipe Hands."""
        import time
        from types import SimpleNamespace

        import mediapipe as mp
        from mediapipe.framework.formats import classification_pb2, landmark_pb2

        # O modo vídeo exige tempos sempre crescentes.
        self._ultimo_ms = max(self._ultimo_ms + 1, int(time.monotonic() * 1000))
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        r = self._landmarker.detect_for_video(imagem, self._ultimo_ms)
        if not r.hand_landmarks:
            return SimpleNamespace(
                multi_hand_landmarks=None,
                multi_handedness=None,
                multi_hand_world_landmarks=None,
            )
        imagem_pb = [
            landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z) for p in mao
                ]
            )
            for mao in r.hand_landmarks
        ]
        mundo_pb = [
            landmark_pb2.LandmarkList(
                landmark=[landmark_pb2.Landmark(x=p.x, y=p.y, z=p.z) for p in mao]
            )
            for mao in r.hand_world_landmarks
        ]
        # O Hand Landmarker dá o lado como se a imagem NÃO fosse espelhada (o
        # Hands antigo, como se fosse). A imagem chega espelhada, então o lado
        # é invertido: no vídeo das letras só com a mão direita (26/10/2026),
        # ele disse "Left" em 449 de 449 quadros; o Hands, "Right".
        lados_pb = [
            classification_pb2.ClassificationList(
                classification=[
                    classification_pb2.Classification(
                        index=1 - c[0].index,
                        score=c[0].score,
                        label=_LADO_OPOSTO.get(c[0].category_name, c[0].category_name),
                    )
                ]
            )
            for c in r.handedness
        ]
        return SimpleNamespace(
            multi_hand_landmarks=imagem_pb,
            multi_handedness=lados_pb,
            multi_hand_world_landmarks=mundo_pb,
        )

    def desenhar(
        self,
        quadro_bgr: np.ndarray,
        resultado: Any,
        estilo: dict[str, Any] | None = None,
    ) -> None:
        """Desenha as mãos. O parâmetro estilo controla apenas a aparência."""
        estilo = estilo or {}

        listas = getattr(resultado, "multi_hand_landmarks", None)
        lateralidades = getattr(resultado, "multi_handedness", None)

        if not listas or not lateralidades:
            return

        altura, largura = quadro_bgr.shape[:2]

        espessura = max(1, min(8, int(estilo.get("maos_espessura_linha", 2))))
        raio = max(1, min(8, int(estilo.get("maos_tamanho_ponto", 3))))
        cor_texto = _cor_bgr(
            estilo.get("maos_cor_texto", "#FFFFFF"),
            (255, 255, 255),
        )
        espessura_texto = max(1, min(8, int(estilo.get("maos_espessura_texto", 2))))

        # O mesmo lado ("esq"/"dir") que a IA usa para cada mão.
        lado_do_indice = lado_de_cada_indice(resultado)
        for indice, landmarks in enumerate(listas):
            lado = lado_do_indice.get(indice)
            if lado is None:
                continue
            sufixo = "esquerda" if lado == "esq" else "direita"

            cor_linha = _cor_bgr(
                estilo.get(f"maos_cor_linha_{sufixo}"),
                config.CORES_MAOS_BGR.get(lado, (0, 255, 0)),
            )
            cor_ponto = _cor_bgr(
                estilo.get(f"maos_cor_ponto_{sufixo}"),
                cor_linha,
            )

            self._desenho.draw_landmarks(
                quadro_bgr,
                landmarks,
                self._hands_mp.HAND_CONNECTIONS,
                self._desenho.DrawingSpec(
                    color=cor_ponto,
                    thickness=espessura,
                    circle_radius=raio,
                ),
                self._desenho.DrawingSpec(
                    color=cor_linha,
                    thickness=espessura,
                ),
            )

            pulso = landmarks.landmark[0]
            x = int(pulso.x * largura)
            y = max(25, int(pulso.y * altura) - 20)

            cv2.putText(
                quadro_bgr,
                NOMES_LADO.get(lado, str(lado).capitalize()),
                (x - 25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cor_texto,
                espessura_texto,
                cv2.LINE_AA,
            )

    def fechar(self) -> None:
        """Libera os recursos do MediaPipe."""
        hands = getattr(self, "_hands", None)
        if hands is not None:
            hands.close()
            self._hands = None
            logger.info("MediaPipe Hands encerrado.")
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None
            logger.info("Hand Landmarker encerrado.")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_excecao) -> None:
        self.fechar()


def tem_maos(resultado: Any) -> bool:
    """Retorna True quando o resultado contém ao menos uma mão."""
    return bool(getattr(resultado, "multi_hand_landmarks", None))
