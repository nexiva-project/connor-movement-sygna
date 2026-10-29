"""Anti-pulo: segura no lugar um dedo que pula sozinho na malha da mão.

Por que existe
--------------
Com a mão borrando no movimento, na frente do rosto ou contra uma janela
clara, o MediaPipe às vezes erra um dedo só por um quadro: na tela, um dedo
da malha "mexe sozinho" com o dedo de verdade parado. No vídeo
``defeito_malha`` (24/10/2026) foram 11 pulos em 602 quadros (polegar,
anelar, mindinho...).

Como funciona
-------------
Para cada mão (pelo lado, esquerda ou direita), compara a forma com a do
quadro anterior, em tamanhos de palma (não depende da distância da câmera):

* se **um dedo** andou muito (mais de ``config.ANTI_PULO_LIMIAR``) e **o
  resto da mão** ficou quase parado, é um pulo: o dedo volta para onde
  estava, preso na palma de agora;
* um dedo que continua "diferente" por mais de ``config.ANTI_PULO_MAX_QUADROS``
  quadros seguidos é movimento de verdade e passa.

Vale para os pontos da imagem (o desenho) e para os do mundo (a forma que a
IA recebe). ``config.ANTI_PULO_LIMIAR = 0`` desliga.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np

from kara_sygna import config
from kara_sygna.visao.caracteristicas import lado_de_cada_indice

#: Pontos de cada dedo no MediaPipe (polegar, indicador, médio, anelar, mindinho).
DEDOS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
#: Pulso e base dos dedos: a "palma", que tem de ficar parada.
PALMA = (0, 1, 5, 9, 13, 17)
#: Quanto o resto da mão pode andar (em palmas) para ainda contar como parada.
RESTO_PARADO = 0.12
_CAMPOS = (
    "multi_hand_landmarks",
    "multi_handedness",
    "multi_hand_world_landmarks",
    "lados",
    "descartadas",
)


def _pontos(lista: Any) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in lista.landmark], dtype=np.float64)


def _forma(pontos: np.ndarray) -> np.ndarray:
    """Pontos em relação ao pulso, em tamanhos de palma (pulso até a base do médio)."""
    palma = np.linalg.norm(pontos[9] - pontos[0]) or 1e-6
    return (pontos - pontos[0]) / palma


def _com_pontos(lista: Any, pontos: np.ndarray) -> Any:
    """Cópia da lista de pontos do MediaPipe com as coordenadas dadas."""
    nova = copy.deepcopy(lista)
    for ponto, (x, y, z) in zip(nova.landmark, pontos, strict=True):
        ponto.x, ponto.y, ponto.z = float(x), float(y), float(z)
    return nova


class FiltroAntiPulo:
    """Segura no lugar o dedo que pulou sozinho (veja o módulo).

    Args:
        limiar: Pulo mínimo de uma ponta de dedo, em palmas. Padrão:
            ``config.ANTI_PULO_LIMIAR`` (0 = desligado).
        max_quadros: Por quantos quadros seguidos um dedo pode ser segurado.
            Padrão: ``config.ANTI_PULO_MAX_QUADROS``.
    """

    def __init__(
        self, limiar: float | None = None, max_quadros: int | None = None
    ) -> None:
        """Cria o filtro sem histórico (veja a classe)."""
        self.limiar = config.ANTI_PULO_LIMIAR if limiar is None else limiar
        self.max_quadros = (
            config.ANTI_PULO_MAX_QUADROS if max_quadros is None else max_quadros
        )
        # Por mão: forma anterior (imagem, mundo) e quadros segurados de cada dedo.
        self._antes: dict[Any, tuple[np.ndarray, np.ndarray | None]] = {}
        self._segurados: dict[Any, list[int]] = {}
        #: Dedos segurados no último quadro (para o diagnóstico e os testes).
        self.segurados_agora = 0
        self._proporcao = 1.0

    def aplicar(self, resultado: Any, proporcao: float = 1.0) -> Any:
        """Devolve o resultado com os dedos que pularam de volta no lugar.

        Args:
            resultado: Mãos do MediaPipe (já conferidas pelos braços).
            proporcao: Largura / altura da imagem: os pontos vêm de 0 a 1 nos
                dois eixos, e sem isso as distâncias de lado e de cima ficam
                em escalas diferentes (640x480 = 4/3).

        O resultado original não é alterado; sem pulo, ele mesmo é devolvido.
        """
        self._proporcao = proporcao
        self.segurados_agora = 0
        maos = list(getattr(resultado, "multi_hand_landmarks", None) or [])
        if self.limiar <= 0 or self.max_quadros <= 0 or not maos:
            self._antes.clear()
            self._segurados.clear()
            return resultado
        mundos = list(getattr(resultado, "multi_hand_world_landmarks", None) or [])
        # Pelo lado ("esq"/"dir"), como a IA vê; antes, sem os braços, era pela
        # ordem na lista, e quando ela trocava uma mão era comparada com a outra.
        lado_do_indice = lado_de_cada_indice(resultado)
        chaves = [lado_do_indice.get(i, i) for i in range(len(maos))]

        novas_maos, novos_mundos, mudou = list(maos), list(mundos), False
        for i, (chave, mao) in enumerate(zip(chaves, maos, strict=True)):
            if len(getattr(mao, "landmark", [])) < 21:
                continue
            escala = np.array([self._proporcao, 1.0, 1.0])
            imagem = _pontos(mao) * escala
            tem_mundo = i < len(mundos) and mundos[i] is not None
            mundo = _pontos(mundos[i]) if tem_mundo else None
            anterior = self._antes.get(chave, (None, None))
            segurados = self._segurados.setdefault(chave, [[0] * len(DEDOS)] * 2)
            # O pulo é procurado nos dois: no desenho (imagem, 2D) e nos pontos
            # 3D que a IA usa. O polegar pula sobretudo em profundidade (em
            # 24/10/2026, 38 dos 40 pulos gravados), quase invisível em 2D.
            dedo_imagem = self._dedo_que_pulou(imagem[:, :2], anterior[0], segurados[0])
            dedo_mundo = (
                None
                if mundo is None
                else self._dedo_que_pulou(mundo, anterior[1], segurados[1])
            )
            novos_segurados = []
            for espaco, dedo in enumerate((dedo_imagem, dedo_mundo)):
                contagem = [0] * len(DEDOS)
                if dedo is not None:
                    contagem[dedo] = segurados[espaco][dedo] + 1
                novos_segurados.append(contagem)
            self._segurados[chave] = novos_segurados
            if dedo_imagem is not None:
                imagem = _segurar(imagem, anterior[0], dedo_imagem)
                novas_maos[i] = _com_pontos(mao, imagem / escala)
            if dedo_mundo is not None:
                mundo = _segurar(mundo, anterior[1], dedo_mundo)
                novos_mundos[i] = _com_pontos(mundos[i], mundo)
            if dedo_imagem is not None or dedo_mundo is not None:
                self.segurados_agora += 1
                mudou = True
            self._antes[chave] = (
                _forma(imagem[:, :2]),
                None if mundo is None else _forma(mundo),
            )
        for chave in set(self._antes) - set(chaves):  # mão que saiu: esquece
            self._antes.pop(chave, None)
            self._segurados.pop(chave, None)
        if not mudou:
            return resultado
        campos = {c: getattr(resultado, c) for c in _CAMPOS if hasattr(resultado, c)}
        campos["multi_hand_landmarks"] = novas_maos
        campos["multi_hand_world_landmarks"] = novos_mundos or None
        return SimpleNamespace(**campos)

    def _dedo_que_pulou(self, pontos, forma_antes, segurados) -> int | None:
        """O dedo que pulou sozinho neste quadro (ou ``None``)."""
        if forma_antes is None:
            return None
        mov = np.linalg.norm(_forma(pontos) - forma_antes, axis=1)
        if np.median(mov[list(PALMA)]) > RESTO_PARADO:
            return None  # a mão toda mexeu
        pontas = np.array([mov[d[-1]] for d in DEDOS])
        dedo = int(pontas.argmax())
        # Só este dedo passou do limite (os outros podem tremer um pouco, como
        # em profundidade nos pontos 3D); vários juntos é a mão abrindo/fechando.
        if (pontas > self.limiar).sum() == 1 and segurados[dedo] < self.max_quadros:
            return dedo
        return None


def _segurar(pontos: np.ndarray, forma_antes: np.ndarray, dedo: int) -> np.ndarray:
    """Põe o dedo na forma anterior, preso no pulso e na palma de agora.

    ``forma_antes`` pode ter só x, y (imagem): aí o z do dedo fica como está.
    """
    dims = forma_antes.shape[1]
    pontos = pontos.copy()
    palma = np.linalg.norm(pontos[9, :dims] - pontos[0, :dims]) or 1e-6
    indices = list(DEDOS[dedo])
    pontos[indices, :dims] = pontos[0, :dims] + forma_antes[indices] * palma
    return pontos
