"""Kara Sygna — FaceMesh para detecção e extração facial."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Self

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PONTOS_ROSTO = 468
COORDENADAS_POR_PONTO = 3
FEATURES_ROSTO = PONTOS_ROSTO * COORDENADAS_POR_PONTO

INDICE_NARIZ = 1
INDICE_LATERAL_ESQUERDA = 234
INDICE_LATERAL_DIREITA = 454

REGIOES_ROSTO: dict[str, tuple[int, ...]] = {
    "testa": (10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338),
    "olho_esquerdo": (33, 133, 159, 145, 160, 144, 158, 153, 157),
    "olho_direito": (362, 263, 386, 374, 387, 373, 385, 380, 384),
    "sobrancelha_esquerda": (46, 53, 52, 65, 55, 63, 70, 105),
    "sobrancelha_direita": (276, 283, 282, 295, 285, 293, 300, 334),
    "nariz": (1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 326, 327, 328, 358, 457),
    "boca": (
        0,
        13,
        14,
        17,
        37,
        39,
        40,
        61,
        78,
        80,
        81,
        82,
        84,
        87,
        88,
        91,
        146,
        178,
        181,
        185,
        191,
        267,
        269,
        270,
        291,
        308,
        310,
        311,
        312,
        314,
        317,
        318,
        321,
        375,
        402,
        405,
        409,
    ),
    "queixo": (152,),
}


def _cor_bgr(valor: Any, padrao: tuple[int, int, int]) -> tuple[int, int, int]:
    """Converte hexadecimal #RRGGBB para BGR."""
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


@dataclass(frozen=True)
class ResultadoRosto:
    """Resultado de uma detecção facial."""

    landmarks: Any | None
    vetor: np.ndarray
    presente: bool

    @property
    def quantidade_pontos(self) -> int:
        if not self.presente or self.landmarks is None:
            return 0
        return len(self.landmarks.landmark)


class DetectorRosto:
    """Detector FaceMesh com extração de características."""

    def __init__(
        self,
        max_rostos: int = 1,
        confianca_deteccao: float = 0.6,
        confianca_rastreio: float = 0.5,
        desenhar_malha: bool = True,
    ) -> None:
        if max_rostos < 1:
            raise ValueError("max_rostos precisa ser pelo menos 1.")
        if not 0.0 <= confianca_deteccao <= 1.0:
            raise ValueError("confianca_deteccao deve estar entre 0 e 1.")
        if not 0.0 <= confianca_rastreio <= 1.0:
            raise ValueError("confianca_rastreio deve estar entre 0 e 1.")

        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Esta versão do MediaPipe não oferece mp.solutions. "
                "Use mediapipe==0.10.21 com uma versão compatível do Python."
            )

        self._mp = mp
        self._desenho = mp.solutions.drawing_utils
        self._conexoes = mp.solutions.face_mesh
        self._desenhar_malha = desenhar_malha
        self._fechado = False

        self._face_mesh = self._conexoes.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_rostos,
            refine_landmarks=False,
            min_detection_confidence=confianca_deteccao,
            min_tracking_confidence=confianca_rastreio,
        )

        logger.info("FaceMesh inicializado: %d pontos por rosto.", PONTOS_ROSTO)

    @staticmethod
    def vetor_zeros() -> np.ndarray:
        """Cria um vetor facial vazio de tamanho fixo."""
        return np.zeros(FEATURES_ROSTO, dtype=np.float32)

    @staticmethod
    def _coordenadas_normalizadas(
        landmarks: Any,
        largura: int,
        altura: int,
    ) -> np.ndarray | None:
        """Normaliza os pontos em relação ao nariz e à largura do rosto."""
        pontos = getattr(landmarks, "landmark", None)
        if pontos is None or len(pontos) != PONTOS_ROSTO:
            return None

        coordenadas = np.asarray(
            [(p.x * largura, p.y * altura, p.z * largura) for p in pontos],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(coordenadas)):
            return None

        origem = coordenadas[INDICE_NARIZ].copy()
        distancia = np.linalg.norm(
            coordenadas[INDICE_LATERAL_DIREITA] - coordenadas[INDICE_LATERAL_ESQUERDA]
        )

        if not np.isfinite(distancia) or distancia < 1e-6:
            return None

        normalizadas = (coordenadas - origem) / distancia
        if not np.all(np.isfinite(normalizadas)):
            return None

        return normalizadas.astype(np.float32)

    def extrair_vetor(
        self,
        landmarks: Any | None,
        largura: int,
        altura: int,
    ) -> np.ndarray:
        """Extrai 1.404 valores faciais normalizados."""
        if landmarks is None or largura <= 0 or altura <= 0:
            return self.vetor_zeros()

        coordenadas = self._coordenadas_normalizadas(landmarks, largura, altura)
        if coordenadas is None:
            return self.vetor_zeros()

        vetor = coordenadas.reshape(FEATURES_ROSTO)
        if vetor.size != FEATURES_ROSTO:
            return self.vetor_zeros()

        return vetor

    def detectar(self, quadro_bgr: np.ndarray) -> ResultadoRosto:
        """Detecta o rosto e extrai as características do quadro."""
        if self._fechado:
            raise RuntimeError("O DetectorRosto já foi fechado.")

        if (
            not isinstance(quadro_bgr, np.ndarray)
            or quadro_bgr.ndim != 3
            or quadro_bgr.shape[2] != 3
            or quadro_bgr.size == 0
        ):
            raise ValueError("quadro_bgr precisa ser uma imagem BGR válida.")

        altura, largura = quadro_bgr.shape[:2]
        quadro_rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        quadro_rgb.flags.writeable = False

        resultado = self._face_mesh.process(quadro_rgb)
        rostos = getattr(resultado, "multi_face_landmarks", None) or []

        if not rostos:
            return ResultadoRosto(
                landmarks=None,
                vetor=self.vetor_zeros(),
                presente=False,
            )

        landmarks = rostos[0]
        vetor = self.extrair_vetor(landmarks, largura, altura)

        if not np.any(vetor):
            logger.debug("Rosto detectado, mas as coordenadas são inválidas.")
            return ResultadoRosto(
                landmarks=None,
                vetor=self.vetor_zeros(),
                presente=False,
            )

        return ResultadoRosto(
            landmarks=landmarks,
            vetor=vetor,
            presente=True,
        )

    def desenhar(
        self,
        quadro_bgr: np.ndarray,
        resultado: ResultadoRosto,
        desenhar_malha: bool | None = None,
        desenhar_pontos: bool = True,
        desenhar_rotulos: bool = True,
        estilo: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Desenha o FaceMesh com estilo personalizável."""
        tela = quadro_bgr.copy()
        estilo = estilo or {}

        if not resultado.presente or resultado.landmarks is None:
            cv2.putText(
                tela,
                "ROSTO NAO DETECTADO",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                _cor_bgr(estilo.get("face_cor_status"), (0, 0, 255)),
                2,
                cv2.LINE_AA,
            )
            return tela

        altura, largura = tela.shape[:2]
        landmarks = resultado.landmarks
        usar_malha = self._desenhar_malha if desenhar_malha is None else desenhar_malha

        espessura_linha = max(1, min(8, int(estilo.get("face_espessura_linha", 1))))
        tamanho_ponto = max(1, min(8, int(estilo.get("face_tamanho_ponto", 1))))
        tamanho_regiao = max(1, min(8, int(estilo.get("face_tamanho_regiao", 4))))
        espessura_texto = max(1, min(8, int(estilo.get("face_espessura_texto", 1))))

        cor_malha = _cor_bgr(estilo.get("face_cor_malha"), (80, 100, 80))
        cor_contorno = _cor_bgr(estilo.get("face_cor_contorno"), (0, 220, 255))
        cor_ponto = _cor_bgr(estilo.get("face_cor_ponto"), (0, 0, 255))
        cor_regiao_ponto = _cor_bgr(estilo.get("face_cor_regiao_ponto"), (255, 255, 0))
        cor_regiao_texto = _cor_bgr(estilo.get("face_cor_regiao_texto"), (255, 255, 0))
        cor_status = _cor_bgr(estilo.get("face_cor_status"), (0, 255, 0))

        if usar_malha:
            self._desenho.draw_landmarks(
                image=tela,
                landmark_list=landmarks,
                connections=self._conexoes.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._desenho.DrawingSpec(
                    color=cor_malha,
                    thickness=espessura_linha,
                    circle_radius=0,
                ),
            )
            self._desenho.draw_landmarks(
                image=tela,
                landmark_list=landmarks,
                connections=self._conexoes.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._desenho.DrawingSpec(
                    color=cor_contorno,
                    thickness=espessura_linha,
                    circle_radius=0,
                ),
            )

        if desenhar_pontos:
            for ponto in landmarks.landmark:
                x = int(ponto.x * largura)
                y = int(ponto.y * altura)
                if 0 <= x < largura and 0 <= y < altura:
                    cv2.circle(tela, (x, y), tamanho_ponto, cor_ponto, -1)

        if desenhar_rotulos:
            indices = {
                INDICE_NARIZ: "NARIZ",
                INDICE_LATERAL_ESQUERDA: "LADO A",
                INDICE_LATERAL_DIREITA: "LADO B",
                10: "TESTA",
                13: "BOCA",
                152: "QUEIXO",
            }

            for indice, rotulo in indices.items():
                ponto = landmarks.landmark[indice]
                x = int(ponto.x * largura)
                y = int(ponto.y * altura)

                if not (0 <= x < largura and 0 <= y < altura):
                    continue

                cv2.circle(tela, (x, y), tamanho_regiao, cor_regiao_ponto, -1)
                cv2.putText(
                    tela,
                    f"{rotulo} {indice}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    cor_regiao_texto,
                    espessura_texto,
                    cv2.LINE_AA,
                )

        cv2.putText(
            tela,
            f"FACEMESH: {PONTOS_ROSTO} PONTOS",
            (15, altura - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            cor_status,
            2,
            cv2.LINE_AA,
        )
        return tela

    @staticmethod
    def regiao_mais_proxima(
        x: float,
        y: float,
        landmarks: Any | None,
        largura: int,
        altura: int,
    ) -> tuple[str, float] | None:
        """Retorna a região facial mais próxima do ponto informado."""
        if landmarks is None or largura <= 0 or altura <= 0:
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None

        px, py = x * largura, y * altura
        melhor: tuple[str, float] | None = None

        for nome, indices in REGIOES_ROSTO.items():
            pontos_regiao = []
            for indice in indices:
                if indice >= len(landmarks.landmark):
                    continue
                ponto = landmarks.landmark[indice]
                pontos_regiao.append((ponto.x * largura, ponto.y * altura))

            if not pontos_regiao:
                continue

            centro = np.mean(np.asarray(pontos_regiao), axis=0)
            distancia = float(np.linalg.norm(centro - (px, py)))

            if melhor is None or distancia < melhor[1]:
                melhor = (nome, distancia)

        return melhor

    def fechar(self) -> None:
        """Libera os recursos internos do MediaPipe."""
        if self._fechado:
            return
        self._face_mesh.close()
        self._fechado = True
        logger.info("FaceMesh encerrado.")

    def __enter__(self) -> Self:
        if self._fechado:
            raise RuntimeError("O DetectorRosto já foi fechado.")
        return self

    def __exit__(self, *_args: object) -> None:
        self.fechar()


__all__ = [
    "COORDENADAS_POR_PONTO",
    "FEATURES_ROSTO",
    "INDICE_LATERAL_DIREITA",
    "INDICE_LATERAL_ESQUERDA",
    "INDICE_NARIZ",
    "PONTOS_ROSTO",
    "REGIOES_ROSTO",
    "DetectorRosto",
    "ResultadoRosto",
]
