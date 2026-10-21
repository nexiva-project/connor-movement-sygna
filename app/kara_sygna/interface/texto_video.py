"""Texto com acentos sobre a imagem da câmera.

O ``cv2.putText`` só conhece letras sem acento: "avião" saía "avi??o". Aqui o
texto é desenhado com uma fonte TrueType (Pillow), só no pedaço da imagem onde
ele fica, para não pesar a cada quadro.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#: Fontes tentadas, em ordem (Windows, Linux, macOS).
FONTES = ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf")


@lru_cache(maxsize=8)
def _fonte(tamanho: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for nome in FONTES:
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default(tamanho)


def escrever(
    quadro: np.ndarray,
    texto: str,
    origem: tuple[int, int],
    cor_bgr: tuple[int, int, int],
    tamanho: int = 22,
) -> None:
    """Escreve ``texto`` no quadro BGR, com a linha de base em ``origem``.

    Texto sem acento vai direto pelo OpenCV (mais rápido); com acento, pela
    fonte TrueType.
    """
    if texto.isascii():
        escala = tamanho / 34
        cv2.putText(
            quadro,
            texto,
            origem,
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            cor_bgr,
            2,
            cv2.LINE_AA,
        )
        return

    fonte = _fonte(tamanho)
    x, y = origem
    esquerda, topo, direita, baixo = fonte.getbbox(texto, anchor="ls")
    x0, y0 = max(0, x + esquerda - 2), max(0, y + topo - 2)
    x1 = min(quadro.shape[1], x + direita + 2)
    y1 = min(quadro.shape[0], y + baixo + 2)
    if x1 <= x0 or y1 <= y0:
        return
    pedaco = Image.fromarray(cv2.cvtColor(quadro[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pedaco).text(
        (x - x0, y - y0), texto, font=fonte, fill=cor_bgr[::-1], anchor="ls"
    )
    quadro[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(pedaco), cv2.COLOR_RGB2BGR)
