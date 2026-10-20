"""Cores e fontes da interface (um só lugar para mudar o visual)."""

from __future__ import annotations

from kara_sygna.aplicacao.controlador import Nivel

FUNDO = "#181a20"
FUNDO_PAINEL = "#20232b"
FUNDO_VIDEO = "#101116"
TEXTO = "#ffffff"
TEXTO_SUAVE = "#dddddd"
TEXTO_APAGADO = "#777777"
PERIGO = "#9b1c1c"

FONTE_TITULO = ("Arial", 20, "bold")
FONTE_CAIXA = ("Arial", 10, "bold")
FONTE_TEXTO_SINAIS = ("Arial", 18)
FONTE_LOG = ("DejaVu Sans Mono", 10)

#: Cor (BGR, para o OpenCV) do texto escrito sobre o vídeo, por nível.
COR_VIDEO_BGR = {
    Nivel.NEUTRO: (220, 220, 220),
    Nivel.ENSINO: (0, 220, 255),
    Nivel.OK: (0, 255, 0),
    Nivel.INCERTO: (0, 165, 255),
}

#: Cor do texto na janela de log, por categoria ou nível.
COR_LOG = {
    "APP": "#222222",
    "CAMERA": "#1769aa",
    "VISION": "#00796b",
    "DATA": "#2e7d32",
    "MODEL": "#7b1fa2",
    "WARNING": "#b36b00",
    "ERROR": "#cc2222",
}
