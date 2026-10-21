"""Texto sobre a câmera, com e sem acentos."""

import numpy as np

from kara_sygna.interface.texto_video import escrever


def test_texto_com_acento_e_desenhado_perto_da_origem():
    quadro = np.zeros((80, 400, 3), dtype=np.uint8)
    escrever(quadro, "SINAL: avião (99%)", (20, 42), (255, 255, 255))
    linhas, colunas = np.nonzero(quadro.any(axis=2))
    assert len(linhas) > 100  # escreveu alguma coisa
    assert linhas.max() <= 50 and colunas.min() >= 15


def test_texto_sem_acento_continua_pelo_opencv():
    quadro = np.zeros((80, 400, 3), dtype=np.uint8)
    escrever(quadro, "SINAL: carro", (20, 42), (0, 255, 0))
    assert quadro[..., 1].any() and not quadro[..., 0].any()


def test_texto_fora_da_imagem_nao_quebra():
    quadro = np.zeros((10, 10, 3), dtype=np.uint8)
    escrever(quadro, "avião", (500, 500), (255, 255, 255))
    assert not quadro.any()
