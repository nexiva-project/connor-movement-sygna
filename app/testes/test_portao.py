"""Testes do portão de movimento (sem câmera)."""

import numpy as np

from kara_sygna.ia.portao import PortaoMovimento
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    TOTAL_FEATURES_QUADRO,
)


def _quadro(x_pulso: float, corpo: bool = True) -> np.ndarray:
    vetor = np.zeros(TOTAL_FEATURES_QUADRO)
    vetor[COLUNA_CORPO_PRESENTE] = 1.0 if corpo else 0.0
    vetor[COLUNAS_PULSO_BRACO["dir"][0]] = x_pulso
    return vetor


def _rodar(portao, posicoes, qps=30, corpo=True):
    return [
        portao.atualizar(_quadro(x, corpo), i / qps) for i, x in enumerate(posicoes)
    ]


def test_mao_parada_fecha_o_portao():
    portao = PortaoMovimento(limite=0.8, janela_s=0.5)
    assert not any(_rodar(portao, [0.0] * 60)[10:])


def test_mao_em_movimento_abre_o_portao():
    portao = PortaoMovimento(limite=0.8, janela_s=0.5)
    # 1,5 larguras de ombro por segundo, a 30 quadros por segundo.
    abertos = _rodar(portao, [1.5 * i / 30 for i in range(60)])
    assert all(abertos[10:])


def test_portao_fecha_depois_da_janela():
    portao = PortaoMovimento(limite=0.8, janela_s=0.5)
    movendo = [1.5 * i / 30 for i in range(30)]
    parado = [movendo[-1]] * 60
    abertos = _rodar(portao, movendo + parado)
    assert abertos[35]  # logo depois de parar, ainda aberto (janela de 0,5 s)
    assert not abertos[-1]  # 2 s depois de parar, fechado


def test_sem_corpo_nao_bloqueia():
    portao = PortaoMovimento(limite=0.8, janela_s=0.5)
    assert all(_rodar(portao, [0.0] * 30, corpo=False))


def test_limite_zero_desliga():
    portao = PortaoMovimento(limite=0, janela_s=0.5)
    assert not portao.ligado
    assert all(_rodar(portao, [0.0] * 30))
