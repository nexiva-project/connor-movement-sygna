"""Testes do portão de movimento (sem câmera)."""

import numpy as np

from kara_sygna.ia.portao import PortaoMovimento
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    IDX_FORMA,
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


def _com_mao(x_pulso: float) -> np.ndarray:
    vetor = _quadro(x_pulso)
    vetor[IDX_FORMA["dir"]] = 0.5  # a mão direita aparece
    return vetor


def test_pulo_do_pulso_num_quadro_nao_abre():
    # Ao vivo (26/10/2026), o pulso do Pose pulou num quadro só com as mãos
    # paradas em S, e o portão abriu. Pela mediana, o pulo não conta.
    portao = PortaoMovimento(limite=0.8, janela_s=0.5)
    posicoes = [0.0] * 40 + [0.6] + [0.0] * 20
    assert not any(_rodar(portao, posicoes)[10:])


def test_subida_das_maos_nao_abre():
    # As mãos aparecem e sobem até a posição (0,25 s), depois param em S.
    portao = PortaoMovimento(limite=0.8, janela_s=0.5)
    subida = [1.5 * i / 30 for i in range(8)]
    quadros = [_com_mao(x) for x in subida + [subida[-1]] * 30]
    abertos = [portao.atualizar(q, i / 30) for i, q in enumerate(quadros)]
    assert not any(abertos)


def test_movimento_depois_da_subida_abre():
    portao = PortaoMovimento(limite=0.8, janela_s=0.5)
    quadros = [_com_mao(0.0)] * 15 + [_com_mao(1.5 * i / 30) for i in range(30)]
    abertos = [portao.atualizar(q, i / 30) for i, q in enumerate(quadros)]
    assert all(abertos[25:])


def test_sem_corpo_nao_bloqueia():
    portao = PortaoMovimento(limite=0.8, janela_s=0.5)
    assert all(_rodar(portao, [0.0] * 30, corpo=False))


def test_limite_zero_desliga():
    portao = PortaoMovimento(limite=0, janela_s=0.5)
    assert not portao.ligado
    assert all(_rodar(portao, [0.0] * 30))
