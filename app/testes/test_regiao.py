"""Testes da região entre as mãos (sem câmera)."""

import numpy as np

from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.ia.regiao import RegiaoMaos, distancia_entre_maos
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    IDX_FORMA,
    TOTAL_FEATURES_QUADRO,
)


def _quadro(distancia: float, duas_maos: bool = True, corpo: bool = True):
    vetor = np.zeros(TOTAL_FEATURES_QUADRO)
    vetor[COLUNA_CORPO_PRESENTE] = 1.0 if corpo else 0.0
    vetor[COLUNAS_PULSO_BRACO["esq"][0]] = -distancia / 2
    vetor[COLUNAS_PULSO_BRACO["dir"][0]] = distancia / 2
    vetor[IDX_FORMA["dir"]] = 0.1
    if duas_maos:
        vetor[IDX_FORMA["esq"]] = 0.1
    return vetor


def _dados(distancias, sinal, duas_maos=True):
    quadros = np.array([_quadro(d, duas_maos) for d in distancias])
    n = len(quadros)
    return ConjuntoDados(quadros, np.array([sinal] * n), np.zeros(n))


def _permite(regiao, sinal, distancia):
    regiao.atualizar(_quadro(distancia))
    return regiao.permite(sinal)


def test_distancia_em_larguras_de_ombro():
    assert np.isclose(distancia_entre_maos(_quadro(1.2))[0], 1.2)


def test_aprende_o_limite_e_bloqueia_maos_longe():
    regiao = RegiaoMaos.das_gravacoes(
        _dados(np.linspace(0.8, 1.0, 50), "carro"), folga=0.4
    )
    assert 1.3 < regiao.limites["carro"] < 1.45
    assert _permite(regiao, "carro", 1.1)  # mais ou menos perto: vale
    assert _permite(regiao, "carro", 1.35)  # um pouco mais longe: ainda vale
    assert not _permite(regiao, "carro", 2.3)  # braços abertos: não vale


def test_sinal_de_uma_mao_nao_tem_limite():
    regiao = RegiaoMaos.das_gravacoes(
        _dados([0.5] * 30, "oi", duas_maos=False), folga=0.4
    )
    assert "oi" not in regiao.limites
    assert _permite(regiao, "oi", 3.0)


def test_sem_corpo_nao_bloqueia():
    regiao = RegiaoMaos({"carro": 1.0})
    regiao.atualizar(_quadro(3.0))
    regiao.atualizar(_quadro(3.0, corpo=False))
    assert regiao.permite("carro")
    assert regiao.distancia is None


def test_folga_negativa_desliga():
    regiao = RegiaoMaos.das_gravacoes(_dados([0.9] * 30, "carro"), folga=-1)
    assert regiao.limites == {}


def test_quadro_perto_solto_nao_libera_maos_longe():
    regiao = RegiaoMaos({"carro": 1.7}, memoria=15)
    for _ in range(5):
        regiao.atualizar(_quadro(2.3))
    assert not _permite(regiao, "carro", 1.5)  # longe há pouco: ainda não vale
    for _ in range(15):
        regiao.atualizar(_quadro(1.0))
    assert regiao.permite("carro")  # perto por meio segundo: vale de novo
