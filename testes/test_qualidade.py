"""Testes do medidor de qualidade da gravação."""

import numpy as np

from kara_sygna import config
from kara_sygna.aplicacao.qualidade import QualidadeGravacao
from testes.conftest import quadro_falso, vetor_de_duas_maos, vetor_de_uma_mao

MUITOS = 3 * config.QUADROS_POR_JANELA


def _medir(quadros, gravados=MUITOS):
    q = QualidadeGravacao()
    for quadro in quadros:
        q.adicionar(quadro)
    return q, q.avisos(gravados)


def test_gravacao_boa_de_duas_maos(rng):
    q, avisos = _medir([quadro_falso(vetor_de_duas_maos(rng)) for _ in range(20)])
    assert avisos == [] and "Boa gravação" in q.resumo(MUITOS)
    assert "2 mãos em 100%" in q.resumo(MUITOS)


def test_sinal_de_uma_mao_nao_recebe_aviso_de_segunda_mao(rng):
    _, avisos = _medir([quadro_falso(vetor_de_uma_mao(rng)) for _ in range(20)])
    assert avisos == []


def test_segunda_mao_sumindo_e_maos_fora_e_sem_rosto(rng):
    duas = [quadro_falso(vetor_de_duas_maos(rng)) for _ in range(10)]
    uma = [quadro_falso(vetor_de_uma_mao(rng)) for _ in range(10)]
    _, avisos = _medir(duas + uma)
    assert avisos == ["a segunda mão sumiu muitas vezes"]

    vazios = [quadro_falso(None, rosto=False) for _ in range(30)]
    q, avisos = _medir(duas + vazios, gravados=5)
    assert "as mãos ficaram fora da câmera muitas vezes" in avisos
    assert "o rosto não apareceu bem" in avisos
    assert "a gravação foi curta" in avisos
    assert "ATENÇÃO" in q.resumo(5)


def test_sem_quadros():
    q = QualidadeGravacao()
    assert q.avisos(0) == ["nenhuma mão foi gravada"]
    assert "nenhum quadro" in q.resumo(0)
    assert np.isfinite(q.quadros)
