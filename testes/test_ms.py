"""Testes do alinhamento DTW e do MS (rodam sem câmera e sem as gravações reais)."""

import numpy as np

from lab_ms.alinhamento import alinhar, alinhar_trecho
from lab_ms.ms_knn import ClassificadorMS
from lab_ms.kara import preparar_import

preparar_import()
from kara_sygna.visao.quadro import (  # noqa: E402
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    IDX_FORMA,
    TOTAL_FEATURES_QUADRO,
)


def test_alinhar_igual_e_zero():
    a = np.random.default_rng(0).normal(size=(20, 3))
    assert alinhar(a, a) == 0.0


def test_alinhar_aceita_outra_velocidade():
    t = np.linspace(0, 2 * np.pi, 40)
    lento = np.c_[np.sin(t), np.cos(t)]
    rapido = lento[::2]
    outro = np.c_[np.sin(3 * t), np.cos(t)]
    assert alinhar(lento, rapido) < alinhar(lento, outro)


def test_subsequencia_acha_o_trecho():
    longa = np.arange(50, dtype=float)[:, None]
    custo, fim = alinhar_trecho(longa[20:30], longa)
    assert custo == 0.0
    assert fim == 29


def _gravacao(forma: float, giro: float, n: int = 40) -> np.ndarray:
    quadros = np.zeros((n, TOTAL_FEATURES_QUADRO))
    quadros[:, IDX_FORMA["esq"]] = forma
    quadros[:, COLUNA_CORPO_PRESENTE] = 1.0
    t = np.linspace(0, 2 * np.pi, n)
    quadros[:, COLUNAS_PULSO_BRACO["esq"][0]] = giro * np.sin(t)
    return quadros


def test_knn_separa_dois_sinais():
    modelo = ClassificadorMS().fit([_gravacao(0.1, 1.0), _gravacao(0.9, 0.0)], ["carro", "casa"])
    assert next(iter(modelo.distancias_gravacao(_gravacao(0.12, 1.0, n=30)))) == "carro"
    assert next(iter(modelo.distancias_janela(_gravacao(0.88, 0.0)[:10]))) == "casa"


def test_apagar_e_congelar_maos():
    from lab_ms.buracos import apagar_maos, congelar
    from kara_sygna.visao.quadro import maos_presentes

    quadros = _gravacao(0.5, 1.0, n=100)
    apagados = apagar_maos(quadros, 0.3, np.random.default_rng(1), buraco_min=20, buraco_max=20)
    sem_mao = ~maos_presentes(apagados)[:, 0]
    assert 0.3 <= sem_mao.mean() <= 0.5
    # A memória cobre 9 dos 20 quadros de cada buraco; o resto continua sem mão.
    congelados = congelar(apagados)
    assert 0 < (~maos_presentes(congelados)[:, 0]).mean() < sem_mao.mean()


def test_tabela_de_buracos_da_camera():
    from camera_teste import buracos

    assert buracos([True, False, False, True, False]) == [2, 1]
    assert buracos([True, True]) == []
