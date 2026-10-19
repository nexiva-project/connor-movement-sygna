"""Testes da máscara no treino (mão sumindo nas cópias das gravações)."""

import numpy as np

from kara_sygna import config
from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.ia.mascara import apagar_maos, congelar, copias_mascaradas
from kara_sygna.ia.sequencia import montar_amostras
from kara_sygna.visao.quadro import IDX_FORMA, TOTAL_FEATURES_QUADRO, maos_presentes


def _gravacao(n: int = 100) -> np.ndarray:
    quadros = np.zeros((n, TOTAL_FEATURES_QUADRO))
    quadros[:, IDX_FORMA["esq"]] = 0.5
    quadros[:, IDX_FORMA["dir"]] = 0.3
    return quadros


def test_apagar_cobre_a_fracao_pedida():
    apagados = apagar_maos(_gravacao(), 0.3, np.random.default_rng(1))
    sem_mao = ~maos_presentes(apagados)
    assert 0.3 <= sem_mao[:, 0].mean() <= 0.5
    assert 0.3 <= sem_mao[:, 1].mean() <= 0.5


def test_congelar_cobre_so_buracos_curtos():
    quadros = _gravacao(40)
    quadros[10:30, IDX_FORMA["esq"]] = 0.0  # 20 quadros sem a mão esquerda
    congelados = maos_presentes(congelar(quadros, limite=9))[:, 0]
    assert congelados[10:19].all()  # os 9 primeiros repetem a última mão
    assert not congelados[19:30].any()  # o resto continua sem mão


def test_copias_sao_iguais_com_a_mesma_semente():
    fracoes = (0.15, 0.3, 0.5)
    a = copias_mascaradas(_gravacao(), semente=7, fracoes=fracoes)
    b = copias_mascaradas(_gravacao(), semente=7, fracoes=fracoes)
    assert len(a) == len(fracoes)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_mascara_vem_desligada():
    # Desligada depois dos testes de 19/10/2026 (ver config.MASCARA_FRACOES).
    assert config.MASCARA_FRACOES == ()
    dados = ConjuntoDados(
        X=_gravacao(), y=np.array(["carro"] * 100), grupos=np.array([1] * 100)
    )
    assert not montar_amostras(dados).mascarada.any()


def test_montar_amostras_inclui_as_copias():
    dados = ConjuntoDados(
        X=_gravacao(), y=np.array(["carro"] * 100), grupos=np.array([1] * 100)
    )
    sem = montar_amostras(dados, mascara=())
    com = montar_amostras(dados, mascara=(0.3,))
    assert len(com) > len(sem)
    assert set(com.grupos.tolist()) == {1}
    assert not sem.mascarada.any()
    assert com.mascarada.any() and not com.mascarada.all()
