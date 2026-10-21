"""Sinais de direção livre (ex.: "avião"): o movimento vale para qualquer lado."""

import numpy as np

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais
from kara_sygna.dados.repositorio import ConjuntoDados, RepositorioSinais
from kara_sygna.ia.sequencia import girar_movimento, montar_amostras
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    IDX_FORMA,
    TOTAL_FEATURES_QUADRO,
)

PULSO = list(COLUNAS_PULSO_BRACO["dir"])


def _voo_para_a_direita(n=20):
    """Mão direita em Y andando para a direita (x cresce), com corpo."""
    quadros = np.zeros((n, TOTAL_FEATURES_QUADRO))
    quadros[:, COLUNA_CORPO_PRESENTE] = 1.0
    quadros[:, IDX_FORMA["dir"]] = 0.1
    quadros[:, PULSO[0]] = np.linspace(-0.5, 0.5, n)
    quadros[:, PULSO[1]] = 0.2
    return quadros


def test_girar_90_graus_leva_o_movimento_para_cima():
    girado = girar_movimento(_voo_para_a_direita(), 90)
    deslocamento = girado[-1, PULSO] - girado[0, PULSO]
    # Na imagem, y cresce para baixo: "para cima" é y diminuindo.
    assert np.allclose(deslocamento, [0.0, -1.0], atol=1e-9)
    # O ponto médio e a forma da mão não mudam.
    assert np.allclose(girado[:, PULSO].mean(axis=0), [0.0, 0.2])
    assert np.array_equal(
        girado[:, IDX_FORMA["dir"]], _voo_para_a_direita()[:, IDX_FORMA["dir"]]
    )


def test_girar_nao_mexe_em_quadros_sem_corpo():
    quadros = _voo_para_a_direita()
    quadros[:5, COLUNA_CORPO_PRESENTE] = 0.0
    girado = girar_movimento(quadros, 180)
    assert np.array_equal(girado[:5], quadros[:5])


def test_sinal_de_direcao_livre_entra_girado_no_treino():
    quadros = _voo_para_a_direita()
    dados = ConjuntoDados(quadros, np.array(["avião"] * 20), np.array([1] * 20))
    normal = montar_amostras(dados, tamanho=10, passo=2, velocidades=(), mascara=())
    livre = montar_amostras(
        dados, tamanho=10, passo=2, velocidades=(), mascara=(), direcao_livre={"avião"}
    )
    copias = len(config.ANGULOS_DIRECAO_LIVRE)
    assert len(livre) == len(normal) * (1 + copias)
    assert livre.mascarada.sum() == len(normal) * copias  # cópias fora do teste
    assert set(livre.grupos.tolist()) == {1}


def test_marcar_direcao_livre_fica_guardado_e_pede_treino(tmp_path):
    repo = RepositorioSinais(tmp_path / "dados.csv")
    repo.anexar_quadros("avião", 1, [(0, np.zeros(TOTAL_FEATURES_QUADRO))])
    repo.marcar_direcao_livre("avião", True)
    assert RepositorioSinais(tmp_path / "dados.csv").sinais_direcao_livre() == {"avião"}
    repo.apagar_sinal("avião")
    assert repo.sinais_direcao_livre() == set()


def test_controlador_mostra_a_direcao_livre_na_lista(tmp_path):
    controlador = ControladorSinais(
        repositorio=RepositorioSinais(tmp_path / "dados.csv"),
        caminho_modelo=tmp_path / "modelo.pkl",
    )
    controlador.repositorio.anexar_quadros(
        "avião", 1, [(0, np.zeros(TOTAL_FEATURES_QUADRO))]
    )
    controlador.marcar_direcao_livre("Avião", True)
    (info,) = controlador.listar_sinais()
    assert info.direcao_livre and not info.treinado
