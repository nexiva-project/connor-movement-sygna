"""Testes das regras da aplicação (sem Tk, câmera ou MediaPipe)."""

import os

import numpy as np
import pytest

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais, Modo, Nivel
from kara_sygna.aplicacao.erros import (
    ModeloAusenteError,
    ModeloDesatualizadoError,
    PalavraInvalidaError,
    SemDadosError,
)
from kara_sygna.aplicacao.previsualizacao import Reproducao, completar_com_espelho
from kara_sygna.dados.repositorio import RepositorioSinais
from kara_sygna.visao.caracteristicas import IDX_DIR, IDX_ESQ
from testes.conftest import quadro_falso, vetor_de_duas_maos, vetor_de_uma_mao
from testes.test_reconhecimento import ModeloFalso


def _quadro(rng, uma_mao=False):
    """Vetor de quadro (214 valores) com uma ou duas mãos."""
    return quadro_falso(vetor_de_uma_mao(rng) if uma_mao else vetor_de_duas_maos(rng))


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    # A maioria dos testes usa todos os quadros; os modos de escolha de
    # quadros têm testes próprios.
    monkeypatch.setattr(config, "MODO_QUADROS", "fixo")
    monkeypatch.setattr(config, "USAR_1_A_CADA_QUADROS", 1)
    return ControladorSinais(
        RepositorioSinais(tmp_path / "d.csv"),
        tmp_path / "m.pkl",
        carregador_modelo=lambda _caminho: ModeloFalso([0.9, 0.05, 0.05]),
    )


def _preparar_modelo(ctrl, rng):
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    ctrl.caminho_modelo.write_text("modelo")
    os.utime(
        ctrl.caminho_modelo, (os.path.getmtime(ctrl.repositorio.caminho) + 10,) * 2
    )


def test_palavra_vazia_e_recusada(ctrl):
    with pytest.raises(PalavraInvalidaError):
        ctrl.iniciar_ensino("   ")
    assert ctrl.modo is Modo.PARADO


def test_ensino_grava_os_quadros_em_sequencia(ctrl, rng):
    ctrl.iniciar_ensino("  OI ")
    r1 = ctrl.processar_quadro(_quadro(rng, uma_mao=True))
    r2 = ctrl.processar_quadro(_quadro(rng))
    r3 = ctrl.processar_quadro(quadro_falso(None))  # sem mão no meio do sinal
    ctrl.processar_quadro(_quadro(rng))
    assert ctrl.total_gravado == 4 and r1.nivel is Nivel.ENSINO
    assert r2.texto_video == "ENSINANDO: oi (2)"
    assert "Mostre as mãos" in r3.status and "Mostre as mãos" not in r2.status
    assert ctrl.parar().startswith("Ensino de “oi” parado: 4")
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 4}
    assert ctrl.repositorio.carregar().X.shape[0] == 4


def test_ensino_nao_grava_o_comeco_e_o_fim_sem_mao(ctrl, rng):
    ctrl.iniciar_ensino("oi")
    for _ in range(5):
        ctrl.processar_quadro(quadro_falso(None))  # antes de levantar a mão
    ctrl.processar_quadro(_quadro(rng))
    ctrl.processar_quadro(_quadro(rng))
    for _ in range(5):
        ctrl.processar_quadro(quadro_falso(None))  # depois de baixar a mão
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 2}
    assert ctrl.repositorio.carregar().X.shape[0] == 2


def test_ensino_usa_1_a_cada_n_quadros(ctrl, rng, monkeypatch):
    monkeypatch.setattr(config, "USAR_1_A_CADA_QUADROS", 2)  # modo "fixo"
    ctrl.iniciar_ensino("oi")
    for _ in range(10):
        ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 5}


def test_reconhecimento_usa_1_a_cada_n_quadros_sem_repetir_confirmacao(
    ctrl, rng, monkeypatch
):
    monkeypatch.setattr(config, "USAR_1_A_CADA_QUADROS", 2)  # modo "fixo"
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    vetor = _quadro(rng)
    total = 2 * (config.QUADROS_POR_JANELA + 10)
    respostas = [ctrl.processar_quadro(vetor) for _ in range(total)]
    assert [r.sinal_confirmado for r in respostas].count("a") == 1
    # o quadro pulado repete o que está na tela
    assert respostas[-1].texto_video == respostas[-2].texto_video


def test_ensino_grava_em_lotes_e_o_resto_ao_parar(ctrl, rng):
    lote = config.QUADROS_POR_LOTE_GRAVACAO
    ctrl.iniciar_ensino("oi")
    for _ in range(lote + 2):
        ctrl.processar_quadro(_quadro(rng))
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": lote}
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": lote + 2}


def test_quadro_que_nao_pode_ser_analisado_nao_grava(ctrl):
    ctrl.iniciar_ensino("oi")
    assert ctrl.processar_quadro(None).texto_video == ""
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {}


def test_modo_parado_ignora_quadros(ctrl, rng):
    assert ctrl.processar_quadro(_quadro(rng)).texto_video == ""


def test_rodadas_diferentes_a_cada_ensino(ctrl, rng, monkeypatch):
    ids = iter([111, 222])
    monkeypatch.setattr(
        "kara_sygna.aplicacao.controlador.novo_id_rodada", lambda: next(ids)
    )
    for _ in range(2):
        ctrl.iniciar_ensino("a")
        ctrl.processar_quadro(_quadro(rng))
        ctrl.parar()
    assert sorted(set(ctrl.repositorio.carregar().grupos.tolist())) == [111, 222]


def test_reconhecer_exige_dados_modelo_e_treino_atual(ctrl, rng):
    with pytest.raises(SemDadosError):
        ctrl.iniciar_reconhecimento()
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    with pytest.raises(ModeloAusenteError):
        ctrl.iniciar_reconhecimento()
    ctrl.caminho_modelo.write_text("x")
    os.utime(ctrl.caminho_modelo, (1, 1))
    with pytest.raises(ModeloDesatualizadoError):
        ctrl.iniciar_reconhecimento()


def test_reconhecimento_espera_a_janela_encher(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    vetor = _quadro(rng)
    respostas = [ctrl.processar_quadro(vetor) for _ in range(config.QUADROS_POR_JANELA)]
    assert all(r.texto_video == "PREPARANDO..." for r in respostas[:-1])
    assert respostas[-1].texto_video.startswith("SINAL: a")
    assert ctrl._modelo.entrada.shape[0] == 1  # uma previsão por quadro, sem rotações


def test_reconhecimento_confirma_sinal_e_ignora_neutro(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    vetor = _quadro(rng)
    total = config.QUADROS_POR_JANELA + 10
    confirmados = [ctrl.processar_quadro(vetor).sinal_confirmado for _ in range(total)]
    assert confirmados.count("a") == 1

    ctrl.limpar_reconhecimento()
    ctrl._modelo.classes_ = np.array(["neutro", "b", "c"])
    assert all(ctrl.processar_quadro(vetor).sinal_confirmado is None for _ in range(10))


def test_reconhecimento_sem_mao_e_incerto(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    assert ctrl.processar_quadro(None).texto_video == "MOSTRE UM SINAL"
    assert ctrl.processar_quadro(quadro_falso(None)).texto_video == "MOSTRE UM SINAL"
    ctrl._modelo.probabilidades = np.array([0.4, 0.3, 0.3])
    for _ in range(config.QUADROS_POR_JANELA):
        r = ctrl.processar_quadro(_quadro(rng))
    assert r.nivel is Nivel.INCERTO and r.texto_video.startswith("INCERTO")


def test_iniciar_reconhecimento_esvazia_a_janela(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    for _ in range(config.QUADROS_POR_JANELA):
        ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    ctrl.iniciar_reconhecimento()
    assert ctrl.processar_quadro(_quadro(rng)).texto_video == "PREPARANDO..."


def test_texto_do_video_e_so_ascii(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    ctrl._modelo = type(
        "M",
        (),
        {
            "classes_": np.array(["a", "b", "c"]),
            "predict_proba": lambda self, X: np.tile([0.9, 0.05, 0.05], (len(X), 1)),
        },
    )()
    for _ in range(config.QUADROS_POR_JANELA):
        resposta = ctrl.processar_quadro(_quadro(rng))
    assert resposta.texto_video.startswith("SINAL") and resposta.texto_video.isascii()


def test_listar_sinais_marca_treinados_so_se_modelo_esta_atualizado(
    ctrl, rng, monkeypatch
):
    _preparar_modelo(ctrl, rng)
    monkeypatch.setattr(
        "kara_sygna.aplicacao.controlador.classes_do_modelo", lambda _c: {"a"}
    )
    assert [(s.palavra, s.treinado) for s in ctrl.listar_sinais()] == [("a", True)]
    os.utime(ctrl.caminho_modelo, (1, 1))
    assert [s.treinado for s in ctrl.listar_sinais()] == [False]


def test_apagar_sinal_durante_o_ensino_grava_antes_e_para(ctrl, rng):
    ctrl.iniciar_ensino("b")
    ctrl.processar_quadro(_quadro(rng))
    ctrl.apagar_sinal("b")
    assert ctrl.modo is Modo.PARADO and ctrl.repositorio.contagem_por_sinal() == {}


def test_apagar_sinal_remove_modelo_antigo(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    mensagem = ctrl.apagar_sinal("a")
    assert (
        "apagado" in mensagem
        and not ctrl.caminho_modelo.exists()
        and ctrl.listar_sinais() == []
    )


def test_treinar_sem_dados(ctrl):
    with pytest.raises(SemDadosError):
        ctrl.treinar()


def test_criar_reproducao(ctrl, rng):
    with pytest.raises(SemDadosError):
        ctrl.repositorio.garantir_arquivo() or ctrl.criar_reproducao("x")
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(_quadro(rng, uma_mao=True))
    ctrl.processar_quadro(quadro_falso(None))  # sem mão: fica fora da reprodução
    ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    reproducao = ctrl.criar_reproducao("a")
    assert len(reproducao.vetores) == 2


def test_completar_com_espelho_so_muda_quadros_de_uma_mao(rng):
    um = vetor_de_uma_mao(rng, "dir")
    duas = vetor_de_duas_maos(rng)
    resultado = completar_com_espelho([um, duas, np.zeros(126)])
    assert len(resultado) == 3
    assert np.array_equal(resultado[0][IDX_DIR], um[IDX_DIR])
    espelho = resultado[0][IDX_ESQ].reshape(21, 3)
    assert np.allclose(espelho[:, 0], -um[IDX_DIR].reshape(21, 3)[:, 0])
    assert np.array_equal(resultado[1], duas)
    assert np.array_equal(resultado[2], np.zeros(126))


def test_reproducao_avanca_em_ciclo_e_gira():
    r = Reproducao("oi", [np.zeros(126), np.ones(126)])
    assert r.legenda == "oi - exemplo 1/2" and r.legenda.isascii()
    r.avancar()
    r.avancar()
    assert r.indice == 0 and r.angulo > 0
    with pytest.raises(ValueError):
        Reproducao("x", [])


def test_modo_mudanca_nao_grava_mao_parada(ctrl, rng, monkeypatch):
    monkeypatch.setattr(config, "MODO_QUADROS", "mudanca")
    monkeypatch.setattr(config, "PARADO_USAR_A_CADA_S", 0)
    parado = _quadro(rng)
    ctrl.iniciar_ensino("oi")
    for _ in range(20):
        ctrl.processar_quadro(parado)
    for _ in range(3):
        ctrl.processar_quadro(_quadro(rng))  # mudou: grava
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 4}


def test_modo_mudanca_confirma_sinal_parado(ctrl, rng, monkeypatch):
    """Com a mão parada a janela quase não recebe quadros, mas cada quadro vota."""
    monkeypatch.setattr(config, "MODO_QUADROS", "mudanca")
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    for _ in range(config.QUADROS_POR_JANELA):
        ctrl.processar_quadro(_quadro(rng))  # quadros diferentes enchem a janela
    parado = _quadro(rng)
    confirmados = [ctrl.processar_quadro(parado).sinal_confirmado for _ in range(10)]
    assert confirmados.count("a") == 1
