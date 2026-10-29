"""IA escolhida e redes neurais (GRU, LSTM, Conv1D); as redes só com o PyTorch."""

import numpy as np
import pytest

from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.ia import tipos_ia
from kara_sygna.ia.sequencia import (
    TOTAL_FEATURES_SEQUENCIA,
    features_sequencia,
    montar_amostras,
    sequencias_de_uma_gravacao,
)
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    IDX_FORMA,
    TOTAL_FEATURES_QUADRO,
)


def _gravacao(sinal_x: float, n=30, rng=None):
    """Mão direita parada (forma ~ sinal_x) com o pulso andando para a direita."""
    rng = rng or np.random.default_rng(0)
    quadros = np.zeros((n, TOTAL_FEATURES_QUADRO))
    quadros[:, COLUNA_CORPO_PRESENTE] = 1.0
    quadros[:, IDX_FORMA["dir"]] = sinal_x + 0.01 * rng.standard_normal((n, 63))
    quadros[:, COLUNAS_PULSO_BRACO["dir"][0]] = np.linspace(0, 1, n)
    return quadros


def test_escolha_da_ia_fica_guardada(tmp_path):
    assert tipos_ia.tipo_escolhido(tmp_path) == tipos_ia.RANDOMFOREST
    tipos_ia.escolher("gru", tmp_path)
    esperado = "gru" if tipos_ia.torch_disponivel() else tipos_ia.RANDOMFOREST
    assert tipos_ia.tipo_escolhido(tmp_path) == esperado
    assert tipos_ia.caminho_modelo("gru", tmp_path).name == "modelo_sinais_gru.pt"
    assert tipos_ia.tipo_do_caminho(tmp_path / "modelo_sinais_conv1d.pt") == "conv1d"
    assert tipos_ia.tipo_do_caminho(tmp_path / "modelo_sinais.pkl") == "randomforest"
    gru_lstm = tipos_ia.caminho_modelo("gru_lstm", tmp_path)
    assert tipos_ia.tipo_do_caminho(gru_lstm) == "gru_lstm"
    with pytest.raises(ValueError):
        tipos_ia.escolher("nada")


def test_sequencia_tem_o_caminho_do_pulso_e_nao_o_lugar():
    janela = _gravacao(0.2, n=10)
    seq = features_sequencia(janela)
    assert seq.shape == (10, TOTAL_FEATURES_SEQUENCIA)
    # O pulso entra em relação ao último quadro: o último é zero, o primeiro -1.
    assert np.isclose(seq[-1, -2], 0.0) and np.isclose(seq[0, -2], -1.0)
    deslocada = janela.copy()
    deslocada[:, COLUNAS_PULSO_BRACO["dir"][0]] += 5.0  # outro lugar, mesmo caminho
    assert np.allclose(features_sequencia(deslocada), seq)
    assert sequencias_de_uma_gravacao(janela[:3], 10, 2).shape == (
        0,
        10,
        TOTAL_FEATURES_SEQUENCIA,
    )


@pytest.mark.parametrize("tipo", tipos_ia.REDES)
def test_rede_aprende_dois_sinais_e_volta_do_arquivo(tipo, tmp_path):
    pytest.importorskip("torch")
    from kara_sygna.ia.modelo_io import carregar_modelo
    from kara_sygna.ia.redes import salvar_rede, treinar_rede

    rng = np.random.default_rng(1)
    partes = [
        (_gravacao(x, rng=rng), s, g)
        for g, (x, s) in enumerate([(0.2, "a"), (0.8, "b")] * 2)
    ]
    dados = ConjuntoDados(
        np.vstack([q for q, _, _ in partes]),
        np.concatenate([[s] * len(q) for q, s, _ in partes]),
        np.concatenate([[g] * len(q) for q, _, g in partes]),
    )
    amostras = montar_amostras(
        dados,
        tamanho=10,
        passo=2,
        velocidades=(),
        mascara=(),
        extrair=sequencias_de_uma_gravacao,
    )
    rede, acuracia = treinar_rede(tipo, amostras, epocas=15)
    assert acuracia == 1.0
    caminho = tmp_path / f"modelo_sinais_{tipo}.pt"
    salvar_rede(rede, caminho)
    carregada = carregar_modelo(caminho)
    p = carregada.predict_proba(amostras.X[:3])
    assert p.shape == (3, 2) and np.allclose(p.sum(axis=1), 1.0)
    assert list(carregada.classes_) == ["a", "b"] and carregada.usa_sequencia


def test_escolha_do_detector_de_maos_fica_junto_da_ia(tmp_path):
    assert tipos_ia.detector_escolhido(tmp_path) == "hands"
    tipos_ia.escolher("randomforest", tmp_path, detector_maos="landmarker")
    assert tipos_ia.detector_escolhido(tmp_path) == "landmarker"
    tipos_ia.escolher("randomforest", tmp_path)  # sem detector: mantém o anterior
    assert tipos_ia.detector_escolhido(tmp_path) == "landmarker"
    with pytest.raises(ValueError):
        tipos_ia.escolher("randomforest", tmp_path, detector_maos="outro")
