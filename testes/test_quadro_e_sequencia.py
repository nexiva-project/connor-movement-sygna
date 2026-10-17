"""Testes do vetor de cada quadro (mãos + rosto + posição) e das janelas de movimento."""

from types import SimpleNamespace

import numpy as np
import pytest

from kara_sygna import config
from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.ia import treinamento
from kara_sygna.ia.sequencia import (
    TOTAL_FEATURES_JANELA,
    BufferQuadros,
    features_janela,
    janelas_de_uma_gravacao,
    montar_amostras,
    nomes_features_janela,
)
from kara_sygna.visao.caracteristicas import montar_vetor_duas_maos
from kara_sygna.visao.quadro import (
    COLUNA_ROSTO_PRESENTE,
    COLUNAS_POSICAO,
    IDX_MAOS,
    IDX_POSICAO,
    IDX_ROSTO_MAOS,
    TOTAL_FEATURES_QUADRO,
    espelhar_quadros,
    maos_presentes,
    montar_vetor_quadro,
    nomes_features_quadro,
    tem_mao,
)
from testes.conftest import (
    gravacao_falsa,
    mao_falsa,
    quadro_falso,
    resultado_falso,
    vetor_de_duas_maos,
    vetor_de_uma_mao,
)

LARGURA, ALTURA = 640, 480


def rosto_falso(rng, nariz=(0.5, 0.4), largura_rosto=0.25):
    """ResultadoRosto falso: 468 pontos, com nariz e laterais do rosto definidos."""
    pontos = rng.random((468, 3)) * 0.1 + np.array(
        [nariz[0] - 0.05, nariz[1] - 0.05, 0]
    )
    pontos[1, :2] = nariz
    pontos[234, :2] = (nariz[0] - largura_rosto / 2, nariz[1])
    pontos[454, :2] = (nariz[0] + largura_rosto / 2, nariz[1])
    landmarks = SimpleNamespace(
        landmark=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in pontos]
    )
    return SimpleNamespace(landmarks=landmarks, presente=True)


SEM_ROSTO = SimpleNamespace(landmarks=None, presente=False)


# ---------------------------------------------------------------- vetor do quadro


def test_nomes_do_quadro_sao_unicos_e_batem_com_o_tamanho():
    nomes = nomes_features_quadro()
    assert len(nomes) == len(set(nomes)) == TOTAL_FEATURES_QUADRO == 214


def test_bloco_das_maos_e_o_mesmo_vetor_de_antes(rng):
    maos = resultado_falso((mao_falsa(rng), "Right"), (mao_falsa(rng, 0.3), "Left"))
    vetor = montar_vetor_quadro(maos, rosto_falso(rng), LARGURA, ALTURA)
    assert vetor.shape == (TOTAL_FEATURES_QUADRO,)
    assert np.allclose(vetor[IDX_MAOS], montar_vetor_duas_maos(maos))
    assert tem_mao(vetor)


def test_posicao_do_pulso_e_medida_em_larguras_de_rosto(rng):
    mao = mao_falsa(rng)
    pulso = mao.landmark[0]
    pulso.x, pulso.y = 0.75, 0.55  # nariz em (0.5, 0.4); rosto com 0.25 de largura
    vetor = montar_vetor_quadro(
        resultado_falso((mao, "Left")), rosto_falso(rng), LARGURA, ALTURA
    )
    x, y, escala = (vetor[c] for c in COLUNAS_POSICAO["dir"])  # "Left" invertido
    largura_rosto_px = 0.25 * LARGURA
    assert x == pytest.approx((0.75 - 0.5) * LARGURA / largura_rosto_px)
    assert y == pytest.approx((0.55 - 0.4) * ALTURA / largura_rosto_px)
    assert escala > 0
    assert np.all(vetor[list(COLUNAS_POSICAO["esq"])] == 0)


def test_sem_rosto_a_posicao_e_o_rosto_ficam_zerados(rng):
    vetor = montar_vetor_quadro(
        resultado_falso((mao_falsa(rng), "Left")), SEM_ROSTO, LARGURA, ALTURA
    )
    assert vetor[COLUNA_ROSTO_PRESENTE] == 0
    assert np.all(vetor[IDX_POSICAO] == 0)
    assert tem_mao(vetor)


def test_sem_maos_o_quadro_nao_tem_mao(rng):
    vetor = montar_vetor_quadro(resultado_falso(), rosto_falso(rng), LARGURA, ALTURA)
    assert not tem_mao(vetor) and vetor[COLUNA_ROSTO_PRESENTE] == 1


def test_mao_degenerada_e_ausente_em_todos_os_blocos(rng):
    mao = SimpleNamespace(
        landmark=[SimpleNamespace(x=0.5, y=0.5, z=0) for _ in range(21)]
    )
    vetor = montar_vetor_quadro(
        resultado_falso((mao, "Left")), rosto_falso(rng), LARGURA, ALTURA
    )
    assert not tem_mao(vetor)
    assert vetor[IDX_ROSTO_MAOS.start + 1] == vetor[IDX_ROSTO_MAOS.start + 2] == 0


def test_lados_das_maos_batem_entre_os_blocos(rng):
    """A mão no slot "dir" das formas também é a "dir" no bloco do rosto."""
    vetor = montar_vetor_quadro(
        resultado_falso((mao_falsa(rng), "Left")), rosto_falso(rng), LARGURA, ALTURA
    )
    presentes = maos_presentes(vetor)[0]
    inicio = IDX_ROSTO_MAOS.start
    assert presentes.tolist() == [False, True]
    assert (vetor[inicio + 1], vetor[inicio + 2]) == (0.0, 1.0)


def test_espelhar_duas_vezes_devolve_o_original(rng):
    maos = resultado_falso((mao_falsa(rng), "Right"), (mao_falsa(rng, 0.3), "Left"))
    vetor = montar_vetor_quadro(maos, rosto_falso(rng), LARGURA, ALTURA)
    assert np.allclose(espelhar_quadros(espelhar_quadros(vetor)), vetor)


def test_espelhar_troca_as_maos_e_inverte_o_x(rng):
    vetor = quadro_falso(vetor_de_uma_mao(rng, "dir"), posicao_dir=(0.8, 0.3))
    espelho = espelhar_quadros(vetor)
    assert maos_presentes(espelho)[0].tolist() == [True, False]
    x, y, _ = (espelho[c] for c in COLUNAS_POSICAO["esq"])
    assert (x, y) == pytest.approx((-0.8, 0.3))
    forma_dir = vetor[63:126].reshape(21, 3)
    forma_esq = espelho[0:63].reshape(21, 3)
    assert np.allclose(forma_esq[:, 0], -forma_dir[:, 0])
    assert np.allclose(forma_esq[:, 1:], forma_dir[:, 1:])


def test_espelhar_aceita_matriz_e_recusa_tamanho_errado(rng):
    matriz = np.vstack([quadro_falso(vetor_de_duas_maos(rng)) for _ in range(3)])
    assert espelhar_quadros(matriz).shape == matriz.shape
    with pytest.raises(ValueError):
        espelhar_quadros(np.zeros(126))


# ---------------------------------------------------------------- janelas


def test_nomes_da_janela_sao_unicos():
    nomes = nomes_features_janela()
    assert len(nomes) == len(set(nomes)) == TOTAL_FEATURES_JANELA


def test_janela_comeca_com_o_ultimo_quadro(rng):
    gravacao = gravacao_falsa(vetor_de_duas_maos(rng), rng, 12)
    vetor = features_janela(gravacao)
    assert vetor.shape == (TOTAL_FEATURES_JANELA,)
    assert np.array_equal(vetor[:TOTAL_FEATURES_QUADRO], gravacao[-1])


def _movimento(vetor, lado="dir"):
    nomes = nomes_features_janela()
    return {
        nome.removeprefix(f"mov_{lado}_"): vetor[i]
        for i, nome in enumerate(nomes)
        if nome.startswith(f"mov_{lado}_")
    }


def test_sinal_parado_tem_movimento_zero(rng):
    mov = _movimento(features_janela(gravacao_falsa(vetor_de_uma_mao(rng), rng, 12)))
    assert mov["dx0"] == mov["dy0"] == mov["caminho"] == 0
    assert mov["presenca"] == 1


def test_movimento_mostra_de_onde_o_pulso_veio(rng):
    gravacao = gravacao_falsa(vetor_de_uma_mao(rng), rng, 12, movimento=(1.0, 0.0))
    mov = _movimento(features_janela(gravacao))
    assert mov["dx0"] == pytest.approx(-1.0)  # começou 1 rosto à esquerda
    assert mov["dy0"] == pytest.approx(0.0)
    assert mov["caminho"] == pytest.approx(1.0)


def test_quadros_sem_mao_ou_sem_rosto_usam_a_posicao_mais_proxima(rng):
    gravacao = gravacao_falsa(vetor_de_uma_mao(rng), rng, 12, movimento=(1.0, 0.0))
    gravacao[0] = quadro_falso(None)
    mov = _movimento(features_janela(gravacao))
    assert mov["dx0"] == pytest.approx(-1.0 + 1 / 11)  # o quadro 1 vale pelo 0
    assert mov["presenca"] == pytest.approx(11 / 12)


def test_mao_ausente_no_ultimo_quadro_zera_o_movimento(rng):
    gravacao = gravacao_falsa(vetor_de_uma_mao(rng), rng, 12, movimento=(1.0, 0.0))
    gravacao[-1] = quadro_falso(None)
    mov = _movimento(features_janela(gravacao))
    assert mov["dx0"] == mov["caminho"] == 0


def test_janela_com_formato_errado_e_recusada():
    with pytest.raises(ValueError):
        features_janela(np.zeros((12, 126)))


def test_gravacao_vira_janelas_so_onde_ha_mao(rng):
    gravacao = gravacao_falsa(vetor_de_duas_maos(rng), rng, 20)
    assert janelas_de_uma_gravacao(gravacao, tamanho=12, passo=2).shape == (
        5,
        TOTAL_FEATURES_JANELA,
    )
    gravacao[-1] = quadro_falso(None)
    assert len(janelas_de_uma_gravacao(gravacao, tamanho=12, passo=2)) == 4
    assert len(janelas_de_uma_gravacao(gravacao[:5], tamanho=12, passo=2)) == 0


def test_montar_amostras_inclui_o_espelho_no_mesmo_grupo(rng):
    gravacao = gravacao_falsa(vetor_de_uma_mao(rng, "dir"), rng, 20)
    dados = ConjuntoDados(gravacao, np.array(["oi"] * 20), np.array([7] * 20))
    amostras = montar_amostras(dados, tamanho=12, passo=2)
    assert len(amostras) == 10 and set(amostras.grupos.tolist()) == {7}
    presentes = maos_presentes(amostras.X[:, :TOTAL_FEATURES_QUADRO])
    assert presentes[:5, 1].all() and presentes[5:, 0].all()  # original e espelho


def test_buffer_so_fica_cheio_com_a_janela_completa(rng):
    buffer = BufferQuadros(tamanho=3)
    for _ in range(2):
        buffer.adicionar(quadro_falso(vetor_de_duas_maos(rng)))
    assert not buffer.cheio
    buffer.adicionar(quadro_falso(vetor_de_duas_maos(rng)))
    assert buffer.cheio and buffer.features().shape == (TOTAL_FEATURES_JANELA,)
    buffer.limpar()
    assert not buffer.cheio
    with pytest.raises(ValueError):
        buffer.adicionar(np.zeros(126))


# ---------------------------------------------------------------- de ponta a ponta


@pytest.fixture
def config_rapida(monkeypatch):
    monkeypatch.setattr(config, "ARVORES", 30)


def _gravacoes(rng, sinais, rodadas=3, quadros=24):
    """Sinais com a MESMA forma de mão; só o movimento ou o lugar muda."""
    forma = vetor_de_uma_mao(np.random.default_rng(5), "dir")
    X, y, g = [], [], []
    rodada = 0
    for nome, (movimento, inicio) in sinais.items():
        for _ in range(rodadas):
            rodada += 1
            gravacao = gravacao_falsa(forma, rng, quadros, movimento)
            col_x, col_y, _ = COLUNAS_POSICAO["dir"]
            gravacao[:, col_x] += inicio[0]
            gravacao[:, col_y] += inicio[1]
            X.append(gravacao)
            y += [nome] * quadros
            g += [rodada] * quadros
    return ConjuntoDados(np.vstack(X), np.array(y), np.array(g))


def test_ia_distingue_sinais_que_so_diferem_no_movimento(tmp_path, rng, config_rapida):
    dados = _gravacoes(
        rng,
        {
            "direita": ((1.5, 0.0), (0.0, 1.0)),
            "baixo": ((0.0, 1.5), (0.0, 1.0)),
            "parado": ((0.0, 0.0), (0.0, 1.0)),
        },
    )
    resultado = treinamento.treinar(dados, tmp_path / "m.pkl")
    assert resultado.acuracia_validacao is not None
    assert resultado.acuracia_validacao > 0.9


def test_ia_distingue_sinais_que_so_diferem_no_lugar(tmp_path, rng, config_rapida):
    dados = _gravacoes(
        rng,
        {
            "testa": ((0.0, 0.0), (0.0, -0.4)),
            "queixo": ((0.0, 0.0), (0.0, 0.6)),
            "peito": ((0.0, 0.0), (0.0, 2.0)),
        },
    )
    resultado = treinamento.treinar(dados, tmp_path / "m.pkl")
    assert resultado.acuracia_validacao is not None
    assert resultado.acuracia_validacao > 0.9
