"""Anti-pulo: o dedo que pula sozinho volta para o lugar (sem câmera)."""

from types import SimpleNamespace

import numpy as np

from kara_sygna.visao.anti_pulo import FiltroAntiPulo


def _mao(polegar_x=0.0, deslocar=0.0):
    """Mão aberta: pulso em (0.5, 0.8), dedos para cima; o polegar pode ir para o lado."""
    pontos = []
    for i in range(21):
        dedo, junta = (i - 1) // 4, (i - 1) % 4
        if i == 0:
            x, y = 0.5, 0.8
        else:
            x, y = 0.40 + 0.05 * dedo, 0.70 - 0.04 * junta
            if dedo == 0:
                x += polegar_x * (junta + 1) / 4
        pontos.append(SimpleNamespace(x=x + deslocar, y=y, z=0.0))
    return SimpleNamespace(landmark=pontos)


def _resultado(mao):
    return SimpleNamespace(multi_hand_landmarks=[mao], lados=["dir"])


def _polegar(resultado):
    return resultado.multi_hand_landmarks[0].landmark[4].x


def test_polegar_que_pula_sozinho_volta_para_o_lugar():
    filtro = FiltroAntiPulo(limiar=0.35, max_quadros=3)
    filtro.aplicar(_resultado(_mao()))
    pulou = _resultado(_mao(polegar_x=0.08))
    corrigido = filtro.aplicar(pulou)
    assert filtro.segurados_agora == 1
    assert np.isclose(_polegar(corrigido), _polegar(_resultado(_mao())), atol=1e-6)
    assert _polegar(pulou) != _polegar(corrigido)  # o original não muda


def test_mao_inteira_andando_nao_e_pulo():
    filtro = FiltroAntiPulo(limiar=0.35, max_quadros=3)
    filtro.aplicar(_resultado(_mao()))
    andou = _resultado(_mao(deslocar=0.1))
    assert filtro.aplicar(andou) is andou and filtro.segurados_agora == 0


def test_dedo_que_continua_mexido_passa_depois_de_alguns_quadros():
    filtro = FiltroAntiPulo(limiar=0.35, max_quadros=2)
    filtro.aplicar(_resultado(_mao()))
    segurados = [
        (filtro.aplicar(_resultado(_mao(polegar_x=0.08))), filtro.segurados_agora)[1]
        for _ in range(4)
    ]
    assert segurados[:2] == [1, 1] and segurados[2] == 0  # movimento de verdade


def _rotulo(nome):
    return SimpleNamespace(classification=[SimpleNamespace(label=nome)])


def test_maos_que_trocam_de_ordem_na_lista_nao_viram_pulo():
    # Sem os braços (sem "lados"): a ordem da lista troca, mas cada mão é
    # comparada com ela mesma (pelo lado), e não com a outra.
    filtro = FiltroAntiPulo(limiar=0.35, max_quadros=3)
    esquerda, direita = _mao(deslocar=-0.3), _mao(polegar_x=0.08, deslocar=0.3)
    filtro.aplicar(
        SimpleNamespace(
            multi_hand_landmarks=[esquerda, direita],
            multi_handedness=[_rotulo("Left"), _rotulo("Right")],
        )
    )
    trocado = SimpleNamespace(
        multi_hand_landmarks=[direita, esquerda],
        multi_handedness=[_rotulo("Right"), _rotulo("Left")],
    )
    assert filtro.aplicar(trocado) is trocado and filtro.segurados_agora == 0


def test_max_quadros_zero_desliga():
    filtro = FiltroAntiPulo(limiar=0.35, max_quadros=0)
    filtro.aplicar(_resultado(_mao()))
    pulou = _resultado(_mao(polegar_x=0.08))
    assert filtro.aplicar(pulou) is pulou and filtro.segurados_agora == 0


def test_limiar_zero_desliga():
    filtro = FiltroAntiPulo(limiar=0)
    filtro.aplicar(_resultado(_mao()))
    pulou = _resultado(_mao(polegar_x=0.08))
    assert filtro.aplicar(pulou) is pulou
