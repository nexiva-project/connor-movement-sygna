"""Lado de cada mão no corpo inteiro (RTMW): vem do modelo, não da tela."""

import numpy as np

from kara_sygna.visao.caracteristicas import indices_dos_slots
from kara_sygna.visao.detector_maos import DetectorMaos, _sem_copia

LARGURA, ALTURA = 640, 480


def _mao(x: float, y: float) -> np.ndarray:
    """21 pontos de uma mão com o pulso em (x, y) e palma de 40 px."""
    pontos = np.zeros((21, 2))
    pontos[:, 0] = x + np.linspace(0, 20, 21)
    pontos[:, 1] = y - np.linspace(0, 40, 21)
    pontos[9] = (x, y - 40)
    return pontos


def _corpo(direita=None, esquerda=None, pulso9=(0, 0), pulso10=(0, 0)):
    """133 pontos e notas, com as mãos nos blocos 91 (direita) e 112 (esquerda)."""
    pontos = np.zeros((133, 2))
    notas = np.zeros(133)
    pontos[9], pontos[10] = pulso9, pulso10
    for mao, a in ((direita, 91), (esquerda, 112)):
        if mao is not None:
            pontos[a : a + 21] = mao
            notas[a : a + 21] = 0.9
    return pontos, notas


def _detector(pontos, notas) -> DetectorMaos:
    """DetectorMaos do RTMW-x sem carregar o modelo."""
    det = object.__new__(DetectorMaos)
    det.tipo = "rtmw"
    det._rtmlib = lambda _q: (pontos[None], notas[None])
    det._filtros, det._pulsos, det._proxima_chave = {}, {}, 0
    return det


def _lados(pontos, notas) -> dict:
    det = _detector(pontos, notas)
    resultado = det._detectar_rtmlib(np.zeros((ALTURA, LARGURA, 3), np.uint8))
    return indices_dos_slots(resultado)


def test_mao_direita_do_outro_lado_da_tela_continua_direita():
    # Antes o lado saía da metade da tela: aqui a direita estava à esquerda.
    pontos, notas = _corpo(direita=_mao(100, 300), pulso9=(100, 300))
    assert _lados(pontos, notas) == {"esq": None, "dir": 0}


def test_maos_cruzadas_mantem_o_lado():
    pontos, notas = _corpo(
        direita=_mao(150, 300), esquerda=_mao(450, 300), pulso9=(150, 300),
        pulso10=(450, 300),
    )
    assert _lados(pontos, notas) == {"dir": 0, "esq": 1}


def test_copia_da_mao_direita_e_tirada():
    direita = _mao(400, 300)
    copia = _mao(405, 305)
    pontos, notas = _corpo(
        direita=direita, esquerda=copia, pulso9=(400, 300), pulso10=(200, 470)
    )
    assert _lados(pontos, notas) == {"esq": None, "dir": 0}


def test_maos_encostadas_ficam_as_duas():
    # Pulsos juntos (menos de uma palma), cada mão na ponta do próprio braço.
    pontos, notas = _corpo(
        direita=_mao(330, 300), esquerda=_mao(310, 300), pulso9=(335, 300),
        pulso10=(305, 300),
    )
    assert _lados(pontos, notas) == {"dir": 0, "esq": 1}


def _lista(pontos):
    from types import SimpleNamespace

    return SimpleNamespace(
        landmark=[SimpleNamespace(x=x / LARGURA, y=y / ALTURA, z=0.0) for x, y in pontos]
    )


def _lado(nome, nota):
    from types import SimpleNamespace

    return SimpleNamespace(
        classification=[SimpleNamespace(label=nome, score=nota)]
    )


def test_sem_corpo_copia_da_mesma_mao_e_tirada():
    from types import SimpleNamespace

    from kara_sygna.visao.maos_pelo_corpo import conferir_maos

    resultado = SimpleNamespace(
        multi_hand_landmarks=[_lista(_mao(150, 300)), _lista(_mao(153, 302))],
        multi_handedness=[_lado("Left", 0.99), _lado("Left", 0.7)],
    )
    conferido = conferir_maos(resultado, None, LARGURA, ALTURA)
    assert len(conferido.multi_hand_landmarks) == 1 and conferido.descartadas == 1
    assert conferido.multi_handedness[0].classification[0].score == 0.99


def test_sem_corpo_duas_maos_de_verdade_ficam():
    from types import SimpleNamespace

    from kara_sygna.visao.maos_pelo_corpo import conferir_maos

    resultado = SimpleNamespace(
        multi_hand_landmarks=[_lista(_mao(150, 300)), _lista(_mao(450, 300))],
        multi_handedness=[_lado("Left", 0.9), _lado("Right", 0.9)],
    )
    assert conferir_maos(resultado, None, LARGURA, ALTURA) is resultado


def test_sem_copia_mantem_duas_maos_longe():
    maos = [
        ("dir", _mao(400, 300), np.full(21, 0.9)),
        ("esq", _mao(200, 300), np.full(21, 0.9)),
    ]
    pontos, _ = _corpo(pulso9=(400, 300), pulso10=(200, 300))
    assert len(_sem_copia(maos, pontos)) == 2


def test_sem_copia_ignora_mao_com_nota_baixa():
    maos = [
        ("dir", _mao(400, 300), np.full(21, 0.9)),
        ("esq", _mao(402, 302), np.full(21, 0.1)),
    ]
    pontos, _ = _corpo(pulso9=(400, 300), pulso10=(200, 300))
    assert _sem_copia(maos, pontos) is maos
