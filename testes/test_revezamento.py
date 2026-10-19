"""Testes dos detectores de mãos em revezamento (sem câmera nem MediaPipe)."""

import threading
import time

import numpy as np
import pytest

from kara_sygna.visao.revezamento import LeitorEmThread, VisaoEmRevezamento


class _Detector:
    """Detector falso: anota os quadros que viu e a thread que o usou."""

    def __init__(self, nome, atraso=0.0):
        self.nome, self.atraso, self.vistos, self.threads = nome, atraso, [], set()

    def detectar(self, quadro):
        time.sleep(self.atraso)
        self.vistos.append(int(quadro[0, 0]))
        self.threads.add(threading.get_ident())
        return (self.nome, int(quadro[0, 0]))


class _Realce:
    def __init__(self, ativo=False):
        self.ativo, self.calibrados = ativo, []

    def calibrar(self, quadro, rosto):
        self.calibrados.append(int(quadro[0, 0]))

    def aplicar(self, quadro):
        return quadro


def _quadro(n):
    return np.full((2, 2), n, dtype=np.int32)


def _esvaziar(visao):
    saida = []
    while True:
        try:
            saida.append(visao.pegar())
        except RuntimeError:
            return saida


def test_maos_alternam_rosto_e_corpo_veem_todos_e_a_ordem_se_mantem():
    maos_a, maos_b = _Detector("A", atraso=0.03), _Detector("B")  # A mais lenta
    rosto, corpo = _Detector("rosto"), _Detector("corpo")
    visao = VisaoEmRevezamento([maos_a, maos_b], rosto, _Realce(), corpo)
    recebidos = []
    for n in range(6):
        if visao.cheia:
            recebidos.append(visao.pegar())
        visao.enviar(_quadro(n))
    recebidos += _esvaziar(visao)
    visao.fechar()
    assert [d.maos[1] for d in recebidos] == [0, 1, 2, 3, 4, 5]  # em ordem
    assert [d.rosto[1] for d in recebidos] == [0, 1, 2, 3, 4, 5]
    assert maos_a.vistos == [0, 2, 4] and maos_b.vistos == [1, 3, 5]  # pares/ímpares
    # Rosto e corpo: um detector só, numa thread só, vendo todos os quadros em ordem.
    assert rosto.vistos == corpo.vistos == [0, 1, 2, 3, 4, 5]
    assert len(rosto.threads) == 1 and rosto.threads == corpo.threads
    assert maos_a.threads != maos_b.threads


def test_com_realce_as_maos_esperam_o_rosto_do_mesmo_quadro():
    rosto = _Detector("rosto", atraso=0.03)  # rosto lento: as mãos chegariam antes
    realce = _Realce(ativo=True)
    ordem = []

    class Maos(_Detector):
        def detectar(self, quadro):
            ordem.append((int(quadro[0, 0]), list(realce.calibrados)))
            return super().detectar(quadro)

    visao = VisaoEmRevezamento([Maos("A"), Maos("B")], rosto, realce)
    visao.enviar(_quadro(7))
    _esvaziar(visao)
    visao.fechar()
    assert ordem == [(7, [7])]  # o realce já tinha sido calibrado com o quadro 7


def test_cheia_impede_mais_quadros_que_detectores_de_maos():
    lento = _Detector("A", atraso=0.05)
    visao = VisaoEmRevezamento(
        [lento, _Detector("B", atraso=0.05)], _Detector("r"), _Realce()
    )
    visao.enviar(_quadro(0))
    visao.enviar(_quadro(1))
    assert visao.cheia
    with pytest.raises(RuntimeError):
        visao.enviar(_quadro(2))
    visao.pegar()
    assert not visao.cheia
    visao.fechar()


def test_leitor_entrega_cada_quadro_novo_uma_vez():
    class Camera:
        def __init__(self):
            self.n = 0

        def ler(self):
            time.sleep(0.01)
            self.n += 1
            return _quadro(self.n)

    leitor = LeitorEmThread(Camera())
    time.sleep(0.05)
    primeiro = leitor.pegar_novo()
    assert primeiro is not None
    assert leitor.pegar_novo() is None  # o mesmo quadro não volta
    time.sleep(0.03)
    segundo = leitor.pegar_novo()
    leitor.parar()
    assert segundo is not None and segundo[0, 0] > primeiro[0, 0]
