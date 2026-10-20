"""Detectores de mãos em revezamento: dois quadros analisados ao mesmo tempo.

Por que existe
--------------
O detector de mãos do MediaPipe leva ~46 ms por quadro, e com o rosto e os
braços cada quadro passa de 60 ms. Analisando um quadro por vez, o app tratava
~15 dos 30 quadros por segundo da câmera, e sinais curtos se perdiam: num
vídeo com trocas rápidas carro→casa, só 2 de ~16 trocas viraram texto.

Como funciona
-------------
* Só as MÃOS se revezam: vários detectores de mãos, cada um na sua thread; o
  quadro k vai para o detector k % N. Cada um vê um quadro sim, outro não, e
  o rastreio do MediaPipe aguenta (nos vídeos de teste, mão sumida 7% contra
  8% com um detector vendo todos os quadros).
* Rosto, realce e braços (Pose) ficam numa thread só, vendo TODOS os quadros
  em ordem (~18 ms por quadro, cabem nos 33 ms da câmera). Com um Pose por
  thread, cada um suavizava o pulso do seu jeito, o pulso "tremia" entre
  quadros e a IA via movimento onde não havia: a casa caiu de 90% para 41%.
* Com realce ligado, as mãos de um quadro esperam o rosto DESSE quadro, porque
  a cor da pele vem do rosto.

O resto do caminho (conferir as mãos pelos braços, memória, IA, desenho) fica
com quem chama, na ordem dos quadros: :meth:`VisaoEmRevezamento.pegar` sempre
devolve o quadro mais antigo enviado.

Medido no laboratório (kara-sygna-lab, 19/10/2026): de 17,5 para 30,6 quadros
por segundo com 2 detectores de mãos.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Deteccao:
    """O que os detectores viram num quadro."""

    quadro: np.ndarray
    rosto: Any
    maos: Any
    corpo: Any | None


class VisaoEmRevezamento:
    """Distribui os quadros entre os detectores de mãos e devolve em ordem.

    Args:
        maos: Detectores de mãos, um por thread. ``len(maos)`` quadros podem
            estar em análise ao mesmo tempo.
        rosto: Detector de rosto (thread comum, todos os quadros).
        realce: Realce de pele, calibrado na thread comum.
        corpo: Detector dos braços (thread comum), ou ``None``.
    """

    def __init__(
        self, maos: list[Any], rosto: Any, realce: Any, corpo: Any | None = None
    ) -> None:
        """Cria a thread comum e uma thread por detector de mãos (veja a classe)."""
        if not maos:
            raise ValueError("É preciso pelo menos um detector de mãos.")
        self.maos, self.rosto, self.realce, self.corpo = maos, rosto, realce, corpo
        self._comum = ThreadPoolExecutor(1, thread_name_prefix="visao-rosto-corpo")
        self._threads_maos = [
            ThreadPoolExecutor(1, thread_name_prefix=f"visao-maos{i}")
            for i in range(len(maos))
        ]
        self._fila: deque[tuple[np.ndarray, Future, Future]] = deque()
        self._proxima = 0

    def _rosto_e_corpo(self, quadro: np.ndarray) -> tuple[Any, Any]:
        rosto = self.rosto.detectar(quadro)
        self.realce.calibrar(quadro, rosto)
        # O corpo usa a imagem original: o realce escurece os braços e a roupa.
        corpo = self.corpo.detectar(quadro) if self.corpo is not None else None
        return rosto, corpo

    def _maos(self, i: int, quadro: np.ndarray, comum: Future) -> Any:
        if getattr(self.realce, "ativo", True):
            comum.result()  # a cor da pele deste quadro vem do rosto dele
        return self.maos[i].detectar(self.realce.aplicar(quadro))

    @property
    def cheia(self) -> bool:
        """Todos os detectores de mãos ocupados: o próximo quadro tem de esperar."""
        return len(self._fila) >= len(self.maos)

    def enviar(self, quadro: np.ndarray) -> None:
        """Manda o quadro para análise (não espera o resultado)."""
        if self.cheia:
            raise RuntimeError("Todos os detectores de mãos estão ocupados.")
        i = self._proxima
        comum = self._comum.submit(self._rosto_e_corpo, quadro)
        maos = self._threads_maos[i].submit(self._maos, i, quadro, comum)
        self._fila.append((quadro, comum, maos))
        self._proxima = (i + 1) % len(self.maos)

    def pronta(self) -> bool:
        """O quadro mais antigo já foi analisado."""
        return bool(self._fila) and self._fila[0][1].done() and self._fila[0][2].done()

    def pegar(self) -> Deteccao:
        """Devolve o quadro mais antigo (espera, se ainda não acabou).

        Raises:
            RuntimeError: Se nada foi enviado.
            Exception: O erro de um detector, se a análise falhou.
        """
        if not self._fila:
            raise RuntimeError("Nenhum quadro em análise.")
        quadro, comum, maos = self._fila.popleft()
        rosto, corpo = comum.result()
        return Deteccao(quadro, rosto, maos.result(), corpo)

    def fechar(self) -> None:
        """Espera as análises em andamento e encerra as threads."""
        for thread in (self._comum, *self._threads_maos):
            thread.shutdown(wait=True)
        self._fila.clear()


class LeitorEmThread:
    """Lê a câmera numa thread e guarda só o quadro mais novo.

    A leitura da webcam espera o próximo quadro (até ~33 ms, mais com pouca
    luz). Fora da janela, essa espera não trava a tela nem as análises.
    """

    def __init__(self, camera: Any) -> None:
        """Começa a ler (veja a classe)."""
        self._camera = camera
        self._trava = threading.Lock()
        self._quadro: np.ndarray | None = None
        self._novo = False
        self._falhas = 0
        self._parar = threading.Event()
        self._thread = threading.Thread(
            target=self._ler_sempre, name="camera", daemon=True
        )
        self._thread.start()

    def _ler_sempre(self) -> None:
        while not self._parar.is_set():
            try:
                quadro = self._camera.ler()
            except Exception:  # drivers de câmera podem lançar qualquer coisa
                logger.exception("Erro ao ler a câmera")
                quadro = None
            with self._trava:
                if quadro is None:
                    self._falhas += 1
                else:
                    self._quadro, self._novo, self._falhas = quadro, True, 0
            if quadro is None:
                self._parar.wait(0.1)

    @property
    def falhando(self) -> bool:
        """A câmera não entrega imagem há várias leituras seguidas."""
        with self._trava:
            return self._falhas >= 10

    def pegar_novo(self) -> np.ndarray | None:
        """O quadro mais novo, se chegou um desde a última vez; senão ``None``."""
        with self._trava:
            if not self._novo:
                return None
            self._novo = False
            return self._quadro

    def parar(self) -> None:
        """Para a thread (antes de liberar a câmera)."""
        self._parar.set()
        self._thread.join(timeout=2.0)
