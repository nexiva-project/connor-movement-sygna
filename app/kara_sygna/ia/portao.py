"""Portão de movimento: a IA só vota quando as mãos se mexeram há pouco.

Por que existe
--------------
O carro é feito com as mãos em S. Com a mão parada em S (o pré-sinal), a IA
escrevia "carro" sem o carro ter sido feito, e a mão relaxada às vezes virava
sinal. Os dois ficam quase parados; o carro e a casa, não. Medido nos vídeos de
teste (velocidade mediana do pulso): S 0,26 · mão relaxada 0,42 · B 0,43 ·
carro calmo 1,03 · casa 1,08–1,39 · carro rápido 1,8–4,7 larguras de ombro/s.

Ideia do projeto satyaidk/sign-language-recognition (GitHub). No laboratório
(kara-sygna-lab, 20/10/2026), com limite 0,8 e janela 0,5 s: o S parado deixou
de virar carro, o carro logo depois do S passou a sair em 0,15 s, a mão
relaxada escreveu 3 vezes em vez de 4, e o carro e a casa ficaram iguais
(carro 86% → 85% de voto certo).

Como funciona
-------------
* Velocidade = quanto o pulso andou (pelo braço, MediaPipe Pose) em 3 quadros,
  em larguras de ombro por segundo, a mais rápida das duas mãos, mediana dos
  últimos 0,3 s. (Era a média: ao vivo, em 26/10/2026, um pulo do pulso do
  Pose num quadro só abriu o portão com as mãos paradas em S, e o carro
  saiu. A mediana ignora o pulo; nos vídeos de teste o carro e a casa ficaram
  iguais e a mão relaxada abriu o portão em 26% dos quadros, antes 37%.)
* Ao levantar as mãos até a posição (quando elas aparecem), o movimento dos
  primeiros ``config.PORTAO_SUBIDA_S`` não conta: subir as mãos já em S não é
  o movimento do carro.
* Portão ABERTO se a velocidade passou de ``config.PORTAO_MOVIMENTO_LIMITE``
  nos últimos ``config.PORTAO_MOVIMENTO_JANELA_S`` segundos.
* Sem corpo na imagem (no escuro o Pose some) não dá para medir: fica aberto,
  para não bloquear o sinal.
* ``PORTAO_MOVIMENTO_LIMITE = 0`` desliga (sempre aberto).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from kara_sygna import config
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    maos_presentes,
)

_LADOS = ("esq", "dir")


class PortaoMovimento:
    """Diz, quadro a quadro, se as mãos se mexeram nos últimos segundos.

    Args:
        limite: Velocidade mínima do pulso, em larguras de ombro por segundo.
            ``0`` (ou menos) desliga o portão. Padrão:
            ``config.PORTAO_MOVIMENTO_LIMITE``.
        janela_s: Quanto tempo o portão fica aberto depois do último movimento.
            Padrão: ``config.PORTAO_MOVIMENTO_JANELA_S``.
        passo: De quantos quadros é medido o deslocamento do pulso.
        media_s: Tempo da média da velocidade (tira o tremido).
    """

    def __init__(
        self,
        limite: float | None = None,
        janela_s: float | None = None,
        passo: int = 3,
        media_s: float = 0.3,
    ) -> None:
        """Cria o portão (veja a classe para os argumentos)."""
        self.limite = float(
            config.PORTAO_MOVIMENTO_LIMITE if limite is None else limite
        )
        self.janela_s = float(
            config.PORTAO_MOVIMENTO_JANELA_S if janela_s is None else janela_s
        )
        self.passo = passo
        self.media_s = media_s
        self.reiniciar()

    @property
    def ligado(self) -> bool:
        """``False`` quando o limite é 0: o portão fica sempre aberto."""
        return self.limite > 0

    def reiniciar(self) -> None:
        """Esquece o movimento visto (ao começar a reconhecer)."""
        self._pulsos: deque = deque(maxlen=self.passo + 1)
        self._velocidades: deque = deque()
        self._ultimo_movimento = float("-inf")
        self._maos_desde: float | None = None  # quando as mãos apareceram
        self.velocidade = 0.0

    def atualizar(self, vetor: np.ndarray, agora: float) -> bool:
        """Registra o quadro e diz se o portão está aberto.

        Args:
            vetor: Vetor do quadro (223 valores).
            agora: Momento do quadro, em segundos.
        """
        if not self.ligado:
            return True
        if vetor[COLUNA_CORPO_PRESENTE] <= 0.5:
            self._pulsos.clear()
            return True  # sem corpo não dá para medir: não bloqueia
        if not maos_presentes(vetor).any():
            self._maos_desde = None
        elif self._maos_desde is None:
            self._maos_desde = agora
        subindo = (
            self._maos_desde is not None
            and agora - self._maos_desde < config.PORTAO_SUBIDA_S
        )
        pulsos = np.concatenate(
            [vetor[list(COLUNAS_PULSO_BRACO[lado])] for lado in _LADOS]
        )
        self._pulsos.append((agora, pulsos))
        if len(self._pulsos) > self.passo and not subindo:
            (t0, p0), (t1, p1) = self._pulsos[0], self._pulsos[-1]
            d = p1 - p0
            v = max(np.hypot(d[0], d[1]), np.hypot(d[2], d[3])) / max(t1 - t0, 1e-3)
            self._velocidades.append((agora, float(v)))
        while self._velocidades and agora - self._velocidades[0][0] > self.media_s:
            self._velocidades.popleft()
        self.velocidade = (
            float(np.median([v for _, v in self._velocidades]))
            if self._velocidades
            else 0.0
        )
        if self.velocidade > self.limite:
            self._ultimo_movimento = agora
        return agora - self._ultimo_movimento <= self.janela_s
