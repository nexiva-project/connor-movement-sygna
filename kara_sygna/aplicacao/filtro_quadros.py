"""Filtra quadros repetidos durante o ensino de sinais."""

from __future__ import annotations

import numpy as np

from kara_sygna.visao.caracteristicas import (
    TOTAL_FEATURES,
    dividir_maos,
    eh_vetor_zerado,
)


class FiltroQuadros:
    """Evita gravar quadros quase idênticos na mesma rodada."""

    def __init__(self, limiar: float = 0.015) -> None:
        if not np.isfinite(limiar) or limiar < 0:
            raise ValueError("O limiar precisa ser finito e não negativo.")

        self.limiar = float(limiar)
        self._ultimo_aceito: np.ndarray | None = None

    def reiniciar(self) -> None:
        """Limpa o histórico ao iniciar uma nova rodada."""
        self._ultimo_aceito = None

    def aceitar(self, vetor: np.ndarray) -> bool:
        """Retorna True se o quadro deve ser gravado."""
        atual = np.asarray(vetor, dtype=np.float64).reshape(-1)

        if atual.size != TOTAL_FEATURES:
            raise ValueError(
                f"Esperados {TOTAL_FEATURES} valores; recebidos {atual.size}."
            )

        if not np.all(np.isfinite(atual)) or eh_vetor_zerado(atual):
            return False

        anterior = self._ultimo_aceito

        if anterior is None:
            self._ultimo_aceito = atual.copy()
            return True

        maos_atual = dividir_maos(atual)
        maos_anteriores = dividir_maos(anterior)

        # Preserva mudanças na presença ou ausência de uma mão.
        mudou_presenca = any(
            eh_vetor_zerado(mao_atual) != eh_vetor_zerado(mao_anterior)
            for mao_atual, mao_anterior in zip(maos_atual, maos_anteriores, strict=True)
        )

        if mudou_presenca:
            self._ultimo_aceito = atual.copy()
            return True

        diferenca = float(np.mean(np.abs(atual - anterior)))

        if diferenca < self.limiar:
            return False

        self._ultimo_aceito = atual.copy()
        return True
