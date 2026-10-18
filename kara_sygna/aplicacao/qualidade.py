"""Qualidade de uma gravação de ensino: diz na hora se vale gravar de novo.

Conta, em todos os quadros da câmera durante o ensino, em quantos apareceu
alguma mão, as duas mãos e o rosto. A IA aprende mal com gravações em que as
mãos somem ou o rosto não aparece (o lugar do sinal é medido no rosto).
"""

from __future__ import annotations

import numpy as np

from kara_sygna import config
from kara_sygna.visao.quadro import COLUNA_ROSTO_PRESENTE, maos_presentes

#: Abaixo disso, a gravação recebe um aviso.
MINIMO_COM_MAO = 0.6
MINIMO_ROSTO = 0.7
#: Se as duas mãos aparecem em parte dos quadros (entre os dois valores), o
#: sinal provavelmente é de duas mãos e a segunda sumiu muitas vezes. Abaixo
#: do mínimo, é um sinal de uma mão só e está tudo bem.
DUAS_MAOS_MINIMO_SINAL = 0.2
DUAS_MAOS_BOM = 0.75


class QualidadeGravacao:
    """Acumula a contagem de uma rodada de ensino."""

    def __init__(self) -> None:
        """Começa a contagem zerada."""
        self.quadros = 0
        self.com_mao = 0
        self.duas_maos = 0
        self.com_rosto = 0

    def adicionar(self, vetor_quadro: np.ndarray) -> None:
        """Conta um quadro da câmera (vetor de 223 valores)."""
        vetor = np.asarray(vetor_quadro, dtype=np.float64).reshape(1, -1)
        presentes = maos_presentes(vetor)[0]
        self.quadros += 1
        self.com_mao += int(presentes.any())
        self.duas_maos += int(presentes.all())
        self.com_rosto += int(vetor[0, COLUNA_ROSTO_PRESENTE] > 0.5)

    def avisos(self, gravados: int) -> list[str]:
        """Problemas encontrados (lista vazia = gravação boa).

        Args:
            gravados: Quantos quadros foram para o CSV nesta rodada.
        """
        if not self.quadros or not gravados:
            return ["nenhuma mão foi gravada"]
        avisos = []
        if self.com_mao / self.quadros < MINIMO_COM_MAO:
            avisos.append("as mãos ficaram fora da câmera muitas vezes")
        if self.com_mao and (
            DUAS_MAOS_MINIMO_SINAL <= self.duas_maos / self.com_mao < DUAS_MAOS_BOM
        ):
            avisos.append("a segunda mão sumiu muitas vezes")
        if self.com_rosto / self.quadros < MINIMO_ROSTO:
            avisos.append("o rosto não apareceu bem")
        if gravados < 2 * config.QUADROS_POR_JANELA:
            avisos.append("a gravação foi curta")
        return avisos

    def resumo(self, gravados: int) -> str:
        """Texto curto para a barra de status."""
        avisos = self.avisos(gravados)
        if self.quadros:
            numeros = (
                f"mãos em {self.com_mao / self.quadros:.0%}, "
                f"2 mãos em {self.duas_maos / self.quadros:.0%}, "
                f"rosto em {self.com_rosto / self.quadros:.0%}"
            )
        else:
            numeros = "nenhum quadro"
        if avisos:
            return f"{numeros}. ATENÇÃO, grave de novo: {'; '.join(avisos)}."
        return f"{numeros}. Boa gravação."
