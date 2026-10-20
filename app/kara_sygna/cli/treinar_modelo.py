"""Treina a IA pelo terminal.

Uso:  python -m kara_sygna.cli.treinar_modelo
"""

from __future__ import annotations

import sys

from kara_sygna.aplicacao.controlador import ControladorSinais
from kara_sygna.aplicacao.erros import ErroAplicacao
from kara_sygna.aplicacao.inicializacao import preparar_ambiente
from kara_sygna.dados.repositorio import DadosInvalidosError
from kara_sygna.ia.treinamento import TreinoImpossivelError


def main() -> int:
    """Treina com o CSV atual e imprime o relatório.

    Returns:
        0 em caso de sucesso, 1 se não foi possível treinar.
    """
    preparar_ambiente()
    try:
        resultado = ControladorSinais().treinar(progresso=print)
    except (ErroAplicacao, TreinoImpossivelError, DadosInvalidosError) as erro:
        print(f"\n{erro}", file=sys.stderr)
        return 1
    print("\n" + resultado.relatorio)
    if resultado.acuracia_validacao is not None:
        print(f"Acurácia na validação por rodada: {resultado.acuracia_validacao:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
