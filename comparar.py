"""Compara o RandomForest do kara-sygna com o MS nas gravações reais.

Uso (com o Python da .venv do kara-sygna-py, que já tem as bibliotecas)::

    python comparar.py                       # usa o CSV atual do app
    python comparar.py --csv outro.csv       # outro arquivo de gravações
    python comparar.py --peso-movimento 2    # dá mais peso ao movimento no MS
    python comparar.py --buracos 0.3 --mascara-treino   # mão sumindo 30% ao vivo

O relatório sai na tela e em ``resultados/comparacao_<data>.md``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from lab_ms.experimento import RITMOS, Resultado, carregar_gravacoes, comparar

PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"


def _pct(certas: int, total: int) -> str:
    return f"{100 * certas / total:.1f}% ({certas}/{total})" if total else "—"


def relatorio(resultados: dict[str, dict[str, Resultado]], gravacoes, opcoes: dict) -> str:
    """Monta o relatório em Markdown."""
    linhas = ["# MS × RandomForest", ""]
    contagem: dict[str, int] = {}
    for g in gravacoes:
        contagem[g.rotulo] = contagem.get(g.rotulo, 0) + 1
    linhas.append(
        "Gravações: " + ", ".join(f"{r} ×{n}" for r, n in sorted(contagem.items()))
    )
    linhas.append(f"Opções: {opcoes}")
    linhas.append("")

    titulos = {
        "todas": "Treino com todas as outras rodadas (como o app hoje)",
        "uma_por_sinal": "Treino com só 1 rodada de cada sinal (poucos dados)",
    }
    for cenario, metodos in resultados.items():
        linhas += [f"## {titulos.get(cenario, cenario)}", ""]
        linhas.append("| Método | Janelas certas | Gravações certas | " + " | ".join(RITMOS) + " | ms/janela (mediana) |")
        linhas.append("|---" * (4 + len(RITMOS)) + "|")
        for nome, r in metodos.items():
            ritmos = [_pct(r.por_ritmo[x], r.por_ritmo_total[x]) for x in RITMOS]
            ms = f"{statistics.median(r.ms_por_janela):.1f}" if r.ms_por_janela else "—"
            linhas.append(
                f"| {nome} | {_pct(r.janelas_certas, r.janelas_total)} | "
                f"{_pct(r.gravacoes_certas, r.gravacoes_total)} | " + " | ".join(ritmos) + f" | {ms} |"
            )
        linhas.append("")
        for nome, r in metodos.items():
            erros = {par: n for par, n in r.confusao.items() if par[0] != par[1]}
            if erros:
                texto = ", ".join(f"{real}→{prev} ×{n}" for (real, prev), n in sorted(erros.items()))
                linhas.append(f"- Erros do {nome} (real→previsto): {texto}")
        linhas.append("")
    return "\n".join(linhas)


def main() -> None:
    """Lê os argumentos, roda a comparação e salva o relatório."""
    sys.stdout.reconfigure(encoding="utf-8")  # o console do Windows não mostra "→"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=None, help="CSV de gravações (padrão: o do app)")
    parser.add_argument("--peso-forma", type=float, default=1.0)
    parser.add_argument("--peso-movimento", type=float, default=1.0)
    parser.add_argument("--sem-espelho", action="store_true", help="não espelha os modelos do MS")
    parser.add_argument("--buracos", type=float, default=0.0, help="parte dos quadros sem mão no teste (ex.: 0.3)")
    parser.add_argument("--mascara-treino", action="store_true", help="também testa o RF treinado com máscara")
    args = parser.parse_args()

    opcoes = {
        "peso_forma": args.peso_forma,
        "peso_movimento": args.peso_movimento,
        "espelhar": not args.sem_espelho,
    }
    gravacoes = carregar_gravacoes(args.csv)
    inicio = time.perf_counter()
    resultados = comparar(gravacoes, buracos=args.buracos, mascara_treino=args.mascara_treino, **opcoes)
    texto = relatorio(resultados, gravacoes, {**opcoes, "buracos": args.buracos, "mascara_treino": args.mascara_treino})
    texto += f"\nTempo total: {time.perf_counter() - inicio:.0f} s\n"

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    arquivo = PASTA_RESULTADOS / f"comparacao_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print()
    print(texto)
    print(f"Relatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
