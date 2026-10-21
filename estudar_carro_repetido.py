"""Carro repetido: S → carro → S → carro escreve carro duas vezes? (teste 7 de 20/10/2026)

No vídeo presinal_s_carro o carro saía uma vez só, mas o vídeo não tem o
segundo S: aos 11 s e 13 s (quando a tela pedia o S) as duas mãos ainda faziam
o carro. Aqui o S → carro → S → carro é montado colando pedaços de vídeos
(S do presinal_s, carro dos vídeos de carro), com o S de vários tamanhos,
para ver de quanto S o app precisa entre dois carros.

Modelo do app hoje (MS.1 + S como nada), IA de pré-sinais do MS.1, portão de
movimento e região das mãos ligados. Nada muda no app.

Uso::

    python estudar_carro_repetido.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter

from lab_ms.kara import preparar_import

preparar_import()

import estudar_presinal
from estudar_presinal import carregar, colar
from estudar_tudo import tabela
from estudar_videos import ARMAZENAMENTO, PASTA_RESULTADOS

CARROS = ("carro_luz_normal_calmo", "carro_luz_normal_rapido", "carro_perto")


def trecho(nome: str, de: float, ate: float):
    """Pedaço de um vídeo, de ``de`` a ``ate`` segundos (tempo recomeça do zero)."""
    v, t = carregar(nome)[:2]
    dentro = (t >= de) & (t < ate)
    return v[dentro], t[dentro] - t[dentro][0]


def escritos(*partes) -> str:
    v, t = colar(*partes)
    lista = estudar_presinal.reconhecer(v, t, True)[0]
    contagem = Counter(s for _, s in lista)
    momentos = " ".join(f"{s[:2]}@{m:.1f}" for m, s in lista)
    return f"carro×{contagem['carro']}" + (f" (+{sum(contagem.values()) - contagem['carro']} outros)" if len(contagem) > 1 else "") + f" [{momentos}]"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    estudar_presinal.MODELO_SINAIS = ARMAZENAMENTO / "modelo_sinais.pkl"
    linhas_tabela = {}
    for carro in CARROS:
        c = trecho(carro, 0.0, 4.0)
        linha = {"carro → carro (sem nada no meio)": escritos(c, c)}
        for duracao in (3.0, 1.0, 0.5, 0.3):
            s = trecho("presinal_s", 2.0, 2.0 + duracao)
            linha[f"S {duracao:g} s → carro → S {duracao:g} s → carro"] = escritos(s, c, s, c)
        linhas_tabela[carro] = linha
        print(f"  ok: {carro}", flush=True)

    linhas = [
        "# Carro repetido: S → carro → S → carro (vídeos colados)",
        "",
        "Cada carro: os 4 primeiros segundos do vídeo. Ideal: carro×2 (carro→carro sem nada: ×1).",
        "Entre colchetes: sinal@segundo em que foi escrito.",
        "",
        *tabela("Carro repetido", linhas_tabela),
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"carro_repetido_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
