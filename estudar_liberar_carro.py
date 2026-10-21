"""Liberar o carro depois de um S curto (continuação do teste 7 de 20/10/2026).

Com o S de 1 s ou menos entre dois carros, o 2º carro não sai: durante o S a
IA ainda acha "carro", mas o voto é bloqueado (região das mãos ou portão) e
vira "sem voto". Para liberar o carro já escrito, o app pede 8 votos de outro
sinal ou 60 quadros (2 s) sem o carro.

Correções testadas:

* voto bloqueado vira "nada" (a tela já mostra NADA): conta como outro sinal;
* liberar sem voto com 30 ou 20 quadros (hoje 60);
* as duas juntas.

Os 11 vídeos de teste dizem se o carro volta a ser escrito várias vezes
(foi por isso que o 60 entrou).

Uso::

    python estudar_liberar_carro.py
"""

from __future__ import annotations

import sys
import time

from lab_ms.kara import preparar_import

preparar_import()

import estudar_presinal
import estudar_tudo
import estudar_videos
from estudar_carro_repetido import CARROS, trecho
from estudar_presinal import colar
from estudar_tudo import _esperados, resumir, tabela
from estudar_videos import ARMAZENAMENTO, PASTA_RESULTADOS, PASTA_VIDEOS, quadros_do_video, reconhecer

MODELO = ARMAZENAMENTO / "modelo_sinais.pkl"
BLOQUEIOS = ("parado", "longe")


def correcao(bloqueado_vira_nada: bool, sem_voto: int | None):
    """Função ``ajustar`` para o controlador com a correção escolhida."""

    def ajustar(controlador):
        estab = controlador.estabilizador
        if sem_voto is not None:
            estab.quadros_para_liberar_sem_voto = sem_voto
        if bloqueado_vira_nada:
            original = estab.atualizar

            def atualizar(sinal):
                if sinal is None and controlador._diag.get("motivo") in BLOQUEIOS:
                    sinal = "nada"
                return original(sinal)

            estab.atualizar = atualizar

    return ajustar


VARIANTES = {
    "app hoje (sem voto 60)": correcao(False, None),
    "bloqueado vira nada": correcao(True, None),
    "sem voto 30": correcao(False, 30),
    "sem voto 20": correcao(False, 20),
    "bloqueado vira nada + sem voto 30": correcao(True, 30),
}


def carros_colados(ajustar) -> dict:
    linha = {}
    for duracao in (1.0, 0.5):
        s = trecho("presinal_s", 2.0, 2.0 + duracao)
        total = 0
        for carro in CARROS:
            c = trecho(carro, 0.0, 4.0)
            v, t = colar(s, c, s, c)
            total += sum(e == "carro" for e in reconhecer(v, t, MODELO, ajustar).escritos)
        linha[f"S {duracao:g} s entre carros: carro escrito (ideal {2 * len(CARROS)})"] = total
    c = [trecho(carro, 0.0, 4.0) for carro in CARROS]
    linha[f"carro→carro sem nada (ideal {len(CARROS)})"] = sum(
        sum(e == "carro" for e in reconhecer(*colar(x, x), MODELO, ajustar).escritos) for x in c
    )
    return linha


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    estudar_videos.MODELO_PRE = ARMAZENAMENTO / "MS.1" / "modelo_sinais_pre.pkl"
    estudar_presinal.MODELO_SINAIS = MODELO
    lista = sorted(v for v in PASTA_VIDEOS.glob("*.avi") if not v.stem.startswith(("_", "presinal")))
    base = {v.stem: quadros_do_video(v, refazer=False, opcoes=estudar_presinal.VISAO) for v in lista}

    linhas_tabela = {}
    for nome, ajustar in VARIANTES.items():
        resultados = {
            stem: (reconhecer(vet, tem, MODELO, ajustar), _esperados(info), vet)
            for stem, (vet, tem, info) in base.items()
        }
        geral = resumir(resultados)
        linhas_tabela[nome] = {
            **carros_colados(ajustar),
            **{k: geral[k] for k in ("carro: escritas (ideal 1/vídeo)", "casa: escritas", "neutro: escritas (ideal 0)", "carro→casa→carro")},
        }
        print(f"  ok: {nome}", flush=True)

    linhas = [
        "# Liberar o carro depois de um S curto",
        "",
        "Modelo do app hoje (MS.1 + S como nada), IA de pré-sinais do MS.1, portão e região ligados.",
        "S colado do presinal_s; carros: os 4 primeiros segundos de " + ", ".join(CARROS) + ".",
        "",
        *tabela("Liberar o carro", linhas_tabela),
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"liberar_carro_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    estudar_tudo  # noqa: B018 (carrega o módulo antes, como nos outros estudos)
    main()
