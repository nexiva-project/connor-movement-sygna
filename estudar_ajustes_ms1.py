"""Mais testes com o MS.1 nos vídeos: ajustes da visão e votos para liberar.

1. Ir além do que funcionou: memória da mão 1,0 s (hoje 0,8) e detecção 0,2
   (hoje 0,3), sozinhas e juntas.
2. Votos de OUTRO sinal para liberar o sinal escrito: 8 (hoje), 12 e 15. Ao
   vivo, uma casa rápida demais (8 votos) liberou o carro, que saiu de novo.

Tudo com o modelo e a IA de pré-sinais do MS.1, e a visão do app hoje (3
detectores de mãos, realce desligado).

Uso::

    python estudar_ajustes_ms1.py
"""

from __future__ import annotations

import sys
import time

from lab_ms.kara import preparar_import

preparar_import()

import estudar_tudo
import estudar_videos
from estudar_tudo import preparar_caches, resumir, tabela, testes_visao
from estudar_videos import (
    ARMAZENAMENTO,
    PASTA_RESULTADOS,
    PASTA_VIDEOS,
    OpcoesVisao,
    quadros_do_video,
    reconhecer,
)

MS1 = ARMAZENAMENTO / "MS.1"
APP = {"detectores_maos": 3, "realce": False}


def videos() -> list:
    """Os 11 vídeos de teste (sem os de pré-sinal, que não são de sinal)."""
    return sorted(v for v in PASTA_VIDEOS.glob("*.avi") if not v.stem.startswith(("_", "presinal")))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    estudar_tudo.MODELO = MS1 / "modelo_sinais.pkl"
    estudar_videos.MODELO_PRE = MS1 / "modelo_sinais_pre.pkl"
    estudar_tudo.VARIANTES_VISAO = {
        12: ("Ajustes da visão com o MS.1", [
            ("app hoje (memória 0,8 s, detecção 0,3)", OpcoesVisao("cp_maos3", memoria_s=0.8, confianca_deteccao=0.3, **APP)),
            ("memória 1,0 s", OpcoesVisao("cp_maos3_mem10", memoria_s=1.0, confianca_deteccao=0.3, **APP)),
            ("detecção 0,2", OpcoesVisao("cp_maos3_det02", memoria_s=0.8, confianca_deteccao=0.2, **APP)),
            ("memória 1,0 s + detecção 0,2", OpcoesVisao("cp_maos3_mem10_det02", memoria_s=1.0, confianca_deteccao=0.2, **APP)),
        ]),
    }
    lista = videos()
    preparar_caches(lista)
    linhas = ["# Ajustes e votos com o MS.1, nos vídeos de teste", "", *testes_visao(lista)]

    base = estudar_tudo.VARIANTES_VISAO[12][1][0][1]
    dados_videos = {v.stem: quadros_do_video(v, refazer=False, opcoes=base) for v in lista}
    linhas_votos = {}
    for votos in (8, 12, 15):

        def ajustar(controlador, votos=votos):
            controlador.estabilizador.quadros_para_liberar = votos

        resultados = {
            stem: (
                reconhecer(vet, tem, estudar_tudo.MODELO, ajustar),
                {s for s in dados["sinal"].split(",") if s},
                vet,
            )
            for stem, (vet, tem, dados) in dados_videos.items()
        }
        linhas_votos[f"{votos} votos{' (app hoje)' if votos == 8 else ''}"] = resumir(resultados)
        print(f"votos {votos} ok", flush=True)
    linhas += tabela("13. Votos de outro sinal para liberar", linhas_votos)
    linhas += [
        "No vídeo carro→casa→carro há ~16 trocas reais (carro e casa alternando a cada ~2 s).",
        "",
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"ajustes_ms1_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
