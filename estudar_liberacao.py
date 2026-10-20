"""Testa nos vídeos quanto tempo a mão sumida leva para liberar o sinal.

Depois de escrever um sinal, ele só sai de novo quando é "liberado". Antes,
8 quadros sem voto (mão borrada, IA incerta) já liberavam, e o carro rápido
era escrito a cada volta. Aqui o mesmo modelo (sem máscara) e os mesmos
vídeos rodam com valores diferentes de ``quadros_para_liberar_sem_voto``;
voto em OUTRO sinal continua liberando em 8 quadros.

Uso::

    python estudar_liberacao.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter

from estudar_videos import (
    ARMAZENAMENTO,
    MODELOS,
    PASTA_RESULTADOS,
    PASTA_VIDEOS,
    quadros_do_video,
    reconhecer,
)

#: Quadros sem voto para liberar (30 = 1 s a 30 qps). 10**9 = nunca.
VALORES = {"8 (antes)": 8, "15 (0,5 s)": 15, "30 (1 s)": 30, "60 (2 s)": 60, "nunca": 10**9}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    modelo = ARMAZENAMENTO / MODELOS["sem mascara"]
    videos = sorted(v for v in PASTA_VIDEOS.glob("*.avi") if not v.stem.startswith("_"))
    linhas = [
        "# Liberação do sinal com a mão sumida",
        "",
        "Quantas vezes cada sinal viraria texto em cada vídeo (20 s), com o modelo sem máscara.",
        "O ideal é 1 por sinal feito (carro→casa→carro: 3; neutro: 0).",
        "",
        "| Vídeo | " + " | ".join(VALORES) + " |",
        "|---" * (len(VALORES) + 1) + "|",
    ]
    for video in videos:
        vetores, tempos, _ = quadros_do_video(video, refazer=False)
        celulas = []
        for valor in VALORES.values():

            def ajustar(controlador, valor=valor):
                controlador.estabilizador.quadros_para_liberar_sem_voto = valor

            escritos = Counter(reconhecer(vetores, tempos, modelo, ajustar).escritos)
            celulas.append(", ".join(f"{s}×{n}" for s, n in escritos.items()) or "nada")
        linhas.append(f"| {video.stem} | " + " | ".join(celulas) + " |")
        print(linhas[-1], flush=True)

    texto = "\n".join(linhas)
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    arquivo = PASTA_RESULTADOS / f"liberacao_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
