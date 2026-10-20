"""Testa nos vídeos dois detectores de mãos em revezamento.

Ao vivo, o detector de mãos leva ~46 ms por quadro, e o app trata ~15 dos 30
quadros por segundo da câmera. Com dois detectores em revezamento (pares no
A, ímpares no B, em threads), a velocidade medida foi de 17,5 para 30,6 qps.
Aqui se mede a QUALIDADE: cada detector vê só um quadro sim, um não, o que pode
atrapalhar o rastreio da mão.

Todas as versões usam os ajustes da cópia: realce desligado, memória 0,8 s e
detecção 0,3 (nas mãos e no Pose, como o app).

Uso::

    python estudar_revezamento.py
"""

from __future__ import annotations

import sys
import time

from lab_ms.kara import preparar_import

preparar_import()

import estudar_tudo
from estudar_tudo import (
    PASTA_RESULTADOS,
    preparar_caches,
    testes_visao,
    videos_de_teste,
)
from estudar_videos import OpcoesVisao

COPIA = {"realce": False, "memoria_s": 0.8, "confianca_deteccao": 0.3}

VARIANTES = {
    11: ("Dois detectores de mãos em revezamento", [
        ("ao vivo hoje: 1 detector, 1 quadro a cada 2 (~15 qps)", OpcoesVisao("cp_metade", a_cada=2, **COPIA)),
        ("2 detectores em revezamento, todos os quadros (30 qps)", OpcoesVisao("cp_revez", revezamento=True, **COPIA)),
        ("referência: 1 detector, todos os quadros", OpcoesVisao("cp_todos", **COPIA)),
        ("3 detectores de mãos em revezamento, todos os quadros", OpcoesVisao("cp_maos3", detectores_maos=3, **COPIA)),
        ("4 detectores de mãos em revezamento, todos os quadros", OpcoesVisao("cp_maos4", detectores_maos=4, **COPIA)),
        ("app com 2 equipes completas (rosto, braços, mãos)", OpcoesVisao("cp_eq2", equipes=2, **COPIA)),
    ]),
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    estudar_tudo.VARIANTES_VISAO = VARIANTES
    videos = videos_de_teste()
    preparar_caches(videos)
    linhas = [
        "# Dois detectores de mãos em revezamento, nos vídeos de teste",
        "",
        "Ajustes da cópia (realce desligado, memória 0,8 s, detecção 0,3), modelo sem máscara.",
        "As regras contadas em quadros (votação, liberação) valem por quadro TRATADO, como ao vivo.",
        "",
        *testes_visao(videos),
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"revezamento_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
