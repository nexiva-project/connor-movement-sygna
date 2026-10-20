"""Testa nos vídeos as melhores ideias do estudo juntas.

Do ``estudar_tudo.py``: realce desligado, memória 0,8 s e detecção 0,3
ajudaram cada uma sozinha. Aqui elas rodam juntas e aos pares, para ver se
somam ou se uma atrapalha a outra (ex.: sem realce e com detecção baixa, a
IA pode achar mãos que não existem e o neutro piorar).

Uso::

    python estudar_combinacao.py
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
from estudar_videos import VISAO_DO_APP, OpcoesVisao

COMBINACOES = {
    10: ("Combinação das melhores", [
        ("app de hoje", VISAO_DO_APP),
        ("só realce desligado", OpcoesVisao("sem_realce", realce=False)),
        ("realce desligado + memória 0,8 s", OpcoesVisao("sem_realce_mem08", realce=False, memoria_s=0.8)),
        ("realce desligado + detecção 0,3", OpcoesVisao("sem_realce_det03", realce=False, confianca_deteccao=0.3)),
        (
            "as três juntas",
            OpcoesVisao("combo", realce=False, memoria_s=0.8, confianca_deteccao=0.3),
        ),
    ]),
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    estudar_tudo.VARIANTES_VISAO = COMBINACOES  # preparar_caches e testes_visao leem daqui
    videos = videos_de_teste()
    preparar_caches(videos)
    linhas = [
        "# Combinação das melhores ideias nos vídeos de teste",
        "",
        "Mesmos 11 vídeos, modelo sem máscara, liberação sem voto 60 quadros.",
        "",
        *testes_visao(videos),
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    arquivo = PASTA_RESULTADOS / f"combinacao_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print()
    print(texto)
    print(f"\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
