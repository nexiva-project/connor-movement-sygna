"""Tirar o rosto da IA dos sinais (teste 2 da lista de 20/10/2026).

No danielelvs/islr-subset (Libras com MediaPipe), o melhor conjunto foi corpo
+ duas mãos, sem o rosto. Com ``USAR_LUGAR = False`` o app já não usa a
distância dos dedos até o rosto nem a posição das mãos; do rosto ainda entram:

* se o rosto foi visto (1 coluna);
* a expressão: olhos, boca, sobrancelhas, inclinação (9 colunas);
* o tamanho da palma, medido em larguras de rosto (2 colunas).

Modelos treinados com as gravações do app hoje (MS.1 + S como "nada"):

1. app hoje (com tudo);
2. sem a expressão do rosto (sai o rosto visto e a expressão);
3. sem rosto nenhum (sai também o tamanho da palma).

A IA de pré-sinais fica a do MS.1, igual. Nada muda no app.

Uso::

    python estudar_sem_rosto.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna.dados.repositorio import RepositorioSinais
from kara_sygna.ia import sequencia, treinamento
from kara_sygna.visao.quadro import COLUNAS_POSICAO, IDX_ROSTO_MAOS

import estudar_videos
from estudar_preencher_ao_vivo import VISAO_APP, _escritos, videos
from estudar_tudo import _esperados, resumir, tabela
from estudar_videos import ARMAZENAMENTO, PASTA_RESULTADOS, quadros_do_video, reconhecer

PASTA = ARMAZENAMENTO / "sem_rosto"
DADOS = ARMAZENAMENTO / "s_como_nada" / "dados_ms1_mais_S_nada.csv"
MODELO_APP = ARMAZENAMENTO / "modelo_sinais.pkl"

EXPRESSAO = [IDX_ROSTO_MAOS.start, *range(IDX_ROSTO_MAOS.start + 3, IDX_ROSTO_MAOS.start + 12)]
PALMA = [COLUNAS_POSICAO["esq"][2], COLUNAS_POSICAO["dir"][2]]
VARIANTES = {
    "app hoje (com tudo)": None,
    "sem a expressão do rosto": EXPRESSAO,
    "sem rosto nenhum": EXPRESSAO + PALMA,
}

_ORIGINAL = sequencia._colunas_quadro


def usar_sem(colunas: list[int] | None) -> None:
    """Faz a janela da IA deixar de fora estas colunas do quadro."""
    if colunas is None:
        sequencia._colunas_quadro = _ORIGINAL
        return
    fora = np.array(colunas)
    sequencia._colunas_quadro = lambda usar_lugar: np.setdiff1d(_ORIGINAL(usar_lugar), fora)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    PASTA.mkdir(exist_ok=True)
    estudar_videos.MODELO_PRE = ARMAZENAMENTO / "MS.1" / "modelo_sinais_pre.pkl"
    lista = videos()
    base = {v.stem: quadros_do_video(v, refazer=False, opcoes=VISAO_APP) for v in lista}
    dados = RepositorioSinais(DADOS).carregar()

    linhas_tabela = {}
    for nome, colunas in VARIANTES.items():
        usar_sem(colunas)
        try:
            if colunas is None:
                modelo, validacao = MODELO_APP, "-"
            else:
                modelo = PASTA / f"modelo_{len(colunas)}_fora.pkl"
                resultado = treinamento.treinar(dados, modelo)
                validacao = f"{100 * (resultado.acuracia_validacao or 0):.1f}%"
            resultados = {
                stem: (reconhecer(vet, tem, modelo), _esperados(info), vet)
                for stem, (vet, tem, info) in base.items()
            }
        finally:
            usar_sem(None)
        sinais = {s: r for s, r in resultados.items() if not s.startswith("presinal")}
        linha = {"colunas fora": len(colunas or []), "validação": validacao, **resumir(sinais)}
        linha["S sozinho escreve"] = _escritos(resultados, "presinal_s")
        linha["S + carro escreve"] = _escritos(resultados, "presinal_s_carro")
        linhas_tabela[nome] = linha
        print(f"  ok: {nome}", flush=True)

    linhas = [
        "# Tirar o rosto da IA dos sinais (vídeos de teste)",
        "",
        "Gravações do app hoje (MS.1 + S como nada), IA de pré-sinais do MS.1, visão do app",
        "(3 detectores de mãos), portão de movimento e região das mãos ligados.",
        "",
        *tabela("Sem rosto", linhas_tabela),
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"sem_rosto_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
