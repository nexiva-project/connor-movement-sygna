"""Preencher os buracos da mão também ao vivo (teste 1 da lista de 20/10/2026).

Na gravação, o app já preenche com uma curva (PCHIP) a mão que sumiu por até
0,5 s (``kara_sygna.visao.preenchimento``): a IA aprende o caminho completo.
Ao vivo não dá para saber onde a mão volta, então a memória das mãos só
CONGELA a última posição por até 0,8 s. A IA treina com curva e vê mão parada.

Ao vivo, preencher exige esperar: com uma espera de K quadros, um buraco de
até K quadros é preenchido quando a mão volta; os maiores continuam
congelados. Tudo sai K quadros mais tarde (K/30 s).

Variantes, todas com o modelo e as regras do app hoje (portão e região):

* app hoje: memória 0,8 s na visão (congela).
* sem memória: a visão sem memória nenhuma (o buraco fica vazio).
* sem memória + congelar 0,8 s aqui (confere que imita o app).
* sem memória + preencher até K quadros + congelar o resto, K = 3, 5, 15.

Uso::

    python estudar_preencher_ao_vivo.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

from lab_ms.kara import preparar_import

preparar_import()

import estudar_tudo
from estudar_tudo import _esperados, preparar_caches, resumir, tabela
from estudar_videos import (
    ARMAZENAMENTO,
    PASTA_RESULTADOS,
    PASTA_VIDEOS,
    OpcoesVisao,
    quadros_do_video,
    reconhecer,
)
from kara_sygna.visao.preenchimento import preencher_buracos
from kara_sygna.visao.quadro import colunas_da_mao, maos_presentes

MODELO = ARMAZENAMENTO / "modelo_sinais.pkl"  # o do app hoje (S como nada)
APP = {"detectores_maos": 3, "realce": False, "confianca_deteccao": 0.3}
VISAO_APP = OpcoesVisao("cp_maos3", memoria_s=0.8, **APP)
VISAO_SEM_MEMORIA = OpcoesVisao("cp_maos3_mem0", memoria_s=0.0, **APP)
QPS = 30.0


def congelar(quadros: np.ndarray, tempos: np.ndarray, memoria_s: float = 0.8) -> np.ndarray:
    """Copia a última mão vista nos quadros em que ela sumiu, por até ``memoria_s``."""
    quadros = quadros.copy()
    presentes = maos_presentes(quadros)
    for coluna, lado in enumerate(("esq", "dir")):
        colunas = colunas_da_mao(lado)
        ultimo = None
        for i in range(len(quadros)):
            if presentes[i, coluna]:
                ultimo = i
            elif ultimo is not None and tempos[i] - tempos[ultimo] <= memoria_s:
                quadros[i, colunas] = quadros[ultimo, colunas]
    return quadros


def preencher_e_congelar(quadros: np.ndarray, tempos: np.ndarray, espera: int) -> np.ndarray:
    """Buracos de até ``espera`` quadros com curva; os maiores, congelados."""
    preenchidos, _ = preencher_buracos(quadros, tempos, maximo_s=(espera + 1.5) / QPS)
    return congelar(preenchidos, tempos)


def videos() -> list:
    return sorted(v for v in PASTA_VIDEOS.glob("*.avi") if not v.stem.startswith("_"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    lista = videos()
    estudar_tudo.VARIANTES_VISAO = {0: ("", [("app", VISAO_APP), ("sem memória", VISAO_SEM_MEMORIA)])}
    preparar_caches(lista)

    app = {v.stem: quadros_do_video(v, refazer=False, opcoes=VISAO_APP) for v in lista}
    crus = {v.stem: quadros_do_video(v, refazer=False, opcoes=VISAO_SEM_MEMORIA) for v in lista}
    variantes = {
        "app hoje (congela 0,8 s)": {s: (v, t) for s, (v, t, _) in app.items()},
        "sem memória (buraco vazio)": {s: (v, t) for s, (v, t, _) in crus.items()},
        "sem memória + congelar 0,8 s aqui": {s: (congelar(v, t), t) for s, (v, t, _) in crus.items()},
    }
    for espera in (3, 5, 15):
        variantes[f"preencher até {espera} quadros (espera {espera / QPS:.2f} s) + congelar"] = {
            s: (preencher_e_congelar(v, t, espera), t) for s, (v, t, _) in crus.items()
        }

    linhas_tabela = {}
    for nome, dados in variantes.items():
        resultados = {}
        for stem, (vetores, tempos) in dados.items():
            esperados = _esperados(app[stem][2])
            resultados[stem] = (reconhecer(vetores, tempos, MODELO), esperados, vetores)
        sinais = {s: r for s, r in resultados.items() if not s.startswith("presinal")}
        linha = resumir(sinais)
        linha["S sozinho escreve"] = _escritos(resultados, "presinal_s")
        linha["S + carro escreve"] = _escritos(resultados, "presinal_s_carro")
        linhas_tabela[nome] = linha
        print(f"  ok: {nome}", flush=True)

    linhas = [
        "# Preencher os buracos da mão ao vivo (vídeos de teste)",
        "",
        "Modelo do app hoje (S como nada), com portão de movimento e região das mãos.",
        "Com espera de K quadros, ao vivo o sinal sairia K/30 s mais tarde.",
        "",
        *tabela("Preencher ao vivo", linhas_tabela),
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"preencher_ao_vivo_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


def _escritos(resultados: dict, stem: str) -> str:
    if stem not in resultados:
        return "-"
    return ", ".join(f"{s}×{n}" for s, n in Counter(resultados[stem][0].escritos).items()) or "nada"


if __name__ == "__main__":
    main()
