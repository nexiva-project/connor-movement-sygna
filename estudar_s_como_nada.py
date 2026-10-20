"""Ensinar à IA que o S parado (e o B) não é sinal: o S entra no treino como "nada".

No teste do pré-sinal, o S sozinho já era escrito como "carro" (o carro é
feito com as mãos em S). Aqui as gravações de pré-sinal do treino (S ×3, B ×3,
feitas no app) entram também na IA dos SINAIS com o rótulo "nada", que o app
nunca escreve (``ROTULOS_IGNORADOS``). Os vídeos de teste do S e do B NÃO
entram no treino.

Modelos comparados (todos com a IA de pré-sinais do MS.1):

1. MS.1;
2. MS.1 + S como "nada";
3. MS.1 + S e B como "nada".

Uso::

    python estudar_s_como_nada.py
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna.dados.repositorio import RepositorioSinais
from kara_sygna.ia import treinamento

import estudar_presinal
import estudar_videos
from estudar_presinal import carregar, colar, medir
from estudar_tudo import resumir, tabela
from estudar_videos import (
    ARMAZENAMENTO,
    PASTA_RESULTADOS,
    PASTA_VIDEOS,
    quadros_do_video,
    reconhecer,
)

MS1 = ARMAZENAMENTO / "MS.1"
PASTA = ARMAZENAMENTO / "s_como_nada"


def montar(como_nada: set[str]) -> RepositorioSinais:
    """CSV do MS.1 + as gravações de pré-sinal escolhidas, copiadas como sinal "nada"."""
    tabela_csv = pd.read_csv(MS1 / "dados_sinais.csv", dtype={"rotulo": str, "tipo": str, "pre_sinal": str})
    pre = tabela_csv[(tabela_csv.tipo == "pre") & tabela_csv.rotulo.isin(como_nada)].copy()
    pre["tipo"], pre["rotulo"], pre["pre_sinal"] = "sinal", "nada", ""
    pre["rodada"] = pre["rodada"] + 1  # rodada própria (não se mistura com a do pré-sinal)
    caminho = PASTA / f"dados_ms1_mais_{'_'.join(sorted(como_nada))}_nada.csv"
    pd.concat([tabela_csv, pre], ignore_index=True).to_csv(caminho, index=False, encoding="utf-8")
    return RepositorioSinais(caminho)


def presinal(modelo) -> dict[str, dict]:
    """S + carro e carro sozinho, com o modelo dado (IA de pré-sinais do MS.1)."""
    estudar_presinal.MODELO_SINAIS = modelo
    v_s, t_s = carregar("presinal_s")[:2]
    linhas = {}
    for video in ("carro_luz_normal_rapido", "carro_luz_normal_calmo", "carro_perto"):
        v_c, t_c = carregar(video)[:2]
        junto_v, junto_t = colar((v_s, t_s), (v_c, t_c))
        inicio = t_s[-1] + 1 / 30
        sozinho = medir(*estudar_presinal.reconhecer(v_c, t_c, True), 0.0, "carro", "S")
        junto = medir(*estudar_presinal.reconhecer(junto_v, junto_t, True), inicio, "carro", "S")
        linhas[video] = {
            "carro sozinho: 1º escrito": sozinho["1º sinal escrito após"],
            "S sozinho: escrito": junto["escrito durante o pré-sinal"],
            "S + carro: 1º carro após": junto["1º sinal escrito após"],
            "S + carro: voto certo": junto["voto certo no sinal"],
        }
    v, t, _ = carregar("presinal_s_carro")
    escritos = estudar_presinal.reconhecer(v, t, True)[0]
    linhas["etapas S→carro→S→carro"] = {
        "carro sozinho: 1º escrito": "-",
        "S sozinho: escrito": ", ".join(f"{s} {m:.1f} s" for m, s in escritos) or "nada",
        "S + carro: 1º carro após": "(carro entra aos 4 s e aos 14 s)",
        "S + carro: voto certo": "-",
    }
    return linhas


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    PASTA.mkdir(exist_ok=True)
    estudar_videos.MODELO_PRE = MS1 / "modelo_sinais_pre.pkl"

    modelos = {"MS.1": MS1 / "modelo_sinais.pkl"}
    for nome, como_nada in (("MS.1 + S como nada", {"S"}), ("MS.1 + S e B como nada", {"S", "B"})):
        caminho = PASTA / f"modelo_{'_'.join(sorted(como_nada))}_nada.pkl"
        resultado = treinamento.treinar(montar(como_nada).carregar(), caminho)
        modelos[nome] = caminho
        print(f"treinado: {nome} (validação {100 * (resultado.acuracia_validacao or 0):.1f}%)", flush=True)

    linhas = ["# S parado como \"nada\": testes nos vídeos", ""]
    for nome, caminho in modelos.items():
        linhas += tabela(f"Pré-sinal com {nome}", presinal(caminho))
        print(f"pré-sinal ok: {nome}", flush=True)

    videos = sorted(v for v in PASTA_VIDEOS.glob("*.avi") if not v.stem.startswith(("_", "presinal")))
    base = {v.stem: quadros_do_video(v, refazer=False, opcoes=estudar_presinal.VISAO) for v in videos}
    gerais = {}
    for nome, caminho in modelos.items():
        resultados = {
            stem: (reconhecer(vet, tem, caminho), {s for s in dados["sinal"].split(",") if s}, vet)
            for stem, (vet, tem, dados) in base.items()
        }
        gerais[nome] = resumir(resultados)
        votos_nada = sum(r.votos["nada"] for r, _, _ in resultados.values())
        gerais[nome]["votos \"nada\" nos 11 vídeos"] = votos_nada
        print(f"vídeos ok: {nome}", flush=True)
    linhas += tabela("Os 11 vídeos de teste (carro e casa não podem piorar)", gerais)
    linhas.append(f"Tempo total: {time.perf_counter() - inicio:.0f} s")

    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"s_como_nada_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
