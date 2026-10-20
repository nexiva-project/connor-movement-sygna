"""Compara, nos vídeos de teste, modelos treinados com gravações diferentes.

Em 19/10/2026 o carro e a casa foram regravados no app da cópia já com os
ajustes novos (realce desligado, 30 qps). As gravações e o modelo de antes
ficaram em ``app/armazenamento/backup_antes_regravar/``. Aqui são treinados e
comparados:

1. antigo: o modelo de antes (backup);
2. novo: só as gravações de hoje;
3. novo sem a gravação de carro que saiu com aviso (mãos em 48%);
4. antigas + novas juntas;
5. antigas + novas, sem a gravação com aviso.

Todos usam a mesma visão (a da cópia: realce desligado, memória 0,8 s,
detecção 0,3, 3 detectores de mãos) e a mesma IA de pré-sinais (a do backup),
para medir só o efeito das gravações.

Uso::

    python comparar_regravacao.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna.dados.repositorio import RepositorioSinais
from kara_sygna.ia import treinamento

import estudar_videos
from estudar_tudo import resumir, tabela, videos_de_teste
from estudar_videos import (
    ARMAZENAMENTO,
    PASTA_RESULTADOS,
    OpcoesVisao,
    quadros_do_video,
    reconhecer,
)

BACKUP = ARMAZENAMENTO / "backup_antes_regravar"
PASTA = ARMAZENAMENTO / "regravacao"
#: Gravação de carro que saiu com "ATENÇÃO, grave de novo" (mãos em 48%).
RODADA_COM_AVISO = 1792454351262
VISAO = OpcoesVisao("cp_maos3", detectores_maos=3, realce=False, memoria_s=0.8, confianca_deteccao=0.3)


def montar_csvs() -> dict[str, Path]:
    """Escreve o CSV de cada conjunto de gravações e devolve ``{nome: caminho}``."""
    PASTA.mkdir(exist_ok=True)
    atual = pd.read_csv(ARMAZENAMENTO / "dados_sinais.csv")
    antigo = pd.read_csv(BACKUP / "dados_sinais.csv")
    so_antigas = antigo[~antigo.rodada.isin(set(atual.rodada))]  # tudo que foi apagado
    conjuntos = {
        "novo": atual,
        "novo sem a gravação com aviso": atual[atual.rodada != RODADA_COM_AVISO],
        "antigas + novas": pd.concat([so_antigas, atual], ignore_index=True),
        "antigas + novas sem a gravação com aviso": pd.concat(
            [so_antigas, atual[atual.rodada != RODADA_COM_AVISO]], ignore_index=True
        ),
    }
    caminhos = {}
    for nome, tabela_csv in conjuntos.items():
        caminho = PASTA / f"dados_{nome.replace(' ', '_').replace('+', 'mais')}.csv"
        tabela_csv.to_csv(caminho, index=False, encoding="utf-8")
        caminhos[nome] = caminho
    return caminhos


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    # Mesma IA de pré-sinais para todos (a do backup; a atual foi apagada junto com os sinais).
    estudar_videos.MODELO_PRE = BACKUP / "modelo_sinais_pre.pkl"

    # Cópia do modelo antigo: o app recusa modelo mais velho que o CSV atual.
    PASTA.mkdir(exist_ok=True)
    antigo = PASTA / "modelo_antigo.pkl"
    shutil.copyfile(BACKUP / "modelo_sinais.pkl", antigo)
    modelos = {"antigo (backup)": antigo}
    validacoes = {"antigo (backup)": "99,0%"}
    for nome, csv in montar_csvs().items():
        caminho = PASTA / f"modelo_{csv.stem.removeprefix('dados_')}.pkl"
        resultado = treinamento.treinar(RepositorioSinais(csv).carregar(), caminho)
        modelos[nome] = caminho
        acuracia = resultado.acuracia_validacao
        validacoes[nome] = "-" if acuracia is None else f"{100 * acuracia:.1f}%"
        print(f"treinado: {nome} (validação {validacoes[nome]})", flush=True)

    videos = videos_de_teste()
    base = {v.stem: quadros_do_video(v, refazer=False, opcoes=VISAO) for v in videos}
    linhas_tabela = {}
    for nome, caminho in modelos.items():
        resultados = {
            stem: (reconhecer(vetores, tempos, caminho), {s for s in dados["sinal"].split(",") if s}, vetores)
            for stem, (vetores, tempos, dados) in base.items()
        }
        linhas_tabela[nome] = {"validação no treino": validacoes[nome], **resumir(resultados)}
        print(f"vídeos: {nome}", flush=True)

    texto = "\n".join(
        [
            "# Regravação de carro e casa: antigo × novo, nos vídeos de teste",
            "",
            "Visão da cópia (realce desligado, memória 0,8 s, detecção 0,3, 3 detectores de mãos);",
            "mesma IA de pré-sinais para todos. A validação no treino mede gravações do mesmo",
            "conjunto; os vídeos são a prova independente.",
            "",
            *tabela("Modelos", linhas_tabela),
            f"Tempo total: {time.perf_counter() - inicio:.0f} s",
        ]
    )
    arquivo = PASTA_RESULTADOS / f"regravacao_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
