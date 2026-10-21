"""Filtro contra tremida (One-Euro) nos pontos que a IA vê (teste 3 de 20/10/2026).

O app já suaviza os pontos, mas só no DESENHO da tela
(``kara_sygna.visao.suavizacao``); a IA recebe os pontos crus, com a tremida
do MediaPipe. O filtro One-Euro (Casiez et al., 2012; usado no
satyaidk/sign-language-recognition) suaviza muito com a mão quase parada e
quase nada com ela rápida, então não atrasa o movimento.

Filtrados: a forma e a posição/tamanho de cada mão, e cotovelos e pulsos
(Pose). Ao sumir a mão (ou o corpo), o filtro recomeça.

Variantes, com o modelo e as regras do app hoje:

* app hoje (sem filtro);
* filtro só ao vivo (vídeos filtrados, modelo igual): leve, médio e forte;
* filtro também no treino (gravações filtradas e modelo novo), com o melhor.

Uso::

    python estudar_one_euro.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna.dados.repositorio import ConjuntoDados, RepositorioSinais
from kara_sygna.ia import treinamento
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_POSICAO,
    IDX_BRACOS,
    IDX_FORMA,
    maos_presentes,
)

import estudar_videos
from estudar_preencher_ao_vivo import VISAO_APP, _escritos, videos
from estudar_tudo import _esperados, resumir, tabela
from estudar_videos import ARMAZENAMENTO, PASTA_RESULTADOS, quadros_do_video, reconhecer

PASTA = ARMAZENAMENTO / "one_euro"
DADOS = ARMAZENAMENTO / "s_como_nada" / "dados_ms1_mais_S_nada.csv"
MODELO_APP = ARMAZENAMENTO / "modelo_sinais.pkl"
QPS = 30.0

#: (corte mínimo em Hz, beta): corte menor = mais suave parado; beta maior = segue mais rápido.
FORCAS = {"leve": (2.0, 0.5), "médio": (1.0, 0.5), "forte": (0.5, 0.3)}

GRUPOS = [
    # (colunas, função que diz em quais quadros o grupo existe)
    (np.r_[np.arange(IDX_FORMA[lado].start, IDX_FORMA[lado].stop), list(COLUNAS_POSICAO[lado])],
     lambda q, i=i: maos_presentes(q)[:, i])
    for i, lado in enumerate(("esq", "dir"))
] + [(np.arange(IDX_BRACOS.start + 1, IDX_BRACOS.stop), lambda q: q[:, COLUNA_CORPO_PRESENTE] > 0.5)]


def _alfa(corte: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2 * np.pi * corte)
    return 1.0 / (1.0 + tau / dt)


def one_euro(quadros: np.ndarray, tempos: np.ndarray, corte_min: float, beta: float) -> np.ndarray:
    """Filtra os grupos de colunas, quadro a quadro, como seria ao vivo (só o passado)."""
    saida = quadros.copy()
    for colunas, existe in GRUPOS:
        vivos = existe(quadros)
        anterior = derivada = None
        for i in range(len(quadros)):
            if not vivos[i]:
                anterior = None
                continue
            x = quadros[i, colunas]
            if anterior is None:
                anterior, derivada = x, np.zeros_like(x)
                continue
            dt = max(1e-3, tempos[i] - tempos[i - 1])
            derivada = derivada + _alfa(1.0, dt) * ((x - anterior) / dt - derivada)
            corte = corte_min + beta * np.abs(derivada)
            anterior = anterior + _alfa(corte, dt) * (x - anterior)
            saida[i, colunas] = anterior
    return saida


def filtrar_gravacoes(dados: ConjuntoDados, corte_min: float, beta: float) -> ConjuntoDados:
    """As gravações de treino filtradas, cada rodada separada (a 30 quadros/s)."""
    X = dados.X.copy()
    for rodada in np.unique(dados.grupos):
        linhas = np.flatnonzero(dados.grupos == rodada)
        X[linhas] = one_euro(dados.X[linhas], np.arange(len(linhas)) / QPS, corte_min, beta)
    return ConjuntoDados(X, dados.y, dados.grupos)


def avaliar(base: dict, modelo, filtro=None) -> dict:
    resultados = {}
    for stem, (vet, tem, info) in base.items():
        if filtro is not None:
            vet = one_euro(vet, tem, *filtro)
        resultados[stem] = (reconhecer(vet, tem, modelo), _esperados(info), vet)
    sinais = {s: r for s, r in resultados.items() if not s.startswith("presinal")}
    linha = resumir(sinais)
    linha["S sozinho escreve"] = _escritos(resultados, "presinal_s")
    linha["S + carro escreve"] = _escritos(resultados, "presinal_s_carro")
    return linha


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    PASTA.mkdir(exist_ok=True)
    estudar_videos.MODELO_PRE = ARMAZENAMENTO / "MS.1" / "modelo_sinais_pre.pkl"
    base = {v.stem: quadros_do_video(v, refazer=False, opcoes=VISAO_APP) for v in videos()}

    linhas_tabela = {"app hoje (sem filtro)": avaliar(base, MODELO_APP)}
    print("  ok: app hoje", flush=True)
    for nome, filtro in FORCAS.items():
        linhas_tabela[f"só ao vivo, {nome} {filtro}"] = avaliar(base, MODELO_APP, filtro)
        print(f"  ok: só ao vivo {nome}", flush=True)

    dados = RepositorioSinais(DADOS).carregar()
    for nome in ("leve", "médio"):
        filtro = FORCAS[nome]
        modelo = PASTA / f"modelo_one_euro_{nome}.pkl"
        resultado = treinamento.treinar(filtrar_gravacoes(dados, *filtro), modelo)
        linha = avaliar(base, modelo, filtro)
        linha = {"validação": f"{100 * (resultado.acuracia_validacao or 0):.1f}%", **linha}
        linhas_tabela[f"treino + ao vivo, {nome} {filtro}"] = linha
        print(f"  ok: treino + ao vivo {nome}", flush=True)
    for linha in linhas_tabela.values():
        linha.setdefault("validação", "-")
    linhas_tabela = {n: {"validação": l.pop("validação"), **l} for n, l in linhas_tabela.items()}

    linhas = [
        "# Filtro One-Euro nos pontos da IA (vídeos de teste)",
        "",
        "Modelo do app hoje (MS.1 + S como nada), IA de pré-sinais do MS.1, visão do app,",
        "portão de movimento e região das mãos ligados. Filtro: (corte mínimo Hz, beta).",
        "",
        *tabela("One-Euro", linhas_tabela),
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"one_euro_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
