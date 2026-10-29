"""Números 1 a 9 trocando rápido: LSTM contra RandomForest, sem câmera.

Cada número tem uma gravação só. O começo dela (60%) vai para o treino e o fim
(40%) fica de fora, para o teste: a IA nunca viu esse pedaço. Os pedaços de
teste são colados em sequência, ~0,6 s cada número, como se a pessoa trocasse
rápido (duas voltas, em ordens diferentes), e passam pelo app inteiro:
votação, portão de movimento, região das mãos. Os outros sinais (letras,
carro, casa, avião) entram inteiros no treino, como no app.

Os modelos ficam numa pasta temporária: os do app não mudam.

Uso::

    python estudar_numeros_rapido.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais
from kara_sygna.dados.repositorio import RepositorioSinais
from kara_sygna.ia import tipos_ia
from kara_sygna.visao.quadro import nomes_features_quadro

from estudar_videos import ARMAZENAMENTO, PASTA_RESULTADOS, Relogio

NUMEROS = [str(n) for n in range(1, 10)]
PARTE_TREINO = 0.6
QPS = 30.0
QUADROS_POR_NUMERO = 18  # ~0,6 s


def separar(caminho_csv: Path, pasta: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """CSV de treino sem o fim de cada número, e os fins (quadros) para o teste."""
    tabela = pd.read_csv(caminho_csv, dtype={"rotulo": str, "tipo": str, "pre_sinal": str})
    tabela["pre_sinal"] = tabela["pre_sinal"].fillna("")
    teste: dict[str, np.ndarray] = {}
    ficam = np.ones(len(tabela), dtype=bool)
    for numero in NUMEROS:
        linhas = np.flatnonzero((tabela.rotulo == numero) & (tabela.tipo == "sinal"))
        corte = int(len(linhas) * PARTE_TREINO)
        ficam[linhas[corte:]] = False
        teste[numero] = tabela.iloc[linhas[corte:]][nomes_features_quadro()].to_numpy(float)
    destino = pasta / "dados_sinais.csv"
    tabela[ficam].to_csv(destino, index=False, encoding="utf-8")
    return destino, teste


def sequencia_rapida(teste: dict[str, np.ndarray], rng) -> tuple[np.ndarray, list[str]]:
    """Duas voltas por 1..9 (a segunda embaralhada), ~0,6 s cada número."""
    ordem = NUMEROS + [str(n) for n in rng.permutation(NUMEROS)]
    pedacos = []
    for numero in ordem:
        quadros = teste[numero]
        inicio = rng.integers(0, max(1, len(quadros) - QUADROS_POR_NUMERO))
        pedaco = quadros[inicio : inicio + QUADROS_POR_NUMERO]
        if len(pedaco) < QUADROS_POR_NUMERO:  # gravação curta: repete o fim
            falta = QUADROS_POR_NUMERO - len(pedaco)
            pedaco = np.vstack([pedaco, np.repeat(pedaco[-1:], falta, axis=0)])
        pedacos.append(pedaco)
    return np.vstack(pedacos), ordem


def reconhecer(controlador, quadros: np.ndarray, relogio: Relogio) -> list[tuple[float, str]]:
    controlador.iniciar_reconhecimento()
    escritos = []
    for i, vetor in enumerate(quadros):
        relogio.agora = i / QPS
        resposta = controlador.processar_quadro(vetor)
        if resposta.sinal_confirmado:
            escritos.append((relogio.agora, resposta.sinal_confirmado))
    controlador.parar()
    return escritos


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    config.DIAGNOSTICO_ATIVO = False
    rng = np.random.default_rng(7)
    linhas = ["# Números 1 a 9 trocando rápido (sem câmera)", ""]
    with tempfile.TemporaryDirectory() as temporaria:
        pasta = Path(temporaria)
        csv_treino, teste = separar(ARMAZENAMENTO / "dados_sinais.csv", pasta)
        (pasta / "dados_sinais_direcao_livre.json").write_text('["avião"]', encoding="utf-8")
        quadros, ordem = sequencia_rapida(teste, rng)
        tamanhos = [len(q) for q in teste.values()]
        linhas += [
            f"Teste: o fim de cada gravação ({100 - 100 * PARTE_TREINO:.0f}%, de "
            f"{min(tamanhos)} a {max(tamanhos)} quadros), nunca visto no treino.",
            f"Sequência: {' '.join(ordem)} ({QUADROS_POR_NUMERO / QPS:.1f} s cada número).",
            "",
            "| IA | Validação | Números certos na hora certa | Escrito (número@segundo) |",
            "|---|---|---|---|",
        ]
        for tipo in ("randomforest", "lstm", "gru", "conv1d"):
            relogio = Relogio()
            controlador = ControladorSinais(
                repositorio=RepositorioSinais(csv_treino),
                caminho_modelo=tipos_ia.caminho_modelo(tipo, pasta),
                relogio=relogio,
            )
            resultado = controlador.treinar()
            escritos = reconhecer(controlador, quadros, relogio)
            # Certo: o número escrito durante o pedaço dele (ou até 0,3 s depois).
            certos = 0
            for posicao, numero in enumerate(ordem):
                de = posicao * QUADROS_POR_NUMERO / QPS
                ate = de + QUADROS_POR_NUMERO / QPS + 0.3
                certos += any(s == numero and de <= t < ate for t, s in escritos)
            texto = " ".join(f"{s}@{t:.1f}" for t, s in escritos) or "nada"
            linhas.append(
                f"| {tipo} | {100 * (resultado.acuracia_validacao or 0):.1f}% | "
                f"{certos}/{len(ordem)} | {texto} |"
            )
            print(f"  ok: {tipo}", flush=True)
    linhas += ["", f"Tempo total: {time.perf_counter() - inicio:.0f} s"]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"numeros_rapido_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
