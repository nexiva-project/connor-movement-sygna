"""As 4 IAs (RandomForest, LSTM, GRU, Conv1D) em todos os sinais, sem câmera.

Para cada sinal com 2 ou mais gravações, a gravação mais nova fica de fora do
treino e serve de teste (só as janelas originais, sem cópias espelhadas ou
giradas): a IA nunca viu aquela gravação. Sinais com uma gravação só entram
no treino, mas não podem ser testados.

Os modelos ficam só na memória: os do app não mudam.

Uso::

    python estudar_todas_ias.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna.dados.repositorio import ConjuntoDados, RepositorioSinais
from kara_sygna.ia import treinamento
from kara_sygna.ia.redes import treinar_rede
from kara_sygna.ia.sequencia import montar_amostras, sequencias_de_uma_gravacao

from estudar_videos import ARMAZENAMENTO, PASTA_RESULTADOS

IAS = ("randomforest", "lstm", "gru", "conv1d")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    repositorio = RepositorioSinais(ARMAZENAMENTO / "dados_sinais.csv")
    dados = repositorio.carregar()
    livres = repositorio.sinais_direcao_livre()
    teste_grupos = {}
    for sinal in dados.classes:
        grupos = np.unique(dados.grupos[dados.y == sinal])
        if len(grupos) >= 2:
            teste_grupos[sinal] = int(max(grupos))  # a gravação mais nova
    sem_teste = [s for s in dados.classes if s not in teste_grupos]

    acertos: dict[str, dict[str, str]] = {s: {} for s in teste_grupos}
    totais = {}
    for ia in IAS:
        extras = {} if ia == "randomforest" else {"extrair": sequencias_de_uma_gravacao}
        amostras = montar_amostras(dados, direcao_livre=livres, **extras)
        no_teste = np.isin(amostras.grupos, list(teste_grupos.values()))
        originais = ~amostras.mascarada if amostras.mascarada is not None else True
        teste = no_teste & originais
        if ia == "randomforest":
            modelo = treinamento._ajustar(amostras.X[~no_teste], amostras.y[~no_teste])
            previstos = modelo.predict(
                pd.DataFrame(amostras.X[teste], columns=modelo.feature_names_in_)
            )
        else:
            treino = ConjuntoDados(
                amostras.X[~no_teste], amostras.y[~no_teste], amostras.grupos[~no_teste]
            )
            modelo = treinar_rede(ia, treino)[0]
            previstos = modelo.classes_[modelo.predict_proba(amostras.X[teste]).argmax(axis=1)]
        reais = amostras.y[teste]
        totais[ia] = f"{100 * np.mean(previstos == reais):.0f}%"
        for sinal in teste_grupos:
            contagem = Counter(previstos[reais == sinal])
            total = sum(contagem.values())
            certo = contagem[sinal] / total if total else 0.0
            erro = next(((s, n) for s, n in contagem.most_common() if s != sinal), None)
            texto = f"{100 * certo:.0f}%"
            if erro and erro[1] / total >= 0.1:
                texto += f" (vira {erro[0]})"
            acertos[sinal][ia] = texto
        print(f"  ok: {ia}", flush=True)

    linhas = [
        "# As 4 IAs em todos os sinais (gravação nunca vista)",
        "",
        "Teste: a gravação mais nova de cada sinal, fora do treino. "
        "Entre parênteses, o erro mais comum (se ≥10%).",
        "",
        "| Sinal | " + " | ".join(IAS) + " |",
        "|---" * (len(IAS) + 1) + "|",
    ]
    for sinal in teste_grupos:
        linhas.append(f"| {sinal} | " + " | ".join(acertos[sinal][ia] for ia in IAS) + " |")
    linhas.append("| **total** | " + " | ".join(f"**{totais[ia]}**" for ia in IAS) + " |")
    linhas += [
        "",
        f"Sem teste (só 1 gravação): {', '.join(sem_teste) or 'nenhum'}.",
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"todas_ias_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
