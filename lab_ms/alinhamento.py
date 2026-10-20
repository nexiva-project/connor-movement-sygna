"""Alinhamento do MS (técnica: Dynamic Time Warping, DTW), só com numpy.

Duas versões:

* :func:`alinhar` compara duas sequências inteiras (início com início, fim com fim).
  Serve para comparar uma gravação inteira com outra.
* :func:`alinhar_trecho` procura o melhor trecho de uma sequência longa (a
  gravação de exemplo) que combina com uma curta (a janela ao vivo). É o caso
  do reconhecimento: a janela de 10 quadros pode ser qualquer pedaço do sinal.

Os custos são normalizados (por ``n + m`` na inteira e por ``n`` na de
subsequência) para que sequências de tamanhos diferentes sejam comparáveis.
"""

from __future__ import annotations

import numpy as np

_INF = float("inf")


def matriz_custos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distância euclidiana entre cada quadro de ``a`` e cada quadro de ``b``.

    Args:
        a: Matriz (n, d).
        b: Matriz (m, d).

    Returns:
        Matriz (n, m).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    quadrados = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(quadrados, 0.0))


def _ultima_linha(custos: np.ndarray, inicio_livre: bool, faixa: int | None) -> list[float]:
    """Custo acumulado na última linha da matriz de DTW.

    Args:
        custos: Matriz (n, m).
        inicio_livre: Se ``True``, o caminho pode começar em qualquer coluna
            (DTW de subsequência).
        faixa: Largura da faixa de Sakoe-Chiba (em quadros), ou ``None``.
    """
    n, m = custos.shape
    linhas = custos.tolist()  # listas puras: o laço fica bem mais rápido
    anterior = [0.0] * (m + 1) if inicio_livre else [0.0] + [_INF] * m
    for i in range(n):
        atual = [_INF] * (m + 1)
        if faixa is None:
            j0, j1 = 1, m
        else:
            centro = round((i + 1) * m / n)
            j0, j1 = max(1, centro - faixa), min(m, centro + faixa)
        custo_i = linhas[i]
        for j in range(j0, j1 + 1):
            # Diagonal, vertical (repete quadro de b) e horizontal (repete de a).
            melhor = anterior[j - 1]
            if anterior[j] < melhor:
                melhor = anterior[j]
            if atual[j - 1] < melhor:
                melhor = atual[j - 1]
            atual[j] = custo_i[j - 1] + melhor
        anterior = atual
    return anterior[1:]


def alinhar(a: np.ndarray, b: np.ndarray, faixa: int | None = None) -> float:
    """Custo do melhor alinhamento entre duas sequências inteiras.

    Args:
        a: Matriz (n, d).
        b: Matriz (m, d).
        faixa: Limite de Sakoe-Chiba (quadros); ``None`` = sem limite. A faixa
            é ampliada para pelo menos ``|n - m| + 1``, senão não há caminho.

    Returns:
        Custo acumulado dividido por ``n + m``.
    """
    if faixa is not None:
        faixa = max(faixa, abs(len(a) - len(b)) + 1)
    ultima = _ultima_linha(matriz_custos(a, b), inicio_livre=False, faixa=faixa)
    return ultima[-1] / (len(a) + len(b))


def alinhar_trecho(curta: np.ndarray, longa: np.ndarray) -> tuple[float, int]:
    """Melhor trecho de ``longa`` para ``curta`` (início e fim livres em ``longa``).

    Returns:
        ``(custo, fim)``: custo acumulado dividido por ``len(curta)`` e a coluna
        de ``longa`` onde o melhor trecho termina.
    """
    ultima = _ultima_linha(matriz_custos(curta, longa), inicio_livre=True, faixa=None)
    fim = int(np.argmin(ultima))
    return ultima[fim] / len(curta), fim
