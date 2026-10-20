"""Máscara no treino: cópias das gravações com a mão sumindo, como ao vivo.

Por que existe
--------------
As gravações do ensino têm os buracos de até 0,5 s preenchidos
(:mod:`kara_sygna.visao.preenchimento`), então a IA aprende com mãos quase
sempre presentes. Ao vivo, a mão rápida borra e some, e o app só repete a
última mão por ``config.MEMORIA_MAO_S``. No laboratório (``kara-sygna-lab``),
com a mão sumindo em 30% dos quadros do teste, o RandomForest acertou 90,0%
das janelas; treinado também com estas cópias, acertou 98,4%.

Como funciona
-------------
Para cada gravação, :func:`copias_mascaradas` faz cópias em que trechos de
cada mão (forma, distâncias ao rosto e posição) são apagados e depois passam
pela mesma memória do app (:func:`congelar`). O pulso do braço (Pose) fica,
porque o Pose quase nunca o perde. O sorteio usa uma semente fixa, então o
mesmo CSV dá sempre o mesmo treino.
"""

from __future__ import annotations

import numpy as np

from kara_sygna import config
from kara_sygna.visao.quadro import colunas_da_mao, maos_presentes

_LADOS = ("esq", "dir")
#: Quadros por segundo da câmera (MediaFoundation), para converter segundos.
_QPS = 30


def apagar_maos(
    quadros: np.ndarray,
    fracao: float,
    rng: np.random.Generator,
    buraco_min: int = config.MASCARA_BURACO_MIN,
    buraco_max: int = config.MASCARA_BURACO_MAX,
) -> np.ndarray:
    """Apaga trechos seguidos de cada mão até cobrir ``fracao`` dos quadros.

    Args:
        quadros: Matriz (n, 223) de uma gravação.
        fracao: Parte dos quadros em que cada mão some (0 a 1).
        rng: Gerador aleatório.
        buraco_min: Menor buraco, em quadros.
        buraco_max: Maior buraco, em quadros.

    Returns:
        Cópia com as mãos apagadas (zeros, como o app grava mão ausente).
    """
    saida = np.array(quadros, dtype=np.float64, copy=True)
    n = len(saida)
    for lado in _LADOS:
        apagados = np.zeros(n, dtype=bool)
        meta = round(fracao * n)
        for _ in range(1000):
            if apagados.sum() >= meta:
                break
            tamanho = int(rng.integers(buraco_min, buraco_max + 1))
            inicio = int(rng.integers(0, max(1, n - tamanho + 1)))
            apagados[inicio : inicio + tamanho] = True
        saida[np.ix_(apagados, colunas_da_mao(lado))] = 0.0
    return saida


def congelar(quadros: np.ndarray, limite: int | None = None) -> np.ndarray:
    """Repete a última mão vista nos buracos curtos, como a memória do app.

    Args:
        quadros: Matriz (n, 223).
        limite: Maior buraco coberto, em quadros. Padrão:
            ``config.MEMORIA_MAO_S`` a 30 quadros por segundo.
    """
    if limite is None:
        limite = round(config.MEMORIA_MAO_S * _QPS)
    saida = np.array(quadros, dtype=np.float64, copy=True)
    presentes = maos_presentes(saida)
    for k, lado in enumerate(_LADOS):
        colunas = colunas_da_mao(lado)
        ultimo, sem_ver = None, 0
        for i in range(len(saida)):
            if presentes[i, k]:
                ultimo, sem_ver = saida[i, colunas].copy(), 0
            elif ultimo is not None and sem_ver < limite:
                saida[i, colunas] = ultimo
                sem_ver += 1
    return saida


def copias_mascaradas(
    quadros: np.ndarray,
    semente: int,
    fracoes: tuple[float, ...] = config.MASCARA_FRACOES,
) -> list[np.ndarray]:
    """Uma cópia da gravação para cada fração de mão apagada.

    Args:
        quadros: Matriz (n, 223) de uma gravação.
        semente: Semente do sorteio (use algo fixo da gravação, como a rodada).
        fracoes: Partes dos quadros em que cada mão some. ``()`` = nenhuma cópia.
    """
    rng = np.random.default_rng(semente)
    return [congelar(apagar_maos(quadros, fracao, rng)) for fracao in fracoes]
