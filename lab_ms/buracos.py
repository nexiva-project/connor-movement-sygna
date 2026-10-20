"""Simula a mão sumindo por borrão, como acontece ao vivo.

As gravações do app já vêm com os buracos preenchidos (PCHIP), mas ao vivo a
mão rápida some e o app só **congela** a última mão por até 0,3 s. Para testar
isso no laboratório:

* :func:`apagar_maos` apaga trechos seguidos de cada mão (forma, distâncias ao
  rosto e posição). O pulso do braço (Pose) fica, porque o Pose quase nunca o
  perde;
* :func:`congelar` repete a última mão vista por até 9 quadros (0,3 s a
  30 qps), como a memória da mão do app.

A mesma :func:`apagar_maos` serve de **máscara no treino**: cópias das
gravações com mãos apagadas ensinam a IA a lidar com os buracos.
"""

from __future__ import annotations

import numpy as np

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna.visao.quadro import colunas_da_mao, maos_presentes  # noqa: E402

_LADOS = ("esq", "dir")
#: 0,3 s a 30 quadros por segundo (memória da mão do app).
QUADROS_MEMORIA = 9


def apagar_maos(
    quadros: np.ndarray,
    fracao: float,
    rng: np.random.Generator,
    buraco_min: int = 2,
    buraco_max: int = 15,
) -> np.ndarray:
    """Apaga trechos de cada mão até cobrir ``fracao`` dos quadros.

    Args:
        quadros: Matriz (n, 223) de uma gravação.
        fracao: Parte dos quadros em que cada mão some (0 a 1).
        rng: Gerador aleatório (para repetir o mesmo sorteio).
        buraco_min: Menor buraco, em quadros.
        buraco_max: Maior buraco, em quadros (15 ≈ 0,5 s).

    Returns:
        Cópia com as mãos apagadas (zeros, como o app grava mão ausente).
    """
    saida = np.array(quadros, dtype=np.float64, copy=True)
    n = len(saida)
    for lado in _LADOS:
        colunas = colunas_da_mao(lado)
        apagados = np.zeros(n, dtype=bool)
        meta = round(fracao * n)
        tentativas = 0
        while apagados.sum() < meta and tentativas < 1000:
            tamanho = int(rng.integers(buraco_min, buraco_max + 1))
            inicio = int(rng.integers(0, max(1, n - tamanho + 1)))
            apagados[inicio : inicio + tamanho] = True
            tentativas += 1
        saida[np.ix_(apagados, colunas)] = 0.0
    return saida


def congelar(quadros: np.ndarray, limite: int = QUADROS_MEMORIA) -> np.ndarray:
    """Repete a última mão vista nos buracos de até ``limite`` quadros (memória do app)."""
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
