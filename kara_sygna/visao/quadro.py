"""Vetor completo de cada quadro: forma das mãos, rosto e posição das mãos.

Cada quadro vira exatamente 214 números::

    [ forma das mãos (126) | rosto e mãos (82) | posição das mãos (6) ]

* **Forma das mãos** (:mod:`kara_sygna.visao.caracteristicas`): cada mão
  centrada no pulso e dividida pelo tamanho da palma.
* **Rosto e mãos** (:mod:`kara_sygna.visao.caracteristicas_rosto_maos`):
  expressão facial e distância de cada ponta de dedo até testa, olhos, nariz,
  bochechas, boca e queixo. Em Libras, o lugar do sinal faz parte dele.
* **Posição** (6): para cada mão, onde o pulso está em relação ao nariz e o
  tamanho da palma, tudo medido em larguras de rosto. É isso que permite
  acompanhar o movimento da mão (veja :mod:`kara_sygna.ia.sequencia`).

As mãos usam sempre os mesmos lados ("esq"/"dir") em todos os blocos.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kara_sygna.visao.caracteristicas import (
    FEATURES_POR_MAO,
    IDX_DIR,
    IDX_ESQ,
    TOTAL_FEATURES,
    TOTAL_LANDMARKS_MAO,
    eh_vetor_zerado,
    nomes_features,
    normalizar_mao,
    separar_maos,
)
from kara_sygna.visao.caracteristicas_rosto_maos import (
    INDICES_EXPRESSAO,
    NOMES_REGIOES,
    PONTAS_DEDOS,
    TOTAL_FEATURES_ROSTO_MAOS,
    extrair_features_rosto_maos_slots,
    nomes_features_rosto_maos,
)

TOTAL_FEATURES_POSICAO = 6
TOTAL_FEATURES_QUADRO = (
    TOTAL_FEATURES + TOTAL_FEATURES_ROSTO_MAOS + TOTAL_FEATURES_POSICAO
)  # 214

IDX_MAOS = slice(0, TOTAL_FEATURES)
IDX_ROSTO_MAOS = slice(TOTAL_FEATURES, TOTAL_FEATURES + TOTAL_FEATURES_ROSTO_MAOS)
IDX_POSICAO = slice(TOTAL_FEATURES + TOTAL_FEATURES_ROSTO_MAOS, TOTAL_FEATURES_QUADRO)

#: Coluna, dentro do vetor do quadro, que diz se o rosto foi encontrado.
COLUNA_ROSTO_PRESENTE = IDX_ROSTO_MAOS.start

#: Colunas (x, y, escala) da posição de cada mão, dentro do vetor do quadro.
COLUNAS_POSICAO = {
    "esq": (IDX_POSICAO.start, IDX_POSICAO.start + 1, IDX_POSICAO.start + 2),
    "dir": (IDX_POSICAO.start + 3, IDX_POSICAO.start + 4, IDX_POSICAO.start + 5),
}

#: Bloco da forma de cada mão, dentro do vetor do quadro.
IDX_FORMA = {"esq": IDX_ESQ, "dir": IDX_DIR}

# Pontos do FaceMesh usados como referência para a posição das mãos.
_NARIZ = 1
_ROSTO_ESQ = 234
_ROSTO_DIR = 454


def nomes_features_quadro() -> list[str]:
    """Nomes das 214 colunas, na ordem do vetor."""
    posicao = [
        f"pos_{lado}_{eixo}" for lado in ("esq", "dir") for eixo in ("x", "y", "escala")
    ]
    return (
        nomes_features()
        + [f"rm_{nome}" for nome in nomes_features_rosto_maos()]
        + posicao
    )


def _posicao_das_maos(
    rosto: Any,
    maos: dict[str, np.ndarray | None],
    largura: int,
    altura: int,
) -> np.ndarray:
    """Posição do pulso em relação ao nariz e tamanho da palma, em larguras de rosto.

    Sem rosto, ou sem a mão, os valores daquela mão ficam em zero (os blocos
    de presença do vetor dizem se o zero é "ausente" ou "no nariz").
    """
    saida = np.zeros(TOTAL_FEATURES_POSICAO, dtype=np.float64)
    pontos_rosto = getattr(rosto, "landmark", None)
    if pontos_rosto is None or len(pontos_rosto) < 468:
        return saida

    escala_px = np.array([largura, altura], dtype=np.float64)

    def px(indice: int) -> np.ndarray:
        p = pontos_rosto[indice]
        return np.array([p.x, p.y]) * escala_px

    largura_rosto = float(np.linalg.norm(px(_ROSTO_DIR) - px(_ROSTO_ESQ)))
    if not np.isfinite(largura_rosto) or largura_rosto < 1.0:
        return saida
    nariz = px(_NARIZ)

    for indice, lado in enumerate(("esq", "dir")):
        pontos = maos.get(lado)
        if pontos is None:
            continue
        pontos_px = pontos[:, :2] * escala_px
        pulso = pontos_px[0]
        palma = np.mean([np.linalg.norm(pontos_px[i] - pulso) for i in (5, 9, 13, 17)])
        x, y = (pulso - nariz) / largura_rosto
        saida[3 * indice : 3 * indice + 3] = (x, y, palma / largura_rosto)

    return np.nan_to_num(np.clip(saida, -10.0, 10.0))


def montar_vetor_quadro(
    resultado_maos: Any,
    resultado_rosto: Any,
    largura: int,
    altura: int,
) -> np.ndarray:
    """Monta o vetor de 214 valores de um quadro.

    Args:
        resultado_maos: Resultado bruto de ``DetectorMaos.detectar``.
        resultado_rosto: ``ResultadoRosto`` de ``DetectorRosto.detectar``.
        largura: Largura do quadro, em pixels.
        altura: Altura do quadro, em pixels.

    Returns:
        Array de 214 ``float64``. Partes ausentes (mão, rosto) viram zeros
        ou o valor de "indisponível" do bloco do rosto.
    """
    vetor = np.zeros(TOTAL_FEATURES_QUADRO, dtype=np.float64)
    maos = separar_maos(resultado_maos)

    for lado in ("esq", "dir"):
        pontos = maos[lado]
        if pontos is None:
            continue
        normalizados = normalizar_mao(pontos)
        if normalizados is None:
            maos[lado] = None  # mão degenerada: ausente em todos os blocos
        else:
            vetor[IDX_FORMA[lado]] = normalizados.reshape(-1)

    vetor[IDX_ROSTO_MAOS] = extrair_features_rosto_maos_slots(
        resultado_rosto, (maos["esq"], maos["dir"]), largura, altura
    )
    rosto = (
        getattr(resultado_rosto, "landmarks", None)
        if vetor[COLUNA_ROSTO_PRESENTE]
        else None
    )
    vetor[IDX_POSICAO] = _posicao_das_maos(rosto, maos, largura, altura)
    return vetor


def maos_presentes(quadros: np.ndarray) -> np.ndarray:
    """Diz, para cada quadro, se cada mão está presente.

    Args:
        quadros: Matriz (n, 214).

    Returns:
        Matriz booleana (n, 2): coluna 0 = mão "esq", coluna 1 = mão "dir".
    """
    quadros = np.asarray(quadros, dtype=np.float64).reshape(-1, TOTAL_FEATURES_QUADRO)
    return np.stack(
        [
            np.abs(quadros[:, IDX_FORMA[lado]]).sum(axis=1) > 1e-9
            for lado in ("esq", "dir")
        ],
        axis=1,
    )


def tem_mao(vetor_quadro: Any) -> bool:
    """Diz se o quadro tem pelo menos uma mão."""
    vetor = np.asarray(vetor_quadro, dtype=np.float64).reshape(-1)
    return not eh_vetor_zerado(vetor[IDX_MAOS])


# ---------------------------------------------------------------------------
# Espelhamento (para um sinal feito com a mão direita valer para a esquerda)
# ---------------------------------------------------------------------------

_INICIO_EXPRESSAO = 3
_INICIO_DISTANCIAS = _INICIO_EXPRESSAO + len(INDICES_EXPRESSAO)
_DISTANCIAS_POR_MAO = len(PONTAS_DEDOS) * len(NOMES_REGIOES)


def _ordem_espelhada_rosto_maos() -> np.ndarray:
    """Ordem das 82 colunas do bloco rosto+mãos depois de espelhar a imagem.

    Espelhar troca as mãos de lado, os olhos e sobrancelhas, e as bochechas.
    """
    ordem = np.arange(TOTAL_FEATURES_ROSTO_MAOS)
    ordem[[1, 2]] = [2, 1]  # presença das mãos

    expressao = list(INDICES_EXPRESSAO)
    for a, b in (
        ("olho_esquerdo_abertura", "olho_direito_abertura"),
        ("sobrancelha_esquerda", "sobrancelha_direita"),
    ):
        ia = _INICIO_EXPRESSAO + expressao.index(a)
        ib = _INICIO_EXPRESSAO + expressao.index(b)
        ordem[[ia, ib]] = [ib, ia]

    regioes = list(NOMES_REGIOES)
    troca_regiao = np.arange(len(regioes))
    ie, idr = regioes.index("bochecha_esquerda"), regioes.index("bochecha_direita")
    troca_regiao[[ie, idr]] = [idr, ie]

    for mao_destino in range(2):
        mao_origem = 1 - mao_destino
        for d in range(len(PONTAS_DEDOS)):
            for r in range(len(regioes)):
                destino = (
                    _INICIO_DISTANCIAS
                    + mao_destino * _DISTANCIAS_POR_MAO
                    + d * len(regioes)
                    + r
                )
                origem = (
                    _INICIO_DISTANCIAS
                    + mao_origem * _DISTANCIAS_POR_MAO
                    + d * len(regioes)
                    + troca_regiao[r]
                )
                ordem[destino] = origem
    return ordem


_ORDEM_ROSTO_MAOS_ESPELHADA = _ordem_espelhada_rosto_maos()
_COLUNA_INCLINACAO = _INICIO_EXPRESSAO + list(INDICES_EXPRESSAO).index(
    "inclinacao_olhos"
)


def espelhar_quadros(quadros: np.ndarray) -> np.ndarray:
    """Espelha quadros (vetor ou matriz com 214 colunas) como num espelho.

    A mão direita vira a esquerda (e vice-versa) em todos os blocos, e o
    rosto é espelhado junto. Espelhar duas vezes devolve o original.
    """
    entrada = np.asarray(quadros, dtype=np.float64)
    if entrada.shape[-1] != TOTAL_FEATURES_QUADRO:
        raise ValueError(
            f"O último eixo deve ter {TOTAL_FEATURES_QUADRO} valores; "
            f"recebidos {entrada.shape[-1]}."
        )
    v = entrada.reshape(-1, TOTAL_FEATURES_QUADRO)
    saida = np.empty_like(v)

    # Forma das mãos: troca os lados e inverte o eixo x.
    for destino, origem in (("esq", "dir"), ("dir", "esq")):
        mao = v[:, IDX_FORMA[origem]].reshape(-1, TOTAL_LANDMARKS_MAO, 3).copy()
        mao[:, :, 0] *= -1.0
        saida[:, IDX_FORMA[destino]] = mao.reshape(-1, FEATURES_POR_MAO)

    # Rosto e mãos: reordena as colunas e inverte a inclinação dos olhos.
    rosto = v[:, IDX_ROSTO_MAOS][:, _ORDEM_ROSTO_MAOS_ESPELHADA]
    rosto[:, _COLUNA_INCLINACAO] *= -1.0
    saida[:, IDX_ROSTO_MAOS] = rosto

    # Posição: troca os lados e inverte o x.
    posicao = v[:, IDX_POSICAO].reshape(-1, 2, 3)[:, ::-1, :].copy()
    posicao[:, :, 0] *= -1.0
    saida[:, IDX_POSICAO] = posicao.reshape(-1, TOTAL_FEATURES_POSICAO)

    return saida.reshape(entrada.shape)
