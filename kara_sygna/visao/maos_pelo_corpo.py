"""Confere as mãos detectadas pelos braços do corpo (MediaPipe Pose).

Por que existe
--------------
Com a exigência de detecção baixa e o realce de pele, o MediaPipe às vezes
acha uma "segunda mão" que não existe. Num teste com o sinal ``neutro``, só
com a mão esquerda, apareceu uma mão "direita" em 27 de 69 quadros, sempre
perto da mão esquerda, enquanto o braço direito estava parado do outro lado.

Toda mão de verdade fica na ponta de um braço. Então:

* uma mão só vale se o pulso dela estiver perto do pulso de um braço do Pose
  (até ``config.DISTANCIA_MAO_PULSO`` larguras de ombro);
* cada braço fica com no máximo uma mão (a mais perto do pulso);
* o lado da mão ("esq"/"dir") é o lado do braço, e não o palpite do
  MediaPipe, que erra mais (e inverte com as mãos cruzadas).

Sem corpo na imagem, as mãos passam sem mudança.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from kara_sygna import config
from kara_sygna.visao.quadro import POSE_DOS_SLOTS

_VISIBILIDADE_OMBRO = 0.5


def conferir_maos(
    resultado_maos: Any,
    resultado_corpo: Any,
    largura: int,
    altura: int,
    distancia_maxima: float | None = None,
) -> Any:
    """Tira as mãos fantasmas e dá a cada mão o lado do braço dela.

    Args:
        resultado_maos: Resultado bruto de ``DetectorMaos.detectar``.
        resultado_corpo: Resultado de ``DetectorCorpo.detectar`` (ou ``None``).
        largura: Largura do quadro, em pixels.
        altura: Altura do quadro, em pixels.
        distancia_maxima: Distância máxima pulso da mão - pulso do braço, em
            larguras de ombro. Padrão: ``config.DISTANCIA_MAO_PULSO``.

    Returns:
        O próprio ``resultado_maos`` se não houver corpo; senão, um objeto com
        as mãos que sobraram (``multi_hand_landmarks``, ``multi_handedness``,
        ``multi_hand_world_landmarks``), o lado de cada uma em ``lados`` e
        quantas foram descartadas em ``descartadas``.
    """
    listas = list(getattr(resultado_maos, "multi_hand_landmarks", None) or [])
    pontos = getattr(getattr(resultado_corpo, "pose_landmarks", None), "landmark", None)
    if not listas or pontos is None or len(pontos) < 17:
        return resultado_maos

    escala = np.array([largura, altura], dtype=np.float64)

    def px(ponto) -> np.ndarray:
        return np.array([ponto.x, ponto.y]) * escala

    ombros = [pontos[POSE_DOS_SLOTS[lado][0]] for lado in ("esq", "dir")]
    if any(getattr(o, "visibility", 1.0) < _VISIBILIDADE_OMBRO for o in ombros):
        return resultado_maos
    largura_ombros = float(np.linalg.norm(px(ombros[0]) - px(ombros[1])))
    if not np.isfinite(largura_ombros) or largura_ombros < 1.0:
        return resultado_maos

    limite = (
        config.DISTANCIA_MAO_PULSO if distancia_maxima is None else distancia_maxima
    )
    pulsos = {lado: px(pontos[POSE_DOS_SLOTS[lado][2]]) for lado in ("esq", "dir")}

    # Cada par (mão, braço) perto o bastante, do mais perto para o mais longe.
    pares = sorted(
        (float(np.linalg.norm(px(lm.landmark[0]) - pulso)) / largura_ombros, i, lado)
        for i, lm in enumerate(listas)
        if len(getattr(lm, "landmark", [])) >= 21
        for lado, pulso in pulsos.items()
    )
    lado_da_mao: dict[int, str] = {}
    for distancia, indice, lado in pares:
        if distancia > limite:
            break
        if indice not in lado_da_mao and lado not in lado_da_mao.values():
            lado_da_mao[indice] = lado

    lateralidades = list(getattr(resultado_maos, "multi_handedness", None) or [])
    mundos = list(getattr(resultado_maos, "multi_hand_world_landmarks", None) or [])
    ficam = sorted(lado_da_mao)
    return SimpleNamespace(
        multi_hand_landmarks=[listas[i] for i in ficam],
        multi_handedness=[
            lateralidades[i] if i < len(lateralidades) else None for i in ficam
        ],
        multi_hand_world_landmarks=[
            mundos[i] if i < len(mundos) else None for i in ficam
        ],
        lados=[lado_da_mao[i] for i in ficam],
        descartadas=len(listas) - len(ficam),
    )
