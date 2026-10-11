"""Contagem de dedos levantados (demonstração, não faz parte do reconhecimento)."""

from __future__ import annotations

import math
from typing import Any

#: nome do dedo -> (índice da ponta, índice da articulação intermediária)
DEDOS = {
    "indicador": (8, 6),
    "medio": (12, 10),
    "anelar": (16, 14),
    "mindinho": (20, 18),
}


def _distancia(pontos: Any, a: int, b: int) -> float:
    """Distância no plano da imagem entre dois landmarks."""
    return math.hypot(pontos[a].x - pontos[b].x, pontos[a].y - pontos[b].y)


def dedos_levantados(landmarks_da_mao: Any) -> list[str]:
    """Lista os dedos levantados de uma mão.

    O critério é a distância até o pulso (não o eixo Y), então funciona com a
    mão inclinada e não depende de qual mão é.

    Args:
        landmarks_da_mao: Landmarks de UMA mão (objeto com ``.landmark``).

    Returns:
        Nomes dos dedos levantados, por exemplo ``["indicador", "polegar"]``.
    """
    p = landmarks_da_mao.landmark
    levantados = [
        nome
        for nome, (ponta, junta) in DEDOS.items()
        if _distancia(p, ponta, 0) > _distancia(p, junta, 0) * 1.1
    ]
    # Polegar: a ponta fica mais longe da base do mindinho que a junta IP.
    if _distancia(p, 4, 17) > _distancia(p, 3, 17) * 1.05:
        levantados.append("polegar")
    return levantados
