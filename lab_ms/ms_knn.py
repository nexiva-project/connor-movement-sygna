"""Classificador MS (movimento de sinal) sobre os quadros do kara-sygna.

Cada gravação de ensino vira um *modelo* (template). Para reconhecer, a
sequência nova é alinhada com cada modelo por DTW, e ganha o sinal do modelo
mais parecido (1-NN). Não há treino: ensinar um sinal é só guardar a gravação.

Cada quadro vira um vetor curto para o alinhamento:

* **forma** das duas mãos (126 valores, os mesmos do app). Quadros em que a
  mão sumiu repetem a forma válida mais próxima;
* **movimento**: a velocidade do pulso de cada braço (MediaPipe Pose), quadro a
  quadro (4 valores). É a velocidade, e não a posição, para o sinal valer em
  qualquer lugar (como ``USAR_LUGAR = False`` no app).

Os dois blocos são escalados para pesar o mesmo (ajustável por ``peso_*``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lab_ms.alinhamento import alinhar, alinhar_trecho
from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna.visao.quadro import (  # noqa: E402
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    IDX_FORMA,
    espelhar_quadros,
    maos_presentes,
)

_LADOS = ("esq", "dir")


def _repetir_valido(valores: np.ndarray, validos: np.ndarray) -> np.ndarray:
    """Troca as linhas inválidas pela linha válida mais próxima (ou zeros)."""
    if not validos.any():
        return np.zeros_like(valores)
    indices = np.flatnonzero(validos)
    posicoes = np.arange(len(valores))
    proximo = np.clip(np.searchsorted(indices, posicoes), 0, len(indices) - 1)
    anterior = np.clip(proximo - 1, 0, len(indices) - 1)
    usar_anterior = np.abs(indices[anterior] - posicoes) <= np.abs(indices[proximo] - posicoes)
    return valores[np.where(usar_anterior, indices[anterior], indices[proximo])]


def blocos_do_quadro(quadros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separa os quadros do app (n, 223) em forma (n, 126) e movimento (n, 4)."""
    quadros = np.asarray(quadros, dtype=np.float64)
    presentes = maos_presentes(quadros)
    forma = np.hstack(
        [
            _repetir_valido(quadros[:, IDX_FORMA[lado]], presentes[:, k])
            for k, lado in enumerate(_LADOS)
        ]
    )
    corpo = quadros[:, COLUNA_CORPO_PRESENTE] > 0.5
    pulsos = np.hstack([quadros[:, list(COLUNAS_PULSO_BRACO[lado])] for lado in _LADOS])
    pulsos = _repetir_valido(pulsos, corpo)
    velocidade = np.vstack([np.zeros((1, pulsos.shape[1])), np.diff(pulsos, axis=0)])
    return forma, velocidade


@dataclass
class ClassificadorMS:
    """MS: 1-NN por alinhamento DTW, com modelos espelhados (vale para as duas mãos).

    Args:
        peso_forma: Peso do bloco de forma das mãos.
        peso_movimento: Peso do bloco de velocidade dos pulsos.
        espelhar: Também guarda cada gravação espelhada, como o app faz.
        faixa: Faixa de Sakoe-Chiba (quadros) ao comparar gravações inteiras.
    """

    peso_forma: float = 1.0
    peso_movimento: float = 1.0
    espelhar: bool = True
    faixa: int | None = 40
    _modelos: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _rotulos: list[str] = field(default_factory=list, init=False, repr=False)
    _escala: tuple[float, float] = field(default=(1.0, 1.0), init=False, repr=False)

    def fit(self, gravacoes: list[np.ndarray], rotulos: list[str]) -> ClassificadorMS:
        """Guarda as gravações (cada uma (n, 223)) como modelos."""
        versoes: list[tuple[np.ndarray, np.ndarray, str]] = []
        for quadros, rotulo in zip(gravacoes, rotulos, strict=True):
            copias = [quadros, espelhar_quadros(quadros)] if self.espelhar else [quadros]
            for copia in copias:
                forma, mov = blocos_do_quadro(copia)
                versoes.append((forma, mov, rotulo))

        # Escala de cada bloco: raiz da variância total, medida só no treino.
        todas_formas = np.vstack([v[0] for v in versoes])
        todos_movs = np.vstack([v[1] for v in versoes])
        self._escala = (
            float(np.sqrt(todas_formas.var(axis=0).sum())) or 1.0,
            float(np.sqrt(todos_movs.var(axis=0).sum())) or 1.0,
        )
        self._modelos = [self._juntar(f, m) for f, m, _ in versoes]
        self._rotulos = [r for _, _, r in versoes]
        return self

    def _juntar(self, forma: np.ndarray, mov: np.ndarray) -> np.ndarray:
        return np.hstack(
            [
                forma * (self.peso_forma / self._escala[0]),
                mov * (self.peso_movimento / self._escala[1]),
            ]
        )

    def _vetores(self, quadros: np.ndarray) -> np.ndarray:
        return self._juntar(*blocos_do_quadro(quadros))

    def _por_classe(self, distancias: list[float]) -> dict[str, float]:
        """Menor distância de cada sinal."""
        saida: dict[str, float] = {}
        for rotulo, d in zip(self._rotulos, distancias, strict=True):
            saida[rotulo] = min(d, saida.get(rotulo, np.inf))
        return dict(sorted(saida.items(), key=lambda item: item[1]))

    def distancias_janela(self, janela: np.ndarray) -> dict[str, float]:
        """Distância de uma janela ao vivo (n, 223) a cada sinal (DTW de trecho)."""
        consulta = self._vetores(janela)
        return self._por_classe([alinhar_trecho(consulta, m)[0] for m in self._modelos])

    def distancias_gravacao(self, quadros: np.ndarray) -> dict[str, float]:
        """Distância de uma gravação inteira (n, 223) a cada sinal (DTW completo)."""
        consulta = self._vetores(quadros)
        return self._por_classe([alinhar(consulta, m, self.faixa) for m in self._modelos])

    @staticmethod
    def certeza(distancias: dict[str, float]) -> float:
        """Quão à frente o primeiro está do segundo: ``1 - d1/d2`` (0 = empate)."""
        valores = list(distancias.values())
        if len(valores) < 2 or valores[1] <= 0:
            return 1.0
        return 1.0 - valores[0] / valores[1]
