"""Região entre as mãos: num sinal de duas mãos, elas não podem estar longe demais.

Por que existe
--------------
Em Libras o lugar do sinal é livre (por isso ``config.USAR_LUGAR = False``),
mas a relação entre a mão esquerda e a direita faz parte do sinal. No carro as
mãos ficam mais ou menos perto, lado a lado, como num volante; com os braços
abertos, as mãos longe uma da outra, não é carro. Ao vivo (20/10/2026) o app
escrevia "carro" assim, porque a IA olha a forma das mãos e o movimento.

Como funciona
-------------
* Distância entre os pulsos (pelo braço, MediaPipe Pose), em larguras de
  ombro: não depende de onde as mãos estão, nem do tamanho da pessoa.
* O limite de cada sinal vem das gravações dele: a distância que quase sempre
  aparece (percentil 95) mais ``config.REGIAO_MAOS_FOLGA``. Nas gravações de
  20/10/2026: carro até 1,34 (+0,4 = 1,74), casa até 0,82 (+0,4 = 1,22).
* Só vale para sinais de duas mãos (as duas presentes em pelo menos metade dos
  quadros gravados). Sinais de uma mão nunca são bloqueados.
* Vale a maior distância dos últimos ``config.REGIAO_MAOS_MEMORIA`` quadros:
  ao vivo a distância oscila em volta do limite, e quadros soltos "perto"
  somavam votos até confirmar o carro com os braços abertos.
* Sem corpo na imagem não dá para medir: não bloqueia.
* ``config.REGIAO_MAOS_FOLGA`` negativo desliga a regra.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from kara_sygna import config
from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    maos_presentes,
)

#: Parte mínima dos quadros com as duas mãos para o sinal ser "de duas mãos".
MINIMO_DUAS_MAOS = 0.5


def distancia_entre_maos(quadros: np.ndarray) -> np.ndarray:
    """Distância entre os pulsos (Pose), em larguras de ombro, de cada quadro."""
    quadros = np.atleast_2d(np.asarray(quadros, dtype=np.float64))
    esq = quadros[:, list(COLUNAS_PULSO_BRACO["esq"])]
    dir_ = quadros[:, list(COLUNAS_PULSO_BRACO["dir"])]
    return np.hypot(*(dir_ - esq).T)


class RegiaoMaos:
    """Distância máxima entre as mãos para cada sinal de duas mãos.

    Args:
        limites: ``{sinal: distância máxima}`` em larguras de ombro.
        memoria: Quadros recentes olhados. Padrão: ``config.REGIAO_MAOS_MEMORIA``.
    """

    def __init__(
        self, limites: dict[str, float] | None = None, memoria: int | None = None
    ) -> None:
        """Guarda os limites (veja a classe)."""
        self.limites = dict(limites or {})
        memoria = config.REGIAO_MAOS_MEMORIA if memoria is None else memoria
        self._recentes: deque[float] = deque(maxlen=max(1, memoria))
        #: Distância do último quadro com corpo (``None`` = sem corpo).
        self.distancia: float | None = None

    def atualizar(self, vetor: np.ndarray) -> None:
        """Mede a distância entre as mãos neste quadro (chamar em todo quadro)."""
        if vetor[COLUNA_CORPO_PRESENTE] <= 0.5:
            self.distancia = None
            self._recentes.clear()
            return
        self.distancia = float(distancia_entre_maos(vetor)[0])
        self._recentes.append(self.distancia)

    @classmethod
    def das_gravacoes(
        cls, dados: ConjuntoDados, folga: float | None = None
    ) -> RegiaoMaos:
        """Aprende o limite de cada sinal de duas mãos a partir das gravações.

        Args:
            dados: Quadros gravados (``RepositorioSinais.carregar()``).
            folga: Somada ao percentil 95. Padrão: ``config.REGIAO_MAOS_FOLGA``;
                negativa desliga (nenhum limite).
        """
        folga = config.REGIAO_MAOS_FOLGA if folga is None else folga
        if folga < 0 or len(dados) == 0:
            return cls()
        corpo = dados.X[:, COLUNA_CORPO_PRESENTE] > 0.5
        duas = maos_presentes(dados.X).all(axis=1)
        distancias = distancia_entre_maos(dados.X)
        limites = {}
        for sinal in dados.classes:
            do_sinal = dados.y == sinal
            medidos = do_sinal & corpo
            if duas[do_sinal].mean() < MINIMO_DUAS_MAOS or medidos.sum() < 10:
                continue
            limites[sinal] = float(np.percentile(distancias[medidos], 95) + folga)
        return cls(limites)

    def permite(self, sinal: str) -> bool:
        """``False`` se o sinal é de duas mãos e elas ficaram longe demais há pouco."""
        limite = self.limites.get(sinal)
        if limite is None or not self._recentes:
            return True
        return max(self._recentes) <= limite
