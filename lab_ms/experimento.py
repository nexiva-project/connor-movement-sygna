"""Comparação justa entre o RandomForest do app e o MS.

Protocolo (deixar uma rodada de fora):

* Para cada gravação (rodada) de um sinal com 2 ou mais rodadas, os dois
  métodos aprendem com **todas as outras** e são testados nela.
* Os dois veem **as mesmas janelas**: as de 10 quadros que o app usaria
  (``janelas_de_uma_gravacao``, terminando em quadros com mão).
* Sinais com 1 rodada só (ex.: ``neutro``) entram no treino, mas não são testados.

Cenários:

* ``todas``: treino com todas as outras rodadas (como o app treina hoje).
* ``uma_por_sinal``: treino com só 1 rodada de cada sinal (pouquíssimos
  dados, onde o MS deveria brilhar). Cada escolha possível é testada.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lab_ms.buracos import apagar_maos, congelar
from lab_ms.ms_knn import ClassificadorMS
from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna import config  # noqa: E402
from kara_sygna.dados.repositorio import (  # noqa: E402
    TIPO_SINAL,
    ConjuntoDados,
    RepositorioSinais,
)
from kara_sygna.ia import treinamento  # noqa: E402
from kara_sygna.ia.sequencia import features_janela, montar_amostras, nomes_features_janela  # noqa: E402
from kara_sygna.visao.quadro import maos_presentes  # noqa: E402

RITMOS = ("devagar", "médio", "rápido")


@dataclass(frozen=True)
class Gravacao:
    """Uma rodada de um sinal: quadros (n, 223) em ordem de gravação."""

    rodada: int
    rotulo: str
    quadros: np.ndarray
    ritmo: str = ""


@dataclass
class Resultado:
    """Acertos de um método em um cenário."""

    janelas_certas: int = 0
    janelas_total: int = 0
    gravacoes_certas: int = 0
    gravacoes_total: int = 0
    ms_por_janela: list[float] = field(default_factory=list)
    confusao: Counter = field(default_factory=Counter)
    por_ritmo: Counter = field(default_factory=Counter)
    por_ritmo_total: Counter = field(default_factory=Counter)

    def registrar(self, gravacao: Gravacao, previstos_janela: list[str], previsto_gravacao: str) -> None:
        """Soma os acertos de uma gravação de teste."""
        certas = sum(p == gravacao.rotulo for p in previstos_janela)
        self.janelas_certas += certas
        self.janelas_total += len(previstos_janela)
        self.gravacoes_certas += previsto_gravacao == gravacao.rotulo
        self.gravacoes_total += 1
        self.confusao.update((gravacao.rotulo, p) for p in previstos_janela)
        if gravacao.ritmo:
            self.por_ritmo[gravacao.ritmo] += certas
            self.por_ritmo_total[gravacao.ritmo] += len(previstos_janela)


def carregar_gravacoes(caminho_csv: Path | None = None) -> list[Gravacao]:
    """Lê as gravações de sinais do CSV do app, separadas por rodada.

    O ritmo (devagar, médio, rápido) é deduzido pela ordem das rodadas de cada
    sinal, quando o sinal tem exatamente 3 (um Enter grava as 3 em sequência).
    """
    dados = RepositorioSinais(caminho_csv).carregar(TIPO_SINAL)
    gravacoes: list[Gravacao] = []
    for rodada in sorted(set(dados.grupos.tolist())):
        da_rodada = dados.grupos == rodada
        for rotulo in dict.fromkeys(dados.y[da_rodada].tolist()):
            quadros = dados.X[da_rodada & (dados.y == rotulo)]
            gravacoes.append(Gravacao(int(rodada), str(rotulo), quadros))

    ritmados: list[Gravacao] = []
    for g in gravacoes:
        do_sinal = [o for o in gravacoes if o.rotulo == g.rotulo]
        ritmo = RITMOS[do_sinal.index(g)] if len(do_sinal) == len(RITMOS) else ""
        ritmados.append(Gravacao(g.rodada, g.rotulo, g.quadros, ritmo))
    return ritmados


def fins_das_janelas(quadros: np.ndarray) -> list[int]:
    """Onde terminam as janelas que o app monta (mesma regra de ``janelas_de_uma_gravacao``)."""
    tamanho, passo = config.QUADROS_POR_JANELA, config.PASSO_JANELA_TREINO
    com_mao = maos_presentes(quadros).any(axis=1)
    return [fim for fim in range(tamanho - 1, len(quadros), passo) if com_mao[fim]]


def _como_conjunto(gravacoes: list[Gravacao]) -> ConjuntoDados:
    return ConjuntoDados(
        X=np.vstack([g.quadros for g in gravacoes]),
        y=np.concatenate([[g.rotulo] * len(g.quadros) for g in gravacoes]).astype(str),
        grupos=np.concatenate([[g.rodada] * len(g.quadros) for g in gravacoes]),
    )


def testar_random_forest(treino: list[Gravacao], teste: Gravacao, resultado: Resultado) -> None:
    """Treina o RandomForest do app (mesmo aumento de dados) e testa uma gravação."""
    amostras = montar_amostras(_como_conjunto(treino))
    modelo = treinamento._ajustar(amostras.X, amostras.y)
    modelo.set_params(n_jobs=1)  # medir a latência de uma janela, como ao vivo

    tamanho = config.QUADROS_POR_JANELA
    previstos: list[str] = []
    for fim in fins_das_janelas(teste.quadros):
        inicio = time.perf_counter()
        vetor = features_janela(teste.quadros[fim - tamanho + 1 : fim + 1])
        previsto = modelo.predict(pd.DataFrame([vetor], columns=nomes_features_janela()))[0]
        resultado.ms_por_janela.append((time.perf_counter() - inicio) * 1000)
        previstos.append(str(previsto))
    votos = Counter(previstos).most_common(1)
    resultado.registrar(teste, previstos, votos[0][0] if votos else "")


def testar_ms(treino: list[Gravacao], teste: Gravacao, resultado: Resultado, **opcoes) -> None:
    """Guarda as gravações de treino como modelos do MS e testa uma gravação."""
    classificador = ClassificadorMS(**opcoes).fit([g.quadros for g in treino], [g.rotulo for g in treino])

    tamanho = config.QUADROS_POR_JANELA
    previstos: list[str] = []
    for fim in fins_das_janelas(teste.quadros):
        inicio = time.perf_counter()
        distancias = classificador.distancias_janela(teste.quadros[fim - tamanho + 1 : fim + 1])
        resultado.ms_por_janela.append((time.perf_counter() - inicio) * 1000)
        previstos.append(next(iter(distancias)))
    da_gravacao = classificador.distancias_gravacao(teste.quadros)
    resultado.registrar(teste, previstos, next(iter(da_gravacao)))


def cenarios(gravacoes: list[Gravacao]):
    """Gera ``(nome, treino, teste)`` para os dois cenários."""
    rodadas = Counter(g.rotulo for g in gravacoes)
    testaveis = [g for g in gravacoes if rodadas[g.rotulo] >= 2]
    for teste in testaveis:
        outras = [g for g in gravacoes if g is not teste]
        yield "todas", outras, teste

        # Só 1 rodada de cada sinal: cada combinação possível.
        por_sinal = {}
        for g in outras:
            por_sinal.setdefault(g.rotulo, []).append(g)
        for escolha in itertools.product(*por_sinal.values()):
            yield "uma_por_sinal", list(escolha), teste


def comparar(
    gravacoes: list[Gravacao],
    progresso=print,
    buracos: float = 0.0,
    mascara_treino: bool = False,
    semente: int = 42,
    **opcoes_ms,
) -> dict[str, dict[str, Resultado]]:
    """Roda os métodos em todos os cenários.

    Args:
        buracos: Parte dos quadros em que cada mão some no **teste** (0 = como
            gravado). Os buracos passam pela memória de 0,3 s, como ao vivo.
        mascara_treino: Também testa um RandomForest treinado com cópias das
            gravações com mãos apagadas (máscara).
        semente: Semente do sorteio dos buracos (mesma para todos os métodos).

    Returns:
        ``{cenario: {metodo: Resultado}}``.
    """
    saida: dict[str, dict[str, Resultado]] = {}
    lista = list(cenarios(gravacoes))
    for numero, (nome, treino, teste) in enumerate(lista, 1):
        progresso(f"[{numero}/{len(lista)}] {nome}: testando {teste.rotulo} ({teste.ritmo or teste.rodada})")
        rng = np.random.default_rng(semente + numero)
        if buracos > 0:
            teste = dataclasses.replace(teste, quadros=congelar(apagar_maos(teste.quadros, buracos, rng)))
        metodos = saida.setdefault(nome, {"RandomForest": Resultado(), "MS": Resultado()})
        testar_random_forest(treino, teste, metodos["RandomForest"])
        if mascara_treino:
            # Cópias com 15%, 30% e 50% da mão apagada. Cada cópia ganha uma rodada
            # própria, senão montar_amostras juntaria as cópias numa sequência só.
            mascarados = [
                dataclasses.replace(
                    g, rodada=g.rodada * 10 + k, quadros=congelar(apagar_maos(g.quadros, f, rng))
                )
                for g in treino
                for k, f in enumerate((0.15, 0.3, 0.5), 1)
            ]
            testar_random_forest(treino + mascarados, teste, metodos.setdefault("RF + máscara", Resultado()))
        testar_ms(treino, teste, metodos["MS"], **opcoes_ms)
    return saida
