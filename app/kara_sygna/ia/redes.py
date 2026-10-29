"""Redes neurais (GRU, LSTM, Conv1D) que reconhecem os sinais, com PyTorch.

O RandomForest recebe um resumo de cada janela de quadros; as redes recebem a
janela quadro a quadro (:func:`kara_sygna.ia.sequencia.features_sequencia`) e
aprendem o movimento sozinhas:

* **GRU** e **LSTM** leem os quadros em ordem, guardando uma memória;
* **Conv1D** procura pedaços de movimento ao longo dos quadros.

Para o resto do app, a rede se comporta como o RandomForest:
:class:`ModeloRede` tem ``classes_`` e ``predict_proba``. Só este módulo usa o
PyTorch, então o app funciona sem ele enquanto a IA escolhida for o
RandomForest.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kara_sygna import config
from kara_sygna.dados.repositorio import ConjuntoDados

logger = logging.getLogger(__name__)

#: Neurônios da camada escondida.
OCULTO = 64


class _Rede(nn.Module):
    def __init__(self, tipo: str, entradas: int, classes: int, oculto: int) -> None:
        super().__init__()
        self.tipo = tipo
        if tipo == "gru":
            self.leitor = nn.GRU(entradas, oculto, batch_first=True)
        elif tipo == "lstm":
            self.leitor = nn.LSTM(entradas, oculto, batch_first=True)
        elif tipo == "conv1d":
            self.leitor = nn.Sequential(
                nn.Conv1d(entradas, oculto, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(oculto, oculto, 3, padding=1),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Rede desconhecida: {tipo}")
        self.saida = nn.Sequential(nn.Dropout(0.2), nn.Linear(oculto, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (lote, quadros, valores)
        if self.tipo == "conv1d":
            resumo = self.leitor(x.transpose(1, 2)).mean(dim=2)
        else:
            saidas, _ = self.leitor(x)
            resumo = saidas[:, -1]  # o que a rede "lembra" no último quadro
        return self.saida(resumo)


class ModeloRede:
    """Rede treinada, com a mesma cara de um modelo do scikit-learn.

    Args:
        tipo: ``"gru"``, ``"lstm"`` ou ``"conv1d"``.
        classes: Sinais, na ordem das saídas da rede.
        media, desvio: Normalização de cada valor do quadro (do treino).
        estado: Pesos da rede (``state_dict``).
        oculto: Neurônios da camada escondida.
    """

    #: O app entrega a janela quadro a quadro (e não o resumo do RandomForest).
    usa_sequencia = True

    def __init__(
        self,
        tipo: str,
        classes: list[str],
        media: np.ndarray,
        desvio: np.ndarray,
        estado: dict,
        oculto: int = OCULTO,
    ) -> None:
        """Monta a rede com os pesos dados (veja a classe)."""
        self.tipo = tipo
        self.classes_ = np.asarray(classes)
        self.media = np.asarray(media, dtype=np.float32)
        self.desvio = np.asarray(desvio, dtype=np.float32)
        self.oculto = oculto
        self.n_entradas = len(self.media)
        self.rede = _Rede(tipo, self.n_entradas, len(classes), oculto)
        self.rede.load_state_dict(estado)
        self.rede.eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidade de cada sinal para cada janela ``(n, quadros, valores)``."""
        entrada = (np.asarray(X, dtype=np.float32) - self.media) / self.desvio
        with torch.no_grad():
            logitos = self.rede(torch.from_numpy(entrada))
        return torch.softmax(logitos, dim=1).numpy().astype(np.float64)


def _ajustar(
    tipo: str,
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    progresso: Callable[[str], None],
    epocas: int,
    semente: int = 0,
) -> ModeloRede:
    """Treina uma rede nova com as janelas ``X`` e os sinais ``y``."""
    torch.manual_seed(semente)
    X = np.asarray(X, dtype=np.float32)
    media = X.reshape(-1, X.shape[-1]).mean(axis=0)
    desvio = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    entrada = torch.from_numpy((X - media) / desvio)
    indice = {c: i for i, c in enumerate(classes)}
    alvo = torch.tensor([indice[s] for s in y], dtype=torch.long)

    # Sinais com menos gravações pesam mais (como o "balanced" do RandomForest).
    contagem = torch.bincount(alvo, minlength=len(classes)).clamp(min=1).float()
    pesos = len(alvo) / (len(classes) * contagem)

    rede = _Rede(tipo, X.shape[-1], len(classes), OCULTO)
    otimizador = torch.optim.Adam(rede.parameters(), lr=1e-3)
    perda = nn.CrossEntropyLoss(weight=pesos)
    gerador = torch.Generator().manual_seed(semente)
    for epoca in range(1, epocas + 1):
        rede.train()
        ordem = torch.randperm(len(alvo), generator=gerador)
        total = 0.0
        for inicio in range(0, len(ordem), config.LOTE_REDE):
            lote = ordem[inicio : inicio + config.LOTE_REDE]
            otimizador.zero_grad()
            erro = perda(rede(entrada[lote]), alvo[lote])
            erro.backward()
            otimizador.step()
            total += erro.item() * len(lote)
        if epoca % 5 == 0 or epoca == epocas:
            progresso(
                f"{tipo.upper()}: época {epoca}/{epocas}, erro {total / len(alvo):.3f}"
            )
    rede.eval()
    return ModeloRede(tipo, classes, media, desvio, rede.state_dict())


def treinar_rede(
    tipo: str,
    dados: ConjuntoDados,
    progresso: Callable[[str], None] = lambda _m: None,
    epocas: int | None = None,
) -> tuple[ModeloRede, float | None]:
    """Valida e treina a rede com as janelas quadro a quadro.

    Validação: de cada sinal com 2 ou mais gravações, a última gravação fica de
    fora do treino e serve de teste (só as janelas originais, sem cópias).
    Depois, a rede final é treinada com tudo.

    Args:
        dados: Janelas de :func:`kara_sygna.ia.sequencia.montar_amostras` com
            ``extrair=sequencias_de_uma_gravacao``.

    Returns:
        ``(modelo, acurácia da validação ou None)``.
    """
    epocas = config.EPOCAS_REDE if epocas is None else epocas
    # carregar_rede limita o PyTorch a 2 núcleos (para reconhecer sem travar o
    # app), e o limite vale para o processo inteiro: sem isto, treinar depois de
    # abrir o app com uma rede usava só 2 núcleos.
    antes = torch.get_num_threads()
    torch.set_num_threads(os.cpu_count() or antes)
    try:
        return _treinar_rede(tipo, dados, progresso, epocas)
    finally:
        torch.set_num_threads(antes)


def _treinar_rede(tipo, dados, progresso, epocas):
    """O treino de :func:`treinar_rede` (validação e rede final)."""
    classes = dados.classes
    teste_grupos = set()
    for sinal in classes:
        grupos = list(dict.fromkeys(dados.grupos[dados.y == sinal].tolist()))
        if len(grupos) >= 2:
            teste_grupos.add(grupos[-1])
    acuracia = None
    if teste_grupos:
        no_teste = np.isin(dados.grupos, list(teste_grupos))
        copias = (
            dados.mascarada
            if dados.mascarada is not None
            else np.zeros(len(dados), dtype=bool)
        )
        progresso(
            f"Validando a {nome(tipo)} (uma gravação de cada sinal fica de fora)..."
        )
        modelo = _ajustar_tipo(
            tipo, dados.X[~no_teste], dados.y[~no_teste], classes, progresso, epocas
        )
        teste = no_teste & ~copias
        previstos = modelo.classes_[modelo.predict_proba(dados.X[teste]).argmax(axis=1)]
        acuracia = float(np.mean(previstos == dados.y[teste]))
        progresso(f"Validação da {nome(tipo)}: {100 * acuracia:.1f}%")
    progresso(f"Treinando a {nome(tipo)} final com todas as gravações...")
    return _ajustar_tipo(tipo, dados.X, dados.y, classes, progresso, epocas), acuracia


#: IAs feitas de várias redes juntas: a resposta é a média das probabilidades.
#: Nas gravações de 24/10/2026, GRU + LSTM acertou 90% (89% cada uma sozinha).
COMITES = {"gru_lstm": ("gru", "lstm")}


def nome(tipo: str) -> str:
    """Nome para mostrar: ``"gru_lstm"`` -> ``"GRU + LSTM"``."""
    return tipo.replace("_", " + ").upper()


class ModeloComite:
    """Várias redes juntas (ex.: GRU + LSTM): a média das probabilidades delas.

    Cada rede erra em lugares um pouco diferentes; juntas, uma corrige a outra.
    """

    usa_sequencia = True

    def __init__(self, tipo: str, membros: list[ModeloRede]) -> None:
        """Junta as redes (todas com os mesmos sinais, na mesma ordem)."""
        self.tipo = tipo
        self.membros = membros
        self.classes_ = membros[0].classes_
        self.n_entradas = membros[0].n_entradas

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Média das probabilidades das redes."""
        return np.mean([m.predict_proba(X) for m in self.membros], axis=0)


def _ajustar_tipo(tipo, X, y, classes, progresso, epocas):
    """Uma rede, ou cada rede de um comitê (``COMITES``)."""
    if tipo in COMITES:
        return ModeloComite(
            tipo, [_ajustar(m, X, y, classes, progresso, epocas) for m in COMITES[tipo]]
        )
    return _ajustar(tipo, X, y, classes, progresso, epocas)


def _para_salvar(modelo: ModeloRede) -> dict:
    return {
        "tipo": modelo.tipo,
        "classes": [str(c) for c in modelo.classes_],
        "media": torch.from_numpy(modelo.media),
        "desvio": torch.from_numpy(modelo.desvio),
        "oculto": modelo.oculto,
        "estado": modelo.rede.state_dict(),
    }


def _de_salvo(dados: dict) -> ModeloRede:
    return ModeloRede(
        dados["tipo"],
        list(dados["classes"]),
        dados["media"].numpy(),
        dados["desvio"].numpy(),
        dados["estado"],
        int(dados["oculto"]),
    )


def salvar_rede(modelo: ModeloRede | ModeloComite, caminho: Path) -> None:
    """Grava a rede (``.pt``) de forma segura (arquivo temporário e troca)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    if isinstance(modelo, ModeloComite):
        conteudo = {
            "tipo": modelo.tipo,
            "membros": [_para_salvar(m) for m in modelo.membros],
        }
    else:
        conteudo = _para_salvar(modelo)
    torch.save(conteudo, temporario)
    os.replace(temporario, caminho)
    logger.info("Rede salva em %s", caminho)


def carregar_rede(caminho: Path) -> ModeloRede | ModeloComite:
    """Carrega uma rede (ou um comitê) salva por :func:`salvar_rede`."""
    dados = torch.load(Path(caminho), map_location="cpu", weights_only=True)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    if "membros" in dados:
        return ModeloComite(dados["tipo"], [_de_salvo(m) for m in dados["membros"]])
    return _de_salvo(dados)
