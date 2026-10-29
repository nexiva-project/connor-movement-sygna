"""Qual IA reconhece os sinais: RandomForest ou uma rede neural (GRU, LSTM, Conv1D).

Cada IA tem o seu arquivo, ao lado das gravações, então trocar de IA não
apaga nenhuma: dá para testar uma rede e voltar ao RandomForest. A escolha
fica em ``ia_escolhida.json`` e é feita na janela que abre antes da tela de
carregamento (:mod:`kara_sygna.interface.janela_escolha_ia`).

As redes neurais precisam do PyTorch (``pip install torch``); sem ele, só o
RandomForest aparece como disponível.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from kara_sygna import config

logger = logging.getLogger(__name__)

RANDOMFOREST = "randomforest"
REDES = ("gru", "lstm", "conv1d", "gru_lstm")
ARQUIVO_ESCOLHA = "ia_escolhida.json"


@dataclass(frozen=True)
class TipoIA:
    """Uma IA da lista de escolha."""

    chave: str
    nome: str
    descricao: str


TIPOS = (
    TipoIA(
        RANDOMFOREST,
        "RandomForest",
        "Floresta de decisões. Rápida e aprende com poucas gravações (a de sempre).",
    ),
    TipoIA("gru", "GRU", "Rede neural que lê os quadros em ordem, com memória."),
    TipoIA("lstm", "LSTM", "Rede neural parecida com a GRU, um pouco mais pesada."),
    TipoIA("conv1d", "Conv1D", "Rede neural que procura pedaços de movimento."),
    TipoIA(
        "gru_lstm",
        "GRU + LSTM",
        "As duas redes juntas (a média delas). Um pouco melhor; treino em dobro.",
    ),
)


#: Detectores de mãos: (chave, nome, descrição). Veja ``config.DETECTOR_MAOS``.
DETECTORES_MAOS = (
    ("hands", "MediaPipe Hands", "o de sempre"),
    ("landmarker", "Hand Landmarker", "novo, do MediaPipe Tasks (em teste)"),
    ("rtmpose", "RTMPose (mão)", "em teste, só 2D, ~30 quadros/s"),
    ("rtmw_m", "RTMW-m (corpo inteiro, leve)", "em teste, só 2D, ~27 quadros/s"),
    ("rtmw", "RTMW-x (corpo inteiro)", "em teste, só 2D, ~18 quadros/s, mão melhor"),
)


def _ler_escolha(pasta: Path) -> dict:
    try:
        dados = json.loads((pasta / ARQUIVO_ESCOLHA).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dados if isinstance(dados, dict) else {}


def detector_escolhido(pasta: Path | None = None) -> str:
    """O detector de mãos escolhido (``"hands"`` se nada foi escolhido)."""
    pasta = Path(pasta) if pasta is not None else config.PASTA_ARMAZENAMENTO
    detector = _ler_escolha(pasta).get("detector_maos", "hands")
    return detector if detector in {d[0] for d in DETECTORES_MAOS} else "hands"


def torch_disponivel() -> bool:
    """Diz se o PyTorch está instalado (as redes neurais precisam dele)."""
    return importlib.util.find_spec("torch") is not None


def caminho_modelo(tipo: str, pasta: Path | None = None) -> Path:
    """Arquivo da IA ``tipo`` (o RandomForest continua em ``modelo_sinais.pkl``)."""
    pasta = Path(pasta) if pasta is not None else config.PASTA_ARMAZENAMENTO
    if tipo == RANDOMFOREST:
        return pasta / config.ARQUIVO_MODELO.name
    if tipo not in REDES:
        raise ValueError(f"IA desconhecida: {tipo}")
    return pasta / f"{config.ARQUIVO_MODELO.stem}_{tipo}.pt"


def tipo_do_caminho(caminho: Path) -> str:
    """Qual IA está no arquivo (pelo nome: ``modelo_sinais_gru.pt`` -> ``"gru"``)."""
    caminho = Path(caminho)
    if caminho.suffix != ".pt":
        return RANDOMFOREST
    base = f"{config.ARQUIVO_MODELO.stem}_"
    tipo = caminho.stem.removeprefix(base) if caminho.stem.startswith(base) else ""
    if tipo not in REDES:
        tipo = caminho.stem.rsplit("_", 1)[-1]
    if tipo not in REDES:
        raise ValueError(f"Não sei qual IA está em {caminho.name}.")
    return tipo


def tipo_escolhido(pasta: Path | None = None) -> str:
    """A IA escolhida da última vez (RandomForest se nada foi escolhido)."""
    pasta = Path(pasta) if pasta is not None else config.PASTA_ARMAZENAMENTO
    try:
        tipo = json.loads((pasta / ARQUIVO_ESCOLHA).read_text(encoding="utf-8"))["tipo"]
    except (OSError, ValueError, KeyError, TypeError):
        return RANDOMFOREST
    if tipo in REDES and not torch_disponivel():
        logger.warning("A IA %s precisa do PyTorch; usando o RandomForest.", tipo)
        return RANDOMFOREST
    return tipo if tipo == RANDOMFOREST or tipo in REDES else RANDOMFOREST


def escolher(
    tipo: str, pasta: Path | None = None, detector_maos: str | None = None
) -> None:
    """Guarda a IA escolhida (e, se dado, o detector de mãos)."""
    if tipo != RANDOMFOREST and tipo not in REDES:
        raise ValueError(f"IA desconhecida: {tipo}")
    pasta = Path(pasta) if pasta is not None else config.PASTA_ARMAZENAMENTO
    pasta.mkdir(parents=True, exist_ok=True)
    escolha = _ler_escolha(pasta) | {"tipo": tipo}
    if detector_maos is not None:
        if detector_maos not in {d[0] for d in DETECTORES_MAOS}:
            raise ValueError(f"Detector de mãos desconhecido: {detector_maos}")
        escolha["detector_maos"] = detector_maos
    (pasta / ARQUIVO_ESCOLHA).write_text(json.dumps(escolha), encoding="utf-8")
    logger.info(
        "IA escolhida: %s (mãos: %s)", tipo, escolha.get("detector_maos", "hands")
    )


def resumo(tipo: str, pasta: Path | None = None) -> dict | None:
    """O resumo do último treino da IA (validação, amostras...), ou ``None``."""
    caminho = caminho_modelo(tipo, pasta)
    if not caminho.exists():
        return None
    try:
        dados = json.loads(caminho.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        dados = {}
    dados["modificado_em"] = caminho.stat().st_mtime
    return dados
