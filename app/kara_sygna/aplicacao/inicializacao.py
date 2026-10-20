"""Preparação feita uma vez, na partida do programa (interface ou terminal)."""

from __future__ import annotations

import logging

from kara_sygna import config
from kara_sygna.dados.migracao import migrar_arquivos_legados
from kara_sygna.logs import configurar_logging

logger = logging.getLogger(__name__)


def preparar_ambiente(nivel_log: int = logging.INFO) -> None:
    """Liga os logs, cria ``armazenamento/`` e atualiza arquivos de versões antigas.

    Args:
        nivel_log: Nível mínimo dos logs no terminal.
    """
    configurar_logging(nivel_log)
    config.PASTA_ARMAZENAMENTO.mkdir(parents=True, exist_ok=True)
    logger.info("Pasta de dados: %s", config.PASTA_ARMAZENAMENTO)
    migrar_arquivos_legados()
