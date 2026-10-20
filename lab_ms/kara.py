"""Acesso ao projeto kara-sygna-py, usado como biblioteca (só leitura).

O laboratório não copia código do kara-sygna: ele importa o mesmo formato de
quadro, as mesmas janelas e o mesmo RandomForest. Assim a comparação vale
para o app de verdade.

A pasta do projeto vem da variável ``KARA_SYGNA_PY`` ou do caminho padrão:
a **cópia** em ``app/``, ao lado do laboratório. O kara-sygna-py original
não é usado nem alterado.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PASTA_PADRAO = Path(__file__).resolve().parent.parent / "app"


def pasta_kara_sygna() -> Path:
    """Pasta do kara-sygna-py (``KARA_SYGNA_PY`` ou o caminho padrão)."""
    pasta = Path(os.environ.get("KARA_SYGNA_PY", PASTA_PADRAO)).expanduser()
    if not (pasta / "kara_sygna" / "__init__.py").exists():
        raise FileNotFoundError(
            f"Não achei o kara-sygna-py em {pasta}. "
            "Aponte a variável KARA_SYGNA_PY para a pasta do projeto."
        )
    return pasta


def preparar_import() -> Path:
    """Deixa ``import kara_sygna`` funcionando e devolve a pasta do projeto."""
    pasta = pasta_kara_sygna()
    if str(pasta) not in sys.path:
        sys.path.insert(0, str(pasta))
    return pasta
