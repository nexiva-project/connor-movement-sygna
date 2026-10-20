"""Testa o pré-sinal com o MS.1: S antes do carro e B antes da casa ajudam?

O carro é feito com as mãos em S, e a casa com as mãos em B. O app usa o
pré-sinal assim: se viu o S nos últimos ``SEGUNDOS_ESPERA_SINAL_PADRAO`` (3 s),
aceita o carro com ``CONFIANCA_MINIMA`` (70%); sem o S, exige
``CONFIANCA_SEM_PRE_SINAL`` (78%).

Os vídeos do S e do B foram gravados sozinhos (``presinal_s``, ``presinal_b``)
e aqui são colados antes dos vídeos de carro e casa que já existiam. Para cada
vídeo de sinal, três situações:

1. sinal sozinho (o app espera o pré-sinal e não vê: exige 78%);
2. pré-sinal + sinal (vê o S/B logo antes: aceita com 70%);
3. pré-sinal + sinal, sem a IA de pré-sinais (referência: 70% sempre).

Também: o vídeo ``presinal_s_carro`` (S → carro → S → carro, gravado em etapas).

Uso::

    python estudar_presinal.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais
from kara_sygna.aplicacao.filtro_quadros import FiltroQuadros

from estudar_videos import (
    ARMAZENAMENTO,
    PASTA_RESULTADOS,
    PASTA_VIDEOS,
    OpcoesVisao,
    Relogio,
    quadros_do_video,
)

MS1 = ARMAZENAMENTO / "MS.1"
#: Modelo dos sinais testado (outro script pode trocar).
MODELO_SINAIS = MS1 / "modelo_sinais.pkl"
#: A visão do app hoje (realce desligado, memória 0,8 s, detecção 0,3, 3 detectores de mãos).
VISAO = OpcoesVisao("cp_maos3", detectores_maos=3, realce=False, memoria_s=0.8, confianca_deteccao=0.3)
CASOS = [
    ("S", "presinal_s", "carro", ["carro_luz_normal_rapido", "carro_luz_normal_calmo", "carro_perto"]),
    ("B", "presinal_b", "casa", ["casa_luz_normal", "casa_luz_escura"]),
]


def carregar(nome: str):
    vetores, tempos, dados = quadros_do_video(PASTA_VIDEOS / f"{nome}.avi", refazer=False, opcoes=VISAO)
    return vetores, tempos - tempos[0], dados


def colar(*partes):
    """Junta vídeos em sequência (o tempo do segundo continua do primeiro)."""
    vetores, tempos, inicio = [], [], 0.0
    for v, t in partes:
        vetores.append(v)
        tempos.append(t + inicio)
        inicio = tempos[-1][-1] + 1 / 30
    return np.vstack(vetores), np.concatenate(tempos)


def reconhecer(vetores, tempos, com_pre: bool):
    """Roda o app no tempo do vídeo; devolve escritos, votos e pré-sinais vistos, com o momento."""
    relogio = Relogio()
    antes = config.DIAGNOSTICO_ATIVO
    config.DIAGNOSTICO_ATIVO = False
    try:
        controlador = ControladorSinais(caminho_modelo=MODELO_SINAIS, relogio=relogio)
        controlador.caminho_modelo_pre = MS1 / ("modelo_sinais_pre.pkl" if com_pre else "nao_existe.pkl")
        controlador.iniciar_reconhecimento()
    finally:
        config.DIAGNOSTICO_ATIVO = antes
    controlador.filtro_quadros = FiltroQuadros(relogio=relogio)
    escritos, votos, pre_vistos = [], [], []
    for vetor, momento in zip(vetores, tempos, strict=True):
        relogio.agora = float(momento)
        resposta = controlador.processar_quadro(vetor)
        votos.append((float(momento), controlador._ultimo_voto or ""))
        if controlador._diag.get("pre_novo"):
            pre_vistos.append((float(momento), controlador._diag["pre_novo"]))
        if resposta.sinal_confirmado:
            escritos.append((float(momento), resposta.sinal_confirmado))
    return escritos, votos, pre_vistos


def medir(escritos, votos, pre_vistos, inicio_sinal: float, sinal: str, pre: str) -> dict:
    """Números do trecho do sinal (a partir de ``inicio_sinal``) e do trecho do pré-sinal."""
    no_sinal = [v for t, v in votos if t >= inicio_sinal]
    escritos_sinal = [s for t, s in escritos if t >= inicio_sinal]
    primeiro = next((t - inicio_sinal for t, s in escritos if t >= inicio_sinal and s == sinal), None)
    escritos_pre = Counter(s for t, s in escritos if t < inicio_sinal)
    return {
        "pré-sinal visto": ", ".join(f"{p} {t:.1f} s" for t, p in pre_vistos if t < inicio_sinal + 3)[:60] or "não",
        "escrito durante o pré-sinal": ", ".join(f"{s}×{n}" for s, n in escritos_pre.items()) or "nada",
        "1º sinal escrito após": "-" if primeiro is None else f"{primeiro:.2f} s",
        "voto certo no sinal": f"{100 * sum(v == sinal for v in no_sinal) / max(1, len(no_sinal)):.0f}%",
        "escritos no sinal": ", ".join(f"{s}×{n}" for s, n in Counter(escritos_sinal).items()) or "nada",
    }


def tabela(titulo: str, linhas: dict[str, dict]) -> list[str]:
    colunas = list(next(iter(linhas.values())))
    saida = [f"### {titulo}", "", "| Situação | " + " | ".join(colunas) + " |", "|---" * (len(colunas) + 1) + "|"]
    saida += [f"| {nome} | " + " | ".join(str(v[c]) for c in colunas) + " |" for nome, v in linhas.items()]
    return [*saida, ""]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    linhas = [
        "# Pré-sinal com o MS.1",
        "",
        "Visão do app hoje (3 detectores de mãos), modelo e IA de pré-sinais do MS.1.",
        "\"1º sinal escrito após\" conta do começo do vídeo do sinal.",
        "",
    ]
    for pre, video_pre, sinal, videos in CASOS:
        v_pre, t_pre = carregar(video_pre)[:2]
        linhas.append(f"## {pre} → {sinal}")
        linhas.append("")
        for video in videos:
            v_sinal, t_sinal = carregar(video)[:2]
            junto_v, junto_t = colar((v_pre, t_pre), (v_sinal, t_sinal))
            inicio_sinal = t_pre[-1] + 1 / 30
            situacoes = {
                f"{sinal} sozinho": medir(*reconhecer(v_sinal, t_sinal, True), 0.0, sinal, pre),
                f"{pre} + {sinal}": medir(*reconhecer(junto_v, junto_t, True), inicio_sinal, sinal, pre),
                f"{pre} + {sinal}, sem IA de pré-sinal": medir(
                    *reconhecer(junto_v, junto_t, False), inicio_sinal, sinal, pre
                ),
            }
            linhas += tabela(video, situacoes)
            print(f"ok: {pre} + {video}", flush=True)

    # Vídeo gravado em etapas: S (0–4 s) → carro (4–10 s) → S (10–14 s) → carro (14–20 s).
    v, t, _ = carregar("presinal_s_carro")
    linhas += ["## Vídeo em etapas: S → carro → S → carro", ""]
    for com_pre in (True, False):
        escritos, _, pre_vistos = reconhecer(v, t, com_pre)
        rotulo = "com IA de pré-sinal" if com_pre else "sem IA de pré-sinal"
        linhas.append(
            f"- {rotulo}: escritos "
            + (", ".join(f"{s} {m:.1f} s" for m, s in escritos) or "nada")
            + "; pré-sinais vistos "
            + (", ".join(f"{p} {m:.1f} s" for m, p in pre_vistos) or "nenhum")
        )
    linhas += ["", f"Tempo total: {time.perf_counter() - inicio:.0f} s"]

    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"presinal_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
