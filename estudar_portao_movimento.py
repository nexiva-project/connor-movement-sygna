"""Portão de movimento: a IA só vota quando a mão se mexeu há pouco.

Ideia do projeto satyaidk/sign-language-recognition (GitHub): só reconhecer
enquanto há movimento. No nosso caso, o S parado virava "carro" e a mão
relaxada virava sinal; os dois ficam quase parados, e o carro e a casa não.

Velocidade = quanto o pulso (pelo braço, MediaPipe Pose) andou nos últimos 3
quadros, em larguras de ombro por segundo, a mão mais rápida das duas, média
dos últimos 0,3 s. Medido nos vídeos (mediana): S 0,26 · neutro 0,42 · B 0,43
· carro calmo 1,03 · casa 1,08–1,39 · carro rápido 1,8–4,7.

O portão fica ABERTO se a velocidade passou do limite nos últimos H segundos,
ou se não há corpo para medir (no escuro o Pose some). Fechado, o voto do
quadro vira "nenhum". Só no estudo: o MS.1 e o app não mudam.

Uso::

    python estudar_portao_movimento.py
"""

from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais
from kara_sygna.aplicacao.filtro_quadros import FiltroQuadros
from kara_sygna.visao.quadro import COLUNA_CORPO_PRESENTE, COLUNAS_PULSO_BRACO

import estudar_videos
from estudar_presinal import VISAO, carregar, colar, medir
from estudar_tudo import resumir, tabela
from estudar_videos import (
    ARMAZENAMENTO,
    PASTA_RESULTADOS,
    PASTA_VIDEOS,
    Relogio,
    Resultado,
    quadros_do_video,
)

MS1 = ARMAZENAMENTO / "MS.1"
#: (limite em larguras de ombro/s, janela H em s); None = sem portão.
VARIANTES = {
    "sem portão (MS.1 hoje)": None,
    "limite 0,6 · janela 0,5 s": (0.6, 0.5),
    "limite 0,6 · janela 1,0 s": (0.6, 1.0),
    "limite 0,8 · janela 0,5 s": (0.8, 0.5),
    "limite 0,8 · janela 1,0 s": (0.8, 1.0),
}


class Portao:
    """Decide, quadro a quadro, se a mão se mexeu nos últimos ``janela_s``."""

    def __init__(self, limite: float, janela_s: float, passo: int = 3, media_s: float = 0.3) -> None:
        self.limite, self.janela_s, self.passo, self.media_s = limite, janela_s, passo, media_s
        self._pulsos: deque = deque(maxlen=passo + 1)
        self._velocidades: deque = deque()
        self._ultimo_movimento = -1e9

    def atualizar(self, vetor: np.ndarray, agora: float) -> bool:
        """Registra o quadro e diz se o portão está aberto."""
        if vetor[COLUNA_CORPO_PRESENTE] <= 0.5:
            self._pulsos.clear()
            return True  # sem corpo não dá para medir: não bloqueia
        pulso = np.concatenate([vetor[list(COLUNAS_PULSO_BRACO[lado])] for lado in ("esq", "dir")])
        self._pulsos.append((agora, pulso))
        if len(self._pulsos) > self.passo:
            (t0, p0), (t1, p1) = self._pulsos[0], self._pulsos[-1]
            d = p1 - p0
            v = max(np.hypot(d[0], d[1]), np.hypot(d[2], d[3])) / max(t1 - t0, 1e-3)
            self._velocidades.append((agora, v))
        while self._velocidades and agora - self._velocidades[0][0] > self.media_s:
            self._velocidades.popleft()
        if self._velocidades and np.mean([v for _, v in self._velocidades]) > self.limite:
            self._ultimo_movimento = agora
        return agora - self._ultimo_movimento <= self.janela_s


def reconhecer_com_portao(vetores, tempos, portao_cfg, com_pre: bool = True):
    """Como o app ao vivo, mas o voto só vale com o portão aberto.

    Returns:
        ``(Resultado, escritos com momento, votos com momento, pré-sinais vistos)``.
    """
    relogio = Relogio()
    antes = config.DIAGNOSTICO_ATIVO
    config.DIAGNOSTICO_ATIVO = False
    try:
        controlador = ControladorSinais(caminho_modelo=MS1 / "modelo_sinais.pkl", relogio=relogio)
        controlador.caminho_modelo_pre = MS1 / ("modelo_sinais_pre.pkl" if com_pre else "nao_existe.pkl")
        controlador.iniciar_reconhecimento()
    finally:
        config.DIAGNOSTICO_ATIVO = antes
    controlador.filtro_quadros = FiltroQuadros(relogio=relogio)

    portao = Portao(*portao_cfg) if portao_cfg else None
    aberto = {"agora": True}
    atualizar_original = controlador.estabilizador.atualizar

    def atualizar_com_portao(sinal):
        return atualizar_original(sinal if aberto["agora"] else None)

    controlador.estabilizador.atualizar = atualizar_com_portao

    resultado, escritos, votos, pre_vistos = Resultado(), [], [], []
    for vetor, momento in zip(vetores, tempos, strict=True):
        relogio.agora = float(momento)
        aberto["agora"] = portao.atualizar(vetor, float(momento)) if portao else True
        resposta = controlador.processar_quadro(vetor)
        voto = (controlador._ultimo_voto or "") if aberto["agora"] else ""
        resultado.quadros += 1
        resultado.votos[voto] += 1
        votos.append((float(momento), voto))
        if controlador._diag.get("pre_novo"):
            pre_vistos.append((float(momento), controlador._diag["pre_novo"]))
        if resposta.sinal_confirmado:
            resultado.escritos.append(resposta.sinal_confirmado)
            escritos.append((float(momento), resposta.sinal_confirmado))
    return resultado, escritos, votos, pre_vistos


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    estudar_videos.MODELO_PRE = MS1 / "modelo_sinais_pre.pkl"
    videos = sorted(v for v in PASTA_VIDEOS.glob("*.avi") if not v.stem.startswith(("_", "presinal")))
    base = {v.stem: quadros_do_video(v, refazer=False, opcoes=VISAO) for v in videos}
    v_s, t_s = carregar("presinal_s")[:2]
    v_b, t_b = carregar("presinal_b")[:2]
    v_et, t_et = carregar("presinal_s_carro")[:2]

    gerais, presinal = {}, {}
    for nome, cfg in VARIANTES.items():
        resultados = {
            stem: (reconhecer_com_portao(vet, tem, cfg)[0], {s for s in dados["sinal"].split(",") if s}, vet)
            for stem, (vet, tem, dados) in base.items()
        }
        gerais[nome] = resumir(resultados)

        linha = {}
        for pre, (vp, tp), video, sinal in (
            ("S", (v_s, t_s), "carro_luz_normal_rapido", "carro"),
            ("B", (v_b, t_b), "casa_luz_normal", "casa"),
        ):
            v_sig, t_sig = carregar(video)[:2]
            junto_v, junto_t = colar((vp, tp), (v_sig, t_sig))
            _, escritos, votos, pre_vistos = reconhecer_com_portao(junto_v, junto_t, cfg)
            m = medir(escritos, votos, pre_vistos, tp[-1] + 1 / 30, sinal, pre)
            linha[f"{pre} sozinho escreve"] = m["escrito durante o pré-sinal"]
            linha[f"{pre} + {sinal}: 1º {sinal} após"] = m["1º sinal escrito após"]
        escritos = reconhecer_com_portao(v_et, t_et, cfg)[1]
        linha["etapas S→carro→S→carro"] = ", ".join(f"{s} {m:.1f} s" for m, s in escritos) or "nada"
        presinal[nome] = linha
        print(f"ok: {nome}", flush=True)

    linhas = [
        "# Portão de movimento nos vídeos de teste (MS.1)",
        "",
        "Velocidade do pulso (Pose) em larguras de ombro/s; portão aberto se passou do",
        "limite nos últimos H segundos, ou sem corpo para medir.",
        "",
        *tabela("Os 11 vídeos", gerais),
        *tabela("Pré-sinal parado e S/B + sinal", presinal),
        f"Tempo total: {time.perf_counter() - inicio:.0f} s",
    ]
    texto = "\n".join(linhas)
    arquivo = PASTA_RESULTADOS / f"portao_movimento_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
