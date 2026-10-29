"""RTMW-x com recorte só da mão: malha sem tremer? (24/10/2026)

O RTMW-x vê o corpo inteiro em 256x192, então a mão vira uns 30x30 pixels
e os dedos tremem. O Hand Landmarker recorta só a mão e por isso a malha fica
firme. Ideia: usar o RTMW-x para achar a mão e, numa segunda passada, rodar
um modelo no recorte só da mão (a mão ocupa a imagem toda).

Variantes, no vídeo do defeito, na placa de vídeo:

* corpo: o app hoje (RTMW-x no corpo inteiro);
* rtmw_mao: RTMW-x de novo, no recorte da mão;
* rtmw_mao_espelho: igual, com a média do recorte e do recorte espelhado;
* rtmpose_mao: RTMPose de mão (256x256) no recorte da mão.

Medidas (pontos crus, sem One-Euro, nos mesmos quadros para todas):
tremida dos dedos (em palmas, tirando o movimento do pulso), pulos (dedo anda
mais de 0,35 palma sem o pulso andar) e tempo por quadro.

Salva em ``resultados/`` um vídeo lado a lado (corpo x melhor variante).

Uso::

    python estudar_rtmw_mao.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rtmlib

RAIZ = Path(__file__).resolve().parent
VIDEO = RAIZ / "videos" / "defeito_malha.avi"
PASTA_MODELOS = Path(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints"))
RTMPOSE_MAO = "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx"
MAOS = {"esq": slice(91, 112), "dir": slice(112, 133)}
PULO = 0.35
NOTA_MAO = 0.45
# Lado do recorte: tantas vezes o maior lado da mão (o rtmlib ainda soma 25%).
TAMANHO_RECORTE = 1.4
LIGACOES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def na_placa(criar):
    """Cria modelos do rtmlib com as sessões do ONNX na placa de vídeo."""
    original = ort.InferenceSession

    def sessao(*args, **kwargs):
        opcoes = ort.SessionOptions()
        opcoes.enable_mem_pattern = False
        opcoes.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        kwargs["sess_options"] = opcoes
        kwargs["providers"] = ["DmlExecutionProvider", "CPUExecutionProvider"]
        return original(*args, **kwargs)

    ort.InferenceSession = sessao
    try:
        return criar()
    finally:
        ort.InferenceSession = original


def palma(mao):
    return float(np.linalg.norm(mao[0] - mao[9])) + 1e-6


def caixa_da_mao(mao, largura, altura):
    """Caixa quadrada em volta da mão (xyxy)."""
    x0, y0 = mao.min(axis=0)
    x1, y1 = mao.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lado = max(x1 - x0, y1 - y0, 40.0) * TAMANHO_RECORTE
    return [
        max(0.0, cx - lado / 2),
        max(0.0, cy - lado / 2),
        min(float(largura), cx + lado / 2),
        min(float(altura), cy + lado / 2),
    ]


def mao_mais_perto(kp, sc, referencia):
    """No recorte o RTMW não vê o corpo: pega a mão (esq/dir) mais perto."""
    melhor = None
    for fatia in MAOS.values():
        d = np.linalg.norm(kp[fatia] - referencia, axis=1).mean()
        if melhor is None or d < melhor[0]:
            melhor = (d, kp[fatia], sc[fatia])
    return melhor[1], melhor[2]


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    quadros = []
    while True:
        ok, q = cap.read()
        if not ok:
            break
        quadros.append(q)
    altura, largura = quadros[0].shape[:2]
    print(f"{VIDEO.name}: {len(quadros)} quadros")

    modos = rtmlib.Wholebody.MODE
    corpo = na_placa(
        lambda: rtmlib.Wholebody(
            det=modos["lightweight"]["det"],
            det_input_size=modos["lightweight"]["det_input_size"],
            pose=modos["balanced"]["pose"],
            pose_input_size=modos["balanced"]["pose_input_size"],
            backend="onnxruntime",
            device="cpu",
        )
    )
    rtmpose_mao = na_placa(
        lambda: rtmlib.RTMPose(
            str(PASTA_MODELOS / RTMPOSE_MAO), model_input_size=(256, 256)
        )
    )

    variantes = ["corpo", "rtmw_mao", "rtmw_mao_espelho", "rtmpose_mao"]
    # pontos[variante][lado][i] = mão (21x2) ou None
    pontos = {v: {lado: [None] * len(quadros) for lado in MAOS} for v in variantes}
    tempos = {v: [] for v in variantes}

    for i, quadro in enumerate(quadros):
        inicio = time.perf_counter()
        kp, sc = corpo(quadro)
        tempo_corpo = time.perf_counter() - inicio
        tempos["corpo"].append(tempo_corpo)
        extra = {v: 0.0 for v in variantes[1:]}
        if len(kp):
            melhor = int(np.argmax(sc.mean(axis=1)))
            kp, sc = kp[melhor], sc[melhor]
            for lado, fatia in MAOS.items():
                if sc[fatia].mean() < NOTA_MAO:
                    continue
                mao = kp[fatia]
                pontos["corpo"][lado][i] = mao
                caixa = caixa_da_mao(mao, largura, altura)

                t = time.perf_counter()
                k2, s2 = corpo.pose_model(quadro, bboxes=[caixa])
                recorte, _ = mao_mais_perto(k2[0], s2[0], mao)
                extra["rtmw_mao"] += time.perf_counter() - t
                pontos["rtmw_mao"][lado][i] = recorte

                t = time.perf_counter()
                espelho = cv2.flip(quadro, 1)
                caixa_e = [largura - caixa[2], caixa[1], largura - caixa[0], caixa[3]]
                k3, s3 = corpo.pose_model(espelho, bboxes=[caixa_e])
                k3 = k3[0].copy()
                k3[:, 0] = largura - 1 - k3[:, 0]
                espelhada, _ = mao_mais_perto(k3, s3[0], mao)
                extra["rtmw_mao_espelho"] += time.perf_counter() - t
                pontos["rtmw_mao_espelho"][lado][i] = (recorte + espelhada) / 2

                t = time.perf_counter()
                k4, _ = rtmpose_mao(quadro, bboxes=[caixa])
                extra["rtmpose_mao"] += time.perf_counter() - t
                pontos["rtmpose_mao"][lado][i] = k4[0]
        tempos["rtmw_mao"].append(tempo_corpo + extra["rtmw_mao"])
        tempos["rtmw_mao_espelho"].append(
            tempo_corpo + extra["rtmw_mao"] + extra["rtmw_mao_espelho"]
        )
        tempos["rtmpose_mao"].append(tempo_corpo + extra["rtmpose_mao"])

    # Mesmos quadros para todas: onde a mão aparece de verdade no corpo.
    resumo = {}
    for v in variantes:
        trem, pulos = [], 0
        for lado in MAOS:
            anterior = None
            for i, mao in enumerate(pontos[v][lado]):
                base = pontos["corpo"][lado][i]
                if mao is None or base is None or base[:, 1].max() >= altura - 10:
                    anterior = None
                    continue
                if anterior is not None:
                    p = palma(mao)
                    rel = (mao - mao[0]) - (anterior - anterior[0])
                    anda = np.linalg.norm(rel[1:], axis=1) / p
                    trem.append(float(np.median(anda)))
                    pulso = np.linalg.norm(mao[0] - anterior[0]) / p
                    if anda.max() > PULO and pulso < PULO / 2:
                        pulos += 1
                anterior = mao
        ms = 1000 * float(np.mean(tempos[v][5:]))
        resumo[v] = (float(np.median(trem)), float(np.mean(trem)), pulos, ms)
        print(
            f"{v:18s} tremida {resumo[v][0]:.4f} palma (media {resumo[v][1]:.4f}), "
            f"pulos {pulos:3d}, {ms:5.1f} ms por quadro"
        )

    melhor = min(variantes[1:], key=lambda v: (resumo[v][2], resumo[v][1]))
    saida = RAIZ / "resultados" / f"rtmw_corpo_vs_{melhor}_20261024.mp4"
    escritor = cv2.VideoWriter(
        str(saida), cv2.VideoWriter_fourcc(*"mp4v"), 30, (2 * largura, altura)
    )
    for i, quadro in enumerate(quadros):
        lados = []
        for v in ("corpo", melhor):
            img = quadro.copy()
            for lado in MAOS:
                mao = pontos[v][lado][i]
                if mao is None:
                    continue
                for a, b in LIGACOES:
                    pa = tuple(int(c) for c in mao[a])
                    pb = tuple(int(c) for c in mao[b])
                    cv2.line(img, pa, pb, (0, 255, 255), 2)
            cv2.putText(img, v, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            lados.append(img)
        escritor.write(np.hstack(lados))
    escritor.release()
    print(f"\nMelhor: {melhor}. Video: {saida}")


if __name__ == "__main__":
    main()
