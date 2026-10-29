"""RTMW-x 256x192 contra 384x288 na placa de vídeo (24/10/2026).

Com a placa de vídeo sobrou tempo, então dá para usar o RTMW-x maior
(384x288, o modo "performance" do rtmlib), que vê a mão com mais pixels. A
pergunta: a malha dos dedos fica mais firme no vídeo do defeito (o dedo que
"mexe sozinho" quando fica atrás da mão)?

Medido nos pontos crus (sem One-Euro), em cada mão vista nos dois modelos:

* nota média dos pontos da mão (a certeza do modelo);
* tremida: quanto cada dedo anda de um quadro para o outro, tirando o
  movimento do pulso (em palmas; mediana);
* pulos: quadros em que algum dedo anda mais de 0,35 palma sem o pulso andar
  junto (o defeito que você mostrou);
* tempo por quadro na placa de vídeo.

Salva em ``resultados/`` uma imagem lado a lado dos quadros em que os dois
modelos mais discordam.

Uso::

    python estudar_rtmw_384.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rtmlib

RAIZ = Path(__file__).resolve().parent
VIDEO = RAIZ / "videos" / "defeito_malha.avi"
MAOS = {"esq": slice(91, 112), "dir": slice(112, 133)}
PULO = 0.35
NOTA_MAO = 0.45


def criar(pose_modo: str):
    """Wholebody com YOLOX-tiny e o RTMW-x do modo pedido, na placa de vídeo."""
    original = ort.InferenceSession

    def sessao(*args, **kwargs):
        opcoes = ort.SessionOptions()
        opcoes.enable_mem_pattern = False
        opcoes.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        kwargs["sess_options"] = opcoes
        kwargs["providers"] = ["DmlExecutionProvider", "CPUExecutionProvider"]
        return original(*args, **kwargs)

    modos = rtmlib.Wholebody.MODE
    ort.InferenceSession = sessao
    try:
        return rtmlib.Wholebody(
            det=modos["lightweight"]["det"],
            det_input_size=modos["lightweight"]["det_input_size"],
            pose=modos[pose_modo]["pose"],
            pose_input_size=modos[pose_modo]["pose_input_size"],
            backend="onnxruntime",
            device="cpu",
        )
    finally:
        ort.InferenceSession = original


def rodar(modelo, quadros):
    """Pontos e notas da melhor pessoa em cada quadro, e o tempo médio."""
    pontos, notas, tempos = [], [], []
    for quadro in quadros:
        inicio = time.perf_counter()
        kp, sc = modelo(quadro)
        tempos.append(time.perf_counter() - inicio)
        if len(kp) == 0:
            pontos.append(None)
            notas.append(None)
            continue
        melhor = int(np.argmax(sc.mean(axis=1)))
        pontos.append(kp[melhor])
        notas.append(sc[melhor])
    return pontos, notas, 1000 * float(np.mean(tempos[5:]))


def palma(mao):
    return float(np.linalg.norm(mao[0] - mao[9])) + 1e-6


def medir(pontos, notas):
    """Nota média, tremida (mediana) e pulos por mão."""
    res = {}
    for lado, fatia in MAOS.items():
        nota, tremida, pulos, vistos = [], [], 0, 0
        anterior = None
        for kp, sc in zip(pontos, notas, strict=True):
            if kp is None or sc[fatia].mean() < NOTA_MAO:
                anterior = None
                continue
            vistos += 1
            mao = kp[fatia]
            nota.append(float(sc[fatia].mean()))
            if anterior is not None:
                p = palma(mao)
                rel = (mao - mao[0]) - (anterior - anterior[0])
                anda = np.linalg.norm(rel[1:], axis=1) / p
                tremida.append(float(np.median(anda)))
                pulso = np.linalg.norm(mao[0] - anterior[0]) / p
                if anda.max() > PULO and pulso < PULO / 2:
                    pulos += 1
            anterior = mao
        res[lado] = {
            "vistos": vistos,
            "nota": float(np.mean(nota)) if nota else 0.0,
            "tremida": float(np.median(tremida)) if tremida else 0.0,
            "pulos": pulos,
        }
    return res


def desenhar(quadro, kp, sc, titulo):
    img = quadro.copy()
    for fatia in MAOS.values():
        if sc[fatia].mean() < NOTA_MAO:
            continue
        for x, y in kp[fatia]:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)
    cv2.putText(img, titulo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    quadros = []
    while True:
        ok, q = cap.read()
        if not ok:
            break
        quadros.append(q)
    print(f"{VIDEO.name}: {len(quadros)} quadros")

    resultados = {}
    for nome, modo in (("256x192", "balanced"), ("384x288", "performance")):
        pontos, notas, ms = rodar(criar(modo), quadros)
        resultados[nome] = (pontos, notas)
        print(f"\nRTMW-x {nome}: {ms:.1f} ms por quadro (deteccao + malha)")
        for lado, m in medir(pontos, notas).items():
            print(
                f"  mao {lado}: vista em {m['vistos']} quadros, nota {m['nota']:.3f}, "
                f"tremida {m['tremida']:.4f} palma, pulos {m['pulos']}"
            )

    # Quadros em que os dois mais discordam (dedos, em palmas).
    (p1, n1), (p2, n2) = resultados["256x192"], resultados["384x288"]
    difs = []
    for i, (a, b) in enumerate(zip(p1, p2, strict=True)):
        if a is None or b is None:
            continue
        for fatia in MAOS.values():
            if n1[i][fatia].mean() >= NOTA_MAO and n2[i][fatia].mean() >= NOTA_MAO:
                d = np.linalg.norm(a[fatia] - b[fatia], axis=1).mean() / palma(a[fatia])
                difs.append((d, i))
    difs.sort(reverse=True)
    # 4 quadros diferentes entre os que mais discordam, longe um do outro.
    escolhidos = []
    for _, i in difs:
        if all(abs(i - j) > 15 for j in escolhidos):
            escolhidos.append(i)
        if len(escolhidos) == 4:
            break
    linhas = [
        np.hstack(
            [
                desenhar(quadros[i], p1[i], n1[i], f"256x192 q{i}"),
                desenhar(quadros[i], p2[i], n2[i], f"384x288 q{i}"),
            ]
        )
        for i in sorted(escolhidos)
    ]
    if linhas:
        saida = RAIZ / "resultados" / "rtmw_256_vs_384_20261024.jpg"
        cv2.imwrite(str(saida), np.vstack(linhas))
        print(f"\nImagem: {saida}")
    if difs:
        print(f"Diferenca media entre os dois: {np.mean([d for d, _ in difs]):.3f} palma")


if __name__ == "__main__":
    main()
