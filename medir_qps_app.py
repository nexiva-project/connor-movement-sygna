"""Mede quantos quadros por segundo a janela do app (cópia) consegue tratar.

Roda a janela de verdade (Tk, detectores, desenho), mas com um vídeo de teste
no lugar da webcam. A câmera falsa imita a real: entrega um quadro a cada
1/30 s, e ``ler()`` ESPERA o próximo quadro (como a webcam medida em
19/10/2026). Assim dá para comparar o laço do app sem depender da luz.

Versões:

* ``A``: como o app está (pausa de ``INTERVALO_QUADRO_MS`` = 15 ms);
* ``B``: pausa de 1 ms;
* ``C``: pausa de 1 ms e câmera lida numa thread (o app pega o quadro mais
  novo sem esperar);
* ``D``: o revezamento da cópia (``REVEZAMENTO_DETECTORES = 2``): duas equipes
  de detectores em threads, câmera em thread.

O qps conta os quadros que chegaram ao fim do caminho (``_depois_da_deteccao``).

Uso::

    python medir_qps_app.py            # as três, 15 s cada
"""

from __future__ import annotations

import itertools
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna import config

VIDEO = Path(__file__).resolve().parent / "videos" / "carro_luz_normal_rapido.avi"
QPS_CAMERA = 30.0
SEGUNDOS = 15.0


class CameraDeVideo:
    """Webcam falsa: um quadro novo a cada 1/30 s; ``ler()`` espera o próximo."""

    def __init__(self, caminho: Path) -> None:
        leitura = cv2.VideoCapture(str(caminho))
        self._quadros = []
        while True:
            ok, quadro = leitura.read()
            if not ok:
                break
            self._quadros.append(quadro)
        leitura.release()
        self.altura, self.largura = self._quadros[0].shape[:2]
        self.largura_original, self.altura_original = self.largura, self.altura
        self.indice, self.proporcao = 0, None
        self.descricao = f"vídeo {caminho.name}"
        self._inicio = time.perf_counter()

    def _numero_agora(self) -> int:
        return int((time.perf_counter() - self._inicio) * QPS_CAMERA)

    def ler(self):
        # Espera o próximo quadro, como a webcam real (a leitura bloqueia).
        proximo = self._numero_agora() + 1
        espera = self._inicio + proximo / QPS_CAMERA - time.perf_counter()
        if espera > 0:
            time.sleep(espera)
        return self._quadros[proximo % len(self._quadros)].copy()

    def esta_aberta(self) -> bool:
        return True

    def liberar(self) -> None:
        pass

    @property
    def recortada(self) -> bool:
        return False


class CameraEmThread:
    """Lê a câmera numa thread; ``ler()`` devolve o quadro mais novo ainda não usado."""

    def __init__(self, camera) -> None:
        self._camera = camera
        self._quadro = None
        self._novo = threading.Event()
        self._parar = False
        for nome in ("altura", "largura", "largura_original", "altura_original", "indice", "proporcao", "descricao"):
            setattr(self, nome, getattr(camera, nome))
        threading.Thread(target=self._laco, daemon=True).start()

    def _laco(self) -> None:
        while not self._parar:
            quadro = self._camera.ler()
            self._quadro = quadro
            self._novo.set()

    def ler(self):
        self._novo.wait(1.0)
        self._novo.clear()
        return self._quadro

    def esta_aberta(self) -> bool:
        return True

    def liberar(self) -> None:
        self._parar = True

    @property
    def recortada(self) -> bool:
        return False


def medir(versao: str) -> dict:
    """Abre a janela, deixa rodar ``SEGUNDOS`` e devolve os tempos do laço."""
    import tkinter as tk

    from kara_sygna.aplicacao.controlador import ControladorSinais
    from kara_sygna.interface.janela_principal import JanelaPrincipal
    from kara_sygna.visao.detector_corpo import criar_detector_corpo
    from kara_sygna.visao.detector_maos import DetectorMaos
    from kara_sygna.visao.detector_rosto import DetectorRosto

    config.INTERVALO_QUADRO_MS = 15 if versao == "A" else 1
    config.REVEZAMENTO_DETECTORES = {"D": 2, "E": 3}.get(versao, 1)
    camera = CameraDeVideo(VIDEO)
    if versao == "C":
        camera = CameraEmThread(camera)

    raiz = tk.Tk()
    janela = JanelaPrincipal(
        raiz, ControladorSinais(), camera, DetectorMaos(), DetectorRosto(), criar_detector_corpo()
    )
    inicios, leituras, telas = [], [], []
    original = janela._depois_da_deteccao
    ler_original, mostrar_original = camera.ler, janela._mostrar

    def ler_medindo():
        a = time.perf_counter()
        quadro = ler_original()
        leituras.append(time.perf_counter() - a)
        return quadro

    def mostrar_medindo(quadro):
        a = time.perf_counter()
        mostrar_original(quadro)
        telas.append(time.perf_counter() - a)

    def terminar_medindo(*args):
        inicios.append(time.perf_counter())
        original(*args)

    camera.ler = ler_medindo
    janela._mostrar = mostrar_medindo
    janela._depois_da_deteccao = terminar_medindo
    raiz.after(int(SEGUNDOS * 1000), raiz.destroy)
    janela.iniciar()
    camera.liberar()

    voltas = [b - a for a, b in itertools.pairwise(inicios[30:])]  # 30 primeiras = aquecimento
    return {
        "qps": len(voltas) / sum(voltas) if voltas else 0.0,
        "volta_ms": 1000 * statistics.median(voltas),
        "espera_camera_ms": 1000 * statistics.median(leituras[30:]),
        "tela_ms": 1000 * statistics.median(telas[30:]),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    versoes = sys.argv[1:] or ["A", "D"]
    nomes = {
        "A": "como está (pausa 15 ms)",
        "B": "pausa 1 ms",
        "C": "pausa 1 ms + câmera em thread",
        "D": "revezamento (2 detectores de mãos)",
        "E": "revezamento (3 detectores de mãos)",
    }
    linhas = ["| Versão | qps | intervalo entre quadros (ms) | espera da câmera (ms) | desenhar a tela (ms) |", "|---|---|---|---|---|"]
    for v in versoes:
        r = medir(v)
        linhas.append(
            f"| {v}: {nomes[v]} | {r['qps']:.1f} | {r['volta_ms']:.1f} | {r['espera_camera_ms']:.1f} | {r['tela_ms']:.1f} |"
        )
        print(linhas[-1], flush=True)
    texto = "\n".join(["# Quadros por segundo do app (câmera falsa de 30 qps)", "", *linhas])
    arquivo = Path(__file__).resolve().parent / "resultados" / f"qps_app_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print(f"\n{texto}\n\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
