"""Grava vídeos de teste: você faz o sinal uma vez, e os testes rodam no vídeo.

Cada vídeo é uma "prova" fixa: todos os testes (modelos, máscara, ajustes do
MediaPipe) veem exatamente o mesmo movimento. Os vídeos são **só para teste**:
não entram no treino da IA.

Uso (com o Python da .venv do kara-sygna-py)::

    python gravar_video.py              # abre a janela com a lista
    python gravar_video.py --segundos 30

Tudo é pela janela, com o mouse: clique no vídeo da lista para gravar (conta
3 s e grava). Clicar durante a gravação cancela. Fechar a janela no X sai.

Para cada vídeo, o programa salva em ``videos/``:

* ``<nome>.avi``: os quadros como a câmera entregou (sem espelhar, sem
  realce), em MJPG de alta qualidade;
* ``<nome>.json``: o que foi pedido, quando, a câmera, o backend, a
  exposição e o tempo de cada quadro (a câmera nem sempre entrega 30 qps).

Tudo fica só neste computador.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PASTA_VIDEOS = Path(__file__).resolve().parent / "videos"
LARGURA, ALTURA = 640, 480  # o mesmo que o app pede (config.LARGURA_CAPTURA)
SEGUNDOS_CONTAGEM = 3
JANELA = "Gravador de videos de teste (clique no video)"
ALTURA_LINHA = 30
TOPO_LISTA = 70


@dataclass(frozen=True)
class Roteiro:
    """Um vídeo da lista: nome do arquivo e o que fazer."""

    nome: str
    sinal: str
    instrucao: str
    grupo: str
    #: Etapas guiadas na tela: (tipo, nome, segundos), tipo "pre" ou "sinal".
    #: Vazio = vídeo livre, com a duração de ``--segundos``.
    etapas: tuple[tuple[str, str, float], ...] = ()

    @property
    def segundos_etapas(self) -> float:
        return sum(s for _, _, s in self.etapas)


#: S (4 s) → carro (6 s), duas vezes: o carro é feito com as mãos em S, e o
#: app usa o S visto há pouco para aceitar o carro com menos certeza.
ETAPAS_S_CARRO = (("pre", "S", 4.0), ("sinal", "carro", 6.0)) * 2
ETAPAS_B_CASA = (("pre", "B", 4.0), ("sinal", "casa", 6.0)) * 2


ROTEIROS = [
    Roteiro("carro_luz_normal_calmo", "carro", "carro devagar, luz normal", "Essenciais"),
    Roteiro("carro_luz_normal_rapido", "carro", "carro rapido, luz normal", "Essenciais"),
    Roteiro("carro_luz_escura_rapido", "carro", "carro rapido, luz ESCURA (apague uma luz)", "Essenciais"),
    Roteiro("casa_luz_normal", "casa", "casa, luz normal", "Essenciais"),
    Roteiro("casa_luz_escura", "casa", "casa, luz ESCURA", "Essenciais"),
    Roteiro("neutro", "neutro", "mao relaxada, SEM fazer sinal", "Essenciais"),
    Roteiro("carro_longe", "carro", "carro LONGE da camera (mao pequena)", "Posicao e distancia"),
    Roteiro("carro_perto", "carro", "carro PERTO da camera", "Posicao e distancia"),
    Roteiro("carro_maos_embaixo", "carro", "carro com as maos EMBAIXO, perto da borda", "Posicao e distancia"),
    Roteiro("carro_contraluz", "carro", "carro com a luz ATRAS de voce", "Luz e fundo"),
    Roteiro("carro_roupa_clara", "carro", "carro com roupa CLARA ou bege", "Luz e fundo"),
    Roteiro("carro_muito_rapido", "carro", "carro MUITO rapido", "Movimento"),
    Roteiro("carro_varias_voltas", "carro", "carro girando VARIAS vezes seguidas", "Movimento"),
    Roteiro("carro_casa_carro", "carro,casa", "carro -> casa -> carro, sem parar", "Movimento"),
    Roteiro("carro_outra_mao", "carro", "carro com a OUTRA mao na frente", "Movimento"),
    Roteiro("outra_pessoa", "carro,casa", "OUTRA pessoa: carro e depois casa", "Outra pessoa"),
    Roteiro("presinal_s", "pre:S", "so o S, girando a mao para varios lados", "Pre-sinal", (("pre", "S", 10.0),)),
    Roteiro("presinal_b", "pre:B", "so o B, girando a mao para varios lados", "Pre-sinal", (("pre", "B", 10.0),)),
    Roteiro(
        "letras_mao_direita",
        "s,b,y",
        "letras S, B e Y com a mao DIREITA (a esquerda abaixada)",
        "Letras",
        (("sinal", "s", 5.0), ("sinal", "b", 5.0), ("sinal", "y", 5.0)),
    ),
    Roteiro(
        "defeito_malha",
        "",
        "mostre a mao com o dedo da malha mexendo sozinho (direita, depois esquerda)",
        "Defeitos",
    ),
]


def ja_gravado(roteiro: Roteiro) -> bool:
    """Diz se o vídeo deste roteiro já existe."""
    return (PASTA_VIDEOS / f"{roteiro.nome}.avi").exists()


def _texto(imagem, linhas: list[str], cor=(255, 255, 255), escala: float = 0.7) -> None:
    y = 32
    for linha in linhas:
        cv2.putText(imagem, linha, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(imagem, linha, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 2, cv2.LINE_AA)
        y += int(36 * escala / 0.7)


def _etapas_no_tempo(roteiro: Roteiro) -> list[dict]:
    """As etapas com o segundo em que começam e terminam no vídeo."""
    saida, inicio = [], 0.0
    for tipo, nome, segundos in roteiro.etapas:
        saida.append({"tipo": tipo, "nome": nome, "inicio_s": inicio, "fim_s": inicio + segundos})
        inicio += segundos
    return saida


def _etapa_na_tela(tela, roteiro: Roteiro, t: float) -> None:
    """Escreve grande a etapa de agora (amarelo = pré-sinal, verde = sinal)."""
    for etapa in _etapas_no_tempo(roteiro):
        if etapa["inicio_s"] <= t < etapa["fim_s"]:
            pre = etapa["tipo"] == "pre"
            texto = f"{'PRE-SINAL' if pre else 'SINAL'}: {etapa['nome']}  ({etapa['fim_s'] - t:.0f} s)"
            cor = (0, 230, 255) if pre else (80, 220, 80)
            altura = tela.shape[0]
            cv2.rectangle(tela, (0, altura - 90), (tela.shape[1], altura - 30), (20, 20, 20), -1)
            cv2.putText(tela, texto, (16, altura - 47), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cor, 3, cv2.LINE_AA)
            return


def abrir_camera(indice: int) -> cv2.VideoCapture:
    """Abre a câmera como o app (MediaFoundation, 640x480, 30 qps)."""
    captura = cv2.VideoCapture(indice, cv2.CAP_MSMF)
    if not captura.isOpened():
        raise SystemExit(f"Nao abriu a camera {indice}.")
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
    captura.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(10):  # os primeiros quadros podem vir escuros
        captura.read()
    return captura


def gravar(captura: cv2.VideoCapture, roteiro: Roteiro, segundos: float, mostrar: bool = True) -> Path:
    """Contagem, grava ``segundos`` e salva o vídeo e os dados dele.

    Returns:
        Caminho do ``.avi`` salvo.
    """
    PASTA_VIDEOS.mkdir(exist_ok=True)
    caminho = PASTA_VIDEOS / f"{roteiro.nome}.avi"
    if roteiro.etapas:
        segundos = roteiro.segundos_etapas
    escritor = None
    tempos: list[float] = []
    cancelado = False
    inicio = time.perf_counter()
    try:
        while True:
            ok, quadro = captura.read()
            agora = time.perf_counter() - inicio
            if not ok or quadro is None:
                raise SystemExit("A camera parou de entregar imagem.")
            gravando = agora >= SEGUNDOS_CONTAGEM
            if gravando:
                if escritor is None:
                    altura, largura = quadro.shape[:2]
                    escritor = cv2.VideoWriter(
                        str(caminho), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (largura, altura)
                    )
                    escritor.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
                    t0 = agora
                escritor.write(quadro)
                tempos.append(round(agora - t0, 4))
                if agora - t0 >= segundos:
                    break

            if mostrar:
                tela = cv2.flip(quadro, 1)  # espelhado só na tela, como no app
                if gravando:
                    qps = (len(tempos) - 1) / tempos[-1] if len(tempos) > 1 and tempos[-1] else 0.0
                    _texto(
                        tela,
                        [f"GRAVANDO {segundos - (agora - t0):4.1f} s | {qps:4.1f} qps", roteiro.instrucao],
                        (0, 0, 255),
                    )
                    if roteiro.etapas:
                        _etapa_na_tela(tela, roteiro, agora - t0)
                else:
                    _texto(tela, [f"PREPARE: {SEGUNDOS_CONTAGEM - agora:.0f}", roteiro.instrucao], (0, 255, 255))
                _texto(tela, ["", "", "", "", "", "", "", "", "", "", "", "", "clique para cancelar"], escala=0.55)
                cv2.imshow(JANELA, tela)
                cv2.waitKey(1)
                if cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1 or CLIQUES:
                    CLIQUES.clear()
                    cancelado = True
                    break
    finally:
        if escritor is not None:
            escritor.release()

    if cancelado:
        caminho.unlink(missing_ok=True)
        raise KeyboardInterrupt("gravacao cancelada")

    duracao = tempos[-1] if tempos else 0.0
    dados = {
        "nome": roteiro.nome,
        "sinal": roteiro.sinal,
        "instrucao": roteiro.instrucao,
        "gravado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "MediaFoundation",
        "largura": LARGURA,
        "altura": ALTURA,
        "exposicao_lida": captura.get(cv2.CAP_PROP_EXPOSURE),
        "quadros": len(tempos),
        "duracao_s": duracao,
        "qps_real": round((len(tempos) - 1) / duracao, 2) if duracao else 0.0,
        "tempos_s": tempos,
        "etapas": _etapas_no_tempo(roteiro),
        "uso": "so teste (nao entra no treino)",
    }
    caminho.with_suffix(".json").write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    return caminho


#: Cliques do mouse na janela (x, y), lidos pelo laço principal.
CLIQUES: list[tuple[int, int]] = []


def _ao_clicar(evento, x, y, *_):
    if evento == cv2.EVENT_LBUTTONDOWN:
        CLIQUES.append((x, y))


def desenhar_menu(aviso: str) -> np.ndarray:
    """Imagem da lista de vídeos; cada linha é um botão."""
    altura = TOPO_LISTA + ALTURA_LINHA * (len(ROTEIROS) + 2) + 40
    tela = np.full((altura, 700, 3), 30, dtype=np.uint8)
    _texto(tela, ["Clique no video para gravar (3 s + 20 s)"], (0, 255, 255), 0.65)
    if aviso:
        cv2.putText(tela, aviso, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
    for i, roteiro in enumerate(ROTEIROS):
        y = TOPO_LISTA + i * ALTURA_LINHA
        feito = ja_gravado(roteiro)
        cv2.rectangle(tela, (8, y + 2), (692, y + ALTURA_LINHA - 2), (60, 90, 60) if feito else (70, 70, 70), -1)
        texto = f"{i + 1:2}. {roteiro.instrucao}" + ("   [gravado]" if feito else "")
        cv2.putText(tela, texto, (16, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y = TOPO_LISTA + (len(ROTEIROS) + 1) * ALTURA_LINHA
    cv2.rectangle(tela, (8, y + 2), (200, y + ALTURA_LINHA - 2), (40, 40, 140), -1)
    cv2.putText(tela, "SAIR", (16, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return tela


def item_clicado(y: int) -> int | None:
    """Índice do roteiro clicado, ``-1`` para SAIR ou ``None``."""
    linha = (y - TOPO_LISTA) // ALTURA_LINHA
    if 0 <= linha < len(ROTEIROS):
        return int(linha)
    if linha == len(ROTEIROS) + 1:
        return -1
    return None


def main() -> None:
    global SEGUNDOS_CONTAGEM
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--segundos", type=float, default=20.0, help="duração de cada vídeo (padrão 20)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--verificar", action="store_true", help="grava 2 s sem janela e apaga (só confere)")
    args = parser.parse_args()

    captura = abrir_camera(args.camera)
    try:
        if args.verificar:
            teste = Roteiro("_verificacao", "", "verificacao", "")
            SEGUNDOS_CONTAGEM = 0
            caminho = gravar(captura, teste, 2.0, mostrar=False)
            dados = json.loads(caminho.with_suffix(".json").read_text(encoding="utf-8"))
            leitura = cv2.VideoCapture(str(caminho))
            lidos = int(leitura.get(cv2.CAP_PROP_FRAME_COUNT))
            leitura.release()
            print(f"gravou {dados['quadros']} quadros ({dados['qps_real']} qps), o arquivo tem {lidos} quadros")
            caminho.unlink()
            caminho.with_suffix(".json").unlink()
            return

        cv2.namedWindow(JANELA)
        cv2.setMouseCallback(JANELA, _ao_clicar)
        aviso = ""
        primeira = True
        while True:
            cv2.imshow(JANELA, desenhar_menu(aviso))
            if primeira:
                # Sem isto a janela pode abrir atrás dos outros programas.
                cv2.moveWindow(JANELA, 60, 40)
                cv2.setWindowProperty(JANELA, cv2.WND_PROP_TOPMOST, 1)
                primeira = False
            cv2.waitKey(30)
            if cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break
            if not CLIQUES:
                continue
            _, y = CLIQUES.pop()
            CLIQUES.clear()
            escolha = item_clicado(y)
            if escolha is None:
                continue
            if escolha == -1:
                break
            roteiro = ROTEIROS[escolha]
            try:
                caminho = gravar(captura, roteiro, args.segundos)
            except KeyboardInterrupt:
                aviso = "Cancelado. Nada foi salvo."
                continue
            dados = json.loads(caminho.with_suffix(".json").read_text(encoding="utf-8"))
            aviso = f"Salvo: {caminho.name} ({dados['quadros']} quadros, {dados['qps_real']} qps)"
            print(aviso, flush=True)
    finally:
        captura.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
