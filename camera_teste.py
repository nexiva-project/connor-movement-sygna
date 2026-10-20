"""Teste de câmera ao vivo: o que ajuda o MediaPipe a não perder a mão rápida?

Abre a webcam com o mesmo detector de mãos do kara-sygna e mostra, na tela,
quantos quadros ficam sem mão. Você muda a exposição, o ganho, a confiança de
rastreio e o realce de pele pelo teclado, faz o sinal (ex.: "carro") e
compara os números de cada configuração.

Uso (com o Python da .venv do kara-sygna-py)::

    python camera_teste.py                    # DirectShow (aceita exposição manual)
    python camera_teste.py --backend msmf     # MediaFoundation (o que o app usa, 30 qps)
    python camera_teste.py --verificar        # só confere se abre (sem janela)

Clique na imagem (ou Enter no terminal) para medir 10 s. As teclas abaixo
valem na janela selecionada ou digitadas no terminal (letra + Enter):

    ESPAÇO   mede 10 s (contagem de 3 s antes): faça o sinal durante a medida
    e / E    exposição mais curta / mais longa (manual)
    a        volta a exposição automática
    g / G    ganho menor / maior
    t        troca a confiança de rastreio (0,3 → 0,5 → 0,6 → 0,7)
    r        liga / desliga o realce de pele
    s        abre a janela de ajustes do driver da câmera (só DirectShow)
    q / ESC  sai e mostra a tabela com todas as medidas

Cada medida também vai para ``resultados/camera_<data>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna import config  # noqa: E402
from kara_sygna.visao.caracteristicas import indices_dos_slots  # noqa: E402
from kara_sygna.visao.detector_corpo import criar_detector_corpo  # noqa: E402
from kara_sygna.visao.detector_maos import DetectorMaos  # noqa: E402
from kara_sygna.visao.detector_rosto import DetectorRosto  # noqa: E402
from kara_sygna.visao.maos_pelo_corpo import conferir_maos  # noqa: E402
from kara_sygna.visao.realce_pele import RealcePele  # noqa: E402

PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
BACKENDS = {"dshow": ("DirectShow", cv2.CAP_DSHOW), "msmf": ("MediaFoundation", cv2.CAP_MSMF)}
CONFIANCAS_RASTREIO = (0.3, 0.5, 0.6, 0.7)
SEGUNDOS_MEDIDA = 10.0
SEGUNDOS_CONTAGEM = 3.0
# No Windows (DirectShow) a exposição é log2 do tempo: -6 = 1/64 s ≈ 15,6 ms.
EXPOSICAO_MIN, EXPOSICAO_MAX = -13, -1


def ms_da_exposicao(valor: float) -> float:
    """Tempo de exposição (ms) de um valor do DirectShow (``2 ** valor`` s)."""
    return 1000.0 * 2.0**valor


@dataclass
class Medida:
    """Resultado de uma medida de 10 s."""

    hora: str
    backend: str
    exposicao: str
    ganho: float
    rastreio: float
    realce: bool
    qps: float
    brilho: float
    sem_mao_pct: float
    sem_esq_pct: float
    sem_dir_pct: float
    duas_maos_pct: float
    maior_buraco_s: float
    buracos: int


def buracos(presente: list[bool]) -> list[int]:
    """Tamanho (em quadros) de cada trecho seguido em que ``presente`` é falso."""
    tamanhos, atual = [], 0
    for p in presente:
        if p:
            if atual:
                tamanhos.append(atual)
            atual = 0
        else:
            atual += 1
    if atual:
        tamanhos.append(atual)
    return tamanhos


def resumir(lados: list[tuple[bool, bool]], tempos: list[float], brilhos: list[float], ajustes: dict) -> Medida:
    """Transforma os quadros de uma medida em uma :class:`Medida`.

    Args:
        lados: Para cada quadro, ``(tem_mao_esquerda, tem_mao_direita)``.
    """
    n = len(lados)
    maos_por_quadro = [esq + dir_ for esq, dir_ in lados]
    duracao = tempos[-1] - tempos[0] if n > 1 else 1.0
    qps = (n - 1) / duracao if n > 1 else 0.0
    # Buraco = quadros sem as duas mãos (o "carro" usa as duas).
    tamanhos = buracos([m >= 2 for m in maos_por_quadro])
    return Medida(
        hora=time.strftime("%H:%M:%S"),
        qps=round(qps, 1),
        brilho=round(float(np.mean(brilhos)), 1),
        sem_mao_pct=round(100 * sum(m == 0 for m in maos_por_quadro) / n, 1),
        sem_esq_pct=round(100 * sum(not esq for esq, _ in lados) / n, 1),
        sem_dir_pct=round(100 * sum(not dir_ for _, dir_ in lados) / n, 1),
        duas_maos_pct=round(100 * sum(m >= 2 for m in maos_por_quadro) / n, 1),
        maior_buraco_s=round(max(tamanhos, default=0) / qps, 2) if qps else 0.0,
        buracos=len(tamanhos),
        **ajustes,
    )


class TesteCamera:
    """Câmera, detectores e ajustes atuais."""

    def __init__(self, backend: str, indice: int) -> None:
        nome, codigo = BACKENDS[backend]
        self.nome_backend = nome
        self.captura = cv2.VideoCapture(indice, codigo)
        if not self.captura.isOpened():
            raise SystemExit(f"Não abriu a câmera {indice} pelo {nome}.")
        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, config.LARGURA_CAPTURA)
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTURA_CAPTURA)
        self.captura.set(cv2.CAP_PROP_FPS, 30)
        self.exposicao: float | None = None  # None = automática
        self._mexeu_exposicao = False
        self.rastreio = config.CONFIANCA_RASTREIO
        self.realce = RealcePele(ativo=config.REALCE_PELE)
        self.rosto = DetectorRosto()
        # Pose: dá a cada mão o lado do braço dela e tira mãos fantasmas, como no app.
        self.corpo = criar_detector_corpo()
        self._paralelo = ThreadPoolExecutor(max_workers=1)
        self.maos = self._novo_detector()
        self.aviso = ""

    def _novo_detector(self) -> DetectorMaos:
        return DetectorMaos(confianca_deteccao=config.CONFIANCA_DETECCAO, confianca_rastreio=self.rastreio)

    # -- ajustes ----------------------------------------------------------

    def mudar_exposicao(self, passo: int) -> None:
        """Exposição manual um passo mais curta (-1) ou mais longa (+1)."""
        if self.exposicao is None:
            atual = self.captura.get(cv2.CAP_PROP_EXPOSURE)
            self.exposicao = float(round(atual)) if EXPOSICAO_MIN <= atual <= EXPOSICAO_MAX else -6.0
        else:
            self.exposicao = float(np.clip(self.exposicao + passo, EXPOSICAO_MIN, EXPOSICAO_MAX))
        self._mexeu_exposicao = True
        self.captura.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0,25 = manual no DirectShow
        self.captura.set(cv2.CAP_PROP_EXPOSURE, self.exposicao)
        # O MediaFoundation devolve True e ignora o valor: vale o que a câmera lê de volta.
        lido = self.captura.get(cv2.CAP_PROP_EXPOSURE)
        if abs(lido - self.exposicao) < 0.5:
            self.aviso = f"exposicao {self.exposicao:+.0f} = {ms_da_exposicao(self.exposicao):.1f} ms"
        else:
            self.aviso = f"a camera nao aceitou {self.exposicao:+.0f} (continua {lido:+.0f}); limite dela ou use --backend dshow"
            self.exposicao = None

    def exposicao_automatica(self) -> None:
        self.captura.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0,75 = automática
        self.exposicao = None
        self.aviso = "exposicao automatica"

    def mudar_ganho(self, passo: float) -> None:
        antes = self.captura.get(cv2.CAP_PROP_GAIN)
        if antes < 0:  # -1 = a câmera não informa o ganho por aqui
            self.aviso = "ganho nao disponivel; tente 's' (ajustes do driver)"
            return
        self.captura.set(cv2.CAP_PROP_GAIN, antes + passo)
        self.aviso = f"ganho {self.captura.get(cv2.CAP_PROP_GAIN):.0f}"

    def trocar_rastreio(self) -> None:
        i = CONFIANCAS_RASTREIO.index(self.rastreio) if self.rastreio in CONFIANCAS_RASTREIO else -1
        self.rastreio = CONFIANCAS_RASTREIO[(i + 1) % len(CONFIANCAS_RASTREIO)]
        self.maos = self._novo_detector()
        self.aviso = f"confianca de rastreio {self.rastreio}"

    def ajustes(self) -> dict:
        """Ajustes atuais, como entram na :class:`Medida`."""
        # No automático, mostra o valor que a câmera escolheu (com mais luz, ele diminui).
        lido = self.captura.get(cv2.CAP_PROP_EXPOSURE)
        valor = lido if self.exposicao is None else self.exposicao
        modo = "auto " if self.exposicao is None else ""
        exposicao = (
            f"{modo}{valor:+.0f} ({ms_da_exposicao(valor):.1f} ms)" if EXPOSICAO_MIN <= valor <= EXPOSICAO_MAX else f"{modo}?"
        )
        return {
            "backend": self.nome_backend,
            "exposicao": exposicao,
            "ganho": round(self.captura.get(cv2.CAP_PROP_GAIN), 1),
            "rastreio": self.rastreio,
            "realce": self.realce.ativo,
        }

    # -- um quadro ----------------------------------------------------------

    def processar(self) -> tuple[np.ndarray, tuple[bool, bool], float] | None:
        """Lê, detecta e desenha um quadro. Devolve (imagem, (esq, dir), brilho)."""
        ok, quadro = self.captura.read()
        if not ok or quadro is None:
            return None
        quadro = cv2.flip(quadro, 1)  # espelhado, como no app
        corpo = self._paralelo.submit(self.corpo.detectar, quadro)  # em paralelo, como no app
        if self.realce.ativo:
            self.realce.calibrar(quadro, self.rosto.detectar(quadro))
        resultado = self.maos.detectar(self.realce.aplicar(quadro))
        altura, largura = quadro.shape[:2]
        resultado = conferir_maos(resultado, corpo.result(), largura, altura)
        slots = indices_dos_slots(resultado)
        lados = (slots["esq"] is not None, slots["dir"] is not None)
        brilho = float(cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY).mean())
        self.maos.desenhar(quadro, resultado)
        return quadro, lados, brilho

    def fechar(self) -> None:
        # A webcam guarda a exposição manual: volta ao automático para o app não abrir escuro.
        if self._mexeu_exposicao:
            self.exposicao_automatica()
        self.captura.release()
        self._paralelo.shutdown()
        self.rosto.fechar()
        self.corpo.fechar()


def _texto(imagem: np.ndarray, linhas: list[str], cor=(255, 255, 255)) -> None:
    y = 22
    for linha in linhas:
        cv2.putText(imagem, linha, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(imagem, linha, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1, cv2.LINE_AA)
        y += 22


def tabela(medidas: list[Medida]) -> str:
    """Tabela em texto com todas as medidas da sessão."""
    titulo = f"{'hora':8} {'backend':15} {'exposicao':18} {'ganho':>5} {'rastr':>5} {'realce':>6} {'qps':>5} {'brilho':>6} {'sem mao':>8} {'sem esq':>8} {'sem dir':>8} {'2 maos':>7} {'buraco max':>10}"
    linhas = [titulo, "-" * len(titulo)]
    for m in medidas:
        linhas.append(
            f"{m.hora:8} {m.backend:15} {m.exposicao:18} {m.ganho:5.0f} {m.rastreio:5.1f} {'sim' if m.realce else 'nao':>6} "
            f"{m.qps:5.1f} {m.brilho:6.0f} {m.sem_mao_pct:7.1f}% {m.sem_esq_pct:7.1f}% {m.sem_dir_pct:7.1f}% {m.duas_maos_pct:6.1f}% {m.maior_buraco_s:9.2f}s"
        )
    return "\n".join(linhas)


def salvar(medida: Medida, arquivo: Path) -> None:
    novo = not arquivo.exists()
    arquivo.parent.mkdir(exist_ok=True)
    with open(arquivo, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(asdict(medida)))
        if novo:
            escritor.writeheader()
        escritor.writerow(asdict(medida))


def verificar(teste: TesteCamera) -> None:
    """Lê ~60 quadros sem janela e mostra qps, brilho e se os ajustes funcionam."""
    tempos, brilhos, lados = [], [], []
    for _ in range(60):
        saida = teste.processar()
        if saida is None:
            raise SystemExit("A câmera abriu, mas não entregou imagem.")
        tempos.append(time.perf_counter())
        lados.append(saida[1])
        brilhos.append(saida[2])
    print(tabela([resumir(lados, tempos, brilhos, teste.ajustes())]))
    teste.mudar_exposicao(-1)  # 1º toque: fixa a exposição atual
    print("Exposição atual (automática):", teste.aviso)
    teste.mudar_exposicao(-2)  # 2º toque: tenta mais curta
    print("Teste de exposição manual:", teste.aviso)
    teste.exposicao_automatica()


def _ler_terminal(comandos: queue.Queue) -> None:
    """Cada linha digitada no terminal vira um comando (a 1ª letra; Enter vazio = medir)."""
    for linha in sys.stdin:
        texto = linha.strip()
        comandos.put(texto[0] if texto else " ")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=list(BACKENDS), default="dshow")
    parser.add_argument("--camera", type=int, default=0, help="índice da câmera (padrão 0)")
    parser.add_argument("--verificar", action="store_true", help="só confere se a câmera abre, sem janela")
    parser.add_argument(
        "--rastreio", type=float, default=None, help="confiança de rastreio das mãos (o app usa 0,3)"
    )
    parser.add_argument(
        "--exposicao", type=int, default=None, help="já abre com exposição manual (ex.: -6 = 15,6 ms; só DirectShow)"
    )
    args = parser.parse_args()

    teste = TesteCamera(args.backend, args.camera)
    if args.rastreio is not None:
        teste.rastreio = args.rastreio
        teste.maos = teste._novo_detector()
        print(f"confianca de rastreio {teste.rastreio}")
    if args.exposicao is not None:
        teste.exposicao = float(args.exposicao)
        teste.mudar_exposicao(0)
        print(teste.aviso)
    if args.verificar:
        try:
            verificar(teste)
        finally:
            teste.fechar()
        return

    arquivo = PASTA_RESULTADOS / f"camera_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    medidas: list[Medida] = []
    janela = "Teste de camera (clique = medir)"
    ultimos = deque(maxlen=90)  # ~3 s de quadros, para o painel ao vivo
    tempos_painel = deque(maxlen=30)
    estado, inicio_estado = "livre", 0.0
    dados: dict[str, list] = {"maos": [], "tempos": [], "brilhos": []}
    na_frente = False
    # As teclas só chegam se a janela estiver selecionada; por isso também dá para
    # clicar na imagem (começa uma medida) ou digitar no terminal.
    comandos: queue.Queue = queue.Queue()
    threading.Thread(target=_ler_terminal, args=(comandos,), daemon=True).start()
    print(
        "\nA janela da camera abriu no canto de cima da tela.\n"
        "  CLIQUE na imagem (ou aperte Enter aqui) = medir 10 s\n"
        "  Aqui no terminal, letra + Enter: e/E exposicao, a automatica, t rastreio,\n"
        "  r realce, s ajustes do driver, q sair (fechar a janela tambem sai)\n"
    )

    try:
        while True:
            saida = teste.processar()
            if saida is None:
                print("A câmera parou de entregar imagem.")
                break
            imagem, lados, brilho = saida
            agora = time.perf_counter()
            ultimos.append(lados)
            tempos_painel.append(agora)
            qps = (len(tempos_painel) - 1) / (tempos_painel[-1] - tempos_painel[0]) if len(tempos_painel) > 1 else 0.0

            if estado == "contagem" and agora - inicio_estado >= SEGUNDOS_CONTAGEM:
                estado, inicio_estado = "medindo", agora
                dados = {"maos": [], "tempos": [], "brilhos": []}
            if estado == "medindo":
                dados["maos"].append(lados)
                dados["tempos"].append(agora)
                dados["brilhos"].append(brilho)
                if agora - inicio_estado >= SEGUNDOS_MEDIDA:
                    medida = resumir(dados["maos"], dados["tempos"], dados["brilhos"], teste.ajustes())
                    medidas.append(medida)
                    salvar(medida, arquivo)
                    teste.aviso = f"medida: sem esq {medida.sem_esq_pct}%, sem dir {medida.sem_dir_pct}%, 2 maos {medida.duas_maos_pct}%"
                    print(tabela(medidas[-1:]).splitlines()[-1])
                    estado = "livre"

            a = teste.ajustes()
            sem_esq = 100 * sum(not e for e, _ in ultimos) / len(ultimos)
            sem_dir = 100 * sum(not d for _, d in ultimos) / len(ultimos)
            duas = 100 * sum(e and d for e, d in ultimos) / len(ultimos)
            painel = [
                f"{a['backend']} | {qps:4.1f} qps | brilho {brilho:3.0f}",
                f"exposicao {a['exposicao']} | ganho {a['ganho']:.0f}",
                f"rastreio {a['rastreio']} | realce {'sim' if a['realce'] else 'nao'}",
                f"agora: esq {'SIM' if lados[0] else 'nao'} | dir {'SIM' if lados[1] else 'nao'}",
                f"ultimos 3 s: sem esq {sem_esq:4.1f}% | sem dir {sem_dir:4.1f}% | 2 maos {duas:4.1f}%",
            ]
            if teste.aviso:
                painel.append(teste.aviso)
            _texto(imagem, painel)
            if estado == "contagem":
                falta = SEGUNDOS_CONTAGEM - (agora - inicio_estado)
                _texto(imagem, ["", "", "", "", "", "", "", "", f"PREPARE O SINAL... {falta:.0f}"], (0, 255, 255))
            elif estado == "medindo":
                falta = SEGUNDOS_MEDIDA - (agora - inicio_estado)
                _texto(imagem, ["", "", "", "", "", "", "", "", f"MEDINDO: faca o sinal ({falta:.0f} s)"], (0, 0, 255))

            cv2.imshow(janela, imagem)
            if not na_frente:
                # Sem isto a janela pode abrir atrás dos outros programas.
                cv2.moveWindow(janela, 60, 60)
                cv2.setWindowProperty(janela, cv2.WND_PROP_TOPMOST, 1)
                cv2.setMouseCallback(
                    janela, lambda evento, *_: evento == cv2.EVENT_LBUTTONDOWN and comandos.put(" ")
                )
                na_frente = True
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 255 and not comandos.empty():
                tecla = ord(comandos.get())
            if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                break  # janela fechada no X
            if tecla in (ord("q"), 27):
                break
            if tecla == ord(" ") and estado == "livre":
                estado, inicio_estado = "contagem", agora
            elif tecla == ord("e"):
                teste.mudar_exposicao(-1)
            elif tecla == ord("E"):
                teste.mudar_exposicao(+1)
            elif tecla == ord("a"):
                teste.exposicao_automatica()
            elif tecla == ord("g"):
                teste.mudar_ganho(-8)
            elif tecla == ord("G"):
                teste.mudar_ganho(+8)
            elif tecla == ord("t"):
                teste.trocar_rastreio()
            elif tecla == ord("r"):
                teste.realce.ativo = not teste.realce.ativo
                teste.aviso = f"realce {'ligado' if teste.realce.ativo else 'desligado'}"
            elif tecla == ord("s"):
                teste.captura.set(cv2.CAP_PROP_SETTINGS, 1)
    finally:
        teste.fechar()
        cv2.destroyAllWindows()

    if medidas:
        print()
        print(tabela(medidas))
        print(f"\nMedidas salvas em {arquivo}")


if __name__ == "__main__":
    main()
