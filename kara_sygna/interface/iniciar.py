"""Ponto de entrada da interface gráfica do Kara Sygna."""

from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # O Tkinter é um pacote separado em algumas distros.
    tk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MENSAGEM_SEM_TK = (
    "O Tkinter não está instalado. No Ubuntu/Debian, instale com:\n"
    "    sudo apt install python3-tk\n"
    "(depois reinicie o programa; se usar uma venv, "
    "ela precisa ter sido criada com o Python do sistema)."
)

FUNDO = "#0B1020"
PAINEL = "#111A2E"
CIANO = "#00D7FF"
TEXTO = "#F0F6FC"
TEXTO_SUAVE = "#93A4BC"
VERMELHO = "#FF7777"


def _liberar_recursos(
    camera: Any = None,
    detector_maos: Any = None,
    detector_rosto: Any = None,
) -> None:
    """Libera os recursos criados durante a inicialização."""
    if camera is not None:
        try:
            camera.liberar()
        except Exception:
            logger.exception("Erro ao liberar a câmera")

    if detector_rosto is not None:
        try:
            detector_rosto.fechar()
        except Exception:
            logger.exception("Erro ao fechar o FaceMesh")

    if detector_maos is not None:
        try:
            detector_maos.fechar()
        except Exception:
            logger.exception("Erro ao fechar o detector de mãos")


def _carregar_em_segundo_plano(
    eventos: queue.Queue[tuple[str, Any]],
    cancelamento: threading.Event,
) -> None:
    """Prepara o ambiente sem bloquear a interface gráfica."""
    camera = None
    detector_maos = None
    detector_rosto = None

    try:
        eventos.put(("etapa", "Preparando o ambiente..."))

        from kara_sygna.aplicacao.inicializacao import preparar_ambiente

        preparar_ambiente()

        if cancelamento.is_set():
            return

        eventos.put(("etapa", "Carregando os componentes de visão..."))

        from kara_sygna.camera import (
            CameraIndisponivelError,
            abrir_camera,
        )
        from kara_sygna.visao.detector_maos import DetectorMaos
        from kara_sygna.visao.detector_rosto import DetectorRosto

        detector_maos = DetectorMaos()

        if cancelamento.is_set():
            _liberar_recursos(
                detector_maos=detector_maos,
            )
            return

        detector_rosto = DetectorRosto()

        if cancelamento.is_set():
            _liberar_recursos(
                detector_maos=detector_maos,
                detector_rosto=detector_rosto,
            )
            return

        eventos.put(("etapa", "Inicializando a câmera..."))
        camera = abrir_camera()

        if cancelamento.is_set():
            _liberar_recursos(
                camera=camera,
                detector_maos=detector_maos,
                detector_rosto=detector_rosto,
            )
            return

        eventos.put(("etapa", "Preparando a interface..."))

        from kara_sygna.aplicacao.controlador import ControladorSinais
        from kara_sygna.interface.janela_principal import JanelaPrincipal

        controlador = ControladorSinais()

        if cancelamento.is_set():
            _liberar_recursos(
                camera=camera,
                detector_maos=detector_maos,
                detector_rosto=detector_rosto,
            )
            return

        # A partir daqui, a interface principal assume os recursos.
        eventos.put(
            (
                "pronto",
                (
                    controlador,
                    camera,
                    detector_maos,
                    detector_rosto,
                    JanelaPrincipal,
                ),
            )
        )

    except Exception as erro:
        logger.exception("Falha ao iniciar o Kara Sygna")

        _liberar_recursos(
            camera=camera,
            detector_maos=detector_maos,
            detector_rosto=detector_rosto,
        )

        if not cancelamento.is_set():
            try:
                from kara_sygna.camera import CameraIndisponivelError

                titulo = (
                    "Câmera indisponível"
                    if isinstance(erro, CameraIndisponivelError)
                    else "Falha na inicialização"
                )
            except ImportError:
                titulo = "Falha na inicialização"

            eventos.put(("erro", (titulo, str(erro))))


def _construir_carregamento(
    raiz: tk.Tk,
) -> tuple[tk.StringVar, tk.Label, ttk.Progressbar, tk.Button]:
    """Desenha a tela inicial de carregamento."""
    raiz.title("Kara Sygna")
    raiz.geometry("580x370")
    raiz.resizable(False, False)
    raiz.configure(bg=FUNDO)

    # Centraliza a janela na tela.
    raiz.update_idletasks()
    x = (raiz.winfo_screenwidth() - 580) // 2
    y = (raiz.winfo_screenheight() - 370) // 2
    raiz.geometry(f"580x370+{x}+{y}")

    estilo = ttk.Style(raiz)
    estilo.theme_use("clam")
    estilo.configure(
        "Kara.Horizontal.TProgressbar",
        troughcolor=PAINEL,
        background=CIANO,
        bordercolor=PAINEL,
        lightcolor=CIANO,
        darkcolor=CIANO,
        thickness=7,
    )

    conteudo = tk.Frame(raiz, bg=FUNDO)
    conteudo.pack(fill="both", expand=True, padx=34, pady=24)

    marca = tk.Frame(conteudo, bg=FUNDO)
    marca.pack(fill="x")

    simbolo = tk.Label(
        marca,
        text="KS",
        bg=CIANO,
        fg=FUNDO,
        font=("Arial", 18, "bold"),
        width=3,
        height=1,
        padx=5,
        pady=5,
    )
    simbolo.pack(side="left", padx=(0, 12))

    nome = tk.Label(
        marca,
        text="KARA SYGNA",
        bg=FUNDO,
        fg=TEXTO,
        font=("Arial", 20, "bold"),
    )
    nome.pack(side="left", anchor="center")

    versao = tk.Label(
        marca,
        text="VISÃO COMPUTACIONAL",
        bg=FUNDO,
        fg=CIANO,
        font=("Arial", 8, "bold"),
    )
    versao.pack(side="right", anchor="center")

    tk.Frame(conteudo, bg=CIANO, height=2).pack(
        fill="x",
        pady=(18, 25),
    )

    tk.Label(
        conteudo,
        text="Seu espaço de sinais está sendo preparado.",
        bg=FUNDO,
        fg=TEXTO,
        font=("Arial", 15, "bold"),
        wraplength=500,
    ).pack(anchor="w")

    tk.Label(
        conteudo,
        text=(
            "Estamos inicializando os componentes necessários "
            "para reconhecer mãos e rosto."
        ),
        bg=FUNDO,
        fg=TEXTO_SUAVE,
        font=("Arial", 10),
        wraplength=490,
        justify="left",
    ).pack(anchor="w", pady=(9, 22))

    status = tk.StringVar(
        master=raiz,
        value="Preparando o ambiente...",
    )

    tk.Label(
        conteudo,
        textvariable=status,
        bg=FUNDO,
        fg=CIANO,
        font=("Arial", 10, "bold"),
        anchor="w",
    ).pack(fill="x", pady=(0, 10))

    progresso = ttk.Progressbar(
        conteudo,
        mode="indeterminate",
        style="Kara.Horizontal.TProgressbar",
    )
    progresso.pack(fill="x")
    progresso.start(12)

    detalhe = tk.Label(
        conteudo,
        text="Aguarde enquanto o aplicativo é inicializado.",
        bg=FUNDO,
        fg=TEXTO_SUAVE,
        font=("Arial", 9),
        anchor="w",
        justify="left",
        wraplength=500,
    )
    detalhe.pack(fill="x", pady=(10, 0))

    tentar_novamente = tk.Button(
        conteudo,
        text="TENTAR NOVAMENTE",
        bg=CIANO,
        fg=FUNDO,
        activebackground="#65E8FF",
        activeforeground=FUNDO,
        relief="flat",
        cursor="hand2",
        font=("Arial", 10, "bold"),
        padx=12,
        pady=7,
    )

    return status, detalhe, progresso, tentar_novamente


def executar() -> int:
    """Abre a tela de carregamento e prepara o aplicativo."""
    if tk is None:
        print(MENSAGEM_SEM_TK, file=sys.stderr)
        return 1

    raiz = tk.Tk()
    eventos: queue.Queue[tuple[str, Any]] = queue.Queue()

    cancelamento: threading.Event | None = None
    carregador: threading.Thread | None = None
    janela_principal = None
    iniciado = False

    status, detalhe, progresso, botao_tentar = _construir_carregamento(raiz)

    def fechar() -> None:
        """Encerra a tela e sinaliza o cancelamento da inicialização."""
        if cancelamento is not None:
            cancelamento.set()

        # Se a interface ainda não assumiu os recursos, libera
        # qualquer conjunto completo que esteja aguardando na fila.
        while True:
            try:
                tipo, dados = eventos.get_nowait()
            except queue.Empty:
                break

            if tipo == "pronto":
                _, camera, detector_maos, detector_rosto, _ = dados
                _liberar_recursos(
                    camera=camera,
                    detector_maos=detector_maos,
                    detector_rosto=detector_rosto,
                )

        try:
            progresso.stop()
            raiz.destroy()
        except tk.TclError:
            pass

    raiz.protocol("WM_DELETE_WINDOW", fechar)

    def iniciar_carregamento() -> None:
        """Inicia uma tentativa de carregamento sem bloquear o Tkinter."""
        nonlocal cancelamento, carregador

        if cancelamento is not None:
            cancelamento.set()

        cancelamento = threading.Event()
        botao_tentar.pack_forget()
        detalhe.configure(
            text="Aguarde enquanto o aplicativo é inicializado.",
            fg=TEXTO_SUAVE,
        )
        status.set("Preparando o ambiente...")
        progresso.start(12)

        carregador = threading.Thread(
            target=_carregar_em_segundo_plano,
            args=(eventos, cancelamento),
            name="kara-sygna-inicializacao",
            daemon=True,
        )
        carregador.start()

    botao_tentar.configure(command=iniciar_carregamento)

    def verificar_eventos() -> None:
        """Recebe os resultados da thread e atualiza a interface."""
        nonlocal iniciado, janela_principal

        if iniciado:
            return

        try:
            while True:
                tipo, dados = eventos.get_nowait()

                if tipo == "etapa":
                    status.set(dados)

                elif tipo == "erro":
                    titulo, mensagem = dados
                    progresso.stop()
                    status.set(titulo)
                    detalhe.configure(
                        text=(f"{mensagem}\n\nConfira a câmera e tente novamente."),
                        fg=VERMELHO,
                    )
                    botao_tentar.pack(anchor="w", pady=(12, 0))

                elif tipo == "pronto":
                    (
                        controlador,
                        camera,
                        detector_maos,
                        detector_rosto,
                        classe_janela,
                    ) = dados

                    progresso.stop()
                    status.set("Tudo pronto! Abrindo o Kara Sygna...")
                    raiz.update_idletasks()

                    # A janela principal reutiliza a mesma raiz Tk.
                    # A thread de carregamento nunca manipula widgets.
                    for widget in raiz.winfo_children():
                        widget.destroy()

                    try:
                        janela_principal = classe_janela(
                            raiz,
                            controlador,
                            camera,
                            detector_maos,
                            detector_rosto,
                        )
                        iniciado = True

                        # O loop Tkinter já está ativo; a janela principal
                        # agenda os próximos quadros sozinha.
                        janela_principal.iniciar_captura()

                    except Exception as erro:
                        # A tela de carregamento já foi desmontada, então
                        # não há como oferecer "tentar novamente" aqui.
                        logger.exception("Falha ao construir a interface principal")
                        _liberar_recursos(
                            camera=camera,
                            detector_maos=detector_maos,
                            detector_rosto=detector_rosto,
                        )
                        iniciado = False
                        messagebox.showerror(
                            "Inicialização",
                            f"Não foi possível abrir o Kara Sygna:\n\n{erro}",
                        )
                        try:
                            raiz.destroy()
                        except tk.TclError:
                            pass

                    return

        except queue.Empty:
            pass

        except Exception as erro:
            logger.exception("Erro ao finalizar a inicialização")
            progresso.stop()
            status.set("Não foi possível abrir a interface.")
            detalhe.configure(
                text=str(erro),
                fg=VERMELHO,
            )
            botao_tentar.pack(anchor="w", pady=(12, 0))

        try:
            raiz.after(100, verificar_eventos)
        except tk.TclError:
            pass

    iniciar_carregamento()
    raiz.after(100, verificar_eventos)
    raiz.mainloop()

    # Se a janela foi fechada durante o carregamento, dá à thread a chance
    # de perceber o cancelamento e liberar a câmera antes de sair.
    if carregador is not None and carregador.is_alive():
        carregador.join(timeout=5)

    return 0 if iniciado else 1


if __name__ == "__main__":
    sys.exit(executar())
