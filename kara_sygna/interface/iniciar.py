"""Ponto de entrada da interface gráfica."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

MENSAGEM_SEM_TK = (
    "O Tkinter não está instalado. No Ubuntu/Debian, instale com:\n"
    "    sudo apt install python3-tk\n"
    "(depois reinicie o programa; se usar uma venv, "
    "ela precisa ter sido criada com o Python do sistema)."
)


def executar() -> int:
    """Abre o Kara Sygna."""

    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        print(MENSAGEM_SEM_TK, file=sys.stderr)
        return 1

    from kara_sygna.aplicacao.controlador import ControladorSinais
    from kara_sygna.aplicacao.inicializacao import preparar_ambiente
    from kara_sygna.camera import CameraIndisponivelError, abrir_camera
    from kara_sygna.interface.janela_principal import JanelaPrincipal
    from kara_sygna.visao.detector_maos import DetectorMaos
    from kara_sygna.visao.detector_rosto import DetectorRosto

    preparar_ambiente()

    raiz = tk.Tk()
    raiz.withdraw()

    detector_maos = None
    detector_rosto = None
    camera = None

    try:
        detector_maos = DetectorMaos()
        detector_rosto = DetectorRosto()

        camera = abrir_camera()

        janela = JanelaPrincipal(
            raiz,
            ControladorSinais(),
            camera,
            detector_maos,
            detector_rosto,
        )

        raiz.deiconify()
        janela.iniciar()
        return 0

    except CameraIndisponivelError as erro:
        logger.error("%s", erro)
        messagebox.showerror("Câmera", str(erro))
        return 1

    except Exception as erro:
        logger.error("Falha ao iniciar o Kara Sygna", exc_info=erro)
        messagebox.showerror(
            "Inicialização",
            f"Não foi possível iniciar o Kara Sygna:\n\n{erro}",
        )
        return 1

    finally:
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

        try:
            raiz.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    sys.exit(executar())
