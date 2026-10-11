"""Janela que mostra o sinal gravado como duas mãos 3D girando."""

from __future__ import annotations

import tkinter as tk

from kara_sygna.aplicacao.previsualizacao import Reproducao
from kara_sygna.interface.imagem_tk import quadro_para_foto
from kara_sygna.visao.render_3d import desenhar_quadro_sinal

TAMANHO = (600, 500)
INTERVALO_MS = 33


class JanelaPrevisualizacao(tk.Toplevel):
    """Anima os exemplos gravados de um sinal (Esc ou fechar a janela para sair).

    O desenho é feito pelo OpenCV, mas exibido em um ``Label`` do Tk: abrir
    uma segunda janela do OpenCV junto com o Tk causa problemas no Linux.

    Args:
        mestre: Janela principal.
        reproducao: Estado da animação (:class:`Reproducao`).
    """

    def __init__(self, mestre: tk.Misc, reproducao: Reproducao) -> None:
        """Cria a janela e começa a animação."""
        super().__init__(mestre)
        self._reproducao = reproducao
        self._foto = None
        self._agendado = None

        self.title(f"Kara Sygna - Pré-visualização: {reproducao.nome}")
        self.resizable(False, False)
        self._imagem = tk.Label(self, bg="#12141c")
        self._imagem.pack()
        self.bind("<Escape>", lambda _e: self.destroy())
        self._passo()

    def _passo(self) -> None:
        """Desenha o quadro atual e agenda o próximo."""
        r = self._reproducao
        quadro = desenhar_quadro_sinal(r.vetor_atual, TAMANHO, r.angulo, r.legenda)
        self._foto = quadro_para_foto(quadro, TAMANHO[0] + 10, TAMANHO[1] + 10)
        self._imagem.configure(image=self._foto)
        r.avancar()
        self._agendado = self.after(INTERVALO_MS, self._passo)

    def destroy(self) -> None:
        """Cancela a animação e fecha a janela."""
        if self._agendado is not None:
            try:
                self.after_cancel(self._agendado)
            except tk.TclError:
                pass
            self._agendado = None
        super().destroy()
