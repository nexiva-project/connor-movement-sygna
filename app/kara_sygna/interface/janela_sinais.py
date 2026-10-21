"""Janela com a lista dos sinais aprendidos."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox

from kara_sygna import config
from kara_sygna.aplicacao.controlador import PREFIXO_PRE, InfoSinal
from kara_sygna.ia.pre_sinais import caminho_modelo_pre
from kara_sygna.interface import tema


def tamanho_legivel(bytes_: int) -> str:
    """Bytes em B, KB, MB ou GB (1 KB = 1024 B), com vírgula: ``"2,3 MB"``."""
    valor = float(bytes_)
    for unidade in ("B", "KB", "MB"):
        if valor < 1024:
            break
        valor /= 1024
    else:
        unidade = "GB"
    if unidade == "B":
        return f"{int(valor)} B"
    return f"{valor:.1f} {unidade}".replace(".", ",")


def tamanho_do_arquivo(caminho: Path) -> int:
    """Bytes do arquivo (0 se ele não existir)."""
    try:
        return caminho.stat().st_size
    except OSError:
        return 0


def abrir_pasta(pasta: Path) -> None:
    """Abre a pasta no gerenciador de arquivos do sistema."""
    pasta.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(pasta)
    else:
        subprocess.Popen(
            ["open" if sys.platform == "darwin" else "xdg-open", str(pasta)]
        )


class JanelaSinais(tk.Toplevel):
    """Lista de sinais com botões para visualizar e apagar.

    Args:
        mestre: Janela principal.
        obter_sinais: Devolve a lista atual de sinais.
        ao_ver: Chamada com a palavra escolhida para visualizar.
        ao_apagar: Chamada com a palavra escolhida para apagar (após confirmação).
        ao_direcao_livre: Chamada com a palavra e o novo estado (``True`` =
            direção livre) para ligar ou desligar. ``None`` = sem o botão.
        pasta: Pasta onde ficam as gravações e a IA. Padrão:
            ``config.PASTA_ARMAZENAMENTO``.
    """

    def __init__(
        self,
        mestre: tk.Misc,
        obter_sinais: Callable[[], list[InfoSinal]],
        ao_ver: Callable[[str], None],
        ao_apagar: Callable[[str], None],
        ao_direcao_livre: Callable[[str, bool], None] | None = None,
        pasta: Path | None = None,
    ) -> None:
        """Cria a janela com a lista e os botões (veja a classe para os argumentos)."""
        super().__init__(mestre)
        self._obter_sinais = obter_sinais
        self._ao_ver = ao_ver
        self._ao_apagar = ao_apagar
        self._ao_direcao_livre = ao_direcao_livre
        self._palavras: list[str] = []
        self._livres: set[str] = set()
        self._pasta = Path(pasta) if pasta is not None else config.PASTA_ARMAZENAMENTO

        self.title("Sinais armazenados")
        self.geometry("600x560")

        conteudo = tk.Frame(self, padx=15, pady=15)
        conteudo.pack(fill="both", expand=True)
        tk.Label(
            conteudo, text="Sinais que a IA conhece", font=("Arial", 13, "bold")
        ).pack(anchor="w")
        tk.Label(
            conteudo,
            text="✓ treinado   |   • precisa treinar IA   |   ↻ direção livre (gira)",
            fg="#666666",
        ).pack(anchor="w", pady=(3, 10))

        area = tk.Frame(conteudo)
        area.pack(fill="both", expand=True)
        self._lista = tk.Listbox(
            area, height=15, font=("Arial", 11), exportselection=False
        )
        self._lista.pack(side="left", fill="both", expand=True)
        self._lista.bind("<Double-Button-1>", lambda _e: self._ver())
        barra = tk.Scrollbar(area, command=self._lista.yview)
        barra.pack(side="right", fill="y")
        self._lista.config(yscrollcommand=barra.set)

        self._total = tk.StringVar(master=self)
        tk.Label(
            conteudo, textvariable=self._total, fg="#444444", justify="left", anchor="w"
        ).pack(fill="x", pady=(8, 0))

        tk.Button(conteudo, text="ATUALIZAR LISTA", command=self.atualizar).pack(
            fill="x", pady=(10, 0)
        )
        tk.Button(
            conteudo, text="ABRIR PASTA DOS SINAIS", command=self._abrir_pasta
        ).pack(fill="x", pady=(7, 0))
        tk.Button(conteudo, text="VER SINAL (SÓ AS MÃOS)", command=self._ver).pack(
            fill="x", pady=(7, 0)
        )
        if ao_direcao_livre is not None:
            tk.Button(
                conteudo,
                text="DIREÇÃO LIVRE (GIRA): LIGAR / DESLIGAR",
                command=self._alternar_direcao_livre,
            ).pack(fill="x", pady=(7, 0))
        tk.Button(
            conteudo,
            text="APAGAR SINAL SELECIONADO",
            command=self._apagar,
            fg=tema.PERIGO,
        ).pack(fill="x", pady=(7, 0))
        self.atualizar()

    def atualizar(self) -> None:
        """Recarrega a lista a partir do CSV."""
        self._lista.delete(0, tk.END)
        self._palavras = []
        self._livres = set()
        self._total.set("")
        try:
            sinais = self._obter_sinais()
        except Exception as erro:  # noqa: BLE001 - mostra o problema em vez de travar a janela
            messagebox.showerror(
                "Sinais", f"Não foi possível ler os sinais:\n{erro}", parent=self
            )
            return
        if not sinais:
            self._lista.insert(tk.END, "Nenhum sinal ensinado ainda.")
            return
        for sinal in sinais:
            estado = "✓ treinado" if sinal.treinado else "• precisa treinar"
            if sinal.eh_pre_sinal:
                nome = f"[pré-sinal] {sinal.palavra}"
            elif sinal.pre_sinal:
                nome = f"{sinal.palavra} (pré-sinal {sinal.pre_sinal})"
            else:
                nome = sinal.palavra
            if sinal.direcao_livre:
                nome = f"↻ {nome}"
                self._livres.add(sinal.chave)
            self._lista.insert(
                tk.END,
                f"{nome} — {sinal.exemplos} exemplos — "
                f"{tamanho_legivel(sinal.tamanho)} — {estado}",
            )
            # Chave: o pré-sinal "S" vira "pre:S" (veja InfoSinal.chave).
            self._palavras.append(sinal.chave)
        self._total.set(self._texto_total(sinais))

    def _texto_total(self, sinais: list[InfoSinal]) -> str:
        """Resumo de quantos sinais há e quanto ocupam (gravações e IA)."""
        n_pre = sum(s.eh_pre_sinal for s in sinais)
        n_sinais = len(sinais) - n_pre
        gravacoes = sum(s.tamanho for s in sinais)
        modelo = self._pasta / config.ARQUIVO_MODELO.name
        ia = tamanho_do_arquivo(modelo) + tamanho_do_arquivo(caminho_modelo_pre(modelo))
        return (
            f"Total: {n_sinais} sinais e {n_pre} pré-sinais — "
            f"gravações {tamanho_legivel(gravacoes)} — IA {tamanho_legivel(ia)}\n"
            f"Pasta: {self._pasta}"
        )

    def _alternar_direcao_livre(self) -> None:
        """Liga ou desliga a direção livre do sinal selecionado."""
        palavra = self._selecionada("marcar como direção livre")
        if not palavra or self._ao_direcao_livre is None:
            return
        if palavra.startswith(PREFIXO_PRE):
            messagebox.showwarning(
                "Direção livre",
                "Direção livre é para sinais, não para pré-sinais.",
                parent=self,
            )
            return
        self._ao_direcao_livre(palavra, palavra not in self._livres)

    def _abrir_pasta(self) -> None:
        """Abre a pasta das gravações e da IA."""
        try:
            abrir_pasta(self._pasta)
        except OSError as erro:
            messagebox.showerror(
                "Sinais", f"Não foi possível abrir a pasta:\n{erro}", parent=self
            )

    def _selecionada(self, acao: str) -> str | None:
        """Palavra selecionada, ou ``None`` (avisando o usuário) se não houver."""
        selecao = self._lista.curselection()
        if not selecao or not self._palavras:
            messagebox.showwarning(
                "Selecione um sinal",
                f"Selecione na lista o sinal que deseja {acao}.",
                parent=self,
            )
            return None
        return self._palavras[selecao[0]]

    def _ver(self) -> None:
        """Pede a pré-visualização do sinal selecionado."""
        palavra = self._selecionada("ver")
        if palavra:
            self._ao_ver(palavra)

    def _apagar(self) -> None:
        """Apaga o sinal selecionado depois de confirmar."""
        palavra = self._selecionada("apagar")
        if not palavra:
            return
        if messagebox.askyesno(
            "Apagar sinal",
            f"Apagar todos os exemplos de “{palavra.removeprefix(PREFIXO_PRE)}”?"
            "\n\nEsta ação não pode ser desfeita.",
            parent=self,
        ):
            self._ao_apagar(palavra)
