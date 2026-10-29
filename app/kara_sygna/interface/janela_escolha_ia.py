"""Janela que abre antes da tela de carregamento: qual IA vai reconhecer os sinais.

Mostra o RandomForest e as redes neurais (GRU, LSTM, Conv1D), cada uma com o
resultado do último treino. A escolha fica guardada
(:func:`kara_sygna.ia.tipos_ia.escolher`) e vem marcada na próxima vez. Cada
IA tem o seu arquivo: trocar não apaga nenhuma.
"""

from __future__ import annotations

import time
import tkinter as tk

from kara_sygna.ia import tipos_ia

FUNDO = "#0B1020"
PAINEL = "#111A2E"
CIANO = "#00D7FF"
TEXTO = "#F0F6FC"
APAGADO = "#8A9BB5"


def situacao(tipo: str, torch_ok: bool) -> str:
    """Uma linha sobre a IA: treinada (com a validação), sem treino ou sem PyTorch."""
    if tipo != tipos_ia.RANDOMFOREST and not torch_ok:
        return "precisa do PyTorch (pip install torch)"
    resumo = tipos_ia.resumo(tipo)
    if resumo is None:
        return "ainda não treinada: escolha e clique em TREINAR IA no app"
    validacao = resumo.get("acuracia_validacao")
    texto = "treinada" if validacao is None else f"validação {100 * validacao:.1f}%"
    quando = time.strftime("%d/%m %H:%M", time.localtime(resumo["modificado_em"]))
    sinais = len(resumo.get("classes") or [])
    return f"{texto} · {sinais} sinais · treinada em {quando}"


def escolher_ia(raiz: tk.Tk) -> bool:
    """Mostra a janela e espera a escolha.

    Returns:
        ``True`` se o usuário escolheu uma IA (fica guardada); ``False`` se
        fechou a janela (o app não abre).
    """
    torch_ok = tipos_ia.torch_disponivel()
    janela = tk.Toplevel(raiz)
    janela.title("Kara Sygna: escolher a IA")
    janela.configure(bg=FUNDO)
    janela.resizable(False, False)
    janela.attributes("-topmost", True)

    escolhida = tk.StringVar(master=janela, value=tipos_ia.tipo_escolhido())
    resultado = {"ok": False}

    tk.Label(
        janela,
        text="Qual IA vai reconhecer os sinais?",
        bg=FUNDO,
        fg=TEXTO,
        font=("Arial", 14, "bold"),
    ).pack(anchor="w", padx=20, pady=(18, 4))
    tk.Label(
        janela,
        text="Cada IA tem o seu arquivo: trocar não apaga nenhuma.",
        bg=FUNDO,
        fg=APAGADO,
        font=("Arial", 9),
    ).pack(anchor="w", padx=20, pady=(0, 12))

    for tipo in tipos_ia.TIPOS:
        disponivel = tipo.chave == tipos_ia.RANDOMFOREST or torch_ok
        cartao = tk.Frame(
            janela, bg=PAINEL, highlightbackground="#243651", highlightthickness=1
        )
        cartao.pack(fill="x", padx=20, pady=4)
        tk.Radiobutton(
            cartao,
            text=tipo.nome,
            value=tipo.chave,
            variable=escolhida,
            state="normal" if disponivel else "disabled",
            bg=PAINEL,
            fg=TEXTO,
            selectcolor="#080D18",
            activebackground=PAINEL,
            activeforeground=CIANO,
            disabledforeground=APAGADO,
            font=("Arial", 11, "bold"),
            cursor="hand2",
        ).pack(anchor="w", padx=10, pady=(8, 0))
        for texto, cor in (
            (tipo.descricao, TEXTO),
            (situacao(tipo.chave, torch_ok), CIANO if disponivel else APAGADO),
        ):
            tk.Label(
                cartao, text=texto, bg=PAINEL, fg=cor, font=("Arial", 9), anchor="w"
            ).pack(fill="x", padx=36, pady=(0, 2))
        tk.Frame(cartao, bg=PAINEL, height=6).pack()

    # Detector de mãos (o Hand Landmarker está em teste).
    detector = tk.StringVar(master=janela, value=tipos_ia.detector_escolhido())
    tk.Label(
        janela,
        text="Detector de mãos",
        bg=FUNDO,
        fg=TEXTO,
        font=("Arial", 11, "bold"),
    ).pack(anchor="w", padx=20, pady=(14, 2))
    for chave, nome, descricao in tipos_ia.DETECTORES_MAOS:
        tk.Radiobutton(
            janela,
            text=f"{nome}  ({descricao})",
            value=chave,
            variable=detector,
            bg=FUNDO,
            fg=TEXTO,
            selectcolor="#080D18",
            activebackground=FUNDO,
            activeforeground=CIANO,
            font=("Arial", 10),
            cursor="hand2",
        ).pack(anchor="w", padx=28)

    def abrir(_evento=None) -> None:
        tipos_ia.escolher(escolhida.get(), detector_maos=detector.get())
        resultado["ok"] = True
        janela.destroy()

    tk.Button(
        janela,
        text="ABRIR COM ESTA IA (ENTER)",
        command=abrir,
        bg="#00BFD9",
        fg="#07111E",
        activebackground="#48E5F5",
        relief="flat",
        font=("Arial", 11, "bold"),
        cursor="hand2",
    ).pack(fill="x", padx=20, pady=(14, 18), ipady=6)

    janela.bind("<Return>", abrir)
    janela.bind("<Escape>", lambda _e: janela.destroy())
    janela.protocol("WM_DELETE_WINDOW", janela.destroy)

    janela.update_idletasks()
    x = (janela.winfo_screenwidth() - janela.winfo_reqwidth()) // 2
    y = (janela.winfo_screenheight() - janela.winfo_reqheight()) // 2
    janela.geometry(f"+{x}+{y}")
    janela.lift()
    janela.after(100, janela.focus_force)
    raiz.wait_window(janela)
    return resultado["ok"]
