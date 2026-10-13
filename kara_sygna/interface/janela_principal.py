"""Janela principal: câmera, painel de controle e configurações visuais."""

from __future__ import annotations

import copy
import json
import logging
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, messagebox, scrolledtext

import cv2
import numpy as np

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais, Modo
from kara_sygna.aplicacao.erros import ErroAplicacao, SemDadosError
from kara_sygna.camera import Camera
from kara_sygna.dados.repositorio import DadosInvalidosError
from kara_sygna.ia.modelo_io import ModeloInvalidoError
from kara_sygna.ia.treinamento import ResultadoTreino, TreinoImpossivelError
from kara_sygna.interface import tema
from kara_sygna.interface.imagem_tk import TAMANHO_MINIMO_PX, quadro_para_foto
from kara_sygna.interface.janela_log import JanelaLog
from kara_sygna.interface.janela_previsualizacao import JanelaPrevisualizacao
from kara_sygna.interface.janela_sinais import JanelaSinais
from kara_sygna.interface.tarefas import ExecutorTk
from kara_sygna.visao.caracteristicas import montar_vetor_duas_maos
from kara_sygna.visao.detector_maos import DetectorMaos, tem_maos
from kara_sygna.visao.detector_rosto import DetectorRosto, ResultadoRosto
from kara_sygna.visao.suavizacao import SuavizadorPontos

logger = logging.getLogger(__name__)

ERROS_DE_USUARIO = (
    ErroAplicacao,
    ModeloInvalidoError,
    DadosInvalidosError,
    TreinoImpossivelError,
    SemDadosError,
)
PADRAO_VISUAL = {
    "maos_cor_linha_esquerda": "#00D7FF",
    "maos_cor_ponto_esquerda": "#00D7FF",
    "maos_cor_linha_direita": "#FF9F43",
    "maos_cor_ponto_direita": "#FF9F43",
    "maos_cor_texto": "#FFFFFF",
    "maos_espessura_linha": 2,
    "maos_tamanho_ponto": 3,
    "maos_espessura_texto": 2,
    "face_cor_malha": "#506450",
    "face_cor_contorno": "#FFDC00",
    "face_cor_ponto": "#FF0000",
    "face_cor_regiao_ponto": "#00FFFF",
    "face_cor_regiao_texto": "#00FFFF",
    "face_cor_status": "#00FF00",
    "face_espessura_linha": 1,
    "face_tamanho_ponto": 1,
    "face_tamanho_regiao": 4,
    "face_espessura_texto": 1,
    "video_cor_texto": "#FFFFFF",
    "texto_cor_sinais": "#FFFFFF",
}


class JanelaPrincipal:
    """Monta a interface e liga a câmera ao ControladorSinais."""

    def __init__(
        self,
        raiz: tk.Tk,
        controlador: ControladorSinais,
        camera: Camera,
        detector: DetectorMaos,
        detector_rosto: DetectorRosto,
    ) -> None:
        self._raiz = raiz
        self._controlador = controlador
        self._camera = camera
        self._detector = detector
        self._detector_rosto = detector_rosto

        self._arquivo_config_suavizacao = Path.home() / ".kara_sygna_suavizacao.json"

        self._config_visual = self._carregar_config_visual()
        self._percentual_maos, self._percentual_rosto = (
            self._carregar_config_suavizacao()
        )

        self._variavel_suavizacao_maos = tk.IntVar(
            master=raiz, value=self._percentual_maos
        )
        self._variavel_suavizacao_rosto = tk.IntVar(
            master=raiz, value=self._percentual_rosto
        )

        self._janela_configuracoes: tk.Toplevel | None = None
        self._variaveis_tamanho: dict[str, tk.IntVar] = {}

        self._suavizador_maos = SuavizadorPontos(
            intensidade=0.30,
            zona_morta=0.002,
            limiar_movimento=0.012,
            resposta_movimento=0.85,
        )
        self._suavizador_rosto = SuavizadorPontos(
            intensidade=0.30,
            zona_morta=0.002,
            limiar_movimento=0.012,
            resposta_movimento=0.85,
        )
        self._aplicar_percentuais_suavizacao()

        self._executor = ExecutorTk(raiz)
        self._log = JanelaLog(raiz)
        self._janela_sinais: JanelaSinais | None = None
        self._janela_previa: JanelaPrevisualizacao | None = None
        self._foto = None
        self._encerrado = False
        self._ultimo_erro_quadro = ""

        raiz.title("Kara Sygna")
        raiz.geometry("1250x850")
        raiz.minsize(1050, 720)
        raiz.protocol("WM_DELETE_WINDOW", self.fechar)

        self._palavra = tk.StringVar(master=raiz)
        self._status = tk.StringVar(
            master=raiz,
            value=f"Escolha um modo para começar. ({camera.descricao})",
        )

        self._construir()
        logger.info("Interface iniciada.")

    # ------------------------------------------------------- salvar configurações

    def _carregar_config_visual(self) -> dict:
        """Carrega as preferências visuais, usando padrões para valores inválidos."""
        dados = dict(PADRAO_VISUAL)

        try:
            with self._arquivo_config_suavizacao.open("r", encoding="utf-8") as arquivo:
                salvo = json.load(arquivo)

            visual = salvo.get("visual", {})
            if isinstance(visual, dict):
                for chave, padrao in PADRAO_VISUAL.items():
                    valor = visual.get(chave, padrao)

                    if (
                        chave.endswith("cor_texto")
                        or chave.startswith("face_cor")
                        or chave == "video_cor_texto"
                        or chave == "texto_cor_sinais"
                    ):
                        if (
                            isinstance(valor, str)
                            and len(valor) == 7
                            and valor.startswith("#")
                        ):
                            try:
                                int(valor[1:], 16)
                                dados[chave] = valor
                            except ValueError:
                                pass
                    else:
                        try:
                            dados[chave] = max(1, min(8, int(valor)))
                        except (TypeError, ValueError):
                            pass

        except (OSError, ValueError, TypeError, AttributeError):
            pass

        return dados

    def _carregar_config_suavizacao(self) -> tuple[int, int]:
        """Carrega os percentuais de suavização."""
        try:
            with self._arquivo_config_suavizacao.open("r", encoding="utf-8") as arquivo:
                dados = json.load(arquivo)

            maos = int(dados.get("maos", 90))
            rosto = int(dados.get("rosto", 90))
        except (OSError, ValueError, TypeError, AttributeError):
            maos = rosto = 90

        return max(0, min(100, maos)), max(0, min(100, rosto))

    def _salvar_config_suavizacao(self) -> None:
        """Salva suavização e aparência no mesmo arquivo, sem perder opções."""
        dados = {
            "maos": self._percentual_maos,
            "rosto": self._percentual_rosto,
            "visual": self._config_visual,
        }

        try:
            with self._arquivo_config_suavizacao.open("w", encoding="utf-8") as arquivo:
                json.dump(dados, arquivo, indent=4, ensure_ascii=False)
        except OSError:
            logger.exception("Não foi possível salvar as configurações.")

    def _aplicar_percentuais_suavizacao(self) -> None:
        """Aplica suavização somente às cópias usadas no desenho."""
        for suavizador, percentual in (
            (self._suavizador_maos, self._percentual_maos),
            (self._suavizador_rosto, self._percentual_rosto),
        ):
            intensidade = 1.0 - (percentual / 100.0) * 0.70
            zona_morta = (percentual / 100.0) * 0.002
            suavizador.intensidade = intensidade
            suavizador.zona_morta = zona_morta
            suavizador.limpar()

    def _alterar_suavizacao_maos(self, valor: str) -> None:
        self._percentual_maos = max(0, min(100, int(float(valor))))
        self._aplicar_percentuais_suavizacao()
        self._salvar_config_suavizacao()

    def _alterar_suavizacao_rosto(self, valor: str) -> None:
        self._percentual_rosto = max(0, min(100, int(float(valor))))
        self._aplicar_percentuais_suavizacao()
        self._salvar_config_suavizacao()

    def _restaurar_suavizacao_padrao(self) -> None:
        self._percentual_maos = 90
        self._percentual_rosto = 90
        self._variavel_suavizacao_maos.set(90)
        self._variavel_suavizacao_rosto.set(90)
        self._aplicar_percentuais_suavizacao()
        self._salvar_config_suavizacao()

    # ------------------------------------------------------------ janela de ajustes

    def _abrir_configuracoes(self) -> None:
        """Abre a janela independente, com rolagem para todos os controles."""
        if (
            self._janela_configuracoes is not None
            and self._janela_configuracoes.winfo_exists()
        ):
            self._janela_configuracoes.deiconify()
            self._janela_configuracoes.lift()
            self._janela_configuracoes.focus_force()
            return

        janela = tk.Toplevel(self._raiz)
        self._janela_configuracoes = janela
        janela.title("Configurações - Kara Sygna")
        janela.geometry("570x760")
        janela.minsize(480, 500)
        janela.configure(bg=tema.FUNDO)
        janela.protocol("WM_DELETE_WINDOW", self._fechar_configuracoes)

        recipiente = tk.Frame(janela, bg=tema.FUNDO)
        recipiente.pack(fill="both", expand=True)

        canvas = tk.Canvas(recipiente, bg=tema.FUNDO, highlightthickness=0)
        barra = tk.Scrollbar(recipiente, orient="vertical", command=canvas.yview)
        conteudo = tk.Frame(canvas, bg=tema.FUNDO)

        janela_interna = canvas.create_window((0, 0), window=conteudo, anchor="nw")
        canvas.configure(yscrollcommand=barra.set)

        conteudo.bind(
            "<Configure>",
            lambda _evento: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda evento: canvas.itemconfigure(janela_interna, width=evento.width),
        )

        canvas.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")

        tk.Label(
            conteudo,
            text="CONFIGURAÇÕES VISUAIS",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            font=tema.FONTE_TITULO,
        ).pack(pady=(15, 5))

        tk.Label(
            conteudo,
            text="As alterações são aplicadas e salvas automaticamente.",
            bg=tema.FUNDO,
            fg=tema.TEXTO_SUAVE,
            wraplength=500,
        ).pack(pady=(0, 12))

        self._titulo_config(conteudo, "MÃOS")
        self._controle_cor(
            conteudo, "Linhas da mão esquerda", "maos_cor_linha_esquerda"
        )
        self._controle_cor(
            conteudo, "Pontos da mão esquerda", "maos_cor_ponto_esquerda"
        )
        self._controle_cor(conteudo, "Linhas da mão direita", "maos_cor_linha_direita")
        self._controle_cor(conteudo, "Pontos da mão direita", "maos_cor_ponto_direita")
        self._controle_cor(conteudo, "Texto das mãos", "maos_cor_texto")
        self._controle_tamanho(
            conteudo, "Espessura das linhas das mãos", "maos_espessura_linha"
        )
        self._controle_tamanho(
            conteudo, "Tamanho dos pontos das mãos", "maos_tamanho_ponto"
        )
        self._controle_tamanho(
            conteudo, "Espessura do texto das mãos", "maos_espessura_texto"
        )

        self._titulo_config(conteudo, "ROSTO / FACEMESH")
        self._controle_cor(conteudo, "Malha facial", "face_cor_malha")
        self._controle_cor(conteudo, "Contorno facial", "face_cor_contorno")
        self._controle_cor(conteudo, "Pontos do rosto", "face_cor_ponto")
        self._controle_cor(
            conteudo,
            "Pontos de referência (nariz, testa etc.)",
            "face_cor_regiao_ponto",
        )
        self._controle_cor(
            conteudo, "Nomes das regiões faciais", "face_cor_regiao_texto"
        )
        self._controle_cor(conteudo, "Texto de estado do FaceMesh", "face_cor_status")
        self._controle_tamanho(
            conteudo, "Espessura das linhas do rosto", "face_espessura_linha"
        )
        self._controle_tamanho(
            conteudo, "Tamanho dos pontos do rosto", "face_tamanho_ponto"
        )
        self._controle_tamanho(
            conteudo, "Tamanho dos pontos de referência", "face_tamanho_regiao"
        )
        self._controle_tamanho(
            conteudo, "Espessura dos nomes faciais", "face_espessura_texto"
        )

        self._titulo_config(conteudo, "TEXTOS")
        self._controle_cor(
            conteudo, "Texto de reconhecimento sobre a câmera", "video_cor_texto"
        )
        self._controle_cor(
            conteudo, "Texto dos sinais reconhecidos", "texto_cor_sinais"
        )

        self._titulo_config(conteudo, "SUAVIZAÇÃO VISUAL")

        tk.Label(
            conteudo,
            text="Suavização do rosto (%)",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(fill="x", padx=20)

        tk.Scale(
            conteudo,
            from_=0,
            to=100,
            resolution=5,
            orient="horizontal",
            variable=self._variavel_suavizacao_rosto,
            command=self._alterar_suavizacao_rosto,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            troughcolor=tema.FUNDO_PAINEL,
            highlightthickness=0,
        ).pack(fill="x", padx=20)

        tk.Label(
            conteudo,
            text="Suavização das mãos (%)",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(fill="x", padx=20)

        tk.Scale(
            conteudo,
            from_=0,
            to=100,
            resolution=5,
            orient="horizontal",
            variable=self._variavel_suavizacao_maos,
            command=self._alterar_suavizacao_maos,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            troughcolor=tema.FUNDO_PAINEL,
            highlightthickness=0,
        ).pack(fill="x", padx=20)

        tk.Button(
            conteudo,
            text="RESTAURAR CORES E TAMANHOS",
            command=self._restaurar_visual,
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", padx=20, pady=(14, 5))

        tk.Button(
            conteudo,
            text="RESTAURAR SUAVIZAÇÃO (90%)",
            command=self._restaurar_suavizacao_padrao,
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", padx=20, pady=(0, 18))

    def _titulo_config(self, mestre: tk.Misc, titulo: str) -> None:
        tk.Label(
            mestre,
            text=titulo,
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO,
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 5))

    def _controle_cor(
        self,
        mestre: tk.Misc,
        rotulo: str,
        chave: str,
    ) -> None:
        linha = tk.Frame(mestre, bg=tema.FUNDO)
        linha.pack(fill="x", padx=20, pady=3)

        tk.Label(
            linha,
            text=rotulo,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        amostra = tk.Label(
            linha,
            text=self._config_visual[chave],
            bg=self._config_visual[chave],
            fg=self._cor_contraste(self._config_visual[chave]),
            width=10,
        )
        amostra.pack(side="right", padx=(5, 0))

        tk.Button(
            linha,
            text="Escolher cor",
            command=lambda: self._escolher_cor(chave, amostra),
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO,
            relief="flat",
            cursor="hand2",
        ).pack(side="right")

    @staticmethod
    def _cor_contraste(cor: str) -> str:
        try:
            r, g, b = (
                int(cor[1:3], 16),
                int(cor[3:5], 16),
                int(cor[5:7], 16),
            )
            return (
                "#000000" if (r * 299 + g * 587 + b * 114) / 1000 > 140 else "#FFFFFF"
            )
        except (ValueError, TypeError, IndexError):
            return "#FFFFFF"

    def _escolher_cor(self, chave: str, amostra: tk.Label) -> None:
        escolhido = colorchooser.askcolor(
            color=self._config_visual[chave],
            title="Escolher cor",
            parent=self._janela_configuracoes,
        )
        if not escolhido or not escolhido[1]:
            return

        self._config_visual[chave] = escolhido[1]
        amostra.configure(
            text=escolhido[1],
            bg=escolhido[1],
            fg=self._cor_contraste(escolhido[1]),
        )
        self._aplicar_cor_texto()
        self._salvar_config_suavizacao()

    def _controle_tamanho(
        self,
        mestre: tk.Misc,
        rotulo: str,
        chave: str,
    ) -> None:
        linha = tk.Frame(mestre, bg=tema.FUNDO)
        linha.pack(fill="x", padx=20, pady=3)

        tk.Label(
            linha,
            text=rotulo,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(fill="x")

        variavel = tk.IntVar(
            master=self._raiz,
            value=int(self._config_visual[chave]),
        )
        self._variaveis_tamanho[chave] = variavel

        tk.Scale(
            linha,
            from_=1,
            to=8,
            resolution=1,
            orient="horizontal",
            variable=variavel,
            command=lambda valor, k=chave: self._alterar_tamanho(k, valor),
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            troughcolor=tema.FUNDO_PAINEL,
            highlightthickness=0,
        ).pack(fill="x")

    def _alterar_tamanho(self, chave: str, valor: str) -> None:
        self._config_visual[chave] = max(1, min(8, int(float(valor))))
        self._salvar_config_suavizacao()

    def _restaurar_visual(self) -> None:
        self._config_visual = dict(PADRAO_VISUAL)

        self._salvar_config_suavizacao()

        if (
            self._janela_configuracoes is not None
            and self._janela_configuracoes.winfo_exists()
        ):
            self._fechar_configuracoes()
            self._abrir_configuracoes()

        self._aplicar_cor_texto()

    def _aplicar_cor_texto(self) -> None:
        """Aplica a cor escolhida à caixa de texto dos sinais."""
        texto = getattr(self, "_texto", None)
        if texto is not None:
            try:
                cor = self._config_visual["texto_cor_sinais"]
                texto.configure(fg=cor, insertbackground=cor)
            except tk.TclError:
                pass

    def _fechar_configuracoes(self) -> None:
        janela = self._janela_configuracoes
        self._janela_configuracoes = None
        if janela is not None:
            try:
                janela.destroy()
            except tk.TclError:
                pass

    # ------------------------------------------------------------------ layout

    def _caixa(
        self,
        mestre: tk.Misc,
        titulo: str,
        fundo: str,
        **opcoes,
    ) -> tk.LabelFrame:
        return tk.LabelFrame(
            mestre,
            text=titulo,
            bg=fundo,
            fg=tema.TEXTO,
            **opcoes,
        )

    def _construir(self) -> None:
        fundo = tk.Frame(self._raiz, bg=tema.FUNDO)
        fundo.pack(fill="both", expand=True)

        tk.Label(
            fundo,
            text="KARA SYGNA",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            font=tema.FONTE_TITULO,
        ).pack(anchor="w", padx=18, pady=(12, 5))

        superior = tk.Frame(fundo, bg=tema.FUNDO)
        superior.pack(fill="both", expand=True, padx=15, pady=(0, 8))

        self._construir_video(superior)
        self._construir_painel(superior)
        self._construir_texto(fundo)

    def _construir_video(self, mestre: tk.Misc) -> None:
        caixa = self._caixa(
            mestre,
            " CÂMERA ",
            tema.FUNDO,
            font=tema.FONTE_CAIXA,
            bd=1,
            relief="solid",
        )
        caixa.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self._area_video = tk.Frame(caixa, bg=tema.FUNDO_VIDEO)
        self._area_video.pack(fill="both", expand=True, padx=5, pady=5)

        self._video = tk.Label(
            self._area_video,
            bg=tema.FUNDO_VIDEO,
            fg="#aaaaaa",
            text="Iniciando câmera...",
            font=("Arial", 14),
        )
        self._video.place(relx=0.5, rely=0.5, anchor="center")

    def _construir_painel(self, mestre: tk.Misc) -> None:
        painel = tk.Frame(mestre, bg=tema.FUNDO_PAINEL, width=300)
        painel.pack(side="right", fill="y")
        painel.pack_propagate(False)

        ensinar = self._caixa(
            painel,
            " MODO ENSINAR ",
            tema.FUNDO_PAINEL,
            padx=10,
            pady=10,
        )
        ensinar.pack(fill="x", padx=10, pady=(10, 8))

        tk.Label(
            ensinar,
            text="Palavra do sinal:",
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO_SUAVE,
        ).pack(anchor="w")

        campo = tk.Entry(ensinar, textvariable=self._palavra, font=("Arial", 11))
        campo.pack(fill="x", pady=(5, 8))
        campo.bind("<Return>", lambda _e: self._iniciar_ensino())

        tk.Button(
            ensinar,
            text="COMEÇAR A ENSINAR",
            command=self._iniciar_ensino,
            height=2,
        ).pack(fill="x")

        tk.Button(
            ensinar,
            text="PARAR",
            command=self._parar,
        ).pack(fill="x", pady=(5, 0))

        self._botao_treinar = tk.Button(
            ensinar, text="TREINAR IA", command=self._treinar, height=2
        )
        self._botao_treinar.pack(fill="x", pady=(8, 0))

        sinal = self._caixa(
            painel,
            " MODO SINAL ",
            tema.FUNDO_PAINEL,
            padx=10,
            pady=10,
        )
        sinal.pack(fill="x", padx=10, pady=(0, 8))

        tk.Button(
            sinal,
            text="INICIAR RECONHECIMENTO",
            command=self._iniciar_reconhecimento,
            height=2,
        ).pack(fill="x")

        tk.Button(
            sinal,
            text="PARAR",
            command=self._parar,
        ).pack(fill="x", pady=(5, 0))

        tk.Button(
            painel,
            text="ABRIR SINAIS APRENDIDOS",
            command=self._abrir_sinais,
            height=2,
        ).pack(fill="x", padx=10, pady=(2, 4))

        tk.Button(
            painel,
            text="ABRIR LOG",
            command=self._log.mostrar,
        ).pack(fill="x", padx=10, pady=(0, 6))

        tk.Button(
            painel,
            text="CONFIGURAÇÕES",
            command=self._abrir_configuracoes,
            height=2,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            activebackground=tema.FUNDO_VIDEO,
            activeforeground=tema.TEXTO,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", padx=10, pady=(0, 6))

        status = self._caixa(
            painel,
            " STATUS ",
            tema.FUNDO_PAINEL,
            padx=8,
            pady=8,
        )
        status.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(
            status,
            textvariable=self._status,
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO_SUAVE,
            justify="left",
            anchor="w",
            wraplength=260,
        ).pack(fill="x")

    def _construir_texto(self, mestre: tk.Misc) -> None:
        caixa = self._caixa(
            mestre,
            " TEXTO DOS SINAIS ",
            tema.FUNDO,
            font=("Arial", 11, "bold"),
            padx=10,
            pady=10,
        )
        caixa.pack(fill="x", padx=15, pady=(0, 15))

        cor_texto = self._config_visual["texto_cor_sinais"]
        self._texto = scrolledtext.ScrolledText(
            caixa,
            height=4,
            font=tema.FONTE_TEXTO_SINAIS,
            bg=tema.FUNDO_VIDEO,
            fg=cor_texto,
            insertbackground=cor_texto,
            wrap="word",
            relief="flat",
        )
        self._texto.pack(fill="both", expand=True)

        botoes = tk.Frame(caixa, bg=tema.FUNDO)
        botoes.pack(fill="x", pady=(8, 0))

        tk.Button(
            botoes,
            text="LIMPAR TEXTO",
            command=self._limpar_texto,
        ).pack(side="right")

    # --------------------------------------------------------------- ações

    def _tratar_erro(self, erro: BaseException, titulo: str) -> None:
        if isinstance(erro, ERROS_DE_USUARIO):
            logger.warning("%s: %s", titulo, erro)
            messagebox.showwarning(titulo, str(erro))
        else:
            logger.error("%s", titulo, exc_info=erro)
            messagebox.showerror(titulo, f"{erro}\n\nVeja os detalhes em ABRIR LOG.")
        self._status.set(f"{titulo}: {erro}")

    def _bloqueado_por_treino(self) -> bool:
        if self._executor.ocupado:
            messagebox.showinfo("Treinando", "Aguarde o treinamento terminar.")
        return self._executor.ocupado

    def _iniciar_ensino(self) -> None:
        if self._bloqueado_por_treino():
            return
        try:
            self._status.set(self._controlador.iniciar_ensino(self._palavra.get()))
        except ErroAplicacao as erro:
            self._tratar_erro(erro, "Não foi possível ensinar")

    def _iniciar_reconhecimento(self) -> None:
        if self._bloqueado_por_treino():
            return
        try:
            self._status.set(self._controlador.iniciar_reconhecimento())
        except ErroAplicacao as erro:
            self._tratar_erro(erro, "Não foi possível reconhecer")

    def _parar(self) -> None:
        estava_ensinando = self._controlador.modo is Modo.ENSINANDO
        self._status.set(self._controlador.parar())
        if estava_ensinando:
            self._atualizar_sinais()

    def _treinar(self) -> None:
        if self._executor.ocupado:
            return

        self._parar()
        self._botao_treinar.config(state="disabled")
        self._status.set("Treinando a IA... pode levar cerca de um minuto.")

        self._executor.executar(
            self._controlador.treinar,
            ao_concluir=self._treino_concluido,
            ao_falhar=self._treino_falhou,
            ao_progresso=self._status.set,
        )

    def _treino_concluido(self, resultado: ResultadoTreino) -> None:
        self._botao_treinar.config(state="normal")
        self._atualizar_sinais()

        detalhe = (
            f"\n\nAcurácia na validação: {resultado.acuracia_validacao:.0%}"
            if resultado.acuracia_validacao is not None
            else f"\n\nAviso: {resultado.relatorio}"
        )

        self._status.set(
            f"IA treinada com {resultado.amostras} exemplos "
            f"e {len(resultado.classes)} sinais."
        )
        messagebox.showinfo(
            "IA treinada",
            "Pronto! Agora use INICIAR RECONHECIMENTO." + detalhe,
        )

    def _treino_falhou(self, erro: BaseException) -> None:
        self._botao_treinar.config(state="normal")
        self._tratar_erro(erro, "Não foi possível treinar")

    def _adicionar_texto(self, palavra: str) -> None:
        self._texto.insert(tk.END, palavra + " ")
        self._texto.see(tk.END)
        logger.info("Sinal adicionado ao texto: %s", palavra)

    def _limpar_texto(self) -> None:
        self._texto.delete("1.0", tk.END)
        self._controlador.limpar_reconhecimento()

    # ------------------------------------------------------------ janelas filhas

    def _atualizar_sinais(self) -> None:
        if self._janela_sinais is not None and self._janela_sinais.winfo_exists():
            self._janela_sinais.atualizar()

    def _abrir_sinais(self) -> None:
        if self._janela_sinais is not None and self._janela_sinais.winfo_exists():
            self._janela_sinais.deiconify()
            self._janela_sinais.lift()
            self._janela_sinais.atualizar()
            return

        self._janela_sinais = JanelaSinais(
            self._raiz,
            self._controlador.listar_sinais,
            self._ver_sinal,
            self._apagar_sinal,
        )

    def _ver_sinal(self, palavra: str) -> None:
        try:
            reproducao = self._controlador.criar_reproducao(palavra)
        except ErroAplicacao as erro:
            self._tratar_erro(erro, "Não foi possível mostrar o sinal")
            return

        if self._janela_previa is not None and self._janela_previa.winfo_exists():
            self._janela_previa.destroy()

        self._janela_previa = JanelaPrevisualizacao(self._raiz, reproducao)

    def _apagar_sinal(self, palavra: str) -> None:
        try:
            self._status.set(self._controlador.apagar_sinal(palavra))
        except ErroAplicacao as erro:
            self._tratar_erro(erro, "Não foi possível apagar")
        self._atualizar_sinais()

    # ------------------------------------------------------------ loop da câmera

    def iniciar(self) -> None:
        self._atualizar_quadro()
        self._raiz.mainloop()

    def _atualizar_quadro(self) -> None:
        if self._encerrado:
            return

        try:
            quadro = self._camera.ler()
            if quadro is None:
                self._status.set("Não foi possível ler a câmera.")
                self._agendar(100)
                return

            self._processar_quadro(cv2.flip(quadro, 1))

        except Exception as erro:
            mensagem = f"{type(erro).__name__}: {erro}"
            if mensagem != self._ultimo_erro_quadro:
                logger.error("Erro no laço da câmera", exc_info=erro)
                self._ultimo_erro_quadro = mensagem
            self._status.set(f"Erro na câmera: {erro}")

        self._agendar(config.INTERVALO_QUADRO_MS)

    def _agendar(self, ms: int) -> None:
        try:
            self._raiz.after(ms, self._atualizar_quadro)
        except tk.TclError:
            pass

    def _processar_quadro(self, quadro) -> None:
        """Suaviza cópias para desenhar; os dados originais vão para a IA."""
        quadro_original = quadro.copy()

        resultado_rosto = self._detector_rosto.detectar(quadro_original)
        resultado_maos = self._detector.detectar(quadro_original)

        if resultado_rosto.presente and resultado_rosto.landmarks is not None:
            rosto_visual = copy.deepcopy(resultado_rosto.landmarks)
            coordenadas = np.asarray(
                [(p.x, p.y, p.z) for p in rosto_visual.landmark],
                dtype=np.float32,
            )
            suavizadas = self._suavizador_rosto.suavizar("rosto", coordenadas)

            for ponto, coordenada in zip(rosto_visual.landmark, suavizadas):
                ponto.x, ponto.y, ponto.z = map(float, coordenada)

            resultado_rosto_visual = ResultadoRosto(
                landmarks=rosto_visual,
                vetor=resultado_rosto.vetor,
                presente=True,
            )
        else:
            self._suavizador_rosto.limpar("rosto")
            resultado_rosto_visual = resultado_rosto

        quadro = self._detector_rosto.desenhar(
            quadro,
            resultado_rosto_visual,
            estilo=self._config_visual,
        )

        if tem_maos(resultado_maos):
            resultado_maos_visual = copy.deepcopy(resultado_maos)

            for indice, mao in enumerate(resultado_maos_visual.multi_hand_landmarks):
                coordenadas = np.asarray(
                    [(p.x, p.y, p.z) for p in mao.landmark],
                    dtype=np.float32,
                )
                suavizadas = self._suavizador_maos.suavizar(
                    f"mao_{indice}", coordenadas
                )

                for ponto, coordenada in zip(mao.landmark, suavizadas):
                    ponto.x, ponto.y, ponto.z = map(float, coordenada)
        else:
            self._suavizador_maos.limpar()
            resultado_maos_visual = resultado_maos

        self._detector.desenhar(
            quadro,
            resultado_maos_visual,
            estilo=self._config_visual,
        )

        # A IA continua recebendo o resultado original, não o visual suavizado.
        vetor = (
            montar_vetor_duas_maos(resultado_maos) if tem_maos(resultado_maos) else None
        )

        try:
            resposta = self._controlador.processar_quadro(vetor)
        except ErroAplicacao as erro:
            self._controlador.parar()
            self._tratar_erro(erro, "Erro no reconhecimento")
            return

        if resposta is not None:
            if resposta.status:
                self._status.set(resposta.status)

            if resposta.sinal_confirmado:
                self._adicionar_texto(resposta.sinal_confirmado)

            if resposta.texto_video:
                self._escrever_no_video(
                    quadro,
                    resposta.texto_video,
                    tema.COR_VIDEO_BGR[resposta.nivel],
                    self._config_visual.get("video_cor_texto", "#FFFFFF"),
                )

        self._mostrar(quadro)

    @staticmethod
    def _escrever_no_video(
        quadro,
        texto: str,
        cor: tuple[int, int, int],
        cor_texto: str = "#FFFFFF",
    ) -> None:
        """Desenha o texto de reconhecimento sobre uma faixa escura."""
        cv2.rectangle(
            quadro,
            (10, 10),
            (min(quadro.shape[1] - 10, 700), 55),
            (15, 15, 18),
            -1,
        )

        try:
            cor_bgr = (
                int(cor_texto[5:7], 16),
                int(cor_texto[3:5], 16),
                int(cor_texto[1:3], 16),
            )
        except (ValueError, TypeError, IndexError):
            cor_bgr = cor

        cv2.putText(
            quadro,
            texto,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            cor_bgr,
            2,
            cv2.LINE_AA,
        )

    def _mostrar(self, quadro) -> None:
        largura = self._area_video.winfo_width()
        altura = self._area_video.winfo_height()

        if largura < TAMANHO_MINIMO_PX or altura < TAMANHO_MINIMO_PX:
            largura = config.LARGURA_AREA_CAMERA
            altura = config.ALTURA_AREA_CAMERA

        self._foto = quadro_para_foto(quadro, largura, altura)
        self._video.configure(image=self._foto, text="")

    # -------------------------------------------------------------------- fechar

    def fechar(self) -> None:
        if self._encerrado:
            return

        self._encerrado = True
        logger.info("Encerrando Kara Sygna.")

        self._fechar_configuracoes()

        try:
            self._detector.fechar()
        except Exception:
            logger.exception("Erro ao fechar o detector de mãos")

        try:
            self._detector_rosto.fechar()
        except Exception:
            logger.exception("Erro ao fechar o FaceMesh")

        try:
            self._camera.liberar()
        except Exception:
            logger.exception("Erro ao liberar a câmera")

        try:
            self._log.fechar()
        except Exception:
            logger.exception("Erro ao fechar a janela de log")

        try:
            self._raiz.destroy()
        except tk.TclError:
            pass
