"""Regras do programa, sem nenhuma interface gráfica."""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from kara_sygna import config
from kara_sygna.aplicacao.erros import (
    ModeloAusenteError,
    ModeloDesatualizadoError,
    PalavraInvalidaError,
    SemDadosError,
)
from kara_sygna.aplicacao.filtro_quadros import FiltroQuadros
from kara_sygna.aplicacao.previsualizacao import Reproducao, completar_com_espelho
from kara_sygna.dados.repositorio import (
    RepositorioSinais,
    novo_id_rodada,
    remover_arquivo_se_existir,
)
from kara_sygna.ia import treinamento
from kara_sygna.ia.modelo_io import (
    carregar_modelo,
    classes_do_modelo,
    modelo_esta_atualizado,
)
from kara_sygna.ia.reconhecimento import Estabilizador, prever
from kara_sygna.ia.sequencia import BufferQuadros
from kara_sygna.visao.quadro import tem_mao

logger = logging.getLogger(__name__)


class Modo(enum.Enum):
    """O que o programa está fazendo com cada quadro da câmera."""

    PARADO = "parado"
    ENSINANDO = "ensinando"
    RECONHECENDO = "reconhecendo"


class Nivel(enum.Enum):
    """Tom do texto mostrado sobre o vídeo."""

    NEUTRO = "neutro"
    ENSINO = "ensino"
    OK = "ok"
    INCERTO = "incerto"


@dataclass(frozen=True)
class ResultadoQuadro:
    """Resultado do processamento de um quadro."""

    texto_video: str = ""
    nivel: Nivel = Nivel.NEUTRO
    status: str | None = None
    sinal_confirmado: str | None = None


@dataclass(frozen=True)
class InfoSinal:
    """Uma linha da lista de sinais aprendidos."""

    palavra: str
    exemplos: int
    treinado: bool


class ControladorSinais:
    """Coordena ensino, treinamento e reconhecimento."""

    def __init__(
        self,
        repositorio: RepositorioSinais | None = None,
        caminho_modelo: Path = config.ARQUIVO_MODELO,
        carregador_modelo: Callable[[Path], Any] = carregar_modelo,
    ) -> None:
        self.repositorio = repositorio or RepositorioSinais()
        self.caminho_modelo = Path(caminho_modelo)
        self._carregador_modelo = carregador_modelo
        self.estabilizador = Estabilizador()
        self.buffer = BufferQuadros()
        self.modo = Modo.PARADO
        self.rotulo_atual = ""
        self.total_gravado = 0
        self._rodada = 0
        self._pendentes: list[tuple[int, np.ndarray]] = []
        # Quadros sem mão depois da última mão; só são gravados se a mão voltar.
        self._lacuna: list[np.ndarray] = []
        self.filtro_quadros = FiltroQuadros()
        self._ultima_resposta = ResultadoQuadro()
        self._ultimo_voto: str | None = None
        self._modelo: Any = None

    # -- ensino ----------------------------------------------------------------

    def iniciar_ensino(self, palavra: str) -> str:
        """Começa a gravar exemplos de um sinal."""
        palavra = palavra.strip().lower()

        if not palavra:
            raise PalavraInvalidaError("Escreva a palavra do sinal antes de começar.")

        self.repositorio.garantir_arquivo()
        self.rotulo_atual = palavra
        self.total_gravado = 0
        self._rodada = novo_id_rodada()
        self._pendentes = []
        self._lacuna = []
        self.filtro_quadros = FiltroQuadros()  # lê o modo atual do config
        self.modo = Modo.ENSINANDO

        logger.info("Ensino iniciado: %s", palavra)
        return f"Ensinando “{palavra}”. Faça o sinal diante da câmera."

    def parar(self) -> str:
        """Para o que estiver em andamento."""
        anterior, self.modo = self.modo, Modo.PARADO

        if anterior is Modo.ENSINANDO:
            self._lacuna = []  # o fim sem mão não é gravado
            self._gravar_pendentes()
            logger.info(
                "Ensino parado: %d linhas gravadas.",
                self.total_gravado,
            )
            return (
                f"Ensino de “{self.rotulo_atual}” parado: "
                f"{self.total_gravado} exemplos gravados."
            )

        if anterior is Modo.RECONHECENDO:
            logger.info("Reconhecimento parado.")
            return "Reconhecimento parado."

        return "Programa parado."

    # -- treinamento -----------------------------------------------------------

    def treinar(
        self,
        progresso: Callable[[str], None] = lambda _m: None,
    ) -> treinamento.ResultadoTreino:
        """Treina a IA com o CSV atual."""
        if not self.repositorio.existe():
            raise SemDadosError("Ensine pelo menos dois sinais antes de treinar a IA.")

        dados = self.repositorio.carregar()
        resultado = treinamento.treinar(
            dados,
            self.caminho_modelo,
            progresso,
        )
        self._modelo = None
        return resultado

    # -- reconhecimento --------------------------------------------------------

    def iniciar_reconhecimento(self) -> str:
        """Carrega o modelo e começa a reconhecer."""
        if not self.repositorio.existe():
            raise SemDadosError("Ensine pelo menos um sinal primeiro.")

        if not self.caminho_modelo.exists():
            raise ModeloAusenteError("Ensine sinais e clique em TREINAR IA primeiro.")

        if not modelo_esta_atualizado(
            self.caminho_modelo,
            self.repositorio.caminho,
        ):
            raise ModeloDesatualizadoError(
                "Os sinais mudaram.\n\nClique em TREINAR IA antes de reconhecer."
            )

        self._modelo = self._carregador_modelo(self.caminho_modelo)
        self.estabilizador.reiniciar()
        self.buffer.limpar()
        self.filtro_quadros = FiltroQuadros()  # lê o modo atual do config
        self._ultima_resposta = ResultadoQuadro("PREPARANDO...")
        self._ultimo_voto = None
        self.modo = Modo.RECONHECENDO

        logger.info(
            "Reconhecimento iniciado. Sinais do modelo: %s",
            list(self._modelo.classes_),
        )
        return "Reconhecendo sinais ao vivo..."

    def limpar_reconhecimento(self) -> None:
        """Esquece o histórico de votos."""
        self.estabilizador.reiniciar()

    # -- quadro a quadro -------------------------------------------------------

    def processar_quadro(
        self,
        vetor: np.ndarray | None,
    ) -> ResultadoQuadro:
        """Processa um quadro de acordo com o modo atual.

        Args:
            vetor: Vetor de 214 valores do quadro
                (:func:`kara_sygna.visao.quadro.montar_vetor_quadro`), mesmo
                sem mãos. ``None`` só se o quadro não pôde ser analisado.
        """
        if self.modo is Modo.ENSINANDO:
            return self._quadro_de_ensino(vetor)

        if self.modo is Modo.RECONHECENDO:
            return self._quadro_de_reconhecimento(vetor)

        return ResultadoQuadro()

    def _quadro_de_ensino(
        self,
        vetor: np.ndarray | None,
    ) -> ResultadoQuadro:
        """Grava os quadros em sequência: a IA aprende o movimento.

        Para o CSV não crescer à toa, só são gravados os quadros escolhidos
        pelo :class:`FiltroQuadros` (``config.MODO_QUADROS``), e os quadros sem
        mão do começo e do fim da gravação ficam de fora.
        """
        if vetor is None:
            return ResultadoQuadro()

        if self.filtro_quadros.aceitar(vetor):
            vetor = np.asarray(vetor, dtype=np.float64)
            if not tem_mao(vetor):
                if self.total_gravado:  # antes da primeira mão, nada é guardado
                    self._lacuna.append(vetor)
            else:
                for quadro in [*self._lacuna, vetor]:
                    self._pendentes.append((self.total_gravado, quadro))
                    self.total_gravado += 1
                self._lacuna = []
                if len(self._pendentes) >= config.QUADROS_POR_LOTE_GRAVACAO:
                    self._gravar_pendentes()

        dica = "" if tem_mao(vetor) else " Mostre as mãos para a câmera."
        return ResultadoQuadro(
            texto_video=f"ENSINANDO: {self.rotulo_atual} ({self.total_gravado})",
            nivel=Nivel.ENSINO,
            status=(
                f"Ensinando “{self.rotulo_atual}”: {self.total_gravado} quadros.{dica}"
            ),
        )

    def _gravar_pendentes(self) -> None:
        """Grava no CSV os quadros do ensino que ainda estão na memória."""
        pendentes, self._pendentes = self._pendentes, []
        self.repositorio.anexar_quadros(self.rotulo_atual, self._rodada, pendentes)

    def _quadro_de_reconhecimento(
        self,
        vetor: np.ndarray | None,
    ) -> ResultadoQuadro:
        """Junta o quadro aos anteriores, prevê o sinal e atualiza a votação.

        Só os quadros escolhidos pelo :class:`FiltroQuadros` (o mesmo modo do
        ensino) entram na janela e geram uma nova previsão. Nos outros, a
        última previsão continua valendo: cada quadro da câmera conta um voto,
        então um sinal parado é confirmado tão rápido quanto um em movimento.
        """
        if vetor is None:
            self.estabilizador.atualizar(None)
            return ResultadoQuadro("MOSTRE UM SINAL")

        if self.filtro_quadros.aceitar(vetor):
            self._ultima_resposta, self._ultimo_voto = self._analisar_janela(vetor)

        confirmado = self.estabilizador.atualizar(self._ultimo_voto)
        if confirmado in config.ROTULOS_IGNORADOS:
            confirmado = None
        return replace(self._ultima_resposta, sinal_confirmado=confirmado)

    def _analisar_janela(self, vetor: np.ndarray) -> tuple[ResultadoQuadro, str | None]:
        """Acrescenta o quadro à janela e prevê o sinal.

        Returns:
            O que mostrar na tela e o voto deste quadro (o sinal aceito, ou
            ``None`` se não há mão, a janela não encheu ou a IA ficou incerta).
        """
        self.buffer.adicionar(vetor)
        if not tem_mao(vetor):
            return ResultadoQuadro("MOSTRE UM SINAL"), None

        if not self.buffer.cheio:
            return ResultadoQuadro("PREPARANDO..."), None

        predicao = prever(self._modelo, self.buffer.features())

        if not predicao.aceito:
            return ResultadoQuadro(
                f"INCERTO ({predicao.confianca:.0%})", Nivel.INCERTO
            ), None

        return ResultadoQuadro(
            f"SINAL: {predicao.sinal} ({predicao.confianca:.0%})",
            Nivel.OK,
            f"Sinal reconhecido: {predicao.sinal} ({predicao.confianca:.0%})",
        ), predicao.sinal

    # -- lista de sinais -------------------------------------------------------

    def listar_sinais(self) -> list[InfoSinal]:
        """Lista sinais, quantidade de exemplos e estado do treinamento."""
        contagem = self.repositorio.contagem_por_sinal()

        if not contagem:
            return []

        atualizado = modelo_esta_atualizado(
            self.caminho_modelo,
            self.repositorio.caminho,
        )
        conhecidos = classes_do_modelo(self.caminho_modelo) if atualizado else set()

        return [
            InfoSinal(palavra, qtd, palavra in conhecidos)
            for palavra, qtd in contagem.items()
        ]

    def apagar_sinal(self, palavra: str) -> str:
        """Apaga um sinal e remove o modelo treinado."""
        if self.modo is Modo.ENSINANDO:
            self.parar()
        self.repositorio.apagar_sinal(palavra)
        remover_arquivo_se_existir(self.caminho_modelo)
        self._modelo = None
        self.modo = Modo.PARADO

        return f"Sinal “{palavra}” apagado. Treine a IA novamente."

    def criar_reproducao(self, palavra: str) -> Reproducao:
        """Prepara a pré-visualização de um sinal gravado."""
        exemplos = self.repositorio.exemplos_do_sinal(palavra)

        if len(exemplos) == 0:
            raise SemDadosError(f"Não há exemplos gravados para “{palavra}”.")

        return Reproducao(palavra, completar_com_espelho(exemplos))
