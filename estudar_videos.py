"""Estuda os vídeos de teste: o mesmo movimento passa por vários modelos.

Para cada vídeo de ``videos/`` (gravados com ``gravar_video.py``):

1. **Visão** (uma vez por vídeo): cada quadro passa pelo mesmo caminho do app
   ao vivo (espelho, rosto, realce de pele, braços, mãos, memória da mão) e
   vira o vetor de 223 números. O resultado fica em ``videos/cache/``.
2. **Reconhecimento** (para cada modelo): os vetores entram num
   ``ControladorSinais`` novo, com o relógio andando no tempo do vídeo, como se
   fosse ao vivo. Conta quantos quadros votaram no sinal certo, no errado ou
   em nenhum, e quais sinais foram escritos.

Uso (com o Python da .venv do kara-sygna-py)::

    python estudar_videos.py                  # todos os vídeos, todos os modelos
    python estudar_videos.py --treinar-15     # antes, treina o modelo com máscara 15%
    python estudar_videos.py --refazer-visao  # ignora o cache da visão

Tudo usa a **cópia** do app em ``app/``. O relatório vai para ``resultados/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lab_ms.kara import preparar_import

PASTA_APP = preparar_import()

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais
from kara_sygna.aplicacao.filtro_quadros import FiltroQuadros
from kara_sygna.visao.quadro import maos_presentes

RAIZ = Path(__file__).resolve().parent
PASTA_VIDEOS = RAIZ / "videos"
PASTA_CACHE = PASTA_VIDEOS / "cache"
PASTA_RESULTADOS = RAIZ / "resultados"
ARMAZENAMENTO = PASTA_APP / "armazenamento"
#: Modelos comparados: nome -> arquivo em ``app/armazenamento``.
MODELOS = {
    "sem mascara": "modelo_sinais_sem_mascara.pkl",
    "mascara 15%": "modelo_sinais_mascara15.pkl",
    "mascara 15/30/50%": "modelo_sinais_mascara.pkl",
}
#: Todos os modelos usam a mesma IA de pré-sinais (a do app).
MODELO_PRE = ARMAZENAMENTO / "modelo_sinais_pre.pkl"


class Relogio:
    """Relógio que anda no tempo do vídeo (em vez do tempo real)."""

    def __init__(self) -> None:
        self.agora = 0.0

    def __call__(self) -> float:
        return self.agora


# -- 1. visão -------------------------------------------------------------------


@dataclass(frozen=True)
class OpcoesVisao:
    """Ajustes da visão para testar.

    Os padrões são os do app ORIGINAL (kara-sygna-py), escritos aqui e não
    lidos do config: o config da cópia muda com os testes, e a variante "base"
    precisa continuar significando o app de antes.
    """

    nome: str = "base"
    memoria_s: float = 0.3
    confianca_deteccao: float = 0.5
    realce: bool = True
    complexidade_pose: int = 0
    #: Dois detectores de mãos em revezamento (pares no A, ímpares no B); só as mãos.
    revezamento: bool = False
    #: Equipes completas em revezamento (rosto, braços, mãos e realce), como o
    #: app com ``REVEZAMENTO_DETECTORES``: o quadro k vai para a equipe k % N.
    equipes: int = 1
    #: Detectores de MÃOS em revezamento, com rosto e braços num só (como o
    #: app com ``REVEZAMENTO_DETECTORES``). Vale o maior entre este,
    #: ``equipes`` e 2 se ``revezamento``.
    detectores_maos: int = 1
    #: Trata 1 quadro a cada N (2 = o app ao vivo a ~15 qps com câmera de 30).
    a_cada: int = 1


ETAPAS = ("rosto", "realce", "corpo", "maos", "vetor")
VISAO_DO_APP = OpcoesVisao()


def extrair_quadros(
    video: Path, tempos: list[float], opcoes: OpcoesVisao = VISAO_DO_APP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Passa o vídeo pela visão do app.

    Returns:
        ``(vetores, ms, tempos)``: vetores (n, 223), o tempo de cada etapa
        (:data:`ETAPAS`) em milissegundos (n, 5) e o momento de cada quadro
        tratado (com ``a_cada > 1`` são menos quadros que o vídeo).
    """
    from kara_sygna.visao.detector_corpo import DetectorCorpo
    from kara_sygna.visao.detector_maos import DetectorMaos
    from kara_sygna.visao.detector_rosto import DetectorRosto
    from kara_sygna.visao.maos_pelo_corpo import conferir_maos
    from kara_sygna.visao.memoria_maos import MemoriaMaos
    from kara_sygna.visao.quadro import montar_vetor_quadro
    from kara_sygna.visao.realce_pele import RealcePele

    relogio = Relogio()
    vetores, medidas, usados = [], [], []
    leitura = cv2.VideoCapture(str(video))
    n_maos = max(opcoes.equipes, opcoes.detectores_maos, 2 if opcoes.revezamento else 1)
    todas_maos = [DetectorMaos(confianca_deteccao=opcoes.confianca_deteccao) for _ in range(n_maos)]
    rostos = [DetectorRosto() for _ in range(opcoes.equipes)]
    # O app usa a mesma CONFIANCA_DETECCAO para as mãos e para os braços (Pose).
    corpos = [
        DetectorCorpo(complexidade=opcoes.complexidade_pose, confianca_deteccao=opcoes.confianca_deteccao)
        for _ in range(opcoes.equipes)
    ]
    realces = [RealcePele(ativo=opcoes.realce) for _ in range(opcoes.equipes)]
    try:
        memoria = MemoriaMaos(memoria_s=opcoes.memoria_s, relogio=relogio)
        for numero, momento in enumerate(tempos):
            ok, quadro = leitura.read()
            if not ok:
                break
            if numero % opcoes.a_cada:
                continue  # o app ainda estava ocupado com o quadro anterior
            k = len(usados)
            maos = todas_maos[k % n_maos]
            rosto_det, corpo_det, realce = (lista[k % opcoes.equipes] for lista in (rostos, corpos, realces))
            usados.append(momento)
            relogio.agora = momento
            quadro = cv2.flip(quadro, 1)  # como no app ao vivo
            marcas = [time.perf_counter()]
            rosto = rosto_det.detectar(quadro)
            marcas.append(time.perf_counter())
            realce.calibrar(quadro, rosto)
            realcado = realce.aplicar(quadro)
            marcas.append(time.perf_counter())
            corpo = corpo_det.detectar(quadro)
            marcas.append(time.perf_counter())
            altura, largura = quadro.shape[:2]
            resultado = memoria.aplicar(conferir_maos(maos.detectar(realcado), corpo, largura, altura))
            marcas.append(time.perf_counter())
            vetores.append(montar_vetor_quadro(resultado, rosto, largura, altura, corpo))
            marcas.append(time.perf_counter())
            medidas.append(np.diff(marcas) * 1000)
    finally:
        for detector in (*todas_maos, *rostos, *corpos):
            detector.fechar()
    leitura.release()
    return (
        np.asarray(vetores, dtype=np.float64),
        np.asarray(medidas, dtype=np.float64),
        np.asarray(usados, dtype=np.float64),
    )


def caminho_cache(video: Path, opcoes: OpcoesVisao) -> Path:
    """Cache da visão; a variante "base" usa o nome antigo (só o vídeo)."""
    sufixo = "" if opcoes.nome == "base" else f"__{opcoes.nome}"
    return PASTA_CACHE / f"{video.stem}{sufixo}.npz"


def quadros_do_video(
    video: Path, refazer: bool, opcoes: OpcoesVisao = VISAO_DO_APP
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Vetores e tempos do vídeo, do cache ou extraídos agora."""
    dados = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    tempos = np.asarray(dados["tempos_s"], dtype=np.float64)
    cache = caminho_cache(video, opcoes)
    if cache.exists() and not refazer and cache.stat().st_mtime > video.stat().st_mtime:
        salvo = np.load(cache)
        return salvo["vetores"], salvo["tempos"], dados
    vetores, ms, tempos = extrair_quadros(video, tempos.tolist(), opcoes)
    PASTA_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, vetores=vetores, tempos=tempos, ms=ms)
    return vetores, tempos, dados


# -- 2. reconhecimento ----------------------------------------------------------


@dataclass
class Resultado:
    """O que um modelo fez num vídeo."""

    quadros: int = 0
    votos: Counter = field(default_factory=Counter)  # sinal votado (ou "") por quadro
    escritos: list[str] = field(default_factory=list)


def reconhecer(
    vetores: np.ndarray, tempos: np.ndarray, caminho_modelo: Path, ajustar=None
) -> Resultado:
    """Reconhece o vídeo com um modelo, como ao vivo, no tempo do vídeo.

    Args:
        ajustar: Função opcional que recebe o ``ControladorSinais`` já pronto,
            para testar uma regra (ex.: mudar o estabilizador).
    """
    relogio = Relogio()
    diagnostico = config.DIAGNOSTICO_ATIVO
    config.DIAGNOSTICO_ATIVO = False  # não espalhar CSVs no app
    try:
        controlador = ControladorSinais(caminho_modelo=caminho_modelo, relogio=relogio)
        controlador.caminho_modelo_pre = MODELO_PRE
        controlador.iniciar_reconhecimento()
    finally:
        config.DIAGNOSTICO_ATIVO = diagnostico
    controlador.filtro_quadros = FiltroQuadros(relogio=relogio)
    if ajustar is not None:
        ajustar(controlador)

    resultado = Resultado()
    for vetor, momento in zip(vetores, tempos, strict=True):
        relogio.agora = float(momento)
        resposta = controlador.processar_quadro(vetor)
        resultado.quadros += 1
        resultado.votos[controlador._ultimo_voto or ""] += 1
        if resposta.sinal_confirmado:
            resultado.escritos.append(resposta.sinal_confirmado)
    return resultado


# -- 3. relatório ---------------------------------------------------------------


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "-"


def linha_resultado(nome: str, r: Resultado, esperados: set[str]) -> str:
    certos = sum(n for s, n in r.votos.items() if s and s in esperados)
    errados = sum(n for s, n in r.votos.items() if s and s not in esperados)
    escritos = ", ".join(f"{s}×{n}" for s, n in Counter(r.escritos).items()) or "nada"
    return (
        f"| {nome} | {_pct(certos, r.quadros)} | {_pct(errados, r.quadros)} | "
        f"{_pct(r.votos[''], r.quadros)} | {escritos} |"
    )


def treinar_mascara_15() -> None:
    """Treina o modelo com máscara só de 15% (o app continua com o dele)."""
    from kara_sygna.dados.repositorio import RepositorioSinais
    from kara_sygna.ia import treinamento

    fracoes = config.MASCARA_FRACOES
    config.MASCARA_FRACOES = (0.15,)
    try:
        resultado = treinamento.treinar(RepositorioSinais().carregar(), ARMAZENAMENTO / MODELOS["mascara 15%"])
    finally:
        config.MASCARA_FRACOES = fracoes
    print(f"Modelo mascara 15% treinado: validacao {100 * (resultado.acuracia_validacao or 0):.1f}%")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--treinar-15", action="store_true", help="treina o modelo com máscara 15% antes")
    parser.add_argument("--refazer-visao", action="store_true", help="ignora o cache da visão")
    args = parser.parse_args()

    if args.treinar_15 or not (ARMAZENAMENTO / MODELOS["mascara 15%"]).exists():
        treinar_mascara_15()

    videos = sorted(v for v in PASTA_VIDEOS.glob("*.avi") if not v.stem.startswith("_"))
    if not videos:
        raise SystemExit("Nenhum vídeo em videos/. Grave com gravar_video.py.")

    inicio = time.perf_counter()
    linhas = ["# Estudo dos vídeos de teste", ""]
    linhas.append(
        "Votos: em quantos quadros a IA votou no sinal certo, num sinal errado ou em "
        "nenhum (sem mão, incerta ou preparando). Escrito: o que viraria texto no app."
    )
    linhas.append("")
    for video in videos:
        print(f"{video.stem}: visao...", flush=True)
        vetores, tempos, dados = quadros_do_video(video, args.refazer_visao)
        esperados = {s for s in dados.get("sinal", "").split(",") if s}
        presentes = maos_presentes(vetores)
        linhas += [
            f"## {video.stem}",
            "",
            (
                f"{dados.get('instrucao', '')} · {len(vetores)} quadros, {dados.get('qps_real', '?')} qps · "
                f"sem mão esq {_pct(int((~presentes[:, 0]).sum()), len(vetores))}, "
                f"sem mão dir {_pct(int((~presentes[:, 1]).sum()), len(vetores))}"
            ),
            "",
            "| Modelo | Voto certo | Voto errado | Sem voto | Escrito |",
            "|---|---|---|---|---|",
        ]
        for nome, arquivo in MODELOS.items():
            caminho = ARMAZENAMENTO / arquivo
            if not caminho.exists():
                linhas.append(f"| {nome} | (modelo não encontrado) | | | |")
                continue
            print(f"  {nome}...", flush=True)
            linhas.append(linha_resultado(nome, reconhecer(vetores, tempos, caminho), esperados))
        linhas.append("")
    linhas.append(f"Tempo total: {time.perf_counter() - inicio:.0f} s")

    texto = "\n".join(linhas)
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    arquivo = PASTA_RESULTADOS / f"estudo_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print()
    print(texto)
    print(f"\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
