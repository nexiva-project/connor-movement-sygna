"""Roda nos vídeos de teste os testes de visão, votação, janela, MS e tempo.

Testes (numeração da lista combinada na conversa):

1. Memória da mão: 0,3 s (app) × 0,5 s × 0,8 s.
3. Confiança de detecção das mãos: 0,5 (app) × 0,4 × 0,3.
4. Realce de pele: ligado (app) × desligado.
5. MediaPipe Pose: leve (app) × completo.
6. Janela de quadros: "só quando muda" (app) × fixo 1 a cada 1 × 1 a cada 2.
7. Votação: 6 votos e 70% (app) × mais votos × mais certeza.
8. MS (movimento de sinal) × RandomForest, janela a janela.
9. Tempo de cada detector por quadro.

O teste 2 (prever a mão com Kalman) não roda: com ``USAR_LUGAR = False`` a
IA não usa a posição da mão, e o movimento vem do pulso do braço (Pose);
mover a mão lembrada não muda nenhum número que a IA vê.

Uso::

    python estudar_tudo.py            # usa o cache da visão; o que faltar roda em paralelo
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lab_ms.kara import preparar_import

preparar_import()

from kara_sygna import config
from kara_sygna.aplicacao.filtro_quadros import FiltroQuadros
from kara_sygna.ia.modelo_io import carregar_modelo
from kara_sygna.ia.reconhecimento import prever
from kara_sygna.ia.sequencia import features_janela
from kara_sygna.visao.quadro import maos_presentes, tem_mao

from estudar_videos import (
    ARMAZENAMENTO,
    ETAPAS,
    MODELOS,
    PASTA_RESULTADOS,
    PASTA_VIDEOS,
    VISAO_DO_APP,
    OpcoesVisao,
    Relogio,
    caminho_cache,
    extrair_quadros,
    quadros_do_video,
    reconhecer,
)

MODELO = ARMAZENAMENTO / MODELOS["sem mascara"]

VARIANTES_VISAO = {
    1: ("Memória da mão", [
        ("0,3 s (app)", VISAO_DO_APP),
        ("0,5 s", OpcoesVisao("memoria_05", memoria_s=0.5)),
        ("0,8 s", OpcoesVisao("memoria_08", memoria_s=0.8)),
    ]),
    3: ("Confiança de detecção das mãos", [
        ("0,5 (app)", VISAO_DO_APP),
        ("0,4", OpcoesVisao("deteccao_04", confianca_deteccao=0.4)),
        ("0,3", OpcoesVisao("deteccao_03", confianca_deteccao=0.3)),
    ]),
    4: ("Realce de pele", [
        ("ligado (app)", VISAO_DO_APP),
        ("desligado", OpcoesVisao("sem_realce", realce=False)),
    ]),
    5: ("MediaPipe Pose", [
        ("leve (app)", VISAO_DO_APP),
        ("completo", OpcoesVisao("pose_completo", complexidade_pose=1)),
    ]),
}


def _ajuste_config(**valores):
    """Muda o config antes do controlador nascer e devolve a função para desfazer."""
    antigos = {k: getattr(config, k) for k in valores}
    for k, v in valores.items():
        setattr(config, k, v)
    return lambda: [setattr(config, k, v) for k, v in antigos.items()]


VARIANTES_RECONHECIMENTO = {
    6: ("Janela de quadros", [
        ("só quando muda (app)", {}, None),
        ("fixo, todo quadro", {"MODO_QUADROS": "fixo", "USAR_1_A_CADA_QUADROS": 1}, None),
        ("fixo, 1 a cada 2", {"MODO_QUADROS": "fixo", "USAR_1_A_CADA_QUADROS": 2}, None),
    ]),
    7: ("Votação e confiança mínima", [
        ("6 votos, 70% (app)", {}, None),
        ("8 votos, 70%", {}, 8),
        ("6 votos, 80%", {"CONFIANCA_MINIMA": 0.80}, None),
        ("6 votos, 90%", {"CONFIANCA_MINIMA": 0.90, "CONFIANCA_SEM_PRE_SINAL": 0.90}, None),
        ("8 votos, 80%", {"CONFIANCA_MINIMA": 0.80}, 8),
    ]),
}


# -- resumo de um conjunto de resultados ----------------------------------------


def videos_de_teste() -> list[Path]:
    return sorted(v for v in PASTA_VIDEOS.glob("*.avi") if not v.stem.startswith("_"))


def resumir(resultados: dict[str, tuple]) -> dict:
    """Números de um modo (variante) somando os vídeos.

    Args:
        resultados: ``{video: (Resultado, esperados, vetores)}``.
    """
    carro = [n for n in resultados if n.startswith("carro_") and n != "carro_casa_carro"]
    casa = [n for n in resultados if n.startswith("casa_")]

    def certos(nome):
        r, esperados, _ = resultados[nome]
        return sum(n for s, n in r.votos.items() if s and s in esperados) / max(1, r.quadros)

    def escritos(nome):
        return Counter(resultados[nome][0].escritos)

    sem_mao = np.mean([(~maos_presentes(v)).mean() for _, _, v in resultados.values()])
    errados_carro = sum(n for nome in carro for s, n in escritos(nome).items() if s != "carro")
    return {
        "sem mão": f"{100 * sem_mao:.0f}%",
        "carro: voto certo": f"{100 * np.mean([certos(n) for n in carro]):.0f}%",
        "carro: escritas (ideal 1/vídeo)": f"{sum(escritos(n)['carro'] for n in carro)}/{len(carro)}"
        + (f" (+{errados_carro} erradas)" if errados_carro else ""),
        "casa: voto certo": f"{100 * np.mean([certos(n) for n in casa]):.0f}%",
        "casa: escritas": ", ".join(
            f"{n.removeprefix('casa_')}: " + (", ".join(f"{s}×{k}" for s, k in escritos(n).items()) or "nada")
            for n in casa
        ),
        "neutro: escritas (ideal 0)": sum(escritos("neutro").values()) if "neutro" in resultados else "-",
        "carro→casa→carro": ", ".join(f"{s}×{k}" for s, k in escritos("carro_casa_carro").items())
        if "carro_casa_carro" in resultados
        else "-",
    }


def tabela(titulo: str, linhas: dict[str, dict]) -> list[str]:
    colunas = list(next(iter(linhas.values())))
    saida = [f"## {titulo}", "", "| Variante | " + " | ".join(colunas) + " |", "|---" * (len(colunas) + 1) + "|"]
    for nome, valores in linhas.items():
        saida.append(f"| {nome} | " + " | ".join(str(valores[c]) for c in colunas) + " |")
    return [*saida, ""]


def _esperados(dados: dict) -> set[str]:
    return {s for s in dados.get("sinal", "").split(",") if s}


# -- testes ---------------------------------------------------------------------


def _extrair(video: str, opcoes: OpcoesVisao) -> str:
    """Trabalho de um processo: preenche o cache de uma variante de um vídeo."""
    quadros_do_video(Path(video), refazer=False, opcoes=opcoes)
    return f"{Path(video).stem} [{opcoes.nome}]"


def preparar_caches(videos: list[Path]) -> None:
    """Roda em paralelo a visão das variantes que ainda não têm cache."""
    trabalhos = {
        (str(v), o)
        for _, variantes in VARIANTES_VISAO.values()
        for _, o in variantes
        for v in videos
        if not caminho_cache(v, o).exists()
    }
    if not trabalhos:
        return
    print(f"Visão: {len(trabalhos)} vídeos×variantes em paralelo...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for pronto in pool.map(_extrair, *zip(*trabalhos, strict=True)):
            print("  ok:", pronto, flush=True)


def testes_visao(videos: list[Path]) -> list[str]:
    linhas = []
    for numero, (titulo, variantes) in VARIANTES_VISAO.items():
        tabela_teste = {}
        for nome, opcoes in variantes:
            resultados = {}
            for video in videos:
                vetores, tempos, dados = quadros_do_video(video, refazer=False, opcoes=opcoes)
                resultados[video.stem] = (reconhecer(vetores, tempos, MODELO), _esperados(dados), vetores)
            tabela_teste[nome] = resumir(resultados)
            print(f"  teste {numero}: {nome}", flush=True)
        linhas += tabela(f"{numero}. {titulo}", tabela_teste)
    return linhas


def testes_reconhecimento(videos: list[Path]) -> list[str]:
    base = {v.stem: quadros_do_video(v, refazer=False) for v in videos}
    linhas = []
    for numero, (titulo, variantes) in VARIANTES_RECONHECIMENTO.items():
        tabela_teste = {}
        for nome, valores, votos in variantes:
            desfazer = _ajuste_config(**valores)
            try:

                def ajustar(controlador, votos=votos):
                    if votos is not None:
                        controlador.estabilizador.votos_minimos = votos

                resultados = {
                    stem: (reconhecer(vetores, tempos, MODELO, ajustar), _esperados(dados), vetores)
                    for stem, (vetores, tempos, dados) in base.items()
                }
            finally:
                desfazer()
            tabela_teste[nome] = resumir(resultados)
            print(f"  teste {numero}: {nome}", flush=True)
        linhas += tabela(f"{numero}. {titulo}", tabela_teste)
    if 6 in VARIANTES_RECONHECIMENTO:
        linhas.append(
            "Obs. do teste 6: o modelo foi treinado com gravações feitas no modo "
            '"só quando muda"; os outros modos mudam também o que a IA vê no treino.'
        )
        linhas.append("")
    return linhas


def janelas_do_video(vetores: np.ndarray, tempos: np.ndarray) -> list[np.ndarray]:
    """As janelas que o app classificaria (mesmo filtro e mesmo buffer do app)."""
    relogio = Relogio()
    filtro = FiltroQuadros(relogio=relogio)
    buffer: list[np.ndarray] = []
    janelas = []
    for vetor, momento in zip(vetores, tempos, strict=True):
        relogio.agora = float(momento)
        if not filtro.aceitar(vetor):
            continue
        buffer = [*buffer, vetor][-config.QUADROS_POR_JANELA :]
        if tem_mao(vetor) and len(buffer) == config.QUADROS_POR_JANELA:
            janelas.append(np.array(buffer))
    return janelas


def teste_ms(videos: list[Path]) -> list[str]:
    from lab_ms.experimento import carregar_gravacoes
    from lab_ms.ms_knn import ClassificadorMS

    gravacoes = carregar_gravacoes()
    ms = ClassificadorMS().fit([g.quadros for g in gravacoes], [g.rotulo for g in gravacoes])
    rf = carregar_modelo(MODELO)
    linhas = [
        "## 8. MS × RandomForest (janela a janela, sem limiar nem votação)",
        "",
        "Em quantas janelas o sinal mais provável é o certo (no neutro, o certo é \"neutro\").",
        "",
        "| Vídeo | Janelas | RandomForest | MS | ms/janela RF | ms/janela MS |",
        "|---|---|---|---|---|---|",
    ]
    totais = Counter()
    for video in videos:
        vetores, tempos, dados = quadros_do_video(video, refazer=False)
        esperados = _esperados(dados)
        janelas = janelas_do_video(vetores, tempos)
        acertos, tempo = Counter(), Counter()
        for janela in janelas:
            inicio = time.perf_counter()
            acertos["rf"] += prever(rf, features_janela(janela)).sinal in esperados
            meio = time.perf_counter()
            acertos["ms"] += next(iter(ms.distancias_janela(janela))) in esperados
            fim = time.perf_counter()
            tempo["rf"] += meio - inicio
            tempo["ms"] += fim - meio
        n = max(1, len(janelas))
        totais.update({"n": len(janelas), "rf": acertos["rf"], "ms": acertos["ms"]})
        linhas.append(
            f"| {video.stem} | {len(janelas)} | {100 * acertos['rf'] / n:.0f}% | {100 * acertos['ms'] / n:.0f}% | "
            f"{1000 * tempo['rf'] / n:.1f} | {1000 * tempo['ms'] / n:.1f} |"
        )
        print(f"  teste 8: {video.stem}", flush=True)
    n = max(1, totais["n"])
    linhas.append(f"| **todos** | {totais['n']} | **{100 * totais['rf'] / n:.0f}%** | **{100 * totais['ms'] / n:.0f}%** | | |")
    return [*linhas, ""]


def teste_tempo(videos: list[Path]) -> list[str]:
    """Mede sozinho (sem outros processos) o tempo de cada etapa da visão."""
    import json

    medidas = []
    for video in videos[:3]:
        dados = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
        _, ms, _ = extrair_quadros(video, dados["tempos_s"][:200])
        medidas.append(ms[10:])  # os primeiros quadros aquecem os modelos
    ms = np.vstack(medidas).mean(axis=0)
    soma = float(ms.sum())
    # No app o Pose roda numa thread ao lado; o resto é em sequência.
    corpo = float(ms[ETAPAS.index("corpo")])
    paralelo = max(soma - corpo, corpo)
    linhas = [
        "## 9. Tempo de cada etapa por quadro (média de 3 vídeos, 1 processo sozinho)",
        "",
        "| Etapa | ms |",
        "|---|---|",
        *[f"| {etapa} | {valor:.1f} |" for etapa, valor in zip(ETAPAS, ms, strict=True)],
        f"| **tudo em sequência** | **{soma:.1f}** (até {1000 / soma:.0f} qps) |",
        f"| **Pose em paralelo (como o app)** | **{paralelo:.1f}** (até {1000 / paralelo:.0f} qps) |",
        "",
        "A câmera entrega no máximo 30 qps (33 ms por quadro).",
        "",
    ]
    return linhas


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    videos = videos_de_teste()
    preparar_caches(videos)

    linhas = [
        "# Estudo nos vídeos de teste: visão, votação, janela, MS e tempo",
        "",
        (f"{len(videos)} vídeos de 20 s · modelo sem máscara · liberação sem voto "
        f"{config.QUADROS_PARA_LIBERAR_SEM_VOTO} quadros."),
        "Voto certo: média dos vídeos de carro (ou casa). Escritas: quantas vezes o sinal viraria texto.",
        "",
        "## 2. Prever a mão (Kalman): não testado",
        "",
        ("Com `USAR_LUGAR = False` a IA não usa a posição da mão, e o movimento vem do pulso do braço "
        "(Pose). Mover a mão lembrada não muda nenhum número que a IA vê."),
        "",
    ]
    print("Testes 1, 3, 4, 5 (visão)...", flush=True)
    linhas += testes_visao(videos)
    print("Testes 6, 7 (reconhecimento)...", flush=True)
    linhas += testes_reconhecimento(videos)
    print("Teste 8 (MS)...", flush=True)
    linhas += teste_ms(videos)
    print("Teste 9 (tempo)...", flush=True)
    linhas += teste_tempo(videos)
    linhas.append(f"Tempo total: {time.perf_counter() - inicio:.0f} s")

    texto = "\n".join(linhas)
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    arquivo = PASTA_RESULTADOS / f"estudo_tudo_{time.strftime('%Y%m%d_%H%M%S')}.md"
    arquivo.write_text(texto, encoding="utf-8")
    print()
    print(texto)
    print(f"\nRelatório salvo em {arquivo}")


if __name__ == "__main__":
    main()
