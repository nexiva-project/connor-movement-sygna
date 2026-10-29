"""Testes das etapas de uma gravação (pré-sinal uma vez, sinal em 3 ritmos)."""

from kara_sygna.aplicacao.etapas import RITMO_NORMAL, montar_etapas
from kara_sygna.dados.repositorio import TIPO_PRE, TIPO_SINAL


def test_pre_sinal_uma_vez_e_sinal_em_tres_ritmos():
    etapas = montar_etapas("carro", "S", 10)
    assert [e.legenda for e in etapas] == [
        "pré-sinal S",
        "carro 1/3: devagar",
        "carro 2/3: médio",
        "carro 3/3: rápido",
    ]
    assert [e.tipo for e in etapas] == [TIPO_PRE] + [TIPO_SINAL] * 3
    assert all(e.pre_sinal == "S" for e in etapas[1:]) and etapas[0].pre_sinal == ""
    assert all(e.duracao == 10 for e in etapas)
    assert [e.legenda_video for e in etapas] == [
        "PRÉ-SINAL S",
        "carro 1/3: DEVAGAR",
        "carro 2/3: MÉDIO",
        "carro 3/3: RÁPIDO",
    ]


def test_neutro_sem_ritmo_e_tempo_padrao_quando_vazio():
    etapas = montar_etapas("neutro")
    assert [e.legenda for e in etapas] == ["neutro 1/3", "neutro 2/3", "neutro 3/3"]
    assert etapas[0].legenda_video == "neutro 1/3"
    assert all(e.duracao is not None for e in etapas)  # a sequência anda sozinha


def test_uma_vez_so_grava_ate_parar():
    etapas = montar_etapas("oi", ritmos=())
    assert len(etapas) == 1 and etapas[0].duracao is None
    assert etapas[0].legenda == "oi 1/1"


def test_legenda_do_video_mantem_os_acentos_do_sinal():
    assert montar_etapas("avião")[0].legenda_video == "avião 1/3: DEVAGAR"


def test_ritmos_desligados_grava_uma_vez_normal():
    etapas = montar_etapas("s", duracao=3, ritmos=RITMO_NORMAL)
    assert [e.legenda for e in etapas] == ["s 1/1: normal"]
    assert etapas[0].legenda_video == "s 1/1: NORMAL" and etapas[0].duracao == 3
    com_pre = montar_etapas("carro", "S", 5, ritmos=RITMO_NORMAL)
    assert [e.legenda for e in com_pre] == ["pré-sinal S", "carro 1/1: normal"]
