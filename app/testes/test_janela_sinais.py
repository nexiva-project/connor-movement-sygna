"""Testes das funções da janela de sinais (sem abrir janela)."""

from kara_sygna.interface.janela_sinais import tamanho_do_arquivo, tamanho_legivel


def test_tamanho_legivel():
    assert tamanho_legivel(0) == "0 B"
    assert tamanho_legivel(512) == "512 B"
    assert tamanho_legivel(2048) == "2,0 KB"
    assert tamanho_legivel(int(5.7 * 1024**2)) == "5,7 MB"
    assert tamanho_legivel(3 * 1024**3) == "3,0 GB"


def test_tamanho_do_arquivo_que_nao_existe(tmp_path):
    assert tamanho_do_arquivo(tmp_path / "nada.pkl") == 0
