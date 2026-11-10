"""Verificação de privacidade da pasta de agregados da versão 2 do depósito público."""

import pytest

from scripts.zenodo_agregados_v2 import checar_privacidade, coluna_proibida


def test_coluna_pair_id_falha(tmp_path):
    arquivo = tmp_path / "por_par.csv"
    arquivo.write_text("pair_id,p_cal\nab12cd34ef56ab78,0.5\n", encoding="utf-8")
    problemas = checar_privacidade(arquivo)
    assert problemas
    assert "pair_id" in problemas[0]
    assert all("ab12cd34ef56ab78" not in p for p in problemas)


@pytest.mark.parametrize("valor", ["03/11/2026", "2026-11-03"])
def test_data_completa_falha(tmp_path, valor):
    arquivo = tmp_path / "datas.csv"
    arquivo.write_text(f"faixa,n,observacao\n<5,10,{valor}\n", encoding="utf-8")
    problemas = checar_privacidade(arquivo)
    assert any("data completa" in p and "observacao" in p for p in problemas)
    assert all(valor not in p for p in problemas)  # nunca reproduz valores


def test_agregado_limpo_passa(tmp_path):
    arquivo = tmp_path / "agregado.csv"
    arquivo.write_text(
        "faixa,n,vinculos,auc,origem\n<5,100,3,0.91,data/m8/x.csv\n5-6,50,4,0.88,gzcmd/y.yaml\n",
        encoding="utf-8")
    assert checar_privacidade(arquivo) == []


def test_caminho_absoluto_falha(tmp_path):
    meta = tmp_path / "x_metadata.json"
    meta.write_text('{"command": "C:\\\\Users\\\\alguem\\\\python.exe s.py"}', encoding="utf-8")
    assert checar_privacidade(meta)
    resumo = tmp_path / "resumo.csv"
    resumo.write_text("mode,path\nvigilancia,D:/git/x/y.json\n", encoding="utf-8")
    assert any("caminho local absoluto em path" in p for p in checar_privacidade(resumo))


@pytest.mark.parametrize("nome", ["COMPREC", "R_DTNASC", "NOME_MAE", "nm_pacient", "Endereço",
                                  "row_index", "index", "dt_obito", "cns"])
def test_colunas_proibidas(nome):
    assert coluna_proibida(nome)


@pytest.mark.parametrize("nome", ["n_vinculos", "fracao_zero", "e_fn", "dimension", "sha256"])
def test_colunas_agregadas_permitidas(nome):
    assert not coluna_proibida(nome)
