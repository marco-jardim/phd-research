"""Faixas da nota final dos recuperados: agregado sem identificadores e tabela gerada a partir dele."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from scripts import perfil_recuperados_faixas as faixas
from scripts import tabelas_banca_b_editorial as editorial

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/perfil_recuperados_faixas.csv"
TABELA = ROOT / "text/latex/tables/editorial/tab_perfil_recuperados.tex"
PROIBIDAS = {"pair_id", "REFREC", "COMPREC", "NOME", "R_DTOBITO", "R_DTNASC", "indice", "index"}


def test_tabular_conta_por_faixa_e_exige_cobertura():
    nota = pd.Series([4.0, 5.0, 5.9, 6.0, 7.99, 8.0, 8.5, 9.0, 10.99, 11.0, 12.0, 6.5])
    recuperados = pd.Series([False, True, True, True, True, False, False, False, False, False, False, False])
    encontrados = pd.Series([False, False, False, False, False, True, True, True, True, True, True, False])
    frame = faixas.tabular(nota, recuperados, encontrados)
    assert frame.set_index("faixa").recuperados.to_dict() == {
        "<5": 0, "5-6": 2, "6-7": 1, "7-8": 1, "8-9": 0, "9-11": 0, ">=11": 0}
    assert frame.set_index("faixa").ja_encontrados.to_dict() == {
        "<5": 0, "5-6": 0, "6-7": 0, "7-8": 0, "8-9": 2, "9-11": 2, ">=11": 2}
    assert (frame.total == frame.recuperados + frame.ja_encontrados).all()
    assert list(frame.columns) == faixas.COLUNAS
    with pytest.raises(ValueError, match="disjuntos"):
        faixas.tabular(nota, recuperados, recuperados)
    with pytest.raises(ValueError, match="ausente"):
        faixas.tabular(nota.where(nota.ne(4.0)), recuperados, encontrados)


def test_csv_agregado_sem_identificadores_e_com_totais_do_teste():
    frame = pd.read_csv(CSV, dtype={"faixa": str})
    assert list(frame.columns) == faixas.COLUNAS
    assert not PROIBIDAS.intersection(frame.columns)
    assert len(frame) == len(faixas.FAIXAS) and frame.faixa.is_unique
    assert int(frame.recuperados.sum()) == faixas.CONTROLES["recuperados"] == 27
    assert int(frame.ja_encontrados.sum()) == 40
    assert (frame.total == frame.recuperados + frame.ja_encontrados).all()
    por_faixa = frame.set_index("faixa")
    assert por_faixa.recuperados[["5-6", "6-7", "7-8"]].tolist() == [3, 10, 14]
    assert int(por_faixa.recuperados[["8-9", "9-11", ">=11"]].sum()) == 0
    assert int(por_faixa.ja_encontrados[["8-9", "9-11", ">=11"]].sum()) == 40
    assert int(por_faixa.ja_encontrados[["<5", "5-6", "6-7", "7-8"]].sum()) == 0
    # Mesmos totais do perfil por conjunto (n de recuperados e já encontrados).
    perfil = pd.read_csv(ROOT / "data/perfil_obitos_agregado.csv").set_index("conjunto")
    assert (int(perfil.loc["recuperados", "n"]), int(perfil.loc["ja_encontrados", "n"])) == (27, 40)


def test_metadata_sem_identificadores_e_com_proveniencia():
    meta = json.loads((ROOT / "data/perfil_recuperados_faixas_metadata.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 42 and meta["limiar"] == 0.5 and meta["corte_nota"] == 8.0
    assert meta["controles"] == faixas.CONTROLES
    for chave in ("fonte", "definicao", "entrada_sha256", "versoes", "fontes_sha256", "csv_sha256"):
        assert meta[chave]
    texto = json.dumps(meta)
    assert not any(nome in texto for nome in ("pair_id", "REFREC", "COMPREC"))


def test_faixas_historicas_do_perfil_batem_com_o_csv_abaixo_de_8():
    historico = json.loads((ROOT / "data/epidemiological_profile.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(CSV, dtype={"faixa": str}).set_index("faixa")
    for grupo, coluna in (("recuperados", "recuperados"), ("ja_encontrados", "ja_encontrados")):
        for chave in ("5-6", "6-7", "7-8", "8-9"):
            assert historico[grupo]["faixas"][chave] == frame.loc[chave, coluna], (grupo, chave)


def test_bloco_de_faixas_e_deterministico_e_soma_os_totais_da_tabela():
    um = editorial.faixas_bloco(CSV, 27, 40)
    dois = editorial.faixas_bloco(CSV, 27, 40)
    assert um == dois
    assert um == [
        r"\quad Faixa 5-6 & 3 (11,1\%) & 0 (0,0\%) \\",
        r"\quad Faixa 6-7 & 10 (37,0\%) & 0 (0,0\%) \\",
        r"\quad Faixa 7-8 & 14 (51,9\%) & 0 (0,0\%) \\",
        r"\quad Faixa $\geq$ 8 & 0 (0,0\%) & 40 (100,0\%) \\"]
    with pytest.raises(ValueError, match="não somam"):
        editorial.faixas_bloco(CSV, 28, 40)


def test_bloco_exibe_faixa_abaixo_de_5_so_quando_ha_pares(tmp_path):
    frame = pd.DataFrame(dict(
        faixa=["<5", "5-6", "6-7", "7-8", "8-9", "9-11", ">=11"],
        nota_min=[None, 5, 6, 7, 8, 9, 11], nota_max_exclusivo=[5, 6, 7, 8, 9, 11, None],
        recuperados=[1, 1, 0, 0, 0, 0, 0], ja_encontrados=[0, 0, 0, 0, 1, 0, 1],
        total=[1, 1, 0, 0, 1, 0, 1]))
    caminho = tmp_path / "f.csv"
    frame.to_csv(caminho, index=False)
    linhas = editorial.faixas_bloco(caminho, 2, 2)
    assert linhas[0].startswith(r"\quad Faixa <5 & 1 (50,0\%)")
    assert linhas[-1] == r"\quad Faixa $\geq$ 8 & 0 (0,0\%) & 2 (100,0\%) \\"
    frame.loc[0, "recuperados"] = 0
    frame.loc[1, "recuperados"] = 2
    frame.to_csv(caminho, index=False)
    assert not any("Faixa <5" in linha for linha in editorial.faixas_bloco(caminho, 2, 2))


def test_tabela_regenerada_igual_a_versionada_e_faixas_somam_os_totais(monkeypatch, tmp_path):
    monkeypatch.setattr(editorial, "OUT", tmp_path)
    editorial.tabela_perfil_recuperados()
    gerada = (tmp_path / TABELA.name).read_bytes()
    assert gerada == TABELA.read_bytes()
    texto = gerada.decode("utf-8")
    assert "data/perfil_recuperados_faixas.csv" in texto and "scripts/perfil_recuperados_faixas.py" in texto
    assert "74 pares verdadeiros" in texto and "74 vínculos" not in texto
    totais = re.search(r"^N & (\d+) & (\d+) \\\\", texto, re.M)
    linhas = re.findall(r"^\\quad Faixa .* & (\d+) \(.*\) & (\d+) \(.*\) \\\\", texto, re.M)
    assert totais and linhas
    assert (sum(int(r) for r, _ in linhas), sum(int(j) for _, j in linhas)) == (int(totais[1]), int(totais[2]))
