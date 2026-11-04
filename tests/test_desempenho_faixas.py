import json
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import desempenho_faixas_ablacao as abl
from scripts import desempenho_faixas_oof_refrec as oof
from scripts import gerar_tabelas_nb03_faixas as tabelas
from gzcmd.llm_dossier import make_pair_id

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTRATO = DATA / "COMPARADORSEMIDENT.csv"


# Ablação (E2)
@pytest.mark.parametrize(
    "nota, esperado",
    [(0.0, "0-3"), (2.99, "0-3"), (3.0, "3-5"), (4.99, "3-5"), (5.0, "5-6"), (6.0, "6-7"),
     (7.5, "7-8"), (8.99, "8-9"), (9.0, "9-10"), (9.99, "9-10"), (10.0, "10+"), (12.0, "10+")],
)
def test_ablacao_faixa_fechada_a_esquerda(nota, esperado):
    assert abl.atribuir_faixa(nota) == esperado


def test_ablacao_agregacao_sintetica():
    nota = pd.Series([1.0, 5.5, 5.6, 6.5, 6.6, 9.5])
    y = np.array([0, 1, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 0, 1, 1])
    res = abl.agregar(nota, y, pred, "M")
    r = res.set_index("faixa")
    assert (r.loc["5-6", "vp"], r.loc["5-6", "fp"], r.loc["5-6", "fn"]) == (1, 1, 0)
    assert r.loc["5-6", "precisao"] == pytest.approx(0.5)
    assert (r.loc["6-7", "vp"], r.loc["6-7", "fn"], r.loc["6-7", "vinculos"]) == (1, 1, 2)
    assert r.loc["6-7", "sensibilidade"] == pytest.approx(0.5)
    assert np.isnan(r.loc["0-3", "sensibilidade"]) and np.isnan(r.loc["0-3", "precisao"])
    tot = r.loc["total"]
    assert (tot.n, tot.vinculos, tot.vp, tot.fp, tot.fn) == (6, 4, 3, 1, 1)
    assert r.drop("total").n.sum() == tot.n
    assert list(res.columns) == abl.COLUNAS


# OOF (E4)
@pytest.mark.parametrize(
    "nota, esperado",
    [(0.0, "<5"), (4.99, "<5"), (5.0, "5-6"), (6.0, "6-7"), (7.0, "7-8"), (8.0, "8-9"), (8.99, "8-9"),
     (9.0, ">=9"), (10.0, ">=9")],
)
def test_oof_faixa_fechada_a_esquerda(nota, esperado):
    assert oof.atribuir_faixa(nota) == esperado


def test_oof_pair_id_formato():
    pid = make_pair_id("1", "2", "3")
    assert len(pid) == 16 and pid == make_pair_id("1", "2", "3") != make_pair_id("2", "1", "3")


def test_oof_agregacao_sintetica():
    df = pd.DataFrame({
        "nota": [4.0, 5.5, 5.6, 5.7, 6.5, 6.6, 9.5],
        "y": [0, 1, 0, 1, 0, 1, 1],
        "p_cal_ml": [0.1, 0.9, 0.6, 0.2, 0.1, 0.8, 0.95],
    })
    r = oof.agregar(df).set_index("faixa")
    assert (r.loc["5-6", "n"], r.loc["5-6", "vinculos"]) == (3, 2)
    assert (r.loc["5-6", "vp"], r.loc["5-6", "fp"], r.loc["5-6", "fn"]) == (1, 1, 1)
    assert r.loc["5-6", "precisao"] == pytest.approx(0.5)
    assert r.loc["5-6", "auc"] == pytest.approx(0.5)
    assert r.loc["5-6", "minoritaria_n"] == 1
    assert r.loc["6-7", "auc"] == pytest.approx(1.0)
    assert np.isnan(r.loc["<5", "auc"]) and np.isnan(r.loc["<5", "sensibilidade"])
    ac = r.loc["area_cinza"]
    assert (ac.n, ac.vinculos) == (5, 3)
    assert r.loc["total", "n"] == 7
    assert r.loc["5-6", "brier"] == pytest.approx(((0.1) ** 2 + (0.6) ** 2 + (0.8) ** 2) / 3)


# Artefatos versionados, independentes do extrato local
def test_ablacao_artefato_totais_e_meta():
    res = pd.read_csv(DATA / "desempenho_faixas_ablacao.csv").set_index(["modelo", "faixa"])
    assert (res.loc[("RF+SMOTE", "total"), "vp"], res.loc[("RF+SMOTE", "total"), "fp"]) == (67, 3)
    assert (res.loc[("GB", "total"), "vp"], res.loc[("GB", "total"), "fp"]) == (60, 4)
    assert [int(res.loc[("RF+SMOTE", f), "vp"]) for f in ("5-6", "6-7", "7-8", "8-9")] == [3, 10, 14, 9]
    meta = json.loads((DATA / "desempenho_faixas_ablacao_meta.json").read_text(encoding="utf-8"))
    assert meta["semente"] == 42 and meta["limiar"] == 0.5 and meta["n_atributos"] == 58
    assert meta["n_teste"] == 18509
    assert meta["limiar_otimizado"] == "selecionado_no_teste_da_ablacao"
    abl.verificar(res.reset_index())


def test_oof_artefato_totais_e_area_cinza():
    res = pd.read_csv(DATA / "desempenho_faixas_oof_refrec.csv").set_index("faixa")
    assert (res.loc["total", "n"], res.loc["total", "vinculos"]) == (61696, 247)
    ac = res.loc["area_cinza"]
    assert (ac.n, ac.vinculos) == (21620, 151)
    assert abs(ac.auc - 0.9951) <= 0.0005 and abs(ac.ap - 0.8623) <= 0.0005
    assert res.drop(["area_cinza", "total"]).n.sum() == 61696
    meta = json.loads((DATA / "desempenho_faixas_oof_refrec_meta.json").read_text(encoding="utf-8"))
    assert meta["limiar"] == 0.5 and meta["limiar_otimizado"] is False
    assert not {"pair_id"} & set(res.reset_index().columns)
    assert meta["fingerprint"]["code_sha256"]
    assert meta["fingerprint"]["source_sha256"]
    assert len(meta["csv_sha256"]) == 64
    assert {"subsample", "colsample_bytree", "scale_pos_weight"} <= meta["hiperparametros"].keys()


def test_ablacao_parametros_iguais_ao_protocolo_original():
    arvore = ast.parse((ROOT / "scripts/ablation_study.py").read_text(encoding="utf-8"))
    modelos = next(n.value for n in ast.walk(arvore) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "models" for t in n.targets))
    chamadas = {ast.literal_eval(k): v.elts[0] for k, v in zip(modelos.keys, modelos.values)}

    def parametros(chamada):
        return {k.arg: abl.RANDOM_STATE if isinstance(k.value, ast.Name) and k.value.id == "RANDOM_STATE"
                else ast.literal_eval(k.value) for k in chamada.keywords}

    assert parametros(chamadas["RF+SMOTE"]) == abl.RF_PARAMS
    assert parametros(chamadas["GB"]) == abl.GB_PARAMS
    smote = next(n for n in ast.walk(arvore) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "SMOTE")
    assert parametros(smote) == abl.SMOTE_PARAMS
    semente = next(n.value for n in arvore.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "RANDOM_STATE" for t in n.targets))
    assert ast.literal_eval(semente) == abl.RANDOM_STATE


@pytest.mark.parametrize("erro", [None, "duplicado", "ausente"])
def test_carregar_cruzado_sintetico(tmp_path, monkeypatch, erro):
    extrato = tmp_path / "extrato.csv"
    entrada_oof = tmp_path / "oof.csv"
    ext = pd.DataFrame({"COMPREC,C,12,0": ["01", "02"], "REFREC": ["03", "04"],
                        "PASSO": ["1", "1"], "PAR": ["1", "0"]})
    ids = [make_pair_id("01", "03", "1"), make_pair_id("02", "04", "1")]
    pred = pd.DataFrame({"pair_id": ids[::-1], "nota": [6.5, 5.5], "p_cal_ml": [0.2, 0.8]})
    if erro == "duplicado":
        ext = pd.concat([ext, ext.iloc[:1]], ignore_index=True)
    if erro == "ausente":
        pred = pred.iloc[:1]
    ext.to_csv(extrato, sep=";", index=False)
    pred.to_csv(entrada_oof, index=False)
    monkeypatch.setattr(oof, "EXTRATO", extrato)
    monkeypatch.setattr(oof, "OOF", entrada_oof)
    if erro:
        with pytest.raises(ValueError, match="duplicado|mesmos pares"):
            oof.carregar_cruzado()
    else:
        m = oof.carregar_cruzado()
        assert m.y.tolist() == [0, 1]
        assert m.nota.tolist() == [6.5, 5.5]
        assert list(m.columns) == ["nota", "p_cal_ml", "y"]


@pytest.mark.parametrize("campo", ["vp", "fp", "fn"])
def test_ablacao_verificar_rejeita_totais_divergentes(campo):
    res = pd.read_csv(DATA / "desempenho_faixas_ablacao.csv")
    res.loc[(res.modelo == "RF+SMOTE") & (res.faixa == "total"), campo] += 1
    with pytest.raises(ValueError, match="RF\\+SMOTE total VP/FP/FN"):
        abl.verificar(res)


def test_oof_verificar_rejeita_total_divergente():
    with pytest.raises(ValueError, match="linhas casadas"):
        oof.verificar(pd.DataFrame({"y": [1]}), pd.DataFrame())


def test_gerador_tex_nd_virgula_ic_e_hifen(tmp_path, monkeypatch):
    monkeypatch.setattr(tabelas, "OUT", tmp_path)
    tabelas.tabela_faixas()
    tabelas.tabela_e4_oof()
    for nome in ("tab_desempenho_faixas.tex", "tab_e4_desempenho_faixas_oof.tex"):
        texto = (tmp_path / nome).read_text(encoding="utf-8")
        assert texto.startswith("% Gerado por scripts/gerar_tabelas_nb03_faixas.py de data/")
        assert texto.endswith("\n")
        assert "n.d." in texto and "0," in texto
        assert "5-6" in texto and "5--6" not in texto
        assert "IC de Wilson" in texto and r"\shortstack" in texto
        assert "Faixa & Pares &" in texto
    texto = (tmp_path / "tab_e4_desempenho_faixas_oof.tex").read_text(encoding="utf-8")
    linha = next(l for l in texto.splitlines() if l.startswith("$<5$ &"))
    assert "& n.d. & n.d. &" in linha
    assert tabelas.discriminacao(pd.Series({"minoritaria_n": 4, "auc": 0.98765}), "auc") == "n.d."
    assert tabelas.discriminacao(pd.Series({"minoritaria_n": 5, "auc": 0.98765}), "auc") == "0,988"


def test_wilson_denominador_zero_e_extremos():
    assert tabelas.razao_ic(0, 0) == "n.d."
    assert "0,000" in tabelas.razao_ic(0, 3)
    assert "1,000" in tabelas.razao_ic(3, 3)


@pytest.mark.parametrize("erro", [None, "csv", "codigo"])
def test_proveniencia_copia_json_e_rejeita_hash_divergente(tmp_path, monkeypatch, erro):
    csv = tmp_path / "oof.csv"
    csv.write_text("pair_id,nota\n", encoding="utf-8")
    original = {"csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                "fingerprint": {"code_sha256": "hash-de-codigo-invalido", "source_sha256": "origem"},
                "hiperparametros": {"subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": "por treino"}}
    if erro == "csv":
        original["csv_sha256"] = "hash-de-csv-invalido"
    elif erro == "codigo":
        del original["hiperparametros"]
    csv.with_suffix(".json").write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(oof, "OOF", csv)
    if erro:
        with pytest.raises(ValueError, match="SHA-256|fingerprint"):
            oof.proveniencia()
    else:
        obtido = oof.proveniencia()
        assert obtido["fingerprint"] == original["fingerprint"]
        assert obtido["csv_sha256"] == original["csv_sha256"]
        assert obtido["hiperparametros"] == original["hiperparametros"]
