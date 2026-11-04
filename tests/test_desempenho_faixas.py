import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import desempenho_faixas_ablacao as abl
from scripts import desempenho_faixas_oof_refrec as oof

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTRATO = DATA / "COMPARADORSEMIDENT.csv"


# ---------------------------------------------------------------- ablação (E2)
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


# ------------------------------------------------------------------- OOF (E4)
@pytest.mark.parametrize(
    "nota, esperado",
    [(0.0, "<5"), (4.99, "<5"), (5.0, "5-6"), (6.0, "6-7"), (7.0, "7-8"), (8.0, "8-9"), (8.99, "8-9"),
     (9.0, ">=9"), (10.0, ">=9")],
)
def test_oof_faixa_fechada_a_esquerda(nota, esperado):
    assert oof.atribuir_faixa(nota) == esperado


def test_oof_pair_id_formato():
    pid = oof.pair_id("1", "2", "3")
    assert len(pid) == 16 and pid == oof.pair_id("1", "2", "3") != oof.pair_id("2", "1", "3")


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
    assert r.loc["6-7", "auc"] == pytest.approx(1.0)
    assert np.isnan(r.loc["<5", "auc"]) and np.isnan(r.loc["<5", "sensibilidade"])
    ac = r.loc["area_cinza"]
    assert (ac.n, ac.vinculos) == (5, 3)
    assert r.loc["total", "n"] == 7
    assert r.loc["5-6", "brier"] == pytest.approx(((0.1) ** 2 + (0.6) ** 2 + (0.8) ** 2) / 3)


# ------------------------------------------- artefatos gerados (extrato local)
@pytest.mark.skipif(not EXTRATO.exists(), reason="extrato COMPARADORSEMIDENT.csv ausente (nao versionado)")
def test_ablacao_artefato_totais_e_meta():
    res = pd.read_csv(DATA / "desempenho_faixas_ablacao.csv").set_index(["modelo", "faixa"])
    assert (res.loc[("RF+SMOTE", "total"), "vp"], res.loc[("RF+SMOTE", "total"), "fp"]) == (67, 3)
    assert (res.loc[("GB", "total"), "vp"], res.loc[("GB", "total"), "fp"]) == (60, 4)
    assert [int(res.loc[("RF+SMOTE", f), "vp"]) for f in ("5-6", "6-7", "7-8", "8-9")] == [3, 10, 14, 9]
    meta = json.loads((DATA / "desempenho_faixas_ablacao_meta.json").read_text(encoding="utf-8"))
    assert meta["semente"] == 42 and meta["limiar"] == 0.5 and meta["n_atributos"] == 58
    assert meta["n_teste"] == 18509


@pytest.mark.skipif(not EXTRATO.exists(), reason="extrato COMPARADORSEMIDENT.csv ausente (nao versionado)")
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
