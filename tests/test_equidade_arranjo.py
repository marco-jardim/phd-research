import json
import re
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from gzcmd.arranjo_corte import sha256
from scripts import equidade_v31 as legacy
from scripts.equidade_arranjo import DECISIONS, SORT_KEYS, analyze
from scripts.tabelas_equidade_comparacao import TABLES, generate
from local_data import DATA_ROOT, requires_files, requires_history, requires_reviewer_sources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = DATA_ROOT


def output(name):
    return pd.read_csv(ROOT / f"data/equidade_arranjo_{name}.csv")


def test_all_groups_account_for_the_whole_universe_including_missing_age():
    table = output("cobertura")
    closed = table.groupby(["decisao", "dimension"])[["n_total", "n_true"]].sum()
    assert len(closed) == 4 * len(legacy.DIMENSOES)
    assert closed.n_total.eq(61696).all()
    assert closed.n_true.eq(247).all()
    primary = output("group_metrics")
    for decision in DECISIONS:
        for dim in legacy.DIMENSOES:
            included = primary.loc[primary.decisao.eq(decision) & primary.dimension.eq(dim)]
            excluded = table.loc[table.decisao.eq(decision) & table.dimension.eq(dim) & ~table.incluido_analise]
            assert int(included.n_total.sum() + excluded.n_total.sum()) == 61696
            assert int(included.n_true.sum() + excluded.n_true.sum()) == 247
    missing = table.loc[table.dimension.eq("faixa_idade") & table.group.eq("MISSING")]
    assert len(missing) == 4
    assert missing.n_total.eq(403).all() and missing.n_true.eq(1).all()
    assert not missing.incluido_analise.any()


@pytest.mark.parametrize("name", list(SORT_KEYS))
def test_published_cascade_reproduces_original_aggregate(name):
    actual = output(name)
    expected = pd.read_csv(ROOT / f"data/equidade_v31_{name}.csv")
    actual = actual.loc[actual.decisao.isin(DECISIONS[2:])]
    expected = expected.loc[expected.decisao.isin(DECISIONS[2:])]
    assert list(actual.columns) == list(expected.columns)
    keys = SORT_KEYS[name]
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                  expected.sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["vigilancia", "confirmacao"])
def test_P_matches_observed_S3_acceptance_and_human_referrals(mode):
    table = output("global")
    row = table.loc[table.decisao.eq(f"arranjo_P_{mode}")].iloc[0]
    metrics = pd.read_csv(ROOT / "data/arranjo_corte_metricas.csv")
    reference = metrics.loc[metrics.modo.eq(mode) & metrics.braco.eq("P") & metrics.faixa.eq("TODAS")].iloc[0]
    assert row.n_total == 61696 and row.n_true == 247
    assert row.tp == reference.aceitos
    assert row.fp == reference.fp
    assert row.fn == reference.perdidos + reference.vinc_humano
    assert row.n_humano == reference.pares_humano
    assert row.n_humano_true == reference.vinc_humano
    assert row.sens == pytest.approx(reference.sens_auto)


def test_same_BH_correction_per_decision_and_published_rejections():
    table = output("pairwise_tests")
    for decision in DECISIONS:
        tests = table.loc[table.decisao.eq(decision)]
        assert len(tests) == 58
        expected = legacy.qw3._benjamini_hochberg(tests.p_value_raw.tolist(), alpha=.05)
        np.testing.assert_allclose(tests.p_value_fdr.to_numpy(),
                                   np.asarray([np.nan if value is None else value for value in expected]),
                                   rtol=1e-12, atol=1e-12, equal_nan=True)
        assert tests.reject_fdr.equals(tests.p_value_fdr.le(.05))
    assert int(table.loc[table.decisao.eq("cascata_v31_vigilancia"), "reject_fdr"].sum()) == 9
    assert int(table.loc[table.decisao.eq("cascata_v31_confirmacao"), "reject_fdr"].sum()) == 1


def test_human_adapter_does_not_change_legacy_function_or_input_frame():
    frame = pd.DataFrame({dim: ["group"] * 60 for dim in legacy.DIMENSOES})
    frame["sexo"] = ["F"] * 30 + ["M"] * 30
    frame["y_true"] = 0
    frame.loc[[0, 1, 35], "y_true"] = 1
    final = pd.DataFrame({"final": ["NONMATCH"] * 60})
    final.loc[0, "final"] = "MATCH"
    final.loc[1, "final"] = "UNSURE"
    final.loc[35, "final"] = "DEFERRED"
    original = frame.copy(deep=True)
    glob, groups, tests = analyze(frame, "arranjo_P_vigilancia", "vigilancia", final)
    assert glob.iloc[0].tp == 1 and glob.iloc[0].fn == 2
    assert glob.iloc[0].n_humano == 2 and glob.iloc[0].n_humano_true == 2
    assert groups.decisao.eq("arranjo_P_vigilancia").all()
    assert tests.decisao.eq("arranjo_P_vigilancia").all()
    pd.testing.assert_frame_equal(frame, original)


def test_no_pair_identifiers_or_PII_in_any_generated_aggregate():
    forbidden = {"pair_id", "refrec", "comprec", "REFREC", "COMPREC", "index", "y_true", "nota",
                 "R_DTNASC", "C_DTNASC", "R_DTOBITO", "NOME", "ENDERECO", "final"}
    for name in ("global", "group_metrics", "pairwise_tests", "maxdiff", "cobertura", "comparacao"):
        assert not forbidden.intersection(output(name).columns)
    groups = output("group_metrics")
    assert groups.n_total.ge(30).all()
    assert "MISSING" not in set(groups.group)
    manifest = json.loads((ROOT / "data/equidade_arranjo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_corte6_predictions"] == 7452
    assert manifest["missing_age_pairs"] == 403 and manifest["missing_age_links"] == 1
    assert manifest["published_reference_reproduced"] is True
    for name, value in manifest["code_sha256"].items():
        assert sha256(ROOT / name) == value


@requires_files(SOURCE / "scripts/equidade_v31.py", SOURCE / "scripts/qw3_fairness_analysis.py")
def test_original_equity_analysis_unchanged():
    assert sha256(ROOT / "scripts/equidade_v31.py") == sha256(SOURCE / "scripts/equidade_v31.py")
    assert sha256(ROOT / "scripts/qw3_fairness_analysis.py") == sha256(SOURCE / "scripts/qw3_fairness_analysis.py")


@requires_history()
def test_original_equity_history_unchanged():
    paths = ["scripts/equidade_v31.py", "scripts/qw3_fairness_analysis.py",
             "text/latex/tables/tab_qw3_fairness_rejections.tex"]
    assert subprocess.check_output(["git", "diff", "b09e979", "--", *paths], cwd=ROOT) == b""


def test_comparisons_labeled():
    assert len(list((ROOT / "text/latex/tables").glob("tab_equidade_arranjo*.tex"))) == 7
    chapter = (ROOT / "text/latex/estudo5_equidade.tex").read_text(encoding="utf-8")
    assert r"\input{tables/editorial/tab_den30_equidade_arranjo.tex}" in re.sub(r"(?m)^%.*$", "", chapter)
    assert "decisão principal" in chapter and "comparação" in chapter
    for mode in ("vigilancia", "confirmacao"):
        table = ROOT / f"text/latex/tables/tab_equidade_arranjo_grupos_{mode}.tex"
        assert "sobre a base inteira, comparação" in table.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", TABLES)
def test_versioned_historical_tables_equal_legacy_generation_with_labels(tmp_path, name):
    generate(ROOT / "data", tmp_path)
    actual = (ROOT / "text/latex/tables" / name).read_text(encoding="utf-8")
    assert actual == (tmp_path / name).read_text(encoding="utf-8")
    assert "comparação" in actual


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", ROOT / ".local/arranjo_corte/oof.npz",
                ROOT / ".local/arranjo_corte/policy_vigilancia_P.csv")
@requires_reviewer_sources()
def test_P_regenerates_all_equity_csvs_and_tables(tmp_path):
    from scripts.equidade_arranjo import run
    prereg = json.loads((ROOT / "data/arranjo_corte_metadata.json").read_text(encoding="utf-8"))["prereg_sha256"]
    run(DATA_ROOT, prereg, output_root=tmp_path)
    for path in (ROOT / "data").glob("equidade_arranjo_*.csv"):
        assert path.read_bytes() == (tmp_path / "data" / path.name).read_bytes()
    for path in (ROOT / "text/latex/tables").glob("tab_equidade_arranjo*.tex"):
        assert path.read_bytes() == (tmp_path / "text/latex/tables" / path.name).read_bytes()
