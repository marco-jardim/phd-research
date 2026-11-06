"""Regression tests for the aggregate-only second-review recalculations."""
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from scripts.ablacao_dossie_json_agregado import aggregate
from scripts.ablacao_dossie_taxa_base import cohort_counts, generate as base_rates
from scripts.dossie_campos_enviados import render
from scripts.equidade_passo_revisor_perfeito import excluded, sensitivity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_private_checkpoint_is_reduced_to_counts_and_original_byte_hash(tmp_path):
    path = tmp_path / "synthetic_controle.json"
    source = {"model": "synthetic", "mode": "vigilancia", "arm": "controle",
              "selected_pair_ids": ["private-a", "private-b"],
              "results": [{"pair_id": "private-a", "decision": "MATCH"}],
              "errors": []}
    path.write_text(json.dumps(source), encoding="utf-8", newline="\n")
    result = aggregate(path)
    assert result["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["n_selected"] == 2
    assert result["n_reviewed"] == 1
    assert result["decisions"] == {"MATCH": 1, "NONMATCH": 0, "UNSURE": 0}
    assert "private-a" not in json.dumps(result)
    assert "pair_id" not in json.dumps(result)


@pytest.mark.parametrize("model,arm,expected", [
    ("claude-haiku-5-5", "controle", (0, 9, 9)),
    ("claude-haiku-5-5", "sem_pcal", (7, 0, 11)),
    ("deepseek-v4.1-flash", "controle", (0, 18, 0)),
    ("deepseek-v4.1-flash", "sem_pcal", (9, 0, 9)),
    ("kimi-k3", "sem_pcal", (1, 1, 1)),
])
def test_published_ablation_counts_and_no_tracked_private_checkpoint(model, arm, expected):
    relative = f"data/sprint3b/ablacao/{model}_{arm}"
    result = json.loads((ROOT / f"{relative}_agregado.json").read_text(encoding="utf-8"))
    assert tuple(result["decisions"][d] for d in ("MATCH", "NONMATCH", "UNSURE")) == expected
    assert sum(expected) == result["n_reviewed"]
    assert len(result["source_sha256"]) == 64
    assert "pair_id" not in json.dumps(result)
    assert not subprocess.check_output(["git", "ls-files", "--", f"{relative}.json"], cwd=ROOT).strip()


def test_published_dossier_field_boundary_and_generated_table():
    fields = pd.read_csv(DATA / "dossie_campos_enviados.csv")
    assert len(fields) == 29
    assert fields.enviado.eq("sim").sum() == 14
    assert fields.enviado.eq("não").sum() == 15
    assert fields.loc[fields.campo.eq("CODMUNRES local igual"), "enviado"].item() == "sim"
    assert fields.loc[fields.campo.eq("DTNASC dt inv dia"), "enviado"].item() == "não"
    assert (ROOT / "text/latex/tables/editorial/tab_dossie_campos.tex").read_text(encoding="utf-8") == render(fields.to_dict("records"))


def test_human_perfect_upper_bound_and_step_eleven_regression():
    groups = pd.read_csv(DATA / "equidade_arranjo_group_metrics.csv")
    actual = sensitivity(groups)
    expected = pd.read_csv(DATA / "equidade_passo_revisor.csv")
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected, check_dtype=False)
    row = actual.loc[actual.decisao.eq("arranjo_P_vigilancia") & actual.passo.eq("passo_11")].iloc[0]
    assert (row.n_true, row.tp, row.fn, row.n_humano_true, row.fn_revisor_perfeito) == (40, 17, 23, 14, 9)
    assert row.sens_automatica == pytest.approx(0.425)
    assert row.sens_revisor_perfeito == pytest.approx(0.775)
    assert actual.sens_revisor_perfeito.ge(actual.sens_automatica).all()
    assert actual.sens_revisor_perfeito.le(1).all()


def test_den30_exclusions_preserve_unfiltered_step_ppv():
    original = pd.read_csv(DATA / "equidade_arranjo_pairwise_tests.csv")
    actual = excluded(original).reset_index(drop=True)
    expected = pd.read_csv(DATA / "equidade_arranjo_den30_excluidos.csv")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert len(actual) + (original.n1.ge(30) & original.n2.ge(30)).sum() == len(original)
    row = actual.loc[actual.decisao.eq("arranjo_P_vigilancia") & actual.dimension.eq("passo")
                     & actual.metric.eq("ppv") & actual.group_b.eq("passo_6")].iloc[0]
    assert (row.n1, row.n2, row.num_a, row.num_b) == (25, 78, 17, 76)
    assert row.rate_a == pytest.approx(0.68)
    assert row.rate_b == pytest.approx(76 / 78)


def test_human_true_links_cannot_exceed_false_negatives():
    frame = pd.DataFrame([dict(dimension="passo", decisao="synthetic", group="passo_11",
                              n_true=1, tp=1, fn=0, n_humano_true=1)])
    with pytest.raises(ValueError, match="subset"):
        sensitivity(frame)


@pytest.mark.parametrize("cohort,expected", [
    ("anthropic", (91, 1321, 433)),
    ("anthropic_sem_mo", (148, 92, 1605)),
    ("anthropic_sem_mo_v4", (177, 1582, 86)),
    ("opencode_go", (123, 1593, 129)),
    ("opencode_go_sem_mo", (267, 43, 1535)),
    ("opencode_go_sem_mo_v4", (541, 0, 1304)),
])
def test_whole_cohort_rates_count_mode_pair_reviews(cohort, expected):
    result, hashes = cohort_counts(DATA / "sprint3b" / cohort)
    assert result["n"] == 1845
    assert tuple(result[d] for d in ("MATCH", "NONMATCH", "UNSURE")) == expected
    assert sum(expected) == 1845
    assert len(hashes) == 2


def test_whole_cohort_generator_regenerates_identical_csv(tmp_path):
    # Only aggregates and six historical checkpoint directories are inputs.
    # A junction/symlink is unnecessary: write the regenerated artifact to a
    # temporary directory through the explicit output argument.
    actual = base_rates(DATA, output=tmp_path / "rates.csv")
    published = pd.read_csv(DATA / "ablacao_dossie_taxa_base.csv")
    pd.testing.assert_frame_equal(actual, published, check_dtype=False)
