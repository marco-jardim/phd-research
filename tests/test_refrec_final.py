import json
from pathlib import Path

import pandas as pd
import pytest

from gzcmd.config import load_config
from scripts.avaliar_reruns_exp2 import load_supplemental_runs, make_latex


def test_classifier_default_is_refrec_without_changing_legacy_costs():
    from gzcmd.runner import build_engine_from_config
    cfg = load_config("gzcmd/gzcmd_v3_config.yaml")
    assert cfg.evaluation.split_by == "refrec"
    old = build_engine_from_config(cfg, mode="vigilancia", policy_version="v3")
    assert old.costs.llm_review == .25
    assert old.human_error.e_fn == 0


def test_supplemental_provenance_and_own_cohort(tmp_path):
    prompt = "same-prompt"
    historical = tmp_path / "data/sprint3b/anthropic_sem_mo_v4/exp2_llm_sync_vigilancia.json"
    historical.parent.mkdir(parents=True)
    historical.write_text(json.dumps({"prompt_sha256": prompt}), encoding="utf-8")
    path = tmp_path / "data/sprint3b/anthropic_sem_mo_v4_lote31_refrec/exp2_llm_sync_vigilancia.json"
    path.parent.mkdir(parents=True)
    run = {"mode": "vigilancia", "model": "claude-haiku-5-5", "prompt_version": "v4",
           "prompt_sha256": prompt, "cohort": "v31_lot", "complete": True,
           "selected_pair_ids": ["new-pair"], "n_review_pairs": 1, "n_reviewed": 1,
           "removed_fields": ["model_outputs." + key for key in ("p_cal", "band", "base_choice", "evr", "guardrail", "guardrail_reason")],
           "results": [{"pair_id": "new-pair", "decision": "NONMATCH"}]}
    path.write_text(json.dumps(run), encoding="utf-8")
    gold = pd.DataFrame({"pair_id": ["new-pair"], "nota_final": [5.5], "y": [0]})
    frames, sources = load_supplemental_runs(tmp_path, gold, {"grey_low": {"min": 5, "max": 6}})
    assert len(frames) == 1 and frames[0].band.tolist() == ["grey_low"]
    assert sources[0]["n_esperado"] == 1 and sources[0]["cohort"] == "lote31_refrec"
    run["prompt_sha256"] = "changed"
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_supplemental_runs(tmp_path, gold, {"grey_low": {"min": 5, "max": 6}})


def test_rerun_table_labels_adaptive_cohort_separately():
    row = {"mode": "vigilancia", "model": "claude-haiku-5-5", "dossie": "sem_mo", "prompt": "v4",
           "cohort": "lote31_refrec", "completo": True, "VP": 1, "FP": 0, "FN": 1, "VN": 10,
           "UNSURE": 0, "sens": .5, "espec": 1., "fracao_unsure": 0.}
    text = make_latex([row])
    assert "v4 (lote REFREC)" in text
    assert "50,0 (9,5; 90,5)" in text
    assert "seleção distinta" in text


def test_cascade_table_is_regenerated_from_cells_not_old_counts(tmp_path, monkeypatch):
    from scripts import fechar_refrec_final as final
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    tables = tmp_path / "text/latex/tables"
    data.mkdir()
    tables.mkdir(parents=True)
    for name in ("funil_gzcmd_final.csv", "funil_gzcmd_final_metricas.csv"):
        (data / name).write_bytes((root / "data" / name.replace(".csv", "_row.csv")).read_bytes())
    metrics = pd.read_csv(data / "funil_gzcmd_final_metricas.csv")
    metrics["sens_com_humano_erro_05"] = (metrics.VP_cadeia + .95 * metrics.vinculos_ao_humano) / (metrics.VP_cadeia + metrics.FN_cadeia)
    metrics.to_csv(data / "funil_gzcmd_final_metricas.csv", index=False)
    (tables / "tab_cap8_cascata_v31.tex").write_bytes((root / "text/latex/tables/tab_cap8_cascata_v31_row.tex").read_bytes())
    monkeypatch.setattr(final, "ROOT", tmp_path)
    final.regenerate_cascade_table()
    text = (tables / "tab_cap8_cascata_v31.tex").read_text(encoding="utf-8")
    assert "OOF por REFREC" in text
    assert "82,2 (76,9; 86,5)" in text
    assert "61.696 (247)" in text
