"""Aggregate publication preserves counts, not selection identifiers."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.ablacao_dossie_agregado import INPUTS, aggregate


def write_inputs(root):
    (root / INPUTS[0]).write_text(
        "pair_id,nota,revisores_erraram,n_revisores_erraram\n"
        "private-id,7.32,DeepSeek V4.1 Flash;Haiku 5.5,2\n", encoding="utf-8")
    (root / INPUTS[1]).write_text(
        "pair_id,modelo,nota,original,controle,sem_pcal,"
        "fn_original_revertido_match,fn_controle_revertido_match\n"
        "private-id,deepseek-v4.1-flash,7.32,NONMATCH,NONMATCH,MATCH,True,True\n"
        "private-id,claude-haiku-5-5,7.32,NONMATCH,UNSURE,NONMATCH,False,False\n",
        encoding="utf-8")


def test_aggregate_deterministic_counts_and_fingerprints_without_identifiers(tmp_path):
    write_inputs(tmp_path)
    result = aggregate(tmp_path)
    assert aggregate(tmp_path) == result
    assert result["selection"] == {"n": 1, "n_revisores_erraram": {"2": 1}}
    assert result["result_rows"] == sum(row["n"] for row in result["categories"]) == 2
    assert result["input_sha256"] == {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in INPUTS}
    serialized = json.dumps(result)
    assert "private-id" not in serialized
    assert "7.32" not in serialized
    assert all(set(row) == {"modelo", "original", "controle", "sem_pcal", "n"}
               for row in result["categories"])


def test_incomplete_or_duplicate_results_are_rejected(tmp_path):
    write_inputs(tmp_path)
    path = tmp_path / INPUTS[1]
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-1]))
    with pytest.raises(ValueError, match="complete selection"):
        aggregate(tmp_path)
    path.write_text("".join(lines + [lines[-1]]))
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(tmp_path)


def test_published_aggregate_matches_historical_summary():
    root = Path(__file__).resolve().parents[1] / "data"
    published = json.loads((root / "ablacao_dossie_agregado.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "sprint3b/ablacao/resumo.json").read_text(encoding="utf-8"))
    assert published["selection"] == {"n": 18, "n_revisores_erraram": {"2": 9, "3": 9}}
    assert published["result_rows"] == 36
    for model in summary["models"]:
        rows = [row for row in published["categories"] if row["modelo"] == model["model"]]
        assert sum(row["n"] for row in rows) == model["n"]
        for arm in ("controle", "sem_pcal"):
            for decision, count in model[arm].items():
                assert sum(row["n"] for row in rows if row[arm] == decision) == count
        for transition, count in model["transitions"].items():
            control, ablated = transition.split("->")
            assert sum(row["n"] for row in rows
                       if row["controle"] == control and row["sem_pcal"] == ablated) == count
