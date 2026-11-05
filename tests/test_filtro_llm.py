import json

import pandas as pd
import pytest
import yaml
from local_data import DATA_ROOT

from scripts import filtro_llm as filtro


def test_synthetic_workload_and_recovery():
    ids = ["a", "b", "c", "d", "e"]
    decisions = dict(zip(ids, ["NONMATCH", "NONMATCH", "MATCH", "UNSURE", "MATCH"]))
    labels = dict(zip(ids, [2, 0, 2, 0, 2]))
    row = filtro.outcomes("synthetic", ids, decisions, labels, {3: 1})
    assert row["H"] == 3
    assert row["par2_topk"] == 3
    assert row["par2_recuperados"] == 2
    assert row["par2_descartados"] == 1
    assert row["nao_vinculos_descartados"] == 1
    assert row["vpn_nonmatch"] == .5
    assert row["vpn_ic95_inf"] < .5 < row["vpn_ic95_sup"]
    assert row["diferenca_par2"] == 1
    assert row["sens_humano_perfeito"] == pytest.approx(213 / 247)
    assert row["sens_humano_05"] == pytest.approx(212.9 / 247)
    assert row["diferenca_sens_humano_05"] == pytest.approx(.95 / 247)


def test_zero_nonmatch_and_zero_human():
    row = filtro.outcomes("test", ["a"], {"a": "MATCH"}, {"a": 2}, {1: 1})
    assert row["H"] == 1
    assert row["vpn_nonmatch"] is None
    assert row["vpn_ic95_inf"] is None
    row = filtro.outcomes("test", ["a"], {"a": "NONMATCH"}, {"a": 2}, {})
    assert row["H"] == 0
    assert row["par2_recuperados"] == row["o3_par2_mesma_carga"] == 0


def test_response_choice_frozen_P_before_extra_sources():
    # Frozen is exactly the dictionary returned by S1 validated_responses,
    # including P's historical last-cohort precedence.
    frozen = {"a": "UNSURE", "b": "MATCH"}
    extra = [{"a": "NONMATCH", "c": "MATCH"},
             {"b": "NONMATCH", "c": "UNSURE", "d": "NONMATCH"}]
    decisions, conflicts = filtro.select_responses(frozen, extra)
    assert decisions == {"a": "UNSURE", "b": "MATCH", "c": "MATCH", "d": "NONMATCH"}
    assert conflicts == {"a", "b", "c"}
    with pytest.raises(ValueError, match="Invalid reviewer decision"):
        filtro.select_responses({}, [{"x": "INVALID"}])


@pytest.mark.parametrize("k", filtro.KS)
def test_stop_rule(k):
    ids = [f"id-{i}" for i in range(k)]
    decisions = {pid: "MATCH" for pid in ids[:-1]}
    row = filtro.coverage_row("test", ids, decisions, set(), k)
    assert row["pendentes"] == 1
    assert not row["completa"]
    assert row["escalate"] == (k <= 125)
    assert row["custo_pendentes_usd_sem_cache"] == .00316
    with pytest.raises(ValueError, match="STOP: incomplete coverage"):
        filtro.outcomes("test", ids, decisions, {}, {})
    decisions[ids[-1]] = "UNSURE"
    row = filtro.coverage_row("test", ids, decisions, set(), k)
    assert row["completa"]
    assert not row["escalate"]


@pytest.mark.parametrize("diffs,expected", [
    ([0, 1], "ajuda"), ([-1, 0], "prejudica"), ([-1, 1], "misto"),
    ([0, 0], "misto"), ([], "indisponivel"),
])
def test_interpretation(diffs, expected):
    assert filtro.interpret(diffs) == expected


def test_incomplete_coverage_never_loads_labels_or_leaks_values(tmp_path, monkeypatch):
    ids = [f"PRIVATE-PAIR-{i}" for i in range(440)]
    decisions = {mode: {ids[0]: "NONMATCH"} for mode in filtro.MODES}
    conflicts = {mode: set() for mode in filtro.MODES}
    spec = {"date": "2026-11-05", "input_sha256": {}, "limitations": []}
    monkeypatch.setattr(filtro, "verify_protocol", lambda root: (spec, "commit"))
    monkeypatch.setattr(filtro, "collect", lambda root: (ids, decisions, conflicts, {}))

    def forbidden(*args, **kwargs):
        pytest.fail("Labels must never be loaded when all k are incomplete")

    monkeypatch.setattr(filtro, "load_raw", forbidden)
    out = tmp_path / "aggregate"
    coverage, table, meta = filtro.generate(tmp_path, out, local=tmp_path / "private")
    assert coverage.escalate.any()
    assert table.empty
    assert meta["evaluated_k"] == {mode: [] for mode in filtro.MODES}
    assert set(meta["interpretation"].values()) == {"indisponivel"}
    for name in filtro.OUTPUTS:
        content = (out / name).read_text(encoding="utf-8")
        assert "PRIVATE-PAIR" not in content
        assert "pair_id" not in content
        assert "COMPREC" not in content
        assert "REFREC" not in content


def test_aggregate_outputs_and_stop_consistency():
    coverage = pd.read_csv(filtro.ROOT / "data" / filtro.OUTPUTS[0])
    table = pd.read_csv(filtro.ROOT / "data" / filtro.OUTPUTS[1])
    meta = json.loads((filtro.ROOT / "data" / filtro.OUTPUTS[2]).read_text(encoding="utf-8"))
    assert list(table) == filtro.RESULT_COLUMNS
    assert len(coverage) == 2 * len(filtro.KS)
    assert (coverage.respostas_validas + coverage.pendentes).eq(coverage.k).all()
    assert coverage.completa.eq(coverage.pendentes.eq(0)).all()
    expected = coverage.loc[coverage.completa, ["modo", "k"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(table[["modo", "k"]], expected, check_dtype=False)
    assert meta["post_hoc"] is True
    assert meta["api_calls"] == meta["model_fits"] == 0
    assert "braço P" in meta["response_reuse_warning"]
    for name in filtro.OUTPUTS[:2]:
        frame = pd.read_csv(filtro.ROOT / "data" / name)
        assert not {"pair_id", "COMPREC", "REFREC", "PAR", "SCORE", "nota final"}.intersection(frame)


def private_inputs_available():
    if not filtro.PROTOCOL.exists():
        return False
    spec = yaml.safe_load(filtro.PROTOCOL.read_text(encoding="utf-8"))
    return all(filtro.resolve(DATA_ROOT, name).exists() for name in spec["input_sha256"])


@pytest.mark.skipif(not private_inputs_available(), reason="Private reviewer runs/extract unavailable")
def test_private_regeneration_identical(tmp_path):
    filtro.generate(DATA_ROOT, tmp_path / "out", local=tmp_path / "private")
    for name in filtro.OUTPUTS:
        expected = (filtro.ROOT / "data" / name).read_bytes().replace(b"\r\n", b"\n")
        assert (tmp_path / "out" / name).read_bytes() == expected
