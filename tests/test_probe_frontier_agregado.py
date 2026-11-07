"""Checks portáveis dos agregados; regeneração privada explicitamente opt-in."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from local_data import DATA_ROOT, requires_files
from scripts import probe_frontier_agregado as probe

ROOT = Path(__file__).resolve().parents[1]


def test_public_schemas_and_no_identifiers_or_field_values():
    for name in probe.ARTIFACTS:
        rows = probe.read_csv(ROOT / "data" / name)
        assert rows
        probe.validate_public(rows)
    for row in ({"pair_id": 1}, {"COMPREC": 1}, {"REFREC": 1},
                {"n": "nome de pessoa"}, {"n": "NaN"}):
        with pytest.raises(ValueError):
            probe.validate_public([row])


def test_probe_strata_sums_and_wilson():
    rows = probe.read_csv(ROOT / "data/probe_frontier_agregado.csv")
    for total in (r for r in rows if r["estrato"] == "TOTAL_AMOSTRA"):
        strata = [r for r in rows if r["model"] == total["model"]
                  and r["modo"] == total["modo"] and r["passada"] == total["passada"]
                  and not r["estrato"].startswith("TOTAL")]
        for column in ("n", "MATCH", "NONMATCH", "UNSURE", "VP", "FP",
                       "PAR2_recuperados", "PAR2_descartados", "PAR2_unsure"):
            assert sum(float(r[column]) for r in strata) == float(total[column])
    for row in rows:
        assert math.isclose(sum(float(row[d]) for d in probe.DECISIONS), float(row["n"]))
        assert float(row["VP"]) + float(row["FP"]) == float(row["MATCH"])
        if row["ponderado"] == "True":
            assert row["VPP_IC95_inf"] == row["VPP_IC95_sup"] == ""
        elif float(row["MATCH"]):
            lo, hi = probe.wilson(float(row["VP"]), float(row["MATCH"]))
            assert float(row["VPP_IC95_inf"]) == pytest.approx(lo)
            assert float(row["VPP_IC95_sup"]) == pytest.approx(hi)


@pytest.mark.parametrize("model,passada,denominator", [
    ("claude-haiku-5-5", "historico", 25),
    ("claude-opus-5-5", "principal", 21),
    ("claude-fable-5-1", "principal", 22),
])
def test_vigilancia_s1_vpp(model, passada, denominator):
    row, = [r for r in probe.read_csv(ROOT / "data/probe_frontier_agregado.csv")
            if (r["model"], r["passada"], r["modo"], r["estrato"])
            == (model, passada, "vigilancia", "S1")]
    assert float(row["VP"]) == 4
    assert float(row["MATCH"]) == denominator
    assert float(row["VPP"]) == pytest.approx(4/denominator)


def test_cohort_totals_partition_and_par2():
    rows = probe.read_csv(ROOT / "data/probe_haiku_sonnet_coorte.csv")
    for total in (r for r in rows if r["faixa"] == "TOTAL"):
        strata = [r for r in rows if r["coorte"] == total["coorte"]
                  and r["model"] == total["model"] and r["modo"] == total["modo"]
                  and r["faixa"] != "TOTAL"]
        for column in ("n", "MATCH", "NONMATCH", "UNSURE", "VP", "FP", "PAR2_total",
                       "PAR2_recuperados", "PAR2_descartados", "PAR2_unsure", "n_concordantes"):
            assert sum(int(r[column]) for r in strata) == int(total[column])
        if total["coorte"] == "original":
            assert int(total["n"]) == {"vigilancia": 1410, "confirmacao": 435}[total["modo"]]
    for row in rows:
        assert sum(int(row[d]) for d in probe.DECISIONS) == int(row["n"])
        assert sum(int(row[c]) for c in ("PAR2_recuperados", "PAR2_descartados", "PAR2_unsure")) == int(row["PAR2_total"])
        assert float(row["agreement"]) == pytest.approx(int(row["n_concordantes"])/int(row["n"]))


def test_agreement_degenerate_and_nontrivial():
    assert probe.agreement(["MATCH"], ["MATCH"]) == (1, 1, None)
    assert probe.agreement(["MATCH", "NONMATCH"], ["MATCH", "UNSURE"]) == (1, .5, 1/3)
    assert probe.band(5) == "[5,7)"
    assert probe.band(7) == "[7,8)"
    assert probe.band(8) == "[8,9)"
    assert probe.band(9) == ">=9"


def test_metadata_hashes_and_lf():
    meta = json.loads((ROOT / "data/probe_frontier_metadata.json").read_text(encoding="utf-8"))
    for name, digest in meta["output_sha256"].items():
        content = (ROOT / "data" / name).read_bytes()
        assert b"\r" not in content
        assert hashlib.sha256(content).hexdigest() == digest
    prompt = (ROOT / meta["prompt_file"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(prompt).hexdigest() == meta["prompt_sha256"]
    assert set(meta["source_sha256"]) == set(probe.SOURCES)
    assert meta["status"] == "exploratório post hoc"
    assert meta["budget_usd"] == 90


def test_cost_accounting():
    costs = probe.read_csv(ROOT / "data/probe_frontier_custos.csv")
    assert sum(float(r["usd_total"]) for r in costs if r["escopo"] == "probe") == pytest.approx(44.4403321)
    for row in costs:
        assert int(row["chamadas_batch"]) + int(row["chamadas_direto"]) == int(row["chamadas"])
        if int(row["pares_revisados"]):
            assert float(row["usd_por_par_revisado"]) == pytest.approx(float(row["usd_total"])/int(row["pares_revisados"]))


PRIVATE = DATA_ROOT / ".local/probe_frontier"
META = json.loads((ROOT / "data/probe_frontier_metadata.json").read_text(encoding="utf-8"))


@requires_files(*(PRIVATE / name for name in probe.SOURCES),
                DATA_ROOT / "data/COMPARADORSEMIDENT.csv",
                *(DATA_ROOT / r["path"] for r in META["historical_sources"]),
                *(DATA_ROOT / Path(r["path"]).parent /
                  ("dossier_field_audit_" + Path(r["path"]).stem.removeprefix("exp2_llm_sync_") + ".json")
                  for r in META["historical_sources"]))
def test_private_regeneration(tmp_path):
    (tmp_path / "gzcmd").mkdir()
    (tmp_path / "gzcmd/gzcmd_v4_llm_prompt.md").write_bytes(
        (ROOT / "gzcmd/gzcmd_v4_llm_prompt.md").read_bytes())
    probe.generate(DATA_ROOT, tmp_path)
    for name in (*probe.ARTIFACTS, "probe_frontier_metadata.json"):
        assert (tmp_path / "data" / name).read_bytes() == (ROOT / "data" / name).read_bytes()
    assert (tmp_path / "docs/plans" / probe.REPORT).read_bytes() == (ROOT / "docs/plans" / probe.REPORT).read_bytes()
