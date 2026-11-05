import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gzcmd.intervals import wilson95
from scripts.arranjo_corte_diagnosticos import ARMS, summarize, generate
from scripts.carga_abaixo_corte import aggregate
from local_data import DATA_ROOT, requires_files, requires_reviewer_sources

ROOT = Path(__file__).resolve().parents[1]


def test_lower_bound_uses_ORL_score_and_includes_ties(monkeypatch):
    import scripts.carga_abaixo_corte as workload
    raw = pd.DataFrame({"PASSO": [6, 6, 6, 9, 9, 9], "PAR": [2, 0, 0, 2, 0, 0],
                        "SCORE": ["-5,65", "-5,65", "-6", "-4", "-5", "-3"],
                        "nota final": ["8", "1", "9", "6", "7", "5"]})
    monkeypatch.setattr(workload, "below_cutoff", lambda _: np.ones(6, dtype=bool))
    result, meta = aggregate(raw)
    total = result.loc[result.passo.eq("TOTAL")].set_index("limiar")
    assert total.pares_acima_limiar.to_dict() == {"global": 5, "por_passo": 4}
    assert total.par2_acima_limiar.eq(2).all()
    assert meta["SCORE_min_PAR2"] == -5.65
    assert meta["nota_CdR_min_PAR2"] == 6


def test_precision_and_mechanisms_are_disjoint_and_expected_FP_is_predecision():
    frame = pd.DataFrame({"abaixo": [True] * 7 + [False], "par": [2, 2, 2, 2, 0, 0, 0, 1],
                          "nota": [4, 6, 7, 8, 7, 7, 6, 9],
                          "action": ["NONMATCH", "NONMATCH", "LLM_REVIEW", "LLM_REVIEW",
                                     "LLM_REVIEW", "LLM_REVIEW", "DEFERRED", "MATCH"],
                          "final": ["NONMATCH", "NONMATCH", "NONMATCH", "MATCH", "MATCH",
                                    "UNSURE", "DEFERRED", "MATCH"]})
    precision, rows = summarize(frame, "test", "P", pd.Series(.1, index=frame.index))
    assert (precision["aceitacoes_llm_abaixo"], precision["VP"], precision["FP"]) == (2, 1, 1)
    assert precision["e_FP_previstas"] == pytest.approx(.2)
    assert (precision["ic95_inf"], precision["ic95_sup"]) == wilson95(1, 2)
    table = pd.DataFrame(rows).set_index("mecanismo")
    assert table.pares.sum() == 7 and table.par2.sum() == 4
    assert table.vinculos_perdidos.sum() == 3
    assert table.loc[["guarda_nota_lt_5", "AUTO_NONMATCH_custo", "NONMATCH_LLM"], "vinculos_perdidos"].eq(1).all()
    frame.loc[frame.action.eq("LLM_REVIEW"), "final"] = "NONMATCH"
    empty, _ = summarize(frame, "test", "P", pd.Series(.1, index=frame.index))
    assert empty["aceitacoes_llm_abaixo"] == 0 and np.isnan(empty["VPP"])


def test_versioned_diagnostics_close_and_reproduce_primary_results():
    precision = pd.read_csv(ROOT / "data/arranjo_corte_vpp_llm.csv")
    mechanisms = pd.read_csv(ROOT / "data/arranjo_corte_perdas_mecanismo.csv")
    assert len(precision) == 12 and set(precision.braco) == set(ARMS)
    assert len(mechanisms) == 96
    assert precision.VP.add(precision.FP).equals(precision.aceitacoes_llm_abaixo)
    primary = pd.read_csv(ROOT / "data/arranjo_corte_par2.csv").set_index(["modo", "braco"])
    for key, group in mechanisms.groupby(["modo", "braco"]):
        assert group.pares.sum() == 61485
        assert group.nao_vinculos.sum() == 61449
        assert group.par2.sum() == 36
        assert group.vinculos_perdidos.sum() == primary.loc[key, "perdidos"]
        assert group.loc[group.mecanismo.isin(["MATCH_LLM", "MATCH_motor"]), "par2"].sum() == primary.loc[key, "aceitos"]
    for mode, counts, losses in [("vigilancia", (4, 21, 25), (1, 6, 6)),
                                  ("confirmacao", (8, 4, 12), (1, 22, 4))]:
        row = precision.loc[precision.modo.eq(mode) & precision.braco.eq("P")].iloc[0]
        assert (row.VP, row.FP, row.aceitacoes_llm_abaixo) == counts
        part = mechanisms.loc[mechanisms.modo.eq(mode) & mechanisms.braco.eq("P")].set_index("mecanismo")
        assert tuple(part.loc[["guarda_nota_lt_5", "AUTO_NONMATCH_custo", "NONMATCH_LLM"], "vinculos_perdidos"]) == losses
    for row in precision.itertuples():
        assert row.VPP == pytest.approx(row.VP / row.aceitacoes_llm_abaixo)
        assert (row.ic95_inf, row.ic95_sup) == pytest.approx(wilson95(row.VP, row.aceitacoes_llm_abaixo))
    for table in (precision, mechanisms):
        assert not {"pair_id", "PAR", "REFREC", "COMPREC", "NOME"}.intersection(table.columns)
    workload = pd.read_csv(ROOT / "data/carga_abaixo_corte.csv")
    total = workload.loc[workload.passo.eq("TOTAL")].set_index("limiar")
    assert total.total_abaixo_corte.eq(61485).all()
    assert total.pares_acima_limiar.to_dict() == {"global": 2522, "por_passo": 2156}


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", ROOT / ".local/arranjo_corte/oof.npz",
                ROOT / ".local/arranjo_corte/policy_confirmacao_P.csv")
@requires_reviewer_sources()
def test_diagnostics_regenerate_exact_csvs(tmp_path):
    from scripts.carga_abaixo_corte import generate as workload
    workload(DATA_ROOT, tmp_path)
    generate(DATA_ROOT, tmp_path)
    for name in ("carga_abaixo_corte.csv", "arranjo_corte_vpp_llm.csv", "arranjo_corte_perdas_mecanismo.csv"):
        assert (tmp_path / name).read_bytes() == (ROOT / "data" / name).read_bytes()


def test_provenance_has_no_absolute_paths():
    for name in ("arranjo_corte_metadata.json", "equidade_arranjo_manifest.json"):
        text = (ROOT / "data" / name).read_text(encoding="utf-8")
        assert "Users" not in text and "D:/git" not in text and "D:\\\\git" not in text
        assert "chapter_tex_modified" not in json.loads(text)


def test_nested_provenance_maps_both_clone_roots_and_rejects_external_sources(tmp_path):
    from scripts.arranjo_paths import portable
    root, data_root = tmp_path / "repo", tmp_path / "private_clone"
    nested = {str(data_root / "data/run.json"): [{"source": str(root / "scripts/generator.py")}]}
    assert portable(nested, root, data_root) == {"data/run.json": [{"source": "scripts/generator.py"}]}
    with pytest.raises(ValueError, match="outside repository"):
        portable(str(tmp_path / "outside.json"), root, data_root)
