"""Portable aggregate contracts and synthetic one-hop person-ID propagation."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from local_data import DATA_ROOT, assert_frozen_csv_hash, requires_files

from scripts.propagacao_pessoa import (
    ROOT, analyze, clean, identifier_diagnostics, propagation, summarize,
)


def synthetic():
    raw = pd.DataFrame({
        "REFREC": ["r", "r", "r", "r", "s", "r", "r"],
        "COMPREC": ["a", "b", "c", "d", "e", "f", "a"],
        "C_IDPESSOA": ["01", "01", "x", "", "01", "z", "01"],
        "C_NOVOIDPESSOA": ["old", "new", "old", "", "old", "01", "old"],
        "PAR": [1, 2, 0, 0, 0, 0, 0], "PASSO": [6, 9, 11, 6, 9, 11, 6],
    })
    return raw, np.array([True, False, False, False, False, False, False])


def test_one_hop_same_column_nonempty_and_distinct_notification():
    raw, above = synthetic()
    masks = propagation(raw, above)
    assert masks["R1_C_IDPESSOA"].tolist() == [False, True, False, False, False, False, False]
    assert masks["R1_C_NOVOIDPESSOA"].tolist() == [False, False, True, False, False, False, False]
    assert masks["R1_UNIAO"].tolist() == [False, True, True, False, False, False, False]
    assert masks["R2_REFREC"].tolist() == [False, True, True, True, False, True, True]
    assert clean(pd.Series([None, "  ", " 001 "])).tolist() == ["", "", "001"]


def test_summary_counts_precision_and_step_partition():
    raw, above = synthetic()
    fp = {"vigilancia": np.array([False, False, True, True, False, False, False]),
          "confirmacao": np.zeros(len(raw), dtype=bool)}
    table = summarize(raw, ~above, propagation(raw, above), fp)
    union = table.loc[table.regra.eq("R1_UNIAO")].set_index("passo")
    assert union.loc["TOTAL", "propagados"] == 2
    assert union.loc["TOTAL", "precisao_PAR"] == .5
    assert union.loc["TOTAL", "fp_P_vigilancia_propagados"] == 1
    assert pd.isna(union.loc["6", "precisao_PAR"])
    for column in ("propagados", "par2", "par0", "fp_P_vigilancia_propagados"):
        assert union.loc[["6", "9", "11"], column].sum() == union.loc["TOTAL", column]


def test_diagnostics_deduplicate_candidates_and_do_not_count_missing_agreement():
    raw, _ = synthetic()
    above = np.array([True, True, True, True, True, False, False])
    raw = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)
    above = np.append(above, False)
    diagnostics = identifier_diagnostics(raw, above)
    d = diagnostics["C_IDPESSOA"]
    assert d["notificacoes_total"] == 6 and d["notificacoes_nao_vazias"] == 5
    assert d["pessoas_distintas"] == 3
    assert d["histograma_notificacoes_por_pessoa"] == {"1": 2, "3": 1}
    assert d["refrec_multiplos_comprec"] == 1
    assert d["refrec_multiplos_id_completo"] == d["refrec_multiplos_id_unico"] == 0
    assert d["pessoas_multiplos_comprec_aceitos"] == 1
    assert d["pessoas_multiplos_comprec_refrec_unico"] == 0
    raw.loc[1, "C_IDPESSOA"] = "inconsistent"
    with pytest.raises(ValueError, match="vary within"):
        identifier_diagnostics(raw, above)


def test_local_comprec_reuse_and_sparse_new_identifier_are_not_imputed():
    raw = pd.DataFrame({"PASSO": [6, 6, 9], "COMPREC": ["a", "a", "a"],
                        "REFREC": ["r", "s", "r"], "C_IDLINHA": ["line1", "line1", "line2"],
                        "C_IDPESSOA": ["p", "p", "q"],
                        "C_NOVOIDPESSOA": ["", "new", ""]})
    above = np.array([True, False, False])
    d = identifier_diagnostics(raw, above)
    assert d["C_IDPESSOA"]["notificacoes_total"] == 2
    assert d["C_IDPESSOA"]["pessoas_distintas"] == 2
    assert d["C_NOVOIDPESSOA"]["pares_nao_vazios"] == 1
    assert d["C_NOVOIDPESSOA"]["notificacoes_nao_vazias"] == 1
    assert d["C_NOVOIDPESSOA"]["por_C_IDLINHA"]["completude_pct"] == 50
    assert not propagation(raw, above)["R1_C_NOVOIDPESSOA"].any()


def test_published_aggregates_privacy_hash_and_accounting():
    path = ROOT / "data/propagacao_pessoa.csv"
    table = pd.read_csv(path, dtype={"passo": str})
    meta = json.loads((ROOT / "data/propagacao_pessoa_metadata.json").read_text(encoding="utf-8"))
    assert len(table) == 16 and not table.duplicated(["regra", "passo"]).any()
    assert meta["baseline"] == {"above_PAR1": 211, "below": 61485, "below_PAR2": 36,
                                "P_FP": {"vigilancia": 21, "confirmacao": 4}}
    assert meta["cpu_only"] and meta["llm_calls"] == meta["model_fits"] == 0
    assert set(meta["identifier_diagnostics"]) == {"C_IDPESSOA", "C_NOVOIDPESSOA"}
    assert not {"pair_id", "COMPREC", "REFREC", "C_IDPESSOA", "C_NOVOIDPESSOA"} & set(table.columns)
    assert_frozen_csv_hash(path, meta["output_sha256"]["data/propagacao_pessoa.csv"])
    assert (table.propagados == table.par2 + table.par0).all()
    totals = table.loc[table.passo.eq("TOTAL")].set_index("regra")
    assert totals.loc["R1_UNIAO", ["propagados", "par2", "par0"]].tolist() == [2, 2, 0]
    assert totals.loc["R2_REFREC", ["propagados", "par2", "par0"]].tolist() == [4582, 6, 4576]
    assert meta["identifier_scope_audit"]["COMPREC_com_multiplos_C_IDLINHA"] == 606
    for _, group in table.groupby("regra"):
        total = group.loc[group.passo.eq("TOTAL")].iloc[0]
        steps = group.loc[~group.passo.eq("TOTAL")]
        for column in ("propagados", "par2", "par0", "fp_P_vigilancia_propagados", "fp_P_confirmacao_propagados"):
            assert steps[column].sum() == total[column]
        assert total.par2 <= 36
        assert total.fp_P_vigilancia_propagados <= 21 and total.fp_P_confirmacao_propagados <= 4
        if total.propagados:
            assert np.isclose(total.precisao_PAR, total.par2 / total.propagados)
        else:
            assert pd.isna(total.precisao_PAR)
    for field, d in meta["identifier_diagnostics"].items():
        histogram = d["histograma_notificacoes_por_pessoa"]
        assert sum(histogram.values()) == d["pessoas_distintas"]
        assert sum(int(k) * n for k, n in histogram.items()) == d["notificacoes_nao_vazias"]
        assert 0 <= d["completude_notificacoes_pct"] <= 100


POLICY_DIR = Path(os.environ.get("PHD_RESEARCH_POLICY_DIR", ROOT / ".local/arranjo_corte"))
RESPONSE_NAMES = json.loads((ROOT / "data/fp_cardinalidade.json").read_text(encoding="utf-8"))["fontes_sha256"]
RESPONSE_PATHS = [next((base / name for base in (DATA_ROOT, POLICY_DIR.parents[1])
                       if (base / name).is_file()), DATA_ROOT / name)
                  for name in RESPONSE_NAMES if name.startswith("data/sprint3b/")]


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", DATA_ROOT / "data/pcal_ml_oof.csv",
                DATA_ROOT / "data/pcal_ml_oof.json", POLICY_DIR / "policy_vigilancia_P.csv",
                POLICY_DIR / "policy_confirmacao_P.csv", *RESPONSE_PATHS)
def test_private_replay_matches_aggregate_output():
    table, metadata = analyze(DATA_ROOT, POLICY_DIR)
    expected = pd.read_csv(ROOT / "data/propagacao_pessoa.csv", dtype={"passo": str})
    pd.testing.assert_frame_equal(table, expected)
    frozen = json.loads((ROOT / "data/propagacao_pessoa_metadata.json").read_text(encoding="utf-8"))
    assert metadata["identifier_diagnostics"] == frozen["identifier_diagnostics"]
