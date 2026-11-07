"""Portable contracts; private-input replay is explicitly optional."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from local_data import DATA_ROOT, requires_files

from scripts.vies_referencia import (
    FP_GRID, PAR_GRID, REPETITIONS, ROOT, SEED, expected_metrics,
    load_inputs, minimum_target_counts, select_changes, simulate, target_counts, turning_points,
)


def test_expected_baseline_and_extreme_scenarios():
    base = SimpleNamespace(aceitos=215, fp=21, perdidos=13, vinc_humano=19)
    observed = expected_metrics(base, 10, 11, 0, 0, 0)
    assert observed["vpp_llm_abaixo"] == 4 / 25
    assert observed["sensibilidade_auto"] == 215 / 247
    all_links = expected_metrics(base, 10, 11, 1, 1, 0)
    assert all_links["vpp_llm_abaixo"] == 1 and all_links["fp"] == 0
    assert all_links["perdidos"] == 13 and all_links["aceitos_vp"] == 236
    removed = expected_metrics(base, 10, 11, 0, 0, .2)
    assert removed["perdidos"] == 10.4 and removed["fp"] == 21.8
    assert np.isclose(removed["vinculos"], 239.8)
    assert np.isclose(removed["aceitos_vp"] + removed["fp"], 236)
    assert np.isclose(removed["aceitos_vp"] + removed["vinculos_humano"] + removed["perdidos"],
                      removed["vinculos"])
    confirmation = SimpleNamespace(aceitos=219, fp=4, perdidos=27, vinc_humano=1)
    assert expected_metrics(confirmation, 0, 4, 0, 0, 0)["vpp_llm_abaixo"] == 8 / 12


def test_fractional_counts_are_unbiased_and_without_replacement():
    group = np.arange(10)
    assert len(select_changes(group, .25, np.arange(10), .2)) == 3
    assert len(select_changes(group, .25, np.arange(10), .8)) == 2
    assert select_changes(group, 0, np.arange(10), 0).size == 0
    assert np.array_equal(select_changes(group, 1, np.arange(10), 0), group)
    assert target_counts(36).tolist() == [9, 23, 36]
    assert target_counts(29).tolist() == [7, 19, 29]
    assert minimum_target_counts(36).tolist() == [9, 24, 36]
    assert minimum_target_counts(29).tolist() == [8, 19, 29]


def test_simulation_paired_differences_reproducible_and_labels_move():
    ranks = {"O1": np.array([1, 2, 3, 4]), "O3": np.array([4, 3, 2, 1]),
             "F4": np.array([2, 1, 4, 3])}
    positives = np.array([0, 2])
    groups = (np.array([1]), np.array([3]), positives)
    baseline = simulate(ranks, positives, groups, (0, 0, 0), 20)
    assert baseline["O1_revisoes_100"].tolist() == [3, 3, 3]
    assert baseline["F4-O1_revisoes_100"].tolist() == [1, 1, 1]
    changed = simulate(ranks, positives, groups, (1, 1, 1), 20)
    assert changed["O1_revisoes_100"].tolist() == [4, 4, 4]
    first = simulate(ranks, positives, groups, (.25, .5, .2), 40)
    second = simulate(ranks, positives, groups, (.25, .5, .2), 40)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert all(v[1] <= v[0] <= v[2] for v in first.values())


def test_grid_turning_point_detects_crossing_not_interpolation():
    rows = [{"modo": "vigilancia", "fp_abaixo": f, "fp_acima": f,
             "par2_nao_vinculo": 0., "metrica": "vpp_llm_abaixo", "mediana": (4 + 21*f)/25}
            for f in FP_GRID]
    turns = turning_points(pd.DataFrame(rows))
    row = turns.loc[turns.eixo.eq("fp_uniforme")].iloc[0]
    assert row.fracao_virada == .5 and row.fracao_anterior == .25


def test_aggregate_outputs_grid_baselines_and_privacy():
    table = pd.read_csv(ROOT / "data/vies_referencia.csv")
    metadata = json.loads((ROOT / "data/vies_referencia_metadata.json").read_text(encoding="utf-8"))
    identifiers = pd.read_csv(ROOT / "data/vies_referencia_identificadores.csv")
    assert len(table[["modo", "fp_abaixo", "fp_acima", "par2_nao_vinculo"]].drop_duplicates()) == 120
    assert set(table.fp_abaixo) == set(FP_GRID) and set(table.par2_nao_vinculo) == set(PAR_GRID)
    assert not table.duplicated(["modo", "fp_abaixo", "fp_acima", "par2_nao_vinculo", "metrica"]).any()
    assert metadata["seed"] == SEED and metadata["repetitions"] == REPETITIONS
    assert metadata["baseline_reproduction"]["passed"]
    assert metadata["cpu_only"] and metadata["llm_calls"] == metadata["model_fits"] == 0
    assert (table.p025 <= table.mediana).all() and (table.mediana <= table.p975).all()
    assert "pair_id" not in table.columns and "pair_id" not in identifiers.columns
    totals = identifiers.loc[identifiers.regiao_limiar_passo.eq("TOTAL")]
    assert totals.n.tolist() == [21, 4, 36, 1000]
    assert totals.sem_conflito_cardinalidade.iloc[:2].tolist() == [20, 4]
    assert (identifiers.ambos <= identifiers.concordancia_maxima_simultanea).all()
    assert (identifiers.ambos <= identifiers.sem_conflito_cardinalidade).all()
    baseline = table.loc[table.modo.eq("vigilancia") & table.fp_abaixo.eq(0)
                         & table.fp_acima.eq(0) & table.par2_nao_vinculo.eq(0)].set_index("metrica")
    assert baseline.loc["O1_revisoes_25", "mediana"] == 51
    assert baseline.loc["O3_revisoes_64", "mediana"] == 125
    assert baseline.loc["O1_revisoes_100", "mediana"] == 2521


POLICY_DIR = Path(os.environ.get("PHD_RESEARCH_POLICY_DIR", ROOT / ".local/arranjo_corte"))
RESPONSE_NAMES = json.loads((ROOT / "data/fp_cardinalidade.json").read_text(encoding="utf-8"))["fontes_sha256"]
RESPONSE_PATHS = [next((base / name for base in (DATA_ROOT, POLICY_DIR.parents[1])
                       if (base / name).is_file()), DATA_ROOT / name)
                  for name in RESPONSE_NAMES if name.startswith("data/sprint3b/")]


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv", DATA_ROOT / "data/pcal_ml_oof.csv",
                DATA_ROOT / "data/pcal_ml_oof.json", POLICY_DIR / "policy_vigilancia_P.csv",
                POLICY_DIR / "policy_confirmacao_P.csv", *RESPONSE_PATHS)
def test_private_replay_reproduces_published_identifier_counts():
    _, _, _, identifiers, _, _ = load_inputs(DATA_ROOT, POLICY_DIR)
    expected = pd.read_csv(ROOT / "data/vies_referencia_identificadores.csv")
    pd.testing.assert_frame_equal(identifiers, expected)
