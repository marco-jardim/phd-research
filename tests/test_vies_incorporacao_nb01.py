"""Contratos sintéticos da análise 4 (viés de incorporação sobre o NB01); sem o extrato real."""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.nb01_bootstrap_ic import (
    DIVERGENCIA_AMBIENTE, JSON_KEYS, MODELS, ReproducaoFalhou, point_estimates, prepare,
    rerun_metrics,
)
from scripts.vies_incorporacao_nb01 import (
    COLUMNS, CONCLUSIONS, METRICS, VARIANTS, cell_selections, check_reproduction,
    conclusion_flags, n_to_flip, perturbed_labels, ranks, run_cell, run_grid, strata_from,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic(n=900, positives=60, seed=5):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=bool)
    y[rng.choice(n, positives, replace=False)] = True
    nota = np.where(y, rng.uniform(4, 15, n), rng.uniform(0, 10, n))
    nota[rng.choice(n, 5, replace=False)] = np.nan
    preds = {}
    for i, name in enumerate(MODELS):
        score = np.round(np.clip(y * 0.6 + rng.normal(0.2, 0.12 + 0.05 * i, n), 0, 1), 2)
        preds[name] = (y.astype(int), (score >= 0.5).astype(int), score)
    return y, nota, preds


def test_strata_follow_label_and_note_bands():
    y, nota, _ = synthetic()
    s = strata_from(y, nota)
    assert y[s["idx_a"]].all() and (nota[s["idx_a"]] >= 9).all()
    assert (~y[s["idx_b"]]).all() and ((nota[s["idx_b"]] >= 5) & (nota[s["idx_b"]] < 9)).all()
    assert s["n_sem_nota"] == 5 and s["n_pos"] == int(y.sum())
    # fronteiras: 9 entra em A, 5 entra em B, 9 não entra em B
    s2 = strata_from(np.array([1, 0, 0, 1]), np.array([9.0, 5.0, 9.0, 8.99]))
    assert s2["idx_a"].tolist() == [0] and s2["idx_b"].tolist() == [1]


def test_n_to_flip_rounds_half_up():
    assert n_to_flip(0.0, 74) == 0
    assert n_to_flip(0.05, 74) == 4  # 3,7
    assert n_to_flip(0.5, 5) == 3  # 2,5 para cima


def test_selection_does_not_depend_on_predictions():
    # cell_selections não recebe predições; assinatura e resultado idênticos
    # para quaisquer predições com os mesmos rótulos e notas.
    params = set(inspect.signature(cell_selections).parameters)
    assert params == {"strata", "k_a", "k_b", "variant", "replicas", "seed"}
    y, nota, _ = synthetic()
    s = strata_from(y, nota)
    for variant in VARIANTS:
        one = list(cell_selections(s, 3, 4, variant, 10, 1))
        two = list(cell_selections(s, 3, 4, variant, 10, 1))
        for (a1, b1), (a2, b2) in zip(one, two, strict=True):
            assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_selection_independent_of_other_cells_and_seed_sensitive():
    y, nota, _ = synthetic()
    s = strata_from(y, nota)
    first = [(a.tolist(), b.tolist()) for a, b in cell_selections(s, 3, 4, "aleatoria", 5, 9)]
    list(cell_selections(s, 2, 2, "aleatoria", 5, 9))  # outra célula não altera esta
    again = [(a.tolist(), b.tolist()) for a, b in cell_selections(s, 3, 4, "aleatoria", 5, 9)]
    other = [(a.tolist(), b.tolist()) for a, b in cell_selections(s, 3, 4, "aleatoria", 5, 10)]
    assert first == again and first != other


def test_perturbation_only_changes_labels_in_the_right_stratum():
    y, nota, _ = synthetic()
    s = strata_from(y, nota)
    for variant in VARIANTS:
        for sel_a, sel_b in cell_selections(s, 4, 6, variant, 20, 3):
            assert len(set(sel_a)) == 4 and len(set(sel_b)) == 6
            assert set(sel_a) <= set(s["idx_a"]) and set(sel_b) <= set(s["idx_b"])
            y_new = perturbed_labels(y, sel_a, sel_b)
            changed = np.flatnonzero(y_new != y)
            assert set(changed) == set(sel_a) | set(sel_b)
            assert not y_new[sel_a].any() and y_new[sel_b].all()
            assert y_new.sum() == y.sum() - 4 + 6
    assert perturbed_labels(y, np.array([], dtype=int), np.array([], dtype=int)).tolist() == y.tolist()


def test_highest_note_variant_picks_top_of_the_band():
    y, nota, _ = synthetic()
    s = strata_from(y, nota)
    k = 5
    band_sorted = np.sort(s["nota_b"])[::-1]
    for _, sel_b in cell_selections(s, 0, k, "nota_mais_alta", 10, 2):
        assert np.sort(nota[sel_b])[::-1].tolist() == band_sorted[:k].tolist()


def test_a_b_zero_reproduces_unperturbed_metrics_exactly():
    y, nota, preds = synthetic()
    preps = {n: prepare(*preds[n]) for n in MODELS}
    table = run_grid(preps, y, nota, (0.0,), (0.0,), VARIANTS, 7, 1)
    m = table[table.tipo == "metrica"]
    assert len(m) == len(VARIANTS) * len(MODELS) * len(METRICS)
    for row in m.itertuples():
        point = point_estimates(preps[row.classificador])[row.metrica]
        assert row.mediana == point and row.p2_5 == point and row.p97_5 == point
        assert row.ponto_sem_perturbacao_rerun == point
    # e as proporções das conclusões são 0 ou 1 (nenhuma perturbação)
    assert set(table[table.tipo == "conclusao"].proporcao_replicas) <= {0.0, 1.0}


def test_run_cell_matches_sklearn_on_perturbed_labels():
    from sklearn.metrics import average_precision_score, f1_score

    y, nota, preds = synthetic()
    preps = {n: prepare(*preds[n]) for n in MODELS}
    s = strata_from(y, nota)
    out = run_cell(preps, y, s, 3, 5, "aleatoria", 4, 11)
    for r, (sel_a, sel_b) in enumerate(cell_selections(s, 3, 5, "aleatoria", 4, 11)):
        y_new = perturbed_labels(y, sel_a, sel_b).astype(int)
        for name in ("Random Forest", "SVM (RBF)"):
            _, pred, score = preds[name]
            assert out[name]["F1"][r] == pytest.approx(f1_score(y_new, pred))
            assert out[name]["AUC-PR"][r] == pytest.approx(average_precision_score(y_new, score), abs=1e-12)


def test_ranks_and_conclusion_flags():
    values = np.array([[0.9, 0.9, 0.5], [0.1, 0.2, 0.3]])
    assert ranks(values).tolist() == [[1, 1, 3], [3, 2, 1]]
    names = list(MODELS)
    base = {n: {"F1": np.array([0.5, 0.5]), "AUC-PR": np.array([0.5, 0.5])} for n in names}
    for n in ("Random Forest", "RF + SMOTE", "Gradient Boosting", "MLP Neural Network"):
        base[n]["F1"] = np.array([0.9, 0.4])
    base["Random Forest"]["F1"] = np.array([0.95, 0.4])  # 2ª réplica: empata com os demais, não maior
    base["RF + SMOTE"]["AUC-PR"] = np.array([0.9, 0.4])
    flags = conclusion_flags(base, names)
    assert flags["RF_maior_F1"].tolist() == [True, False]
    assert flags["RFSMOTE_maior_AUCPR"].tolist() == [True, False]
    assert flags["grupo_cima_acima_baixo_F1"].tolist() == [True, False]


def test_grid_skips_insufficient_stratum_and_flags_turning_point():
    y, nota, preds = synthetic()
    preps = {n: prepare(*preds[n]) for n in MODELS}
    rng = np.random.default_rng(0)
    nota = np.where(y, nota, np.where(rng.random(len(y)) < 0.03, 6.0, 1.0))  # poucos não pares em [5, 9)
    s = strata_from(y, nota)
    # b = 100% de 60 pares verdadeiros excede o estrato B
    assert len(s["idx_b"]) < 60
    table = run_grid(preps, y, nota, (0.0, 0.2), (0.0, 1.0), ("aleatoria",), 5, 2)
    cells = table[table.tipo == "celula"]
    skipped = cells[cells.status == "ignorada_estrato_insuficiente"]
    assert len(skipped) == 2 and (skipped.b == 1.0).all()
    assert not ((table.tipo == "metrica") & (table.b == 1.0)).any()
    turns = table[table.tipo == "virada"]
    assert set(turns.conclusao) == set(CONCLUSIONS)
    assert set(turns.status) <= {"virada_na_grade", "sem_virada_na_grade", "nao_vale_no_baseline"}


def test_deterministic_and_seed_sensitive():
    y, nota, preds = synthetic()
    preps = {n: prepare(*preds[n]) for n in MODELS}
    a = run_grid(preps, y, nota, (0.0, 0.2), (0.0, 0.2), VARIANTS, 6, 4)
    b = run_grid(preps, y, nota, (0.0, 0.2), (0.0, 0.2), VARIANTS, 6, 4)
    c = run_grid(preps, y, nota, (0.0, 0.2), (0.0, 0.2), VARIANTS, 6, 5)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_output_is_aggregate_only():
    y, nota, preds = synthetic()
    preps = {n: prepare(*preds[n]) for n in MODELS}
    table = run_grid(preps, y, nota, (0.0, 0.1), (0.0, 0.1), ("aleatoria",), 5, 1)
    assert list(table.columns) == list(COLUMNS)
    forbidden = {"pair_id", "REFREC", "COMPREC", "indice", "linha"}
    assert not forbidden & set(table.columns)
    assert table.tipo.isin({"celula", "metrica", "posicao", "ordem", "conclusao", "virada"}).all()
    # nenhuma célula guarda listas de posições: tamanho da tabela só depende da grade
    assert len(table) < 500
    for col in ("texto", "empates", "status", "classificador"):
        assert not table[col].dropna().astype(str).str.contains(r"\[|array").any()


def reference_for(preds):
    n = len(next(iter(preds.values()))[0])
    pos = int(next(iter(preds.values()))[0].sum())
    results = [{"Modelo": name, **{JSON_KEYS[m]: round(v, 4) for m, v in rerun_metrics(*preds[name]).items()}}
               for name in MODELS]
    return {"n_test": n, "pos_test": pos, "all_results": results}


def test_reproduction_ok_and_listed_divergence_does_not_abort():
    _, _, preds = synthetic()
    captured = {"preds": preds, "n_test": 900, "pos_test": 60}
    reference = reference_for(preds)
    status, _ = check_reproduction(captured, reference)
    assert all(s["reproduz"] for s in status.values())
    lr = "Logistic Regression"
    assert lr in DIVERGENCIA_AMBIENTE
    for row in reference["all_results"]:
        if row["Modelo"] == lr:
            row["F1-Score"] = round(row["F1-Score"] + 0.05, 4)
    status, _ = check_reproduction(captured, reference)
    assert status[lr]["reproduz"] is False and status[lr]["motivo"] == DIVERGENCIA_AMBIENTE[lr]


def test_reproduction_failure_outside_listed_models_aborts():
    _, _, preds = synthetic()
    captured = {"preds": preds, "n_test": 900, "pos_test": 60}
    reference = reference_for(preds)
    assert "Random Forest" not in DIVERGENCIA_AMBIENTE
    for row in reference["all_results"]:
        if row["Modelo"] == "Random Forest":
            row["Recall"] = round(row["Recall"] + 0.02, 4)
    with pytest.raises(ReproducaoFalhou, match="Random Forest Sensibilidade"):
        check_reproduction(captured, reference)
    captured["pos_test"] = 61  # partição diferente também aborta
    with pytest.raises(ReproducaoFalhou, match="pos_test"):
        check_reproduction(captured, reference_for(preds))


def test_generated_artifacts_are_aggregate_if_present():
    path = ROOT / "data" / "vies_incorporacao_nb01.csv"
    meta_path = ROOT / "data" / "vies_incorporacao_nb01_metadata.json"
    if not path.exists() or not meta_path.exists():
        pytest.skip("artefato ainda não gerado")
    table = pd.read_csv(path)
    assert list(table.columns) == list(COLUMNS)
    assert not {"pair_id", "REFREC"} & set(table.columns)
    base = table[(table.tipo == "metrica") & (table.k_a == 0) & (table.k_b == 0)]
    assert (base.p2_5 <= base.mediana).all() and (base.mediana <= base.p97_5).all()
    assert (base.mediana == base.ponto_sem_perturbacao_rerun).all()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["llm_calls"] == 0 and "limitacoes" in meta
    assert "pair_id" not in json.dumps(meta)
