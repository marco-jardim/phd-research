"""Sensibilidade de O3 ao estimador (XGBoost versus Random Forest), só CPU.

Análise exploratória pós-hoc, autorizada pela autora em 2026-11-09. A recomendação
da tese é ordenar a região abaixo do corte do OpenRecLink pela probabilidade
calibrada fora da amostra (O3, XGBoost aninhado com calibração isotônica 5x5,
partição por REFREC, seed 42; data/pcal_ml_oof.csv). Este script troca SÓ o
estimador por Random Forest (200 árvores, profundidade 10, class_weight balanced,
random_state 42, como o RF do NB01 em scripts/run_nb01_uniform.py), mantendo os 30
atributos, as partições externas e internas, a seed e a calibração isotônica
aninhada de gzcmd/ml_oof.py, e reavalia a ordenação abaixo do corte.

Ordem de execução (qualquer falha de reprodução termina com código 1, antes de
ajustar o RF):
  1. O1 e O3-XGBoost reproduzem data/comparador_mesma_carga.csv (125, 29, 23.443,
     22 e 6 para O3-XGBoost);
  2. o mesmo código com XGBoost reproduz data/pcal_ml_oof.csv (tolerância 1e-9);
  3. gera data/pcal_ml_oof_rf.csv/.json (fingerprint como o do XGBoost);
  4. métricas de ordenação, bootstrap agrupado por REFREC (mesmo procedimento e
     semente de scripts/ordenacao_bootstrap_o3_o1.py) e desempates aleatórios
     (como scripts/ordenacao_empates_o3.py);
  5. grava data/o3_estimador_rf.csv e data/o3_estimador_rf_metadata.json.

Não altera gzcmd/ml_oof.py (o fingerprint de data/pcal_ml_oof.json depende do hash
desse arquivo) nem sobrescreve artefatos existentes. Só agregados, escores e
pair_id: nenhuma PII. Lê data/COMPARADORSEMIDENT.csv (não versionado).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256  # noqa: E402
from gzcmd.loader import resolve_prefixed_column  # noqa: E402
from gzcmd.ml_oof import (  # noqa: E402
    METHOD as METHOD_XGB,
    N_SPLITS,
    SEED as OOF_SEED,
    _classifier,
    nested_oof_probabilities,
    numeric_features,
    pair_ids,
)
from scripts.comparador_mesma_carga import (  # noqa: E402
    align_original,
    cumulative_counts,
    load_raw,
    numeric,
)
from scripts.fusao_robustez import (  # noqa: E402
    FRACTIONS,
    SEED as BOOTSTRAP_SEED,
    TARGETS,
    resampled_queues,
    reviews,
)
from scripts.ordenacao_bootstrap_o3_o1 import (  # noqa: E402
    BASE as BOOT_BASE,
    FUSED as BOOT_FUSED,
    REPLICAS as BOOTSTRAP_REPLICAS,
    summarize_o3_o1,
)
from scripts.ordenacao_empates_o3 import (  # noqa: E402
    REFERENCE_MAP,
    SEED as TIE_SEED,
    atomic_write,
    quantities,
    run_replicas,
    summarize as summarize_ties,
    tie_summary,
)
from scripts.sensibilidade_o3_macd import queue  # noqa: E402

METHOD_RF = "nested-rf-isotonic-5x5-v1"
TIE_REPLICAS = 200
TOLERANCE = 1e-9
SAFETY = 1.5
OUT_OOF = "data/pcal_ml_oof_rf.csv"
OUT_CSV = "data/o3_estimador_rf.csv"
OUT_META = "data/o3_estimador_rf_metadata.json"
REFERENCE = "data/comparador_mesma_carga.csv"
BOOTSTRAP_REFERENCE = "data/ordenacao_bootstrap_o3_o1.csv"
RF_PARAMS = {"n_estimators": 200, "max_depth": 10, "class_weight": "balanced"}
BELOW = "abaixo_corte"
COLUMNS = ("bloco", "estimador", "comparacao", "modo", "meta", "grandeza", "estatistica", "valor")
ORDERINGS = ("O1", "O3_XGB", "O3_RF")
REFERENCE_NAMES = {"O1": "O1_SCORE", "O3_XGB": "O3_p_cal_ml"}
COMPARISONS = (("O3_RF", "O1"), ("O3_RF", "O3_XGB"), ("O3_XGB", "O1"))
# Conferência obrigatória contra data/comparador_mesma_carga.csv e a tese.
EXPECTED = {
    "O3_XGB": {
        ("vigilancia", "k_min_equivalente_P"): 125,
        ("vigilancia", "k_min_36"): 23443,
        ("vigilancia", "par2_topk_K1_humana"): 22,
        ("confirmacao", "k_min_equivalente_P"): 29,
        ("confirmacao", "k_min_36"): 23443,
        ("confirmacao", "par2_topk_K1_humana"): 6,
    },
    "O1": {
        ("vigilancia", "k_min_equivalente_P"): 501,
        ("vigilancia", "k_min_36"): 2521,
        ("vigilancia", "par2_topk_K1_humana"): 19,
        ("confirmacao", "k_min_equivalente_P"): 51,
        ("confirmacao", "k_min_36"): 2521,
        ("confirmacao", "par2_topk_K1_humana"): 3,
    },
}


class ReproductionError(RuntimeError):
    """Uma referência existente não foi reproduzida; a análise não prossegue."""


# --------------------------------------------------------------------------- estimadores


class SerialPredictRandomForest(RandomForestClassifier):
    """RandomForestClassifier com predict_proba serial, bit a bit determinístico.

    O ajuste com n_jobs=-1 é determinístico (cada árvore tem o seu random_state,
    sorteado antes da paralelização), mas predict_proba soma as árvores em threads
    na ordem de chegada; a soma em ponto flutuante varia na ordem de 1e-16 e isso
    pode mudar empates e a fila O3-RF. A predição roda, portanto, com n_jobs=1.
    """

    def predict_proba(self, X):
        n_jobs = self.n_jobs
        self.n_jobs = 1
        try:
            return super().predict_proba(X)
        finally:
            self.n_jobs = n_jobs


def rf_classifier(y, seed):
    """Random Forest com os hiperparâmetros do RF do NB01; y não é usado.

    Ajuste com n_jobs=-1; predict_proba com n_jobs=1 (ver SerialPredictRandomForest).
    """
    return SerialPredictRandomForest(**RF_PARAMS, random_state=seed, n_jobs=-1)


# factory None = comportamento padrão de gzcmd.ml_oof (XGBoost), sem tocar no padrão.
ESTIMATORS = {
    "xgb": {"method": METHOD_XGB, "factory": None},
    "rf": {"method": METHOD_RF, "factory": rf_classifier},
}


def model_factory(name):
    if name not in ESTIMATORS:
        raise ValueError(f"Estimador desconhecido: {name}")
    return ESTIMATORS[name]["factory"] or _classifier


def oof_probabilities(x, y, groups, estimator="xgb"):
    """Mesmo procedimento aninhado 5x5 por REFREC de gzcmd.ml_oof; só o estimador muda."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"Estimador desconhecido: {estimator}")
    return nested_oof_probabilities(
        x, y, split_by="refrec", groups=groups,
        model_factory=ESTIMATORS[estimator]["factory"],
    )


def features_from_raw(raw):
    """Atributos, rótulo (PAR em {1,2}) e grupos REFREC, como gzcmd.ml_oof.fit_xgb_oof."""
    x = numeric_features(raw)
    y = pd.to_numeric(raw["PAR"], errors="coerce").fillna(0).isin([1, 2]).astype(int).to_numpy()
    column = resolve_prefixed_column(raw, "REFREC")
    if column is None:
        raise ValueError("Missing REFREC group column")
    groups = raw[column].astype("string").str.strip()
    if groups.isna().any() or groups.eq("").any():
        raise ValueError("Missing group identity")
    return x, y, groups.to_numpy(dtype=str)


def oof_from_raw(raw, estimator="xgb"):
    x, y, groups = features_from_raw(raw)
    return oof_probabilities(x, y, groups, estimator)


def check_xgb_reproduction(p, folds, expected_p, expected_folds, tolerance=TOLERANCE):
    expected_p = np.asarray(expected_p, dtype=float)
    result = {
        "max_abs_difference": float(np.max(np.abs(np.asarray(p) - expected_p))),
        "folds_identical": bool(np.array_equal(folds, expected_folds)),
        "tolerance": tolerance,
    }
    result["passed"] = result["folds_identical"] and result["max_abs_difference"] <= tolerance
    if not result["passed"]:
        raise ReproductionError(
            "O código com XGBoost não reproduz data/pcal_ml_oof.csv: " + json.dumps(result)
        )
    return result


# --------------------------------------------------------------------------- avaliação


def evaluate_ordering(ids, below, par2, score):
    """Grandezas de data/comparador_mesma_carga.csv, mais AUC e AP abaixo do corte."""
    score = np.asarray(score, dtype=float)
    counts = cumulative_counts(ids[below], score[below], par2[below])
    result = dict(quantities(counts))
    result[(BELOW, "auc_abaixo")] = float(roc_auc_score(par2[below], score[below]))
    result[(BELOW, "ap_abaixo")] = float(average_precision_score(par2[below], score[below]))
    return result


def verify_reference(name, values, reference, expected=None):
    """Compara com os valores fixos da tese e com data/comparador_mesma_carga.csv."""
    expected = EXPECTED[name] if expected is None else expected
    for (mode, grandeza), want in expected.items():
        got = values[(mode, grandeza)]
        if got != want:
            raise ReproductionError(f"{name} {mode}/{grandeza}: obtido {got}, esperado {want}")
        load, column = REFERENCE_MAP[grandeza]
        rows = reference.loc[
            reference.ordenacao.eq(REFERENCE_NAMES[name])
            & reference.modo.eq(mode)
            & reference.carga.eq(load)
        ]
        if len(rows) != 1:
            raise ReproductionError(f"Linha de referência ausente: {name}/{mode}/{load}")
        if int(rows.iloc[0][column]) != got:
            raise ReproductionError(
                f"{name} {mode}/{grandeza}: obtido {got}, comparador {int(rows.iloc[0][column])}"
            )
    return True


def build_universe(ids, below, par2, groups, scores):
    """Filas (índices na ordem lexical dos hashes), rótulos e grupos, como fusao_filas."""
    lexical = np.sort(ids[below])
    if len(np.unique(lexical)) != len(lexical):
        raise ValueError("Universo abaixo do corte sem pair_id único")
    labels = pd.Series(par2[below], index=ids[below]).loc[lexical].to_numpy()
    queues = {
        name: np.searchsorted(lexical, queue(ids, score, below))
        for name, score in scores.items()
    }
    aligned = np.argsort(ids[below])
    codes, uniques = pd.factorize(groups[below][aligned], sort=True)
    return queues, labels, codes, len(uniques)


def bootstrap_runs(queues, labels, codes, group_count, replicas, seed):
    """Mesmo sorteio de bootstrap_o3_o1_once (uma rng.integers por réplica), todas as filas.

    Com a mesma semente, as réplicas são as mesmas de data/ordenacao_bootstrap_o3_o1.csv;
    todas as comparações usam as mesmas amostras (pareadas).
    """
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(replicas):
        draws = rng.integers(0, group_count, size=group_count)
        counts = np.bincount(draws, minlength=group_count)
        sampled, original = resampled_queues(queues, counts[codes])
        sampled_labels = labels[original]
        total = int(sampled_labels.sum())
        if not total:
            continue
        targets = TARGETS if total >= 36 else tuple(int(np.ceil(f * total)) for f in FRACTIONS)
        runs.append((reviews(sampled, sampled_labels, targets), total, len(original)))
    return runs


def summarize_comparisons(runs, comparisons=COMPARISONS):
    """summarize_o3_o1 para cada par (primeiro menos segundo; negativo favorece o primeiro)."""
    frames = []
    for first, second in comparisons:
        sub = [({BOOT_FUSED: r[0][first], BOOT_BASE: r[0][second]}, r[1], r[2]) for r in runs]
        table = summarize_o3_o1(sub)
        table["comparacao"] = f"{first}-{second}"
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def verify_bootstrap_reference(table, reference_path, replicas, seed):
    """O3_XGB-O1 tem de reproduzir data/ordenacao_bootstrap_o3_o1.csv (mesma semente)."""
    path = Path(reference_path)
    if replicas != BOOTSTRAP_REPLICAS or seed != BOOTSTRAP_SEED or not path.exists():
        return {"checked": False, "reason": "réplicas/semente diferentes do arquivo de referência ou arquivo ausente"}
    reference = pd.read_csv(path)
    mine = table.loc[table.comparacao.eq("O3_XGB-O1")].reset_index(drop=True)
    numeric_columns = [c for c in reference.columns if c not in ("comparacao", "proporcao_F4_menor_ambas")]
    if len(mine) != len(reference) or not np.allclose(
        mine[numeric_columns].to_numpy(dtype=float),
        reference[numeric_columns].to_numpy(dtype=float), rtol=0, atol=1e-9,
    ):
        raise ReproductionError("Bootstrap O3_XGB-O1 não reproduz data/ordenacao_bootstrap_o3_o1.csv")
    return {"checked": True, "file": BOOTSTRAP_REFERENCE, "atol": 1e-9}


# --------------------------------------------------------------------------- saída


def long_table(metrics, bootstrap, ties):
    rows = []

    def add(**kw):
        rows.append({c: kw.get(c, "") for c in COLUMNS})

    for estimator, values in metrics.items():
        for (mode, grandeza), value in values.items():
            add(bloco="ordenacao", estimador=estimator, modo=mode, grandeza=grandeza,
                estatistica="valor", valor=value)
    stats = {"replicas_validas": "replicas_validas",
             "proporcao_fusao_menor": "proporcao_primeiro_menor"}
    for r in bootstrap.to_dict("records"):
        base = {"bloco": "bootstrap", "comparacao": r["comparacao"], "meta": int(r["meta_nominal"])}
        for column, name in stats.items():
            add(**base, grandeza="replicas" if name == "replicas_validas" else "frequencia",
                estatistica=name, valor=r[column])
        for grandeza in ("diferenca_revisoes", "reducao_percentual"):
            for stat in ("mediana", "ic95_inf", "ic95_sup"):
                add(**base, grandeza=grandeza, estatistica=stat, valor=r[f"{grandeza}_{stat}"])
    if ties is not None:
        for r in ties.to_dict("records"):
            for stat, column in (("original", "valor_original"), ("minimo", "minimo"),
                                 ("p2_5", "p2_5"), ("mediana", "mediana"),
                                 ("p97_5", "p97_5"), ("maximo", "maximo")):
                add(bloco="empates", estimador="O3_RF", modo=r["modo"], grandeza=r["grandeza"],
                    estatistica=stat, valor=r[column])
    return pd.DataFrame(rows, columns=list(COLUMNS))


def write_json(path, obj):
    atomic_write(path, lambda p: p.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n"))


def rf_fingerprint(source, sklearn_version):
    import hashlib

    return {
        "method": METHOD_RF, "seed": OOF_SEED, "folds": N_SPLITS, "split_by": "refrec",
        "source_sha256": sha256(source),
        "code_sha256": hashlib.sha256((ROOT / "gzcmd/ml_oof.py").read_bytes()).hexdigest(),
        "sklearn": sklearn_version, "estimator": "RandomForestClassifier",
        "estimator_params": {**RF_PARAMS, "random_state": OOF_SEED, "n_jobs": -1},
        "predict_proba_n_jobs": 1,
    }


def write_oof_rf(path, original, p, folds, features, n_groups, fingerprint):
    """Mesmo formato de data/pcal_ml_oof.csv; nota e p_cal_platt não dependem do estimador."""
    result = pd.DataFrame({
        "pair_id": original.pair_id.to_numpy(), "nota": original.nota.to_numpy(),
        "p_cal_platt": original.p_cal_platt.to_numpy(), "p_cal_ml": p, "fold": folds,
    })
    atomic_write(path, lambda t: result.to_csv(t, index=False, lineterminator="\n"))
    meta = {"fingerprint": fingerprint, "features": list(features), "n_groups": int(n_groups),
            "csv_sha256": sha256(path),
            "note": "p_cal_ml = Random Forest aninhado com calibração isotônica; nota e "
                    "p_cal_platt copiados de data/pcal_ml_oof.csv (independem do estimador)."}
    write_json(Path(path).with_suffix(".json"), meta)
    return result


def reuse_rf(path, fingerprint, original):
    path = Path(path)
    meta_path = path.with_suffix(".json")
    if not (path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fingerprint") != fingerprint or meta.get("csv_sha256") != sha256(path):
        return None
    saved = pd.read_csv(path, dtype=str).set_index("pair_id").loc[original.pair_id]
    return (saved.p_cal_ml.astype(float).to_numpy(), saved.fold.astype(int).to_numpy())


# --------------------------------------------------------------------------- execução


def load_universe(data_root):
    source = Path(data_root) / "data/COMPARADORSEMIDENT.csv"
    original_path = Path(data_root) / "data/pcal_ml_oof.csv"
    raw = load_raw(source)
    below = below_cutoff(raw)
    original = align_original(raw, original_path, source)
    meta = json.loads(original_path.with_suffix(".json").read_text(encoding="utf-8"))
    x, y, groups = features_from_raw(raw)
    if list(x) != meta["features"] or len(x.columns) != 30:
        raise ValueError("Allowlist original de 30 atributos divergiu")
    par = numeric(raw.PAR)
    if not par.loc[below].isin([0, 2]).all():
        raise ValueError("Below-cutoff labels must be PAR=0 or PAR=2")
    return {
        "source": source, "original_path": original_path, "raw": raw, "below": below,
        "original": original, "x": x, "y": y, "groups": groups,
        "ids": pair_ids(raw).to_numpy(), "par2": (par == 2).to_numpy(),
        "o1": numeric(raw.SCORE).to_numpy(),
    }


def estimate_minutes(x, y, groups, bootstrap_pilot=None, replicas=BOOTSTRAP_REPLICAS):
    """Um fit-piloto interno por estimador; 30 fits aninhados (5 externos x 5+1)."""
    outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=OOF_SEED)
    train, _ = next(outer.split(x, y, groups))
    inner = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=OOF_SEED + 1)
    fit, pred = next(inner.split(x.iloc[train], y[train], groups[train]))
    pilot = {}
    for name in ESTIMATORS:
        start = time.perf_counter()
        model = model_factory(name)(y[train[fit]], OOF_SEED)
        model.fit(x.iloc[train[fit]], y[train[fit]])
        model.predict_proba(x.iloc[train[pred]])
        pilot[name] = time.perf_counter() - start
    result = {
        "pilot_fit_seconds": pilot, "nested_fits_per_estimator": N_SPLITS * (N_SPLITS + 1),
        "safety_factor": SAFETY,
        "estimated_minutes": {k: v * N_SPLITS * (N_SPLITS + 1) * SAFETY / 60 for k, v in pilot.items()},
    }
    if bootstrap_pilot is not None:
        result["bootstrap_pilot_seconds_per_replica"] = bootstrap_pilot
        result["estimated_minutes"]["bootstrap"] = bootstrap_pilot * replicas * SAFETY / 60
    result["estimated_minutes"]["total"] = sum(result["estimated_minutes"].values())
    return result


def generate(data_root=ROOT, out_dir=None, *, replicas=BOOTSTRAP_REPLICAS, seed=BOOTSTRAP_SEED,
             tie_replicas=TIE_REPLICAS, tie_seed=TIE_SEED, tolerance=TOLERANCE,
             reuse=False, estimate_only=False):
    import sklearn
    import xgboost

    started = time.perf_counter()
    out_dir = Path(out_dir) if out_dir is not None else Path(data_root) / "data"
    u = load_universe(data_root)
    ids, below, par2, groups = u["ids"], u["below"], u["par2"], u["groups"]
    xgb_scores = u["original"].p_cal_ml.to_numpy()

    # 1. Referências existentes (barato): falha antes de qualquer ajuste de modelo.
    reference = pd.read_csv(Path(data_root) / REFERENCE)
    metrics = {"O1": evaluate_ordering(ids, below, par2, u["o1"]),
               "O3_XGB": evaluate_ordering(ids, below, par2, xgb_scores)}
    for name in ("O1", "O3_XGB"):
        verify_reference(name, metrics[name], reference)
    print("O1 e O3-XGBoost reproduzem data/comparador_mesma_carga.csv.", flush=True)

    pilot_queues, labels, codes, group_count = build_universe(
        ids, below, par2, groups, {"O1": u["o1"], "O3_XGB": xgb_scores, "O3_RF": xgb_scores})
    rng = np.random.default_rng(seed)
    pilot_start = time.perf_counter()
    for _ in range(5):
        draws = rng.integers(0, group_count, size=group_count)
        resampled_queues(pilot_queues, np.bincount(draws, minlength=group_count)[codes])
    boot_pilot = (time.perf_counter() - pilot_start) / 5
    estimate = estimate_minutes(u["x"], u["y"], groups, boot_pilot, replicas)
    print("Estimativa de tempo: " + json.dumps(estimate), flush=True)
    if estimate_only:
        return estimate, None

    # 2. O mesmo código com XGBoost reproduz data/pcal_ml_oof.csv.
    t0 = time.perf_counter()
    p_xgb, folds_xgb = oof_probabilities(u["x"], u["y"], groups, "xgb")
    reproduction = check_xgb_reproduction(
        p_xgb, folds_xgb, xgb_scores, u["original"].fold.to_numpy(), tolerance)
    reproduction["seconds"] = time.perf_counter() - t0
    print("XGBoost reproduz data/pcal_ml_oof.csv: " + json.dumps(reproduction), flush=True)

    # 3. Random Forest, mesmo procedimento.
    fingerprint = rf_fingerprint(u["source"], sklearn.__version__)
    rf_path = out_dir / Path(OUT_OOF).name
    cached = reuse_rf(rf_path, fingerprint, u["original"]) if reuse else None
    t0 = time.perf_counter()
    if cached is None:
        p_rf, folds_rf = oof_probabilities(u["x"], u["y"], groups, "rf")
    else:
        p_rf, folds_rf = cached
        print("Random Forest OOF reutilizado de " + str(rf_path), flush=True)
    rf_seconds = time.perf_counter() - t0
    if not np.array_equal(folds_rf, u["original"].fold.to_numpy()):
        raise ReproductionError("Folds do RF divergem do original")
    if not (np.isfinite(p_rf).all() and ((p_rf >= 0) & (p_rf <= 1)).all()):
        raise ValueError("Probabilidades do RF inválidas")
    if cached is None:
        write_oof_rf(rf_path, u["original"], p_rf, folds_rf, u["x"].columns,
                     len(np.unique(groups)), fingerprint)

    # 4. Avaliação, bootstrap e desempates.
    metrics["O3_RF"] = evaluate_ordering(ids, below, par2, p_rf)
    queues, labels, codes, group_count = build_universe(
        ids, below, par2, groups, {"O1": u["o1"], "O3_XGB": xgb_scores, "O3_RF": p_rf})
    runs = bootstrap_runs(queues, labels, codes, group_count, replicas, seed)
    if not runs:
        raise ValueError("Nenhuma réplica com PAR=2")
    bootstrap = summarize_comparisons(runs)
    bootstrap_check = verify_bootstrap_reference(
        bootstrap, Path(data_root) / BOOTSTRAP_REFERENCE, replicas, seed)
    ties = {name: tie_summary(s[below]) for name, s in
            (("O1", u["o1"]), ("O3_XGB", xgb_scores), ("O3_RF", p_rf))}
    tie_table = None
    if ties["O3_RF"][0] > 0:
        tie_table = summarize_ties(
            quantities(cumulative_counts(ids[below], p_rf[below], par2[below])),
            run_replicas(p_rf[below], par2[below], tie_replicas, tie_seed),
            ties["O3_RF"], tie_replicas, tie_seed)

    # 5. Saída.
    table = long_table(metrics, bootstrap, tie_table)
    out_csv = out_dir / Path(OUT_CSV).name
    atomic_write(out_csv, lambda p: table.to_csv(p, index=False, lineterminator="\n"))
    inputs = {"data/COMPARADORSEMIDENT.csv": u["source"], "data/pcal_ml_oof.csv": u["original_path"],
              "data/pcal_ml_oof.json": u["original_path"].with_suffix(".json"),
              REFERENCE: Path(data_root) / REFERENCE, OUT_OOF: rf_path}
    if (Path(data_root) / BOOTSTRAP_REFERENCE).exists():
        inputs[BOOTSTRAP_REFERENCE] = Path(data_root) / BOOTSTRAP_REFERENCE
    metadata = {
        "analysis": "exploratory_post_hoc", "authorized_by_author": "2026-11-09",
        "registered": False, "cpu_only": True, "llm_calls": 0,
        "date_utc": datetime.now(UTC).isoformat(),
        "question": "A recomendação (ordenar abaixo do corte pela probabilidade calibrada, O3) depende do estimador XGBoost?",
        "change": "só o estimador (XGBoost -> Random Forest); mesmos 30 atributos, partições aninhadas 5x5 por REFREC, seed e calibração isotônica de gzcmd/ml_oof.py",
        "methods": {"xgb": METHOD_XGB, "rf": METHOD_RF},
        "rf_params": {**RF_PARAMS, "random_state": OOF_SEED, "n_jobs": -1,
                      "source": "scripts/run_nb01_uniform.py (RF do NB01), sem ajuste de hiperparâmetros",
                      "class": "SerialPredictRandomForest (subclasse de RandomForestClassifier)",
                      "predict_proba_n_jobs": 1,
                      "determinism": "fit com n_jobs=-1 (árvores com random_state próprio, determinístico); predict_proba com n_jobs=1, porque a soma das árvores em threads varia na ordem de 1e-16 e poderia alterar empates e a fila O3-RF"},
        "features": list(u["x"].columns), "oof_seed": OOF_SEED,
        "universe": {"below_cutoff": int(below.sum()), "PAR2": int(par2[below].sum()),
                     "grupos_REFREC_abaixo": int(group_count)},
        "xgb_reproduction": reproduction,
        "reference_checks": {"orderings": ["O1", "O3_XGB"], "file": REFERENCE,
                             "expected": {n: {f"{m}/{g}": v for (m, g), v in e.items()}
                                          for n, e in EXPECTED.items()},
                             "bootstrap_o3_xgb_o1": bootstrap_check},
        "rf_oof_seconds": rf_seconds, "rf_oof_reused": cached is not None,
        "bootstrap": {
            "seed": seed, "replicas": replicas, "replicas_validas": len(runs),
            "rng": "numpy.default_rng PCG64; uma rng.integers por réplica, como scripts/ordenacao_bootstrap_o3_o1.py",
            "targets": list(TARGETS), "fractions_if_less_36": list(FRACTIONS),
            "comparisons": [f"{a}-{b}" for a, b in COMPARISONS],
            "sign": "primeiro menos segundo; negativo favorece o primeiro; proporcao_primeiro_menor = proporção de réplicas com o primeiro menor",
            "confidence_interval": "percentil 2,5/97,5, quantis lineares; condicional às probabilidades OOF e aos rótulos fixos; não mede a variabilidade do treino dos classificadores",
            "paired": "as três filas são reamostradas com os mesmos grupos em cada réplica",
        },
        "ties": {"blocos_empatados_e_maior_bloco": {k: list(v) for k, v in ties.items()},
                 "random_tie_break_run": tie_table is not None, "seed": tie_seed,
                 "replicas": tie_replicas},
        "tie_break": "escore decrescente, pair_id lexical crescente (exatamente k linhas), como scripts/comparador_mesma_carga.py",
        "limitations": [
            "Viés de verificação: os PAR=2 vêm da revisão do serviço, presumivelmente em ordem de SCORE, o que favorece O1.",
            "Hiperparâmetros do RF copiados do NB01, sem ajuste; um único estimador alternativo não esgota a sensibilidade à escolha do modelo.",
            "Análise post hoc, não registrada; revisor perfeito simulado por PAR=2.",
        ],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
        "estimate": estimate,
        "input_sha256": {n: sha256(p) for n, p in inputs.items()},
        "code_sha256": {n: sha256(ROOT / n) for n in (
            "scripts/o3_estimador_rf.py", "tests/test_o3_estimador_rf.py",
            "gzcmd/ml_oof.py", "scripts/comparador_mesma_carga.py",
            "scripts/ordenacao_bootstrap_o3_o1.py", "scripts/ordenacao_empates_o3.py")},
        "output_sha256": {out_csv.name: sha256(out_csv)},
        "elapsed_seconds": time.perf_counter() - started,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    write_json(out_dir / Path(OUT_META).name, metadata)
    return table, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--saida-dir", type=Path, default=None, help="padrão: <data-root>/data")
    parser.add_argument("--replicas", type=int, default=BOOTSTRAP_REPLICAS, help="réplicas do bootstrap")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help="semente do bootstrap")
    parser.add_argument("--replicas-empates", type=int, default=TIE_REPLICAS)
    parser.add_argument("--seed-empates", type=int, default=TIE_SEED)
    parser.add_argument("--tolerancia", type=float, default=TOLERANCE,
                        help="tolerância da reprodução do XGBoost (padrão 1e-9)")
    parser.add_argument("--reutilizar-rf", action="store_true",
                        help="reutiliza data/pcal_ml_oof_rf.csv se o fingerprint conferir")
    parser.add_argument("--estimar-tempo", action="store_true",
                        help="só confere as referências e estima o tempo; não ajusta o OOF completo")
    args = parser.parse_args(argv)
    try:
        table, metadata = generate(
            args.data_root, args.saida_dir, replicas=args.replicas, seed=args.seed,
            tie_replicas=args.replicas_empates, tie_seed=args.seed_empates,
            tolerance=args.tolerancia, reuse=args.reutilizar_rf, estimate_only=args.estimar_tempo)
    except ReproductionError as exc:
        print(f"FALHA: {exc}", file=sys.stderr)
        return 1
    if args.estimar_tempo:
        return 0
    print(table.loc[table.bloco.eq("ordenacao")].to_string(index=False))
    print(table.loc[table.bloco.eq("bootstrap")].to_string(index=False))
    print(f"Concluído em {metadata['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
