"""Sensibilidade ao tratamento do desbalanceamento com Random Forest e XGBoost.

Análise exploratória pós-hoc, autorizada pela autora em 2026-11-09 ("análise 3").
Repete, com XGBoost, a sensibilidade do Estudo 2 (§ sec:e2-robustez, Tabela
tab:imbalance-sensitivity, data/imbalance_sensitivity.csv, gerada por
scripts/robustness_analysis.py::run_imbalance_sensitivity: RF de 150 árvores e
profundidade 12, validação cruzada estratificada de 5 partições por linha com seed
42, limiar 0,5, 58 atributos) e estima a incerteza das diferenças pareadas entre
estratégias.

Ordem de execução (falha de reprodução termina com código 1 antes de qualquer
ajuste de XGBoost; nada é ajustado para "bater"):
  1. Carga do extrato com dtype=str e to_numeric (scripts/nb01_bootstrap_ic.py::
     load_extract) e engenharia de 58 atributos (scripts/feature_engineering.py).
  2. Repetição 0 (seed 42, a mesma partição e a mesma semente do histórico) só com
     Random Forest e as nove estratégias; as médias e os desvios de VPP,
     sensibilidade e F1 têm de bater com data/imbalance_sensitivity.csv (1e-4).
  3. Validação cruzada 5x5 repetida (cinco repetições das cinco partições; seed
     da repetição r = seed + r; mesma partição, mesma reamostragem e mesma semente
     para os dois estimadores), RF e XGBoost.
  4. Métricas por partição (VPP, sensibilidade, F1 e AUC-PR, limiar 0,5) e
     diferenças pareadas, com IC 95% de Nadeau e Bengio (2003).

Só agregados em data/balanceamento_xgb.csv e data/balanceamento_xgb_metadata.json.
Lê data/COMPARADORSEMIDENT.csv (não versionado). Checkpoint em
data/balanceamento_xgb.parcial.json (ignorado pelo git), removido ao concluir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gzcmd.ml_oof import _classifier  # noqa: E402
from scripts.o3_estimador_rf import SerialPredictRandomForest  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
REPS = 5
N_SPLITS = 5
THRESHOLD = 0.5
TOLERANCE = 1e-4
SAFETY = 1.3
CHECKPOINT_MINUTES = 30
SMOTE_NEIGHBORS = 3
RF_PARAMS = {"n_estimators": 150, "max_depth": 12}
SOURCE = "data/COMPARADORSEMIDENT.csv"
HISTORICAL = "data/imbalance_sensitivity.csv"
OUT_CSV = "data/balanceamento_xgb.csv"
OUT_META = "data/balanceamento_xgb_metadata.json"
OUT_PARTIAL = "data/balanceamento_xgb.parcial.json"
COLUMNS = (
    "tipo", "estimador", "estrategia", "metrica", "media", "dp", "diferenca_media",
    "dp_diferenca", "ic95_inf", "ic95_sup", "t", "gl", "p", "n_particoes",
)
METRICS = {"vpp": "VPP", "sensibilidade": "Sensibilidade", "f1": "F1", "auc_pr": "AUC-PR"}
HISTORICAL_COLUMNS = {"vpp": "prec", "sensibilidade": "rec", "f1": "f1"}
BASELINE = "No balancing"


class ReproductionError(RuntimeError):
    """O RF histórico não foi reproduzido; a análise não prossegue."""


@dataclass(frozen=True)
class Strategy:
    key: str  # nome da estratégia em data/imbalance_sensitivity.csv
    label_pt: str
    kind: str | None  # None, smote, borderline, adasyn, smotetomek
    ratio: float | None
    weighted: bool  # RF: class_weight="balanced"; XGBoost: scale_pos_weight=n_neg/n_pos


STRATEGIES = (
    Strategy("No balancing", "Sem balanceamento", None, None, False),
    Strategy("Class weight=balanced", "Ponderação de classes", None, None, True),
    Strategy("SMOTE 0.3", "SMOTE 0,3", "smote", 0.3, False),
    Strategy("SMOTE 0.5", "SMOTE 0,5", "smote", 0.5, False),
    Strategy("SMOTE 1.0", "SMOTE 1,0", "smote", 1.0, False),
    Strategy("BorderlineSMOTE 0.3", "BorderlineSMOTE 0,3", "borderline", 0.3, False),
    Strategy("ADASYN 0.3", "ADASYN 0,3", "adasyn", 0.3, False),
    Strategy("SMOTETomek 0.3", "SMOTETomek 0,3", "smotetomek", 0.3, False),
    Strategy("Class weight + SMOTE 0.3", "Ponderação + SMOTE 0,3", "smote", 0.3, True),
)


# --------------------------------------------------------------------------- estimadores


def class_ratio(y):
    """n_negativos / n_positivos (scale_pos_weight = class_weight balanced para a classe 1)."""
    y = np.asarray(y)
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("Treino sem positivos")
    return float((y == 0).sum() / positives)


def rf_factory(y_fit, weighted, seed):
    """RF do histórico (150 árvores, profundidade 12); predict_proba serial (determinístico)."""
    return SerialPredictRandomForest(
        **RF_PARAMS, class_weight="balanced" if weighted else None,
        random_state=seed, n_jobs=-1)


def xgb_factory(y_fit, weighted, seed):
    """XGBoost de gzcmd/ml_oof.py (_classifier); só scale_pos_weight muda por estratégia.

    _classifier usa scale_pos_weight = n_neg/n_pos do y informado (ponderado). Sem
    ponderação, scale_pos_weight=1; com ponderação, n_neg/n_pos de y_fit (o treino já
    reamostrado, como class_weight="balanced" no RF).
    """
    model = _classifier(np.asarray(y_fit), seed)
    model.set_params(scale_pos_weight=class_ratio(y_fit) if weighted else 1.0)
    return model


ESTIMATORS = {"rf": rf_factory, "xgb": xgb_factory}


# --------------------------------------------------------------------------- dados e partições


def load_xy(path):
    """Extrato com dtype=str + to_numeric, 58 atributos e rótulo PAR em {1, 2}."""
    from feature_engineering import engineer_features
    from scripts.nb01_bootstrap_ic import load_extract

    df = load_extract(Path(path))
    X, y = engineer_features(df)
    return X.to_numpy(dtype=float), y.to_numpy(dtype=int), list(X.columns)


def make_partitions(y, reps=REPS, seed=SEED, n_splits=N_SPLITS):
    """(repetição, partição, treino, teste); StratifiedKFold por linha, seed + repetição.

    A repetição 0 usa random_state=seed, a partição de robustness_analysis.py.
    """
    y = np.asarray(y)
    parts = []
    for rep in range(reps):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for fold, (train, test) in enumerate(skf.split(np.zeros(len(y)), y)):
            parts.append((rep, fold, train, test))
    return parts


def make_sampler(kind, ratio, seed):
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

    if kind == "smote":
        return SMOTE(sampling_strategy=ratio, k_neighbors=SMOTE_NEIGHBORS, random_state=seed)
    if kind == "borderline":
        return BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=SMOTE_NEIGHBORS, random_state=seed)
    if kind == "adasyn":
        return ADASYN(sampling_strategy=ratio, n_neighbors=SMOTE_NEIGHBORS, random_state=seed)
    if kind == "smotetomek":
        return SMOTETomek(
            smote=SMOTE(sampling_strategy=ratio, k_neighbors=SMOTE_NEIGHBORS, random_state=seed),
            random_state=seed)
    raise ValueError(f"Reamostragem desconhecida: {kind}")


def resample_train(X_train, y_train, strategy, seed):
    """Reamostragem só do treino; estratégias sem reamostragem devolvem o treino intacto."""
    if strategy.kind is None:
        return X_train, y_train
    return make_sampler(strategy.kind, strategy.ratio, seed).fit_resample(X_train, y_train)


def score_partition(y_true, proba, threshold=THRESHOLD):
    pred = (proba >= threshold).astype(int)
    return {
        "vpp": float(precision_score(y_true, pred, zero_division=0)),
        "sensibilidade": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true, proba)),
    }


def run_partition(X, y, train, test, seed, estimators, *, rep=0, fold=0,
                  strategies=STRATEGIES, done=frozenset()):
    """Ajusta cada estimador em cada estratégia nesta partição; reamostra uma vez por estratégia."""
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    rows = []
    for strategy in strategies:
        needed = [n for n in estimators if (n, rep, fold, strategy.key) not in done]
        if not needed:
            continue
        X_fit, y_fit = resample_train(X_train, y_train, strategy, seed)
        for name in needed:
            model = estimators[name](y_fit, strategy.weighted, seed)
            model.fit(X_fit, y_fit)
            proba = model.predict_proba(X_test)[:, 1]
            rows.append({
                "estimador": name, "rep": rep, "fold": fold, "estrategia": strategy.key,
                **score_partition(y_test, proba),
                "n_treino": int(len(train)), "n_teste": int(len(test)),
                "n_ajuste": int(len(y_fit)), "n_ajuste_pos": int(np.sum(y_fit == 1)),
            })
    return rows


# --------------------------------------------------------------------------- reprodução e IC


def check_reproduction(records, historical, tolerance=TOLERANCE, strategies=STRATEGIES):
    """Médias e desvios (ddof=1) do RF na repetição 0 contra data/imbalance_sensitivity.csv."""
    frame = pd.DataFrame(records)
    frame = frame[(frame.estimador == "rf") & (frame.rep == 0)]
    hist = historical.set_index("strategy")
    rows, failures = [], []
    for strategy in strategies:
        sub = frame[frame.estrategia == strategy.key].sort_values("fold")
        if strategy.key not in hist.index or sub.empty:
            failures.append(f"{strategy.key}: ausente")
            continue
        for key, prefix in HISTORICAL_COLUMNS.items():
            for stat, column, got in (
                ("media", f"{prefix}_mean", sub[key].mean()),
                ("dp", f"{prefix}_std", sub[key].std(ddof=1)),
            ):
                want = float(hist.loc[strategy.key, column])
                diff = float(abs(got - want))
                rows.append({"estrategia": strategy.key, "metrica": key, "estatistica": stat,
                             "obtido": float(got), "historico": want, "diferenca_abs": diff})
                if not diff <= tolerance:
                    failures.append(f"{strategy.key} {key} {stat}: obtido {got:.6f}, "
                                    f"histórico {want:.6f}, |dif| {diff:.2e}")
    result = {"tolerancia": tolerance, "passou": not failures,
              "max_diferenca_abs": max((r["diferenca_abs"] for r in rows), default=None),
              "n_comparacoes": len(rows)}
    if failures:
        raise ReproductionError(
            "O RF não reproduz data/imbalance_sensitivity.csv (sem ajustes para bater):\n  "
            + "\n  ".join(failures))
    return result


def nadeau_bengio(diffs, n_test, n_train, level=0.95):
    """Teste t corrigido para validação cruzada repetida (Nadeau e Bengio, 2003).

    d: diferenças pareadas, uma por partição (n = k*r). Variância da média corrigida:
    (1/n + n_test/n_train) * s^2, com s^2 amostral (ddof=1); t com n-1 graus de liberdade.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        raise ValueError("Ao menos duas diferenças pareadas")
    mean, var = float(d.mean()), float(d.var(ddof=1))
    if np.ptp(d) == 0.0:  # diferenças todas iguais: a variância é zero (evita resíduo ~1e-32)
        mean, var = float(d[0]), 0.0
    se = math.sqrt((1.0 / n + n_test / n_train) * var)
    gl = n - 1
    crit = float(stats.t.ppf(0.5 + level / 2, gl))
    if se == 0.0:
        t = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
        p = 1.0 if mean == 0.0 else 0.0
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), gl))
    return {"diferenca_media": mean, "dp_diferenca": math.sqrt(var), "se": se, "t": t,
            "gl": gl, "p": p, "ic95_inf": mean - crit * se, "ic95_sup": mean + crit * se,
            "n_particoes": n}


def aggregate(records, strategies=STRATEGIES, estimators=("rf", "xgb")):
    """Tabela longa: desempenho, diferenças contra 'sem balanceamento' e XGBoost menos RF."""
    frame = pd.DataFrame(records)
    parts = frame.drop_duplicates(["rep", "fold"])
    ratio = float(parts.n_teste.mean() / parts.n_treino.mean())
    index = ["rep", "fold"]

    def series(est, strat, metric):
        sub = frame[(frame.estimador == est) & (frame.estrategia == strat)].set_index(index)
        values = sub[metric].sort_index()
        # pandas ignoraria NaN em média e desvio sem avisar; aqui um valor indefinido interrompe.
        if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Métrica indefinida ou ausente: {est}/{strat}/{metric}")
        return values

    rows = []

    def add(**kw):
        rows.append({c: kw.get(c, np.nan) for c in COLUMNS})

    for est in estimators:
        for strategy in strategies:
            for key, name in METRICS.items():
                v = series(est, strategy.key, key)
                add(tipo="desempenho", estimador=est, estrategia=strategy.key, metrica=name,
                    media=float(v.mean()), dp=float(v.std(ddof=1)), n_particoes=len(v))
        for strategy in strategies:
            if strategy.key == BASELINE:
                continue
            for key, name in METRICS.items():
                d = series(est, strategy.key, key) - series(est, BASELINE, key)
                add(tipo="dif_sem_balanceamento", estimador=est, estrategia=strategy.key,
                    metrica=name, **_nb_columns(d, ratio))
    if "rf" in estimators and "xgb" in estimators:
        for strategy in strategies:
            for key, name in METRICS.items():
                d = series("xgb", strategy.key, key) - series("rf", strategy.key, key)
                add(tipo="dif_xgb_menos_rf", estimador="xgb_menos_rf", estrategia=strategy.key,
                    metrica=name, **_nb_columns(d, ratio))
    return pd.DataFrame(rows, columns=list(COLUMNS)), ratio


def _nb_columns(d, ratio):
    # n_test/n_train = ratio: passa-se (ratio, 1) porque só a razão entra na fórmula.
    r = nadeau_bengio(d.to_numpy(), ratio, 1.0)
    return {k: r[k] for k in ("diferenca_media", "dp_diferenca", "ic95_inf", "ic95_sup",
                              "t", "gl", "p", "n_particoes")}


# --------------------------------------------------------------------------- execução


def run_experiment(X, y, *, reps=REPS, seed=SEED, n_splits=N_SPLITS, historical=None,
                   estimators=None, strategies=STRATEGIES, records=None, on_progress=None,
                   tolerance=TOLERANCE):
    """Fase 1 (RF, repetição 0) + conferência; fase 2 (todos os estimadores, todas as repetições).

    `records` (checkpoint) é reaproveitado por chave (estimador, repetição, partição,
    estratégia). Com `historical`, a conferência ocorre antes de qualquer ajuste de
    estimador diferente do RF. Devolve (registros, resultado da reprodução ou None).
    """
    estimators = estimators or ESTIMATORS
    records = list(records or [])
    parts = make_partitions(y, reps, seed, n_splits)

    def keys():
        return {(r["estimador"], r["rep"], r["fold"], r["estrategia"]) for r in records}

    def sweep(selected, names):
        for rep, fold, train, test in parts:
            if rep not in selected:
                continue
            new = run_partition(X, y, train, test, seed + rep,
                                {n: estimators[n] for n in names}, rep=rep, fold=fold,
                                strategies=strategies, done=keys())
            if new:
                records.extend(new)
                if on_progress is not None:
                    on_progress(rep, fold, records)

    reproduction = None
    if historical is not None:
        sweep({0}, ["rf"])
        reproduction = check_reproduction(
            [r for r in records if r["estimador"] == "rf"], historical, tolerance, strategies)
    sweep(set(range(reps)), list(estimators))
    return records, reproduction


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_text(path, text):
    """Grava via arquivo temporário e os.replace, com retry (bloqueio de arquivo no Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.5 * 2**attempt)


FINGERPRINT_IGNORED = ("script_sha256",)


def fingerprints_match(saved, current):
    """Compara as chaves que definem os resultados; o hash do script não invalida o checkpoint.

    Os registros são métricas por partição já calculadas; alterar só a escrita do
    metadata/CSV não os invalida. O hash do script fica registrado, não comparado.
    """
    keep = lambda f: {k: v for k, v in (f or {}).items() if k not in FINGERPRINT_IGNORED}  # noqa: E731
    return keep(saved) == keep(current)


def read_partial(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_partial(path, fingerprint):
    saved = read_partial(path)
    if saved is None or not fingerprints_match(saved.get("fingerprint"), fingerprint):
        return []
    return saved["records"]


def vpp_zero_counts(records):
    """Partições com VPP = 0 (nenhum acerto entre as previsões positivas), por estimador e estratégia.

    VPP é calculada com zero_division=0, como no histórico, o que também vale 0 quando
    não há previsão positiva; a contagem torna visíveis essas partições, que entram nas
    médias com valor 0 (nunca descartadas).
    """
    frame = pd.DataFrame(records)
    zero = frame[frame.vpp == 0]
    return {est: {s: int(n) for s, n in sub.groupby("estrategia").size().items()}
            for est, sub in zero.groupby("estimador")}


def estimate_minutes(X, y, seed=SEED, reps=REPS, n_splits=N_SPLITS):
    """Piloto: a primeira partição, nove estratégias e dois estimadores, extrapolada."""
    rep, fold, train, test = make_partitions(y, 1, seed, n_splits)[0]
    start = time.perf_counter()
    run_partition(X, y, train, test, seed, ESTIMATORS, rep=rep, fold=fold)
    seconds = time.perf_counter() - start
    total = seconds * reps * n_splits * SAFETY / 60
    return {"pilot_partition_seconds": seconds, "partitions": reps * n_splits,
            "safety_factor": SAFETY, "estimated_minutes": total,
            "checkpoint_recommended": total > CHECKPOINT_MINUTES}


def estimator_description(y_example):
    """Parâmetros efetivos (não nulos) dos dois estimadores, só valores simples."""
    def clean(model):
        # XGBClassifier traz missing=nan (marcador de valor ausente), que o JSON estrito
        # recusa: valores não finitos são registrados como texto ("nan"), nunca como 0.
        return {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in model.get_params().items()
                if v is not None and isinstance(v, (int, float, str, bool))}

    return {"rf": clean(rf_factory(y_example, False, SEED)),
            "xgb": clean(xgb_factory(y_example, False, SEED))}


def generate(data_root=ROOT, out_dir=None, *, reps=REPS, seed=SEED, n_splits=N_SPLITS,
             estimate_only=False, restart=False, tolerance=TOLERANCE):
    import imblearn
    import sklearn
    import xgboost

    started = time.perf_counter()
    data_root = Path(data_root)
    out_dir = Path(out_dir) if out_dir is not None else data_root / "data"
    source = data_root / SOURCE
    X, y, columns = load_xy(source)
    print(f"Extrato: {len(y)} linhas, {int(y.sum())} pares verdadeiros, {X.shape[1]} atributos.",
          flush=True)
    estimate = estimate_minutes(X, y, seed, reps, n_splits)
    print("Estimativa de tempo: " + json.dumps(estimate), flush=True)
    if estimate_only:
        return estimate, None

    historical_path = data_root / HISTORICAL
    historical = pd.read_csv(historical_path)
    partial = out_dir / Path(OUT_PARTIAL).name
    script = Path(__file__).resolve()
    fingerprint = {"seed": seed, "reps": reps, "n_splits": n_splits, "n_rows": int(len(y)),
                   "n_pos": int(y.sum()), "strategies": [s.key for s in STRATEGIES],
                   "script_sha256": sha256(script), "tolerance": tolerance}
    saved = [] if restart else load_partial(partial, fingerprint)
    if saved:
        print(f"Checkpoint retomado: {len(saved)} registros.", flush=True)

    def progress(rep, fold, records):
        atomic_write_text(partial, json.dumps({"fingerprint": fingerprint, "records": records}))
        print(f"  repetição {rep} partição {fold}: {len(records)} registros "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    records, reproduction = run_experiment(
        X, y, reps=reps, seed=seed, n_splits=n_splits, historical=historical,
        records=saved, on_progress=progress, tolerance=tolerance)
    print("RF reproduz data/imbalance_sensitivity.csv: " + json.dumps(reproduction), flush=True)

    table, ratio = aggregate(records)
    out_csv = out_dir / Path(OUT_CSV).name
    atomic_write_text(out_csv, table.to_csv(index=False, lineterminator="\n"))
    metadata = {
        "analysis": "exploratory_post_hoc", "authorized_by_author": "2026-11-09",
        "registered": False, "cpu_only": True, "llm_calls": 0,
        "date_utc": datetime.now(UTC).isoformat(),
        "question": "O efeito do tratamento do desbalanceamento observado no RF vale para o XGBoost, e as diferenças entre estratégias se distinguem da variação entre partições?",
        "historical_script": "scripts/robustness_analysis.py::run_imbalance_sensitivity",
        "partition": {
            "unit": "linha (StratifiedKFold estratificado pelo rótulo PAR em {1,2}); não agrupa por óbito (REFREC)",
            "k": n_splits, "repetitions": reps, "seed": seed,
            "seed_by_repetition": {str(r): seed + r for r in range(reps)},
            "same_partition_for_all_estimators_and_strategies": True,
            "repetition_0_is_historical_partition": "random_state=42, shuffle=True",
            "seed_use": "a seed da repetição define a partição, os reamostradores e o estimador",
            "limitation": "linhas do mesmo óbito (REFREC) podem cair em treino e teste; as demais análises da tese que usam partição por REFREC não se aplicam aqui",
        },
        "resampling": "só no treino de cada partição; teste intacto; k_neighbors=3 (n_neighbors=3 no ADASYN); mesma reamostragem para RF e XGBoost",
        "threshold": THRESHOLD, "metrics": list(METRICS.values()),
        "auc_pr": "average_precision_score sobre a probabilidade bruta, sem limiar",
        "features": {"n": len(columns), "source": "scripts/feature_engineering.py::engineer_features (58 atributos, como o histórico)",
                     "note": "o XGBoost de gzcmd/ml_oof.py usa 30 atributos (numeric_features); aqui os 58 do RF, para comparar estimadores com os mesmos atributos"},
        "estimators": {
            "rf": {"class": "SerialPredictRandomForest (RandomForestClassifier com predict_proba serial)",
                   "params": {**RF_PARAMS, "random_state": "seed da repetição", "n_jobs": -1},
                   "weighted": "class_weight='balanced' (calculado sobre o treino já reamostrado)",
                   "unweighted": "class_weight=None"},
            "xgb": {"factory": "gzcmd.ml_oof._classifier (estimador de O3)",
                    "params": "n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric=logloss, n_jobs=8, random_state=seed da repetição",
                    "weighted": "scale_pos_weight=n_neg/n_pos do treino já reamostrado (o padrão de _classifier)",
                    "unweighted": "scale_pos_weight=1,0 (a fábrica de O3 sempre pondera; aqui 'sem balanceamento' a desliga)",
                    "qw1_difference": "o XGBoost do QW1 (scripts/qw1_deep_learning_comparison.py) usa learning_rate=0.1, imputação pela mediana e sem scale_pos_weight; esta análise usa o do GZ-CMD (learning_rate=0.05)"},
            "effective_params_example": estimator_description(y),
        },
        "strategies": {s.key: {"label_pt": s.label_pt, "kind": s.kind, "ratio": s.ratio,
                               "weighted": s.weighted} for s in STRATEGIES},
        "paired_differences": {
            "baseline": BASELINE,
            "comparisons": ["estratégia menos 'No balancing', por estimador, partição a partição",
                            "XGBoost menos RF, por estratégia, partição a partição"],
            "confidence_interval": "Nadeau e Bengio (2003), teste t corrigido para validação cruzada repetida",
            "rule": "d_i = diferença na partição i (n = k*r = 25); média d; s^2 amostral (ddof=1); variância corrigida da média = (1/n + n_teste/n_treino) * s^2; t = d / sqrt(variância corrigida), n-1 graus de liberdade; IC 95% = d ± t_{0,975, n-1} * erro-padrão corrigido; p bilateral",
            "n_test_over_n_train": ratio,
            "reference": "Nadeau, C.; Bengio, Y. Inference for the generalization error. Machine Learning, 52(3), 239-281, 2003",
            "multiplicity": "sem correção para comparações múltiplas (8 estratégias x 4 métricas x 2 estimadores, mais 9 x 4 de XGBoost menos RF); p e IC são descritivos",
            "caveat": "a correção supõe partições de treino sobrepostas e dependência entre repetições; as repetições usam a mesma amostra de linhas e a partição é por linha (não por óbito)",
        },
        "historical_reproduction": reproduction | {"file": HISTORICAL,
                                                   "compared": "repetição 0, RF, VPP/sensibilidade/F1, média e dp (ddof=1) por estratégia"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "imblearn": imblearn.__version__},
        "universe": {"rows": int(len(y)), "positives": int(y.sum())},
        "undefined_values": {
            "rule": "nenhum valor indefinido é descartado ou trocado por 0 sem registro: a agregação interrompe se alguma métrica por partição não for finita; VPP sem previsão positiva usa zero_division=0 (como o histórico, necessário para reproduzi-lo) e entra nas médias como 0",
            "partitions_with_vpp_zero": vpp_zero_counts(records),
            "csv_empty_cells": "célula vazia = não se aplica ao tipo de linha (média e dp só em 'desempenho'; diferença, IC, t, gl e p só nas linhas de diferença); nenhuma linha de desempenho ou de diferença tem valor indefinido",
            "csv_inf": "t = inf/-inf e p = 0 só se todas as diferenças pareadas forem iguais e não nulas (variância zero)",
            "estimator_params_nonfinite": "parâmetros não finitos dos estimadores (XGBoost: missing=nan) aparecem como texto",
        },
        "checkpoint": {
            "resumed_records": len(saved), "total_records": len(records),
            "script_sha256_at_checkpoint": (read_partial(partial) or {}).get("fingerprint", {}).get("script_sha256"),
            "script_sha256_now": fingerprint["script_sha256"],
            "note": "o hash do script não invalida o checkpoint (ver fingerprints_match); semente, repetições, partições, linhas, positivos, estratégias e tolerância invalidam",
        },
        "estimate": estimate,
        "input_sha256": {SOURCE: sha256(source), HISTORICAL: sha256(historical_path)},
        "code_sha256": {n: sha256(ROOT / n) for n in (
            "scripts/balanceamento_xgb.py", "tests/test_balanceamento_xgb.py",
            "gzcmd/ml_oof.py", "scripts/robustness_analysis.py")},
        "output_sha256": {out_csv.name: sha256(out_csv)},
        "privacy": "só agregados; sem linhas do extrato nem pair_id",
        "elapsed_seconds": time.perf_counter() - started,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    atomic_write_text(out_dir / Path(OUT_META).name,
                      json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if partial.exists():
        partial.unlink()
    return table, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--saida-dir", type=Path, default=None, help="padrão: <data-root>/data")
    parser.add_argument("--repeticoes", type=int, default=REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tolerancia", type=float, default=TOLERANCE)
    parser.add_argument("--estimar-tempo", action="store_true",
                        help="piloto de uma partição; não grava nada")
    parser.add_argument("--reiniciar", action="store_true", help="ignora o checkpoint")
    args = parser.parse_args(argv)
    try:
        table, metadata = generate(
            args.data_root, args.saida_dir, reps=args.repeticoes, seed=args.seed,
            estimate_only=args.estimar_tempo, restart=args.reiniciar, tolerance=args.tolerancia)
    except ReproductionError as exc:
        print(f"FALHA: {exc}", file=sys.stderr)
        return 1
    if args.estimar_tempo:
        return 0
    pd.set_option("display.width", 200)
    print(table[table.tipo != "desempenho"].to_string(index=False))
    print(f"Concluído em {metadata['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
