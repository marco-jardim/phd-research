"""Bootstrap agrupado por REFREC da diferença de revisões O3 menos O1, só CPU.

Reaproveita, de scripts/fusao_robustez.py, a mesma semente, as mesmas metas, o
mesmo sorteio de grupos e a mesma expansão por multiplicidade. Cada réplica
consome uma única chamada rng.integers, como bootstrap_once; com a mesma semente,
as réplicas O3-O1 são portanto as mesmas amostras usadas em F4-O1 e F4-O3.
Gera artefatos novos e não toca em data/fusao_robustez_*.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fusao_filas import load_queues, sha256  # noqa: E402
from scripts.fusao_robustez import (  # noqa: E402
    FRACTIONS, SEED, TARGETS, resampled_queues, reviews, verify_reference,
)

REPLICAS = 2000
FUSED, BASE = "O3", "O1"
COMPARISON = f"{FUSED}-{BASE}"
# Mesmas colunas de data/fusao_robustez_bootstrap.csv, na mesma ordem.
COLUMNS = (
    "comparacao", "meta_nominal", "fracao_se_menos_36", "replicas_validas",
    "proporcao_fusao_menor", "diferenca_revisoes_mediana",
    "diferenca_revisoes_ic95_inf", "diferenca_revisoes_ic95_sup",
    "reducao_percentual_mediana", "reducao_percentual_ic95_inf",
    "reducao_percentual_ic95_sup", "proporcao_F4_menor_ambas",
)
OUT_CSV = "data/ordenacao_bootstrap_o3_o1.csv"
OUT_META = "data/ordenacao_bootstrap_o3_o1_metadata.json"


def bootstrap_o3_o1_once(queues, labels, group_codes, group_count, rng):
    """Uma réplica; mesma lógica de sorteio e metas de fusao_robustez.bootstrap_once."""
    draws = rng.integers(0, group_count, size=group_count)
    counts = np.bincount(draws, minlength=group_count)
    sampled, original = resampled_queues(
        {name: queues[name] for name in (FUSED, BASE)}, counts[group_codes])
    sampled_labels = labels[original]
    total = int(sampled_labels.sum())
    if not total:
        return None
    targets = TARGETS if total >= 36 else tuple(
        int(np.ceil(f * total)) for f in FRACTIONS)
    return reviews(sampled, sampled_labels, targets), total, len(original)


def summarize_o3_o1(runs):
    """Mesmas estatísticas de summarize_bootstrap para a comparação O3-O1.

    A coluna proporcao_F4_menor_ambas só tem sentido para F4; fica vazia (NaN).
    """
    fv = np.array([r[0][FUSED] for r in runs])
    bv = np.array([r[0][BASE] for r in runs])
    diff = fv - bv
    reduction = 100 * (bv - fv) / bv
    rows = []
    for j, target in enumerate(TARGETS):
        row = {"comparacao": COMPARISON, "meta_nominal": target,
               "fracao_se_menos_36": FRACTIONS[j], "replicas_validas": len(runs),
               "proporcao_fusao_menor": float((diff[:, j] < 0).mean())}
        for name, values in (("diferenca_revisoes", diff[:, j]),
                             ("reducao_percentual", reduction[:, j])):
            lo, median, hi = np.quantile(values, [0.025, 0.5, 0.975], method="linear")
            row.update({f"{name}_mediana": float(median),
                        f"{name}_ic95_inf": float(lo),
                        f"{name}_ic95_sup": float(hi)})
        row["proporcao_F4_menor_ambas"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=list(COLUMNS))


def run_bootstrap(queues, labels, group_codes, group_count, replicas=REPLICAS, seed=SEED):
    rng = np.random.default_rng(seed)
    runs = [bootstrap_o3_o1_once(queues, labels, group_codes, group_count, rng)
            for _ in range(replicas)]
    return [r for r in runs if r is not None]


def distribution(values):
    return dict(zip(("minimo", "mediana", "maximo"),
                    map(float, np.quantile(values, [0, 0.5, 1]))))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def generate(data_root, replicas=REPLICAS):
    import sklearn
    import xgboost

    started = time.perf_counter()
    queues, labels, data, fit_estimate = load_queues(data_root)
    reproduction = verify_reference(queues, labels)
    print("Reprodução de todas as métricas O1/O3/F3/F4 confirmada.", flush=True)
    below = data["below"]
    aligned = np.argsort(data["ids"][below])
    group_codes, groups = pd.factorize(data["groups"][below][aligned], sort=True)
    valid = run_bootstrap(queues, labels, group_codes, len(groups), replicas)
    if not valid:
        raise ValueError("Nenhuma réplica com PAR=2")
    table = summarize_o3_o1(valid)
    out = ROOT / OUT_CSV
    table.to_csv(out, index=False, lineterminator="\n")
    input_paths = {"data/COMPARADORSEMIDENT.csv": data["source"],
                   **{n: Path(data_root) / n for n in ("data/pcal_ml_oof.csv", "data/pcal_ml_oof.json")},
                   **{n: ROOT / n for n in ("data/fusao_filas.csv", "data/fusao_filas_metadata.json",
                                            "data/fusao_robustez_metadata.json")}}
    code_names = ("scripts/ordenacao_bootstrap_o3_o1.py", "tests/test_ordenacao_bootstrap_o3_o1.py",
                  "scripts/fusao_robustez.py", "scripts/fusao_filas.py")
    metadata = {
        "analysis": "exploratory_post_hoc", "cpu_only": True, "llm_calls": 0,
        "comparison": COMPARISON, "negative_favors": FUSED,
        "date_utc": datetime.now(UTC).isoformat(), "seed": SEED,
        "seed_rule": "mesma semente de scripts/fusao_robustez.py; uma chamada rng.integers por réplica, logo as mesmas amostras de F4-O1 e F4-O3 (réplicas pareadas)",
        "rng": "numpy.default_rng PCG64", "replicas": replicas,
        "replicas_sem_PAR2": replicas - len(valid), "grupos_REFREC": len(groups),
        "below_cutoff": len(labels), "PAR2": int(labels.sum()),
        "PAR2_por_replica": distribution([r[1] for r in valid]),
        "replicas_menos_36": sum(r[1] < 36 for r in valid),
        "pares_por_replica": distribution([r[2] for r in valid]),
        "targets": list(TARGETS), "fractions_if_less_36": list(FRACTIONS),
        "targets_rule": "fixed 9/23/36 if total>=36; otherwise ceil(25%/64%/100% of total); zero-positive replicates excluded",
        "targets_note": "metas contam PAR=2 recuperados; 9 e 23 já fazem parte do conjunto, sem meta adicional",
        "bootstrap": "REFREC sampled with replacement among below-cutoff groups; multiplicities expanded, both arms reranked; no model refit",
        "confidence_interval": "percentile 2.5/97.5, linear quantiles; conditional on fixed OOF scores and labels",
        "empty_column": "proporcao_F4_menor_ambas vazia: definida só para F4",
        "baseline_reproduction": reproduction, "fit_estimate": fit_estimate,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
        "input_sha256": {n: sha256(p) for n, p in input_paths.items()},
        "code_sha256": {n: sha256(ROOT / n) for n in code_names},
        "output_sha256": {OUT_CSV.split("/")[-1]: sha256(out)},
        "elapsed_seconds": time.perf_counter() - started,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    write_json(ROOT / OUT_META, metadata)
    print(table.to_string(index=False), flush=True)
    print(f"Concluído em {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    args = parser.parse_args()
    generate(args.data_root, args.replicas)
