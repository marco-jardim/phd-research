"""Sensibilidade da ordenação O3 ao desempate, só CPU (análise exploratória pós-hoc).

scripts/comparador_mesma_carga.py ordena a região abaixo do corte pela
probabilidade calibrada fora da amostra (O3) e desempata pela ordem lexical do
pair_id. A probabilidade isotônica tem muitos empates. Este script repete a
ordenação O3 com desempates aleatórios dentro de cada bloco de probabilidade
empatada e recalcula as grandezas que o comparador reporta para O3:

* k_min_equivalente_P: revisões para recuperar tantos PAR=2 quanto o braço P
  (23 na vigilância, 9 na confirmação);
* k_min_36: revisões para recuperar os 36 PAR=2 abaixo do corte;
* par2_topk_K1_humana: PAR=2 recuperados com a carga humana de P (96 e 17).

Universo (abaixo do corte), PAR, metas e carga vêm de comparador_mesma_carga.py.
O desempate original (pair_id) é reproduzido e comparado com
data/comparador_mesma_carga.csv; se não bater, o script termina com código 1.
Só agregados: nenhuma linha por par, nenhum PII. Lê data/COMPARADORSEMIDENT.csv
(não versionado) e data/pcal_ml_oof.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256  # noqa: E402
from gzcmd.ml_oof import pair_ids  # noqa: E402
from scripts.comparador_mesma_carga import (  # noqa: E402
    MODES,
    align_original,
    cumulative_counts,
    load_raw,
    numeric,
)

SEED = 20261108
REPLICAS = 200
ORDERING = "O3_p_cal_ml"
OUT_CSV = ROOT / "data/ordenacao_empates_o3.csv"
REFERENCE = ROOT / "data/comparador_mesma_carga.csv"
COLUMNS = (
    "modo",
    "grandeza",
    "valor_original",
    "minimo",
    "p2_5",
    "mediana",
    "p97_5",
    "maximo",
    "n_replicas",
    "seed",
    "blocos_empatados",
    "maior_bloco",
)
# grandeza -> (carga, coluna) em data/comparador_mesma_carga.csv
REFERENCE_MAP = {
    "k_min_equivalente_P": ("k_min_equivalente_P", "k"),
    "k_min_36": ("k_min_36", "k"),
    "par2_topk_K1_humana": ("K1_humana", "par2_topk"),
}


class ReproductionError(RuntimeError):
    """O desempate original não reproduz data/comparador_mesma_carga.csv."""


def quantities(counts):
    """Grandezas por (modo, grandeza) a partir da contagem acumulada de PAR=2."""
    if int(counts[-1]) != 36:
        raise ValueError("Ranking does not contain all 36 PAR=2")
    result = {}
    for mode, p in MODES.items():
        recovered = p["accepted"] + p["referred"]
        result[(mode, "k_min_equivalente_P")] = int(np.searchsorted(counts, recovered) + 1)
        result[(mode, "k_min_36")] = int(np.searchsorted(counts, 36) + 1)
        result[(mode, "par2_topk_K1_humana")] = int(counts[p["human"] - 1])
    return result


def tie_summary(score):
    """Número de blocos de escore empatado (tamanho >= 2) e tamanho do maior."""
    _, sizes = np.unique(np.asarray(score), return_counts=True)
    return int((sizes > 1).sum()), int(sizes.max())


def random_tie_counts(score, par2, rng):
    """Escore decrescente; desempate uniforme dentro de cada bloco empatado."""
    score = np.asarray(score, dtype=float)
    keys = rng.random(len(score))
    order = np.lexsort((keys, -score))
    return np.asarray(par2)[order].astype(int).cumsum()


def run_replicas(score, par2, replicas, seed):
    rng = np.random.default_rng(seed)
    return [quantities(random_tie_counts(score, par2, rng)) for _ in range(replicas)]


def verify_original(original, reference):
    """Falha se o desempate por pair_id não reproduz o CSV do comparador."""
    for (mode, name), value in original.items():
        load, column = REFERENCE_MAP[name]
        rows = reference.loc[
            reference.ordenacao.eq(ORDERING)
            & reference.modo.eq(mode)
            & reference.carga.eq(load)
        ]
        if len(rows) != 1:
            raise ReproductionError(f"Linha de referência ausente: {mode}/{load}")
        expected = int(rows.iloc[0][column])
        if value != expected:
            raise ReproductionError(
                f"{mode}/{name}: reproduzido {value}, comparador {expected}"
            )


def summarize(original, runs, ties, replicas, seed):
    rows = []
    for key, value in original.items():
        values = np.array([run[key] for run in runs])
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975], method="linear")
        rows.append(
            {
                "modo": key[0],
                "grandeza": key[1],
                "valor_original": value,
                "minimo": int(values.min()),
                "p2_5": float(low),
                "mediana": float(median),
                "p97_5": float(high),
                "maximo": int(values.max()),
                "n_replicas": replicas,
                "seed": seed,
                "blocos_empatados": ties[0],
                "maior_bloco": ties[1],
            }
        )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def load_below(data_root):
    """Mesmo universo, PAR e pair_id de comparador_mesma_carga.generate."""
    source = Path(data_root) / "data/COMPARADORSEMIDENT.csv"
    original_path = Path(data_root) / "data/pcal_ml_oof.csv"
    raw = load_raw(source)
    below = below_cutoff(raw)
    par = numeric(raw.PAR)
    if not par.loc[below].isin([0, 2]).all():
        raise ValueError("Below-cutoff labels must be PAR=0 or PAR=2")
    ids = pair_ids(raw).to_numpy()
    original = align_original(raw, original_path, source)
    score = original.p_cal_ml.to_numpy()
    return (
        ids[below],
        score[below],
        par.to_numpy()[below] == 2,
        {"data/COMPARADORSEMIDENT.csv": source, "data/pcal_ml_oof.csv": original_path,
         "data/pcal_ml_oof.json": original_path.with_suffix(".json")},
    )


def atomic_write(path, write):
    """Escreve em arquivo temporário vizinho e troca, com retry (Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def generate(data_root=ROOT, replicas=REPLICAS, seed=SEED, saida=OUT_CSV,
             reference=REFERENCE, write=True):
    started = time.perf_counter()
    ids, score, par2, inputs = load_below(data_root)
    original = quantities(cumulative_counts(ids, score, par2))
    reference_table = pd.read_csv(reference)
    verify_original(original, reference_table)
    ties = tie_summary(score)
    runs = run_replicas(score, par2, replicas, seed)
    table = summarize(original, runs, ties, replicas, seed)
    if not write:
        return table, None
    saida = Path(saida)
    meta_path = saida.with_name(saida.stem + "_metadata.json")
    atomic_write(saida, lambda p: table.to_csv(p, index=False, lineterminator="\n"))
    inputs = {**inputs, "data/comparador_mesma_carga.csv": Path(reference)}
    metadata = {
        "analysis": "exploratory_post_hoc",
        "cpu_only": True,
        "llm_calls": 0,
        "date_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "rng": "numpy.default_rng PCG64; uma chamada rng.random(n) por réplica",
        "n_replicas": replicas,
        "script": "scripts/ordenacao_empates_o3.py",
        "ordering": ORDERING,
        "universe": {"below_cutoff": int(len(score)), "PAR2": int(par2.sum())},
        "ties": {"blocos_empatados": ties[0], "maior_bloco": ties[1]},
        "tie_break": "escore decrescente; ordem aleatória uniforme dentro de cada bloco empatado",
        "original_tie_break": "pair_id lexical ascendente (comparador_mesma_carga.py)",
        "original_reproduction": "valores conferidos contra data/comparador_mesma_carga.csv",
        "quantiles": "mínimo, máximo e percentis 2,5/50/97,5 (método linear) das réplicas",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "code_sha256": {
            name: sha256(ROOT / name)
            for name in ("scripts/ordenacao_empates_o3.py",
                         "scripts/comparador_mesma_carga.py")
        },
        "output_sha256": {saida.name: sha256(saida)},
        "elapsed_seconds": time.perf_counter() - started,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    atomic_write(
        meta_path,
        lambda p: p.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        ),
    )
    return table, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--saida", type=Path, default=OUT_CSV)
    args = parser.parse_args(argv)
    try:
        table, _ = generate(args.data_root, args.replicas, args.seed, args.saida)
    except ReproductionError as exc:
        print(f"FALHA: {exc}", file=sys.stderr)
        return 1
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
