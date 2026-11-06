"""Export aggregate-only dossier ablation counts from private, ignored CSVs.

Usage: python scripts/ablacao_dossie_agregado.py [--data-root PATH]
The input CSVs are intentionally not distributed in Git. SHA-256 fingerprints
record the exact inputs, without retaining identifiers, scores or row-level data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("ablacao_dossie_pares.csv", "ablacao_dossie_resultado.csv")
DECISIONS = {"MATCH", "UNSURE", "NONMATCH"}
MODELS = {"deepseek-v4.1-flash", "claude-haiku-5-5"}
REVIEWERS = {"Kimi K2.5", "DeepSeek V4.1 Flash", "Haiku 5.5"}


def aggregate(data_root: Path) -> dict:
    """Validate the paired sample and return only categorical counts and hashes."""
    raw = {name: (data_root / name).read_bytes() for name in INPUTS}
    pairs, results = [list(csv.DictReader(io.StringIO(raw[n].decode("utf-8-sig"))))
                      for n in INPUTS]
    ids = {row["pair_id"] for row in pairs}
    if not ids or "" in ids or len(ids) != len(pairs):
        raise ValueError("Selection must contain unique, nonempty pair IDs")
    errors = Counter()
    for row in pairs:
        reviewers = row["revisores_erraram"].split(";")
        count = int(row["n_revisores_erraram"])
        if (count not in (2, 3) or len(reviewers) != count
                or len(set(reviewers)) != count or not set(reviewers) <= REVIEWERS):
            raise ValueError("Invalid reviewer category in selection")
        errors[str(count)] += 1
    counts = Counter()
    seen = set()
    for row in results:
        key = (row["pair_id"], row["modelo"])
        decisions = tuple(row[k] for k in ("original", "controle", "sem_pcal"))
        if (key in seen or key[0] not in ids or key[1] not in MODELS
                or not set(decisions) <= DECISIONS):
            raise ValueError("Invalid or duplicate paired result")
        seen.add(key)
        for field, expected in (
            ("fn_original_revertido_match", decisions[0] == "NONMATCH" and decisions[2] == "MATCH"),
            ("fn_controle_revertido_match", decisions[1] == "NONMATCH" and decisions[2] == "MATCH"),
        ):
            if row[field] != str(expected):
                raise ValueError("Inconsistent reversal indicator")
        counts[(key[1], *decisions)] += 1
    if seen != {(pid, model) for pid in ids for model in MODELS}:
        raise ValueError("Both models must cover the complete selection")
    return {
        "schema_version": 1,
        "generator": "scripts/ablacao_dossie_agregado.py",
        "input_sha256": {n: hashlib.sha256(raw[n]).hexdigest() for n in INPUTS},
        "selection": {"n": len(pairs), "n_revisores_erraram": dict(sorted(errors.items()))},
        "result_rows": len(results),
        "categories": [dict(zip(("modelo", "original", "controle", "sem_pcal"), key), n=n)
                       for key, n in sorted(counts.items())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    output = args.data_root / "ablacao_dossie_agregado.json"
    output.write_text(json.dumps(aggregate(args.data_root), ensure_ascii=False, indent=2)
                      + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
