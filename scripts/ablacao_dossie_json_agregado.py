"""D16: aggregate private ablation checkpoints without publishing pair identities.

Run before removing originals from Git. To replay after removal, pass a private
directory with byte-identical originals via --source-dir. No API is called.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ("MATCH", "NONMATCH", "UNSURE")


def aggregate(path: Path) -> dict:
    raw = path.read_bytes()
    source = json.loads(raw)
    selected = source["selected_pair_ids"]
    results = source["results"]
    ids = [row["pair_id"] for row in results]
    if len(selected) != len(set(selected)) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate private pair identities")
    if not set(ids).issubset(selected):
        raise ValueError("Result outside the selected cohort")
    counts = Counter(row["decision"] for row in results)
    if set(counts) - set(DECISIONS):
        raise ValueError("Unknown decision")
    return {
        "schema_version": 1,
        "source_file": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "model": source["model"],
        "mode": source["mode"],
        "arm": source["arm"],
        "n_selected": len(selected),
        "n_reviewed": len(results),
        "n_errors": int(source.get("n_errors", len(source.get("errors", [])))),
        "decisions": {key: counts[key] for key in DECISIONS},
    }


def generate(source_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pattern in ("*_controle.json", "*_sem_pcal.json"):
        for path in sorted(source_dir.glob(pattern)):
            source = json.loads(path.read_bytes())
            if "selected_pair_ids" not in source:
                continue
            destination = output_dir / f"{path.stem}_agregado.json"
            result = aggregate(path)
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8", newline="\n")
            print(json.dumps(result, ensure_ascii=False))
            written.append(destination)
    if not written:
        raise ValueError("No private ablation checkpoints found; supply --source-dir")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/sprint3b/ablacao")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sprint3b/ablacao")
    parser.add_argument("--remove-tracked", action="store_true")
    args = parser.parse_args()
    paths = generate(args.source_dir, args.output_dir)
    if args.remove_tracked:
        for aggregate_path in paths:
            original = args.source_dir / aggregate_path.name.replace("_agregado.json", ".json")
            # Preserve the author's private input, but remove it from the next tree.
            subprocess.run(["git", "rm", "--cached", "--", str(original)],
                           cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
