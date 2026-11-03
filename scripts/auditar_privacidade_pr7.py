"""Remove pair-level labels from PR7 outputs only; never edit legacy or archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("data/exp2_haiku_sem_mo_pares_*.csv", "data/exp2_deepseek_sem_mo_pares_*.csv",
            "data/exp2_sage_*.csv", "data/lote_llm_novos_*.csv")
LEGACY_PATTERNS = ("data/exp2_comparacao_revisores_*.csv", "data/exp2_tres_revisores_*.csv",
                   "data/exp2_gabarito_por_par.csv", "data/sprint3b/*/gabarito_*.csv")
LABELS = {"y", "gabarito", "par"}


def label_columns(columns):
    return [c for c in columns if c.split(",")[0].strip().lower() in LABELS]


def main():
    output_path = ROOT / "data/privacidade_pr7_auditoria.json"
    previous = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    previous_removed = {row["path"]: row["removed_columns"] for row in previous.get("files", [])}
    files = sorted({p for pattern in PATTERNS for p in ROOT.glob(pattern) if "_m3v3" not in p.name})
    rows = []
    for path in files:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        removed = label_columns(table.columns)
        if removed:
            table.drop(columns=removed).to_csv(path, index=False)
        relative = path.relative_to(ROOT).as_posix()
        rows.append({"path": relative, "n": len(table), "removed_columns": sorted(set(removed) | set(previous_removed.get(relative, []))),
                     "label_columns_after": label_columns(pd.read_csv(path, nrows=0).columns)})
    legacy = []
    for path in sorted({p for pattern in LEGACY_PATTERNS for p in ROOT.glob(pattern)}):
        columns = pd.read_csv(path, nrows=0).columns
        legacy.append({"path": path.relative_to(ROOT).as_posix(), "label_columns": label_columns(columns),
                       "sha256_unchanged": hashlib.sha256(path.read_bytes()).hexdigest()})
    tracked = subprocess.run(["git", "ls-files", "data/exp2_gabarito*"], cwd=ROOT, text=True,
                             capture_output=True, check=True).stdout.splitlines()
    output = {"scope": "PR7 outputs only; archived _m3v3 files excluded and kept locally",
              "files": rows, "legacy_not_edited": legacy, "other_gold_files_for_user_review": tracked}
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
