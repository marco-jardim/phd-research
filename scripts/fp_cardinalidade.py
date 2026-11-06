"""Replay P's archived actions and existing LLM responses; publish counts only.

No model training, policy rerun or API calls. --policy-dir must contain the
private historical policy_<mode>_P.csv files, verified against frozen hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gzcmd.arranjo_corte import below_cutoff
from gzcmd.ml_oof import pair_ids
from scripts.arranjo_corte_s2_suplementar import responses


def counts(false_positives: pd.DataFrame, accepted_above: pd.DataFrame) -> dict:
    for field in ("COMPREC", "REFREC"):
        if false_positives[field].isna().any() or accepted_above[field].isna().any():
            raise ValueError("Missing record identifier")
    left = false_positives.COMPREC.isin(set(accepted_above.COMPREC))
    right = false_positives.REFREC.isin(set(accepted_above.REFREC))
    return {"fp_total": len(false_positives), "comprec_ja_aceito": int(left.sum()),
            "refrec_ja_aceito": int(right.sum()), "ambos_ja_aceitos": int((left & right).sum()),
            "comprec_ou_refrec_ja_aceito": int((left | right).sum()),
            "nenhum_ja_aceito": int((~left & ~right).sum())}


def generate(data_root: Path, policy_dir: Path, output: Path) -> dict:
    extract = data_root / "data/COMPARADORSEMIDENT.csv"
    raw = pd.read_csv(extract, sep=";", dtype=str)
    raw.columns = [name.split(",")[0] for name in raw.columns]
    for field in ("COMPREC", "REFREC"):
        raw[field] = raw[field].str.strip()
    below = below_cutoff(raw)
    raw.index = pair_ids(raw)
    if not raw.index.is_unique:
        raise ValueError("Duplicate pair identities in extract")
    raw["par"] = pd.to_numeric(raw.PAR, errors="raise")
    accepted_above = raw.loc[~below & raw.par.eq(1)]
    metadata = json.loads((ROOT / "data/robustez_arranjo_metadata.json").read_text(encoding="utf-8"))
    result = {"braco": "P", "referencia": "PAR=1 acima do corte SCORE por PASSO",
              "n_referencia": len(accepted_above), "modos": {},
              "fontes_sha256": {"data/COMPARADORSEMIDENT.csv": hashlib.sha256(extract.read_bytes()).hexdigest()}}
    subset = raw.loc[below]
    for mode, expected in (("vigilancia", 21), ("confirmacao", 4)):
        logical = f".local/arranjo_corte/policy_{mode}_P.csv"
        path = policy_dir / Path(logical).name
        if not path.exists():
            raise FileNotFoundError(f"Missing archived P policy: {path}; aggregate outcomes and LLM journals alone do not identify P's selected actions")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["registered_sha256"][logical]:
            raise ValueError(f"Historical policy fingerprint mismatch: {mode}")
        policy = pd.read_csv(path).set_index("pair_id")
        if not policy.index.is_unique or set(policy.index) != set(subset.index):
            raise ValueError("Policy universe differs from below-cut extract")
        policy = policy.loc[subset.index]
        known, sources = responses(data_root, mode, include_supplement=True)
        selected = policy.action.eq("LLM_REVIEW")
        decisions = pd.Series(policy.index.map(known), index=policy.index)
        if decisions.loc[selected].isna().any():
            raise ValueError("Missing response for a P-selected LLM review")
        accepted = policy.action.eq("MATCH") | (selected & decisions.eq("MATCH"))
        fp = subset.loc[subset.par.eq(0) & accepted]
        if len(fp) != expected:
            raise ValueError(f"P FP do not reproduce 21/4: {mode}={len(fp)}")
        result["modos"][mode] = counts(fp, accepted_above)
        result["fontes_sha256"][logical] = digest
        for source in sources:
            result["fontes_sha256"][source.relative_to(data_root).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--policy-dir", type=Path, default=ROOT / ".local/arranjo_corte")
    args = parser.parse_args()
    generate(args.data_root, args.policy_dir, ROOT / "data/fp_cardinalidade.json")
