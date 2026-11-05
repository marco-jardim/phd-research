"""Contagens exploratórias de FP de P por região de SCORE, sem saída por par."""
import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff
from gzcmd.ml_oof import pair_ids
from scripts.arranjo_corte_s2_suplementar import responses


def aggregate(frame, mode, global_threshold, step_thresholds):
    rows = []
    false_positive = frame.par.eq(0) & frame.accepted
    for version in ("global", "por_passo"):
        threshold = pd.Series(global_threshold, index=frame.index) if version == "global" else frame.passo.map(step_thresholds)
        if threshold.isna().any():
            raise ValueError("Passo sem limiar de referência")
        above = frame.score.ge(threshold)
        for step in [*sorted(frame.passo.unique()), "TOTAL"]:
            region = pd.Series(True, index=frame.index) if step == "TOTAL" else frame.passo.eq(step)
            for label, mask in (("maior_ou_igual", above), ("menor", ~above)):
                rows.append({"modo": mode, "braco": "P", "limiar": version, "passo": step,
                             "regiao_SCORE": label, "FP": int((false_positive & region & mask).sum()),
                             "exploratorio_post_hoc": True})
    return pd.DataFrame(rows)


def generate(data_root, output=ROOT / "data/fp_arranjo_por_regiao.csv"):
    fields = {"COMPREC", "REFREC", "PASSO", "PAR", "SCORE"}
    raw = pd.read_csv(data_root / "data/COMPARADORSEMIDENT.csv", sep=";", dtype=str,
                      usecols=lambda name: name.split(",")[0] in fields)
    raw.columns = [name.split(",")[0] for name in raw.columns]
    numeric = {name: pd.to_numeric(raw[name].str.strip().str.replace(",", ".", regex=False), errors="raise")
               for name in ("PAR", "PASSO", "SCORE")}
    below = below_cutoff(raw)
    frame = pd.DataFrame({"pair_id": pair_ids(raw), "par": numeric["PAR"],
                          "passo": numeric["PASSO"], "score": numeric["SCORE"]}).set_index("pair_id")
    frame = frame.loc[below]
    meta = json.loads((ROOT / "data/carga_abaixo_corte_metadata.json").read_text(encoding="utf-8"))
    global_threshold = meta["SCORE_min_PAR2"]
    step_thresholds = {int(k): value for k, value in meta["SCORE_min_PAR2_por_passo"].items()}
    if frame.loc[frame.par.eq(2), "score"].min() != global_threshold:
        raise ValueError("Limiar global não reproduz o agregado histórico")
    observed_steps = frame.loc[frame.par.eq(2)].groupby("passo").score.min().to_dict()
    if observed_steps != step_thresholds:
        raise ValueError("Limiares por passo não reproduzem o agregado histórico")
    tables = []
    for mode, expected in (("vigilancia", 21), ("confirmacao", 4)):
        policy = pd.read_csv(ROOT / f".local/arranjo_corte/policy_{mode}_P.csv").set_index("pair_id")
        if not policy.index.is_unique or set(policy.index) != set(frame.index):
            raise ValueError("Universo da política P diverge do extrato abaixo do corte")
        known, _ = responses(data_root, mode, include_supplement=True)
        policy = policy.loc[frame.index]
        llm = policy.action.eq("LLM_REVIEW")
        decisions = pd.Series(policy.index.map(known), index=policy.index)
        if decisions.loc[llm].isna().any():
            raise ValueError("Resposta selecionada ausente")
        work = frame.assign(accepted=policy.action.eq("MATCH") | (llm & decisions.eq("MATCH")))
        if int((work.par.eq(0) & work.accepted).sum()) != expected:
            raise ValueError("FP de P não reproduzem 21/4")
        tables.append(aggregate(work, mode, global_threshold, step_thresholds))
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output, index=False, lineterminator="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PHD_RESEARCH_DATA_ROOT", ROOT)))
    args = parser.parse_args()
    print(generate(args.data_root).to_string(index=False))
