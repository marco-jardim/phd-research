"""Aggregate observed LLM precision and below-cutoff decision mechanisms.

No API calls, fitting, pair-level outputs or updates to frozen reviewer rates.
Expected FP sums frozen P(MATCH | nonlink, band, fold) over ALL nonlinks sent
to the LLM, not over the retrospectively accepted subset. Thus it is directly
comparable with observed LLM FP, but conditional on the reference labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256
from gzcmd.bands import BandAssigner
from gzcmd.config import load_config
from gzcmd.intervals import wilson95
from gzcmd.ml_oof import pair_ids
from gzcmd.reliability import load_band_rates
from scripts.arranjo_corte_s1_simular import LOCAL, MODES, load_raw
from scripts.arranjo_corte_s2_suplementar import responses
from scripts.arranjo_corte_s3_avaliar import build_frames

ARMS = ("P", "P-Platt", "A3", "P-PAR2-grey", "B1", "aposteriori")
MECHANISMS = ("guarda_nota_lt_5", "AUTO_NONMATCH_custo", "NONMATCH_LLM", "outros",
              "MATCH_LLM", "MATCH_motor", "HUMANO_LLM", "HUMANO_motor")


def summarize(frame, mode, arm, rates):
    work = frame.loc[frame.abaixo].copy()
    llm = work.action.eq("LLM_REVIEW")
    accepted = llm & work.final.eq("MATCH")
    negative = work.par.eq(0)
    tp, fp = int((accepted & ~negative).sum()), int((accepted & negative).sum())
    n = tp + fp
    low, high = wilson95(tp, n) if n else (float("nan"), float("nan"))
    expected_fp = float(rates.loc[work.index[llm & negative]].sum())
    precision = {"modo": mode, "braco": arm, "aceitacoes_llm_abaixo": n, "VP": tp, "FP": fp,
                 "VPP": tp / n if n else float("nan"), "ic95_inf": low, "ic95_sup": high,
                 "nao_vinculos_ao_llm": int((llm & negative).sum()), "e_FP_previstas": expected_fp}
    mechanism = pd.Series("outros", index=work.index)
    lost = work.final.eq("NONMATCH")
    mechanism.loc[lost & work.action.eq("NONMATCH") & work.nota.lt(5)] = "guarda_nota_lt_5"
    mechanism.loc[lost & work.action.eq("NONMATCH") & work.nota.ge(5)] = "AUTO_NONMATCH_custo"
    mechanism.loc[lost & llm] = "NONMATCH_LLM"
    mechanism.loc[work.final.eq("MATCH") & llm] = "MATCH_LLM"
    mechanism.loc[work.final.eq("MATCH") & ~llm] = "MATCH_motor"
    human = work.final.isin(["UNSURE", "DEFERRED"])
    mechanism.loc[human & llm] = "HUMANO_LLM"
    mechanism.loc[human & ~llm] = "HUMANO_motor"
    rows = []
    for name in MECHANISMS:
        mask = mechanism.eq(name)
        rows.append({"modo": mode, "braco": arm, "mecanismo": name, "pares": int(mask.sum()),
                     "nao_vinculos": int((mask & negative).sum()), "par2": int((mask & work.par.eq(2)).sum()),
                     "vinculos_perdidos": int((mask & lost & work.par.eq(2)).sum())})
    assert sum(r["pares"] for r in rows) == len(work)
    return precision, rows


def generate(data_root, out=ROOT / "data"):
    raw = load_raw(data_root / "data/COMPARADORSEMIDENT.csv")
    par = pd.to_numeric(raw.PAR).astype(int)
    base = pd.DataFrame({"pair_id": pair_ids(raw), "par": par, "y": par.isin([1, 2]).astype(int),
                         "nota": pd.to_numeric(raw["nota final"].str.replace(",", ".", regex=False)),
                         "abaixo": below_cutoff(raw)}).set_index("pair_id")
    cfg = load_config(ROOT / "gzcmd/gzcmd_v3_config.yaml")
    bands = BandAssigner.from_config(cfg).assign(base.nota)
    oof = np.load(LOCAL / "oof.npz", allow_pickle=False)
    folds = pd.Series(oof["fold"], index=oof["pair_id"]).loc[base.index]
    frozen = pd.read_csv(ROOT / "data/arranjo_corte_taxas_crossfit.csv")
    s1 = json.loads((ROOT / "data/arranjo_corte_s1_metadata.json").read_text(encoding="utf-8"))
    assert s1["frozen_rates_sha256"] == sha256(ROOT / "data/arranjo_corte_taxas_crossfit.csv")
    assert s1["input_fingerprints"]["data/COMPARADORSEMIDENT.csv"] == sha256(data_root / "data/COMPARADORSEMIDENT.csv")
    summary = pd.read_csv(ROOT / "data/arranjo_corte_s1_resumo.csv")
    precision_rows, mechanism_rows = [], []
    for mode in MODES:
        known, _ = responses(data_root, mode, include_supplement=True)
        frames, _, _ = build_frames(data_root, mode, base, known, summary, oof)
        frozen_mode = frozen.loc[frozen.modo.eq(mode)].set_index(["fold", "banda"]).e_fp
        legacy = load_band_rates(cfg.llm_review.error_rates_by_band, mode=mode,
                                 source=ROOT / "gzcmd/error_rates_by_band_v4.yaml")
        for arm in ARMS:
            if arm == "aposteriori":
                rates = bands.map({band: rate.e_fp for band, rate in legacy.items()})
            elif arm == "P-PAR2-grey":
                sensitivity = s1["sensitivity_PAR2_grey_rates"][mode]
                rates = pd.Series([sensitivity[str(f)][b]["e_fp"] for f, b in zip(folds, bands)], index=base.index)
            else:
                rates = pd.Series([frozen_mode.loc[f, b] for f, b in zip(folds, bands)], index=base.index)
            assert rates.notna().all()
            precision, mechanisms = summarize(frames[arm], mode, arm, rates)
            precision_rows.append(precision)
            mechanism_rows.extend(mechanisms)
    precision = pd.DataFrame(precision_rows)
    mechanisms = pd.DataFrame(mechanism_rows)
    precision.to_csv(out / "arranjo_corte_vpp_llm.csv", index=False)
    mechanisms.to_csv(out / "arranjo_corte_perdas_mecanismo.csv", index=False)
    metadata = {"expected_FP": "Sum frozen e_fp over reference nonlinks sent to LLM, before observing LLM decisions. Not sum over acceptances.",
                "rate_sources": {"crossfit": "data/arranjo_corte_taxas_crossfit.csv",
                                 "P-PAR2-grey": "data/arranjo_corte_s1_metadata.json:sensitivity_PAR2_grey_rates",
                                 "aposteriori": "gzcmd/error_rates_by_band_v4.yaml"},
                "frozen_rates_sha256": s1["frozen_rates_sha256"],
                "code_sha256": {name: sha256(ROOT / name) for name in (
                    "scripts/arranjo_corte_diagnosticos.py", "scripts/arranjo_corte_s3_avaliar.py")},
                "scope": "Below ORL cutoff only; acceptance is action=LLM_REVIEW and final=MATCH. Wilson 95% conditional on reference; not external validation.",
                "mechanisms": "Mutually exclusive partition of all below-cutoff pairs, including nonlinks and PAR=2. nota<5 guard takes precedence over cost; remaining engine NONMATCH is Bayesian cost rejection. Human is UNSURE/DEFERRED, not a lost link.",
                "zero_acceptances": "Undefined VPP and Wilson bounds encoded as empty CSV cells."}
    (out / "arranjo_corte_diagnosticos_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(precision.to_string(index=False))
    print(mechanisms.loc[mechanisms.braco.eq("P")].to_string(index=False))
    return precision, mechanisms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    generate(parser.parse_args().data_root)
