"""Exploratory post-hoc, offline same-workload comparator (not registered)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256  # noqa: E402
from gzcmd.ml_oof import pair_ids  # noqa: E402

MAX_K = 2522
MODES = {
    "vigilancia": {"human": 96, "llm": 344, "accepted": 4, "referred": 19, "fp": 21},
    "confirmacao": {"human": 17, "llm": 41, "accepted": 8, "referred": 1, "fp": 4},
}

CURVE_COLUMNS = [
    "ordenacao",
    "k",
    "par2_topk",
    "nao_vinculos_topk",
    "sens_humano_perfeito",
    "sens_humano_05",
]
SUMMARY_COLUMNS = CURVE_COLUMNS + [
    "modo",
    "carga",
    "k_min_equivalente_P",
    "k_min_36",
    "P_par2_aceitos",
    "P_par2_humano",
    "P_par2_recuperados",
    "P_fp",
    "P_pares_humano",
    "P_pares_llm",
    "P_sens_humano_perfeito",
    "P_sens_humano_05",
    "comparador_fp_revisor_perfeito",
]
OUTPUTS = (
    "comparador_mesma_carga.csv",
    "comparador_mesma_carga_curva.csv",
    "comparador_mesma_carga_metadata.json",
)


def numeric(series):
    result = pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False), errors="raise"
    )
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite comparator input")
    return result


def load_raw(path):
    """Preserve record identifiers before numeric conversion, as in S1."""
    raw = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    if not raw.columns.is_unique:
        raise ValueError("Ambiguous normalized headers")
    return raw


def text_sha256(path):
    """Stable text fingerprint across Git's LF/CRLF checkout normalization."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def matches_hash(path, expected):
    import hashlib

    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return expected in {
        hashlib.sha256(content).hexdigest(),
        hashlib.sha256(content.replace(b"\n", b"\r\n")).hexdigest(),
    }


def align_original(raw, path, source):
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    fingerprint = metadata["fingerprint"]
    if (
        fingerprint["source_sha256"] != sha256(source)
        or fingerprint["split_by"] != "refrec"
        or not matches_hash(path, metadata["csv_sha256"])
        or not matches_hash(ROOT / "gzcmd/ml_oof.py", fingerprint["code_sha256"])
    ):
        raise ValueError("Original OOF fingerprint mismatch")
    original = pd.read_csv(path, dtype=str)
    ids = pair_ids(raw)
    if (
        original.pair_id.isna().any()
        or not original.pair_id.is_unique
        or set(original.pair_id) != set(ids)
    ):
        raise ValueError("Original OOF pair identity mismatch")
    original = original.set_index("pair_id").loc[ids].reset_index()
    for column in ("fold", "p_cal_ml"):
        original[column] = numeric(original[column])
    if (
        not original.fold.isin(range(5)).all()
        or not original.p_cal_ml.between(0, 1).all()
    ):
        raise ValueError("Invalid original OOF values")
    return original


def load_principal(path, original, source, original_path, s1_path):
    """Only consume the existing P OOF cache, never fit or call a reviewer."""
    if not path.exists():
        return None, {"available": False, "reason": "oof.npz absent"}
    if not s1_path.exists():
        return None, {"available": False, "reason": "S1 provenance absent"}
    try:
        s1 = json.loads(s1_path.read_text(encoding="utf-8"))
        fingerprints = s1["input_fingerprints"]
        for name, input_path in (
            ("data/COMPARADORSEMIDENT.csv", source),
            ("data/pcal_ml_oof.csv", original_path),
            ("gzcmd/arranjo_corte.py", ROOT / "gzcmd/arranjo_corte.py"),
            ("gzcmd/ml_oof.py", ROOT / "gzcmd/ml_oof.py"),
        ):
            if not matches_hash(input_path, fingerprints[name]):
                raise ValueError("S1 fingerprint mismatch")
        with np.load(path, allow_pickle=False) as cache:
            ids = cache["pair_id"]
            if (
                ids.ndim != 1
                or len(ids) != len(original)
                or len(set(ids)) != len(ids)
                or set(ids) != set(original.pair_id)
            ):
                raise ValueError("P pair identity mismatch")
            aligned = original.set_index("pair_id").loc[ids]
            if not np.array_equal(cache["fold"], aligned.fold.to_numpy()):
                raise ValueError("P fold identity mismatch")
            if not np.allclose(
                cache["B1"], aligned.p_cal_ml.to_numpy(), rtol=0, atol=1e-14
            ):
                raise ValueError("P B1 probability identity mismatch")
            p = cache["P"]
            if (
                p.shape != (len(ids),)
                or not np.isfinite(p).all()
                or ((p < 0) | (p > 1)).any()
            ):
                raise ValueError("Invalid P probabilities")
            result = pd.Series(p, index=ids).loc[original.pair_id].to_numpy()
    except (ValueError, KeyError, OSError) as exc:
        return None, {"available": False, "reason": str(exc)}
    return result, {
        "available": True,
        "identity_checks": [
            "S1 source and code fingerprints",
            "exact unique pair set",
            "aligned REFREC folds",
            "B1 equals original OOF at atol=1e-14",
            "finite P probabilities in [0,1]",
        ],
        "limitation": (
            "Cache has no historical P checksum in S1; current NPZ SHA256 is recorded."
        ),
    }


def cumulative_counts(ids, scores, par2):
    """Descending score, ascending lexical pair hash: exactly k, no tie expansion."""
    frame = pd.DataFrame({"pair_id": ids, "score": scores, "par2": par2})
    if (
        not frame.pair_id.is_unique
        or not np.isfinite(frame.score).all()
        or not frame.par2.isin([False, True]).all()
    ):
        raise ValueError("Invalid ranking inputs")
    ranked = frame.sort_values(["score", "pair_id"], ascending=[False, True])
    return ranked.par2.to_numpy(dtype=int).cumsum()


def metrics(order, k, counts):
    found = int(counts[k - 1])
    return {
        "ordenacao": order,
        "k": k,
        "par2_topk": found,
        "nao_vinculos_topk": k - found,
        "sens_humano_perfeito": (211 + found) / 247,
        "sens_humano_05": (211 + 0.95 * found) / 247,
    }


def check_p_reference(par2_path, metrics_path):
    par2 = pd.read_csv(par2_path)
    reference = pd.read_csv(metrics_path)
    for mode, expected in MODES.items():
        rows = par2.loc[par2.modo.eq(mode) & par2.braco.eq("P")]
        full = reference.loc[
            reference.modo.eq(mode)
            & reference.braco.eq("P")
            & reference.faixa.eq("TODAS")
        ]
        if len(rows) != 1 or len(full) != 1:
            raise ValueError("Unique P reference required")
        row, total = rows.iloc[0], full.iloc[0]
        if (
            int(row.vinculos) != 36
            or int(row.aceitos) != expected["accepted"]
            or int(row.humano) != expected["referred"]
            or int(row.perdidos) != 36 - expected["accepted"] - expected["referred"]
            or int(total.pares_humano) != expected["human"]
            or int(total.pares_llm) != expected["llm"]
            or int(total.fp) != expected["fp"]
        ):
            raise ValueError("P reference differs from requested comparison")


def generate(data_root=ROOT, out=ROOT / "data", *, oof_path=None, analysis_date=None):
    data_root, out = Path(data_root), Path(out)
    source = data_root / "data/COMPARADORSEMIDENT.csv"
    original_path = data_root / "data/pcal_ml_oof.csv"
    oof_path = (
        Path(oof_path)
        if oof_path is not None
        else ROOT / ".local/arranjo_corte/oof.npz"
    )
    s1_path = ROOT / "data/arranjo_corte_s1_metadata.json"
    inputs = {
        "data/COMPARADORSEMIDENT.csv": source,
        "data/pcal_ml_oof.csv": original_path,
        "data/pcal_ml_oof.json": original_path.with_suffix(".json"),
        **{
            f"data/{name}.csv": ROOT / f"data/{name}.csv"
            for name in (
                "arranjo_corte_par2",
                "arranjo_corte_metricas",
                "carga_abaixo_corte",
            )
        },
    }
    check_p_reference(
        inputs["data/arranjo_corte_par2.csv"], inputs["data/arranjo_corte_metricas.csv"]
    )
    raw = load_raw(source)
    below = below_cutoff(raw)
    par = numeric(raw.PAR)
    if not par.loc[below].isin([0, 2]).all():
        raise ValueError("Below-cutoff labels must be PAR=0 or PAR=2")
    ids = pair_ids(raw).to_numpy()
    original = align_original(raw, original_path, source)
    scores = {
        "O1_SCORE": numeric(raw.SCORE).to_numpy(),
        "O2_CdR": numeric(raw["nota final"]).to_numpy(),
        "O3_p_cal_ml": original.p_cal_ml.to_numpy(),
    }
    principal, o4 = load_principal(oof_path, original, source, original_path, s1_path)
    if oof_path.exists():
        inputs[".local/arranjo_corte/oof.npz"] = oof_path
    if s1_path.exists():
        inputs["data/arranjo_corte_s1_metadata.json"] = s1_path
    if principal is not None:
        scores["O4_P_recalibrada"] = principal
    summary, curve, notable = [], [], {}
    for order, score in scores.items():
        counts = cumulative_counts(ids[below], score[below], par.to_numpy()[below] == 2)
        if counts[-1] != 36:
            raise ValueError("Ranking does not contain all 36 PAR=2")
        min_all = int(np.searchsorted(counts, 36) + 1)
        notable[order] = {"k_min_36": min_all}
        curve.extend(metrics(order, k, counts) for k in range(1, MAX_K + 1))
        for mode, p in MODES.items():
            recovered = p["accepted"] + p["referred"]
            min_equivalent = int(np.searchsorted(counts, recovered) + 1)
            notable[order][f"k_min_{mode}_P"] = min_equivalent
            for load, k in (
                ("K1_humana", p["human"]),
                ("K2_total", p["human"] + p["llm"]),
                ("k_min_equivalente_P", min_equivalent),
                ("k_min_36", min_all),
                ("K3_limite", MAX_K),
            ):
                summary.append(
                    {
                        **metrics(order, k, counts),
                        "modo": mode,
                        "carga": load,
                        "k_min_equivalente_P": min_equivalent,
                        "k_min_36": min_all,
                        "P_par2_aceitos": p["accepted"],
                        "P_par2_humano": p["referred"],
                        "P_par2_recuperados": recovered,
                        "P_fp": p["fp"],
                        "P_pares_humano": p["human"],
                        "P_pares_llm": p["llm"],
                        "P_sens_humano_perfeito": (211 + recovered) / 247,
                        "P_sens_humano_05": (211 + p["accepted"] + 0.95 * p["referred"])
                        / 247,
                        "comparador_fp_revisor_perfeito": 0,
                    }
                )
    summary, curve = (
        pd.DataFrame(summary)[SUMMARY_COLUMNS],
        pd.DataFrame(curve)[CURVE_COLUMNS],
    )
    # The service lower bound includes the entire score tie at the last PAR=2.
    # Exact top-k can reach all positives earlier within that tie, never later.
    burden = pd.read_csv(inputs["data/carga_abaixo_corte.csv"])
    global_row = burden.loc[burden.limiar.eq("global") & burden.passo.eq("TOTAL")].iloc[
        0
    ]
    threshold = float(global_row.score_minimo)
    included = below & (scores["O1_SCORE"] >= threshold)
    if (
        int(included.sum()) != MAX_K
        or int(global_row.pares_acima_limiar) != MAX_K
        or int((included & par.eq(2).to_numpy()).sum()) != 36
        or int(
            curve.loc[
                curve.ordenacao.eq("O1_SCORE") & curve.k.eq(MAX_K), "par2_topk"
            ].iloc[0]
        )
        != 36
    ):
        raise ValueError("O1 differs from inclusive service lower bound")
    analysis_date = date.fromisoformat(
        analysis_date or date.today().isoformat()
    ).isoformat()
    metadata = {
        "analysis": "exploratory_post_hoc",
        "registered": False,
        "date": analysis_date,
        "api_calls": 0,
        "llm_calls": 0,
        "model_fits": 0,
        "universe": {
            "below_cutoff": 61485,
            "PAR2": 36,
            "nonlinks": 61449,
            "above_cutoff_links": 211,
            "total_links": 247,
        },
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "code_sha256_lf": {
            name: text_sha256(ROOT / name)
            for name in (
                "scripts/comparador_mesma_carga.py",
                "gzcmd/arranjo_corte.py",
                "gzcmd/ml_oof.py",
            )
        },
        "O4": o4,
        "notable_k": notable,
        "ordering": (
            "Descending numeric score, ascending lexical pair_id on ties; "
            "exactly k rows."
        ),
        "curve": {"k_min": 1, "k_max": MAX_K},
        "inclusion_convention": (
            "ORL accepts SCORE >= step cutoff; comparator uses SCORE < cutoff. "
            "Service global lower bound uses SCORE >= minimum PAR=2 SCORE, "
            "including all ties (2522). Exact top-k breaks ties by pair_id, "
            "so k_min_36 may be smaller than 2522."
        ),
        "assumptions": {
            "perfect_reviewer": (
                "All top-k PAR=2 accepted and all top-k nonlinks rejected: "
                "comparator FP=0. Nonlinks reviewed are workload, NOT false positives; "
                "P_fp are observed automatic false positives."
            ),
            "human_error_05": (
                "Sensitivity scenario only: (211 + 0.95 * PAR2_topk)/247. "
                "The 211 ORL links remain fixed; no 5% false-positive rate "
                "is assumed for nonlinks."
            ),
            "P_recovery": (
                "Accepted + referred PAR=2, assuming perfect review of referrals; "
                "P 5% scenario discounts referrals only, not automatic acceptances."
            ),
            "K2": (
                "Human + LLM review counts of P, not time-equivalent "
                "or cost-equivalent workloads."
            ),
        },
        "verification_bias": (
            "Os PAR=2 foram definidos pela revisao do servico, que se presume "
            "em ordem de SCORE. O1 e favorecido por construcao. O3/O4 sao "
            "comparadores mais justos com o GZ-CMD. Nao e uma estimativa "
            "independente de sensibilidade nem uma analise registrada."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / OUTPUTS[0], index=False, lineterminator="\n")
    curve.to_csv(out / OUTPUTS[1], index=False, lineterminator="\n")
    (out / OUTPUTS[2]).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary, curve, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--oof-path", type=Path, default=ROOT / ".local/arranjo_corte/oof.npz"
    )
    parser.add_argument("--analysis-date", default=date.today().isoformat())
    args = parser.parse_args()
    table, _, metadata = generate(
        args.data_root,
        args.out,
        oof_path=args.oof_path,
        analysis_date=args.analysis_date,
    )
    print(table.to_string(index=False))
    print(
        json.dumps({"O4": metadata["O4"], "notable_k": metadata["notable_k"]}, indent=2)
    )
