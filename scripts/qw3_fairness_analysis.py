"""Quick Win 3: fairness diagnostics for record linkage classifiers.

This script keeps the acceptance criteria in scope explicit:
1. Equalized odds is the operational objective, with TPR and PPV parity.
2. Slicing variables:
   - quartis de ``NOME qtd frag iguais``
   - ``bandeira`` (SITENCe analog via ``C_SITUENCE`` in {3,4})
   - ``faixa_idade``
   - ``sexo``
   - ``UF`` with n>=30 minimum; UF with fewer rows are mapped by region.
3. For each 2x2 slice comparison, the script chooses chi² or Fisher exact
   by expected-cell rule and adds explicit null/alternative statements.
4. Benjamini-Hochberg FDR is applied over all completed 2x2 tests.
5. Effects and CIs are reported (Wilson for rates; OR and CI; difference CI).
6. Score-band stability (<=6, (6,11], (11,25], >25) is exported.

All outputs are written to ``data/qw3``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm


SEED_DEFAULT = 2026
N_MIN_CELL_DEFAULT = 30
ALPHA_DEFAULT = 0.05
CHI2_EXPECTED_MIN = 5.0


BR_STATE_TO_REGION: dict[int, str] = {
    11: "NORTE",
    12: "NORTE",
    13: "NORTE",
    14: "NORTE",
    15: "NORTE",
    16: "NORTE",
    17: "NORTE",
    21: "NORDESTE",
    22: "NORDESTE",
    23: "NORDESTE",
    24: "NORDESTE",
    25: "NORDESTE",
    26: "NORDESTE",
    27: "NORDESTE",
    28: "NORDESTE",
    29: "NORDESTE",
    31: "SUDESTE",
    32: "SUDESTE",
    33: "SUDESTE",
    35: "SUDESTE",
    41: "SUL",
    42: "SUL",
    43: "SUL",
    50: "CENTRO_OESTE",
    51: "CENTRO_OESTE",
    52: "CENTRO_OESTE",
    53: "CENTRO_OESTE",
}


AGE_BINS = [0.0, 18.0, 40.0, 60.0, 80.0, 130.0]
AGE_LABELS = ["0-17", "18-39", "40-59", "60-79", "80+"]


@dataclass(frozen=True)
class Config:
    input_csv: Path
    output_dir: Path
    sep: str
    seed: int
    target_col: str
    target_pos_values: tuple[int, ...]
    score_col: str
    threshold: float
    prediction_col: str | None
    score_band_col: str
    alpha: float
    n_min_cell: int


def _resolve_column(df: pd.DataFrame, canonical: str) -> str:
    """Resolve OpenRecLink-style columns like ``COL,C,12,0``.

    Returns ``canonical`` if present. If not, returns the unique prefixed
    column that starts with ``"{canonical},"``. Raises ValueError on ambiguity.
    """

    if canonical in df.columns:
        return canonical

    prefix = f"{canonical},"
    matches = [c for c in df.columns if str(c).startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise KeyError(f"Missing required column: {canonical}")
    raise KeyError(f"Ambiguous column resolution for {canonical}: {', '.join(matches)}")


def _parse_decimal(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _parse_int(series: pd.Series) -> pd.Series:
    parsed = _parse_decimal(series).round(0)
    return parsed.astype("Int64")


def _parse_date_yyyymmdd(series: pd.Series) -> pd.Series:
    parsed = _parse_int(series)
    txt = parsed.astype("string").str.zfill(8)
    return pd.to_datetime(txt, format="%Y%m%d", errors="coerce")


def _parse_date_ddmmyyyy(series: pd.Series) -> pd.Series:
    parsed = _parse_int(series)
    txt = parsed.astype("string").str.zfill(8)
    return pd.to_datetime(txt, format="%d%m%Y", errors="coerce")


def _safe_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _wilson_ci(
    successes: int, trials: int, alpha: float = ALPHA_DEFAULT
) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan

    if successes < 0 or successes > trials:
        return math.nan, math.nan

    z = norm.ppf(1 - alpha / 2)
    p = successes / trials
    n = trials
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    low = (center - half) / denom
    high = (center + half) / denom
    return max(0.0, low), min(1.0, high)


def _or_and_ci(
    table: np.ndarray, alpha: float = ALPHA_DEFAULT
) -> tuple[float, float, float]:
    a, b = float(table[0, 0]), float(table[0, 1])
    c, d = float(table[1, 0]), float(table[1, 1])

    # Haldane-Anscombe correction to avoid division by zero.
    if a == 0 or b == 0 or c == 0 or d == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    if (a + b) == 0 or (c + d) == 0:
        return math.nan, math.nan, math.nan

    or_value = (a * d) / (b * c)
    log_or = math.log(or_value)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = norm.ppf(1 - alpha / 2)
    low = math.exp(log_or - z * se)
    high = math.exp(log_or + z * se)
    return or_value, low, high


def _difference_ci(
    p1: float, n1: int, p2: float, n2: int, alpha: float = ALPHA_DEFAULT
) -> tuple[float, float, float]:
    if n1 <= 0 or n2 <= 0:
        return math.nan, math.nan, math.nan

    diff = p1 - p2
    lo1, hi1 = _wilson_ci(int(round(p1 * n1)), n1, alpha=alpha)
    lo2, hi2 = _wilson_ci(int(round(p2 * n2)), n2, alpha=alpha)
    diff_lo = lo1 - hi2
    diff_hi = hi1 - lo2
    return diff, diff_lo, diff_hi


def _benjamini_hochberg(
    pvals: list[float], alpha: float = ALPHA_DEFAULT
) -> list[float | None]:
    """Benjamini-Hochberg adjusted p-values.

    Returns list aligned with original order. ``None`` stays as None.
    """

    out: list[float | None] = [None] * len(pvals)
    finite = [
        (i, p) for i, p in enumerate(pvals) if p is not None and not math.isnan(p)
    ]
    if not finite:
        return out

    idx = np.array([i for i, _ in finite], dtype=int)
    values = np.array([p for _, p in finite], dtype=float)
    m = len(values)
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_adjusted = sorted_vals * (m / (np.arange(m) + 1))
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
    sorted_adjusted = np.clip(sorted_adjusted, 0.0, 1.0)

    for original_pos in range(m):
        out[int(idx[order[original_pos]])] = float(sorted_adjusted[original_pos])

    return out


def _recode_sex(df: pd.DataFrame) -> pd.Series:
    def _norm(v: Any) -> str | None:
        if pd.isna(v):
            return None
        s = str(v).strip().upper()
        if s.startswith("F"):
            return "F"
        if s.startswith("M"):
            return "M"
        return None

    r = df["R_SEXO" if "R_SEXO" in df.columns else _resolve_column(df, "R_SEXO")]
    c = df["C_SEXO" if "C_SEXO" in df.columns else _resolve_column(df, "C_SEXO")]

    series = r.where(r.notna(), c).map(_norm)
    return series.astype("string")


def _build_uf_group(
    df: pd.DataFrame, n_min_cell: int
) -> tuple[pd.Series, list[dict[str, Any]]]:
    r = _parse_int(
        df[
            "R_CODMUNRES"
            if "R_CODMUNRES" in df.columns
            else _resolve_column(df, "R_CODMUNRES")
        ]
    )
    c = _parse_int(
        df[
            "C_CODMUNRES"
            if "C_CODMUNRES" in df.columns
            else _resolve_column(df, "C_CODMUNRES")
        ]
    )

    codmun = r.combine_first(c)
    codmun_str = codmun.astype("string")
    uf_code = codmun_str.map(
        lambda x: int(float(x[:2]))
        if isinstance(x, str) and x.isdigit() and len(x) >= 2
        else pd.NA
    )
    uf = uf_code.astype("Int64")
    uf_label = uf.astype("string")

    note_rows: list[dict[str, Any]] = []
    counts = uf_label.value_counts(dropna=False)
    uf_fallback = uf_label.copy()

    for code in sorted(v for v in counts.index if pd.notna(v) and str(v) != "<NA>"):
        n = int(counts.loc[code])
        if n >= n_min_cell:
            continue
        code_int = int(float(code))
        region = BR_STATE_TO_REGION.get(code_int, "OUTRA_REGION")
        replacement = f"REGIAO_{region}"
        uf_fallback.loc[uf_label == code] = replacement
        note_rows.append(
            {
                "original_uf": code,
                "rows": n,
                "action": "aggregated_to_region",
                "region": region,
            }
        )

    # Enforce minimum sample size after UF->region aggregation.
    # Any region bucket still below n_min_cell is collapsed to MISSING so
    # pairwise fairness tests do not run on tiny support groups.
    final_counts = uf_fallback.value_counts(dropna=False)
    for group in sorted(
        v for v in final_counts.index if pd.notna(v) and str(v) != "<NA>"
    ):
        group_s = str(group)
        if not group_s.startswith("REGIAO_"):
            continue
        n = int(final_counts.loc[group])
        if n >= n_min_cell:
            continue
        uf_fallback.loc[uf_fallback == group] = "MISSING"
        note_rows.append(
            {
                "original_uf": group_s,
                "rows": n,
                "action": "collapsed_to_missing_due_small_region",
                "region": group_s.replace("REGIAO_", ""),
            }
        )

    if (uf_fallback == "<NA>").any():
        note_rows.append(
            {
                "original_uf": "missing",
                "rows": int((uf_fallback == "<NA>").sum()),
                "action": "kept_missing",
                "region": "missing",
            }
        )

    return uf_fallback.fillna("MISSING").astype("string"), note_rows


def _build_age_band(df: pd.DataFrame) -> pd.Series:
    birth = _parse_date_yyyymmdd(
        df["R_DTNASC" if "R_DTNASC" in df.columns else _resolve_column(df, "R_DTNASC")]
    )
    death = _parse_date_ddmmyyyy(
        df[
            "R_DTOBITO"
            if "R_DTOBITO" in df.columns
            else _resolve_column(df, "R_DTOBITO")
        ]
    )
    age = (death - birth).dt.days / 365.25

    # Keep only coherent ages.
    age = age.where((age >= 0) & (age <= 130))
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)


def _prepare_fairness_frame(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(cfg.input_csv, sep=cfg.sep, low_memory=False)

    target_col = _resolve_column(df, cfg.target_col)
    score_col = _resolve_column(df, cfg.score_col)
    score_band_col = _resolve_column(df, cfg.score_band_col)
    situ_col = _resolve_column(df, "C_SITUENCE")

    par = _parse_int(df[target_col])
    y_true = par.isin(cfg.target_pos_values).astype("int64")

    score = _parse_decimal(df[score_col]).astype("float64")
    score_band_value = _parse_decimal(df[score_band_col]).astype("float64")

    if cfg.prediction_col:
        pred_raw = _parse_decimal(df[_resolve_column(df, cfg.prediction_col)])
        y_pred = pred_raw.fillna(0.0).round().clip(0, 1).astype("int64")
        score_source = score_col
    else:
        y_pred = (score >= cfg.threshold).astype("int64")
        score_source = score_col

    score_band_source = score_band_col

    band_rules = {
        "0-6": {
            "column": score_band_source,
            "label": "<=6",
            "mask": score_band_value <= 6,
        },
        "6-11": {
            "column": score_band_source,
            "label": "(6,11]",
            "mask": (score_band_value > 6) & (score_band_value <= 11),
        },
        "11-25": {
            "column": score_band_source,
            "label": "(11,25]",
            "mask": (score_band_value > 11) & (score_band_value <= 25),
        },
        ">25": {
            "column": score_band_source,
            "label": ">25",
            "mask": score_band_value > 25,
        },
    }

    nome_qtd = _parse_decimal(df["NOME qtd frag iguais"]).astype("float64")
    nome_quartil = pd.qcut(
        nome_qtd,
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    ).astype("string")

    situ = _parse_int(df[situ_col])
    bandeira = np.where(situ.isin([3, 4]), "tb_origem", "controle")
    bandeira = pd.Series(bandeira, index=df.index, dtype="string")

    faixa_idade = _build_age_band(df).astype("string")

    sexo = _recode_sex(df).astype("string")

    uf, uf_notes = _build_uf_group(df, n_min_cell=cfg.n_min_cell)

    score_band = pd.Series("MISSING", index=df.index, dtype="string")
    for band_name, rule in band_rules.items():
        score_band = score_band.mask(rule["mask"], band_name)

    out = pd.DataFrame(
        {
            "index": df.index,
            "y_true": y_true,
            "y_pred": y_pred,
            "nome_qtd_quartis": nome_quartil,
            "bandeira": bandeira,
            "faixa_idade": faixa_idade,
            "sexo": sexo,
            "uf": uf,
            "score_band": score_band,
            "score_source": score,
            "score_band_source": score_band_source,
        }
    )

    # Keep only rows with labels required for slicing.
    out["nome_qtd_quartis"] = out["nome_qtd_quartis"].fillna("MISSING")
    out["bandeira"] = out["bandeira"].fillna("MISSING")
    out["faixa_idade"] = out["faixa_idade"].fillna("MISSING")
    out["sexo"] = out["sexo"].fillna("MISSING")
    out["uf"] = out["uf"].fillna("MISSING")

    quartil_bins: list[float] | str
    if nome_qtd.notna().sum() > 0:
        quartil_bins = [
            float(x)
            for x in pd.qcut(
                nome_qtd,
                q=4,
                labels=False,
                duplicates="drop",
                retbins=True,
            )[1]
        ]
    else:
        quartil_bins = ""

    notes: list[dict[str, Any]] = [
        {
            "step": "input",
            "rows": int(len(out)),
            "score_col": score_col,
            "prediction_mode": "score_threshold"
            if cfg.prediction_col is None
            else "prediction_col",
            "prediction_source": cfg.prediction_col or score_col,
            "threshold": float(cfg.threshold),
        },
        {
            "step": "slice_counts",
            "target_positive": int(y_true.sum()),
            "positive_rate": float(y_true.mean()) if len(y_true) else math.nan,
        },
        {
            "step": "quartis_name_qtd",
            "bin_edges": quartil_bins,
        },
    ]
    notes.append(
        {
            "step": "uf_aggregation",
            "notes": uf_notes,
        }
    )

    return out, {
        "notes": notes,
        "score_band_rules": {
            name: {
                "column": details["column"],
                "label": details["label"],
            }
            for name, details in band_rules.items()
        },
    }


def _build_group_metrics(df: pd.DataFrame, group_col: str, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for group_name, subset in df.groupby(group_col, dropna=False):
        subset = subset.copy()
        y_true = subset["y_true"].to_numpy()
        y_pred = subset["y_pred"].to_numpy()

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))

        if metric == "tpr":
            num = tp
            den = tp + fn
            rate_name = "tpr"
        elif metric == "ppv":
            num = tp
            den = tp + fp
            rate_name = "ppv"
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        rate = float(num / den) if den > 0 else math.nan
        ci_low, ci_high = _wilson_ci(num, den)

        rows.append(
            {
                "dimension": group_col,
                "metric": rate_name,
                "group": str(group_name),
                "n_total": int(len(subset)),
                "n_true": int((subset["y_true"] == 1).sum()),
                "n_pred_pos": int((subset["y_pred"] == 1).sum()),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "num": num,
                "den": den,
                "rate": rate,
                "wilson_low": ci_low,
                "wilson_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def _build_pairwise_test(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    g1: str,
    g2: str,
    y_col: str = "y_true",
    p_col: str = "y_pred",
    alpha: float = ALPHA_DEFAULT,
) -> dict[str, Any]:
    sub = df[df[group_col].isin([g1, g2])]
    g1_df = sub[sub[group_col] == g1]
    g2_df = sub[sub[group_col] == g2]

    if len(g1_df) < 1 or len(g2_df) < 1:
        return {
            "dimension": group_col,
            "metric": metric,
            "group_a": str(g1),
            "group_b": str(g2),
            "n_group_a": int(len(g1_df)),
            "n_group_b": int(len(g2_df)),
            "test": "skipped",
            "test_reason": "one_or_more_groups_empty",
            "p_value_raw": math.nan,
            "p_value_fdr": math.nan,
            "reject_fdr": False,
            "test_statistic": math.nan,
            "odds_ratio": math.nan,
            "or_ci_low": math.nan,
            "or_ci_high": math.nan,
            "rate_a": math.nan,
            "rate_b": math.nan,
            "rate_diff": math.nan,
            "rate_diff_ci_low": math.nan,
            "rate_diff_ci_high": math.nan,
            "n_cell_min": 0,
            "null_hypothesis": _null_hypothesis(metric),
            "alternative": _alternative_hypothesis(metric),
        }

    if metric == "tpr":
        g1_num = int(((g1_df[y_col] == 1) & (g1_df[p_col] == 1)).sum())
        g1_den = int((g1_df[y_col] == 1).sum())
        g2_num = int(((g2_df[y_col] == 1) & (g2_df[p_col] == 1)).sum())
        g2_den = int((g2_df[y_col] == 1).sum())
    elif metric == "ppv":
        g1_num = int(((g1_df[y_col] == 1) & (g1_df[p_col] == 1)).sum())
        g1_den = int((g1_df[p_col] == 1).sum())
        g2_num = int(((g2_df[y_col] == 1) & (g2_df[p_col] == 1)).sum())
        g2_den = int((g2_df[p_col] == 1).sum())
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if g1_den <= 0 or g2_den <= 0:
        return {
            "dimension": group_col,
            "metric": metric,
            "group_a": str(g1),
            "group_b": str(g2),
            "n_group_a": int(len(g1_df)),
            "n_group_b": int(len(g2_df)),
            "n1": g1_den,
            "n2": g2_den,
            "num_a": g1_num,
            "num_b": g2_num,
            "test": "skipped",
            "test_reason": "invalid_denominator_zero",
            "p_value_raw": math.nan,
            "p_value_fdr": math.nan,
            "reject_fdr": False,
            "test_statistic": math.nan,
            "odds_ratio": math.nan,
            "or_ci_low": math.nan,
            "or_ci_high": math.nan,
            "rate_a": math.nan,
            "rate_b": math.nan,
            "rate_diff": math.nan,
            "rate_diff_ci_low": math.nan,
            "rate_diff_ci_high": math.nan,
            "n_cell_min": int(min(g1_num, g2_num, g1_den - g1_num, g2_den - g2_num)),
            "null_hypothesis": _null_hypothesis(metric),
            "alternative": _alternative_hypothesis(metric),
        }

    table2x2 = np.array(
        [
            [g1_num, int(g1_den - g1_num)],
            [g2_num, int(g2_den - g2_num)],
        ],
        dtype=float,
    )

    n_cells = table2x2.flatten()
    n_cell_min = int(n_cells.min()) if n_cells.size else 0

    if (table2x2 < 0).any():
        return {
            "dimension": group_col,
            "metric": metric,
            "group_a": str(g1),
            "group_b": str(g2),
            "n_group_a": int(len(g1_df)),
            "n_group_b": int(len(g2_df)),
            "n1": g1_den,
            "n2": g2_den,
            "num_a": g1_num,
            "num_b": g2_num,
            "test": "skipped",
            "test_reason": "invalid_negative_cell",
            "p_value_raw": math.nan,
            "p_value_fdr": math.nan,
            "reject_fdr": False,
            "test_statistic": math.nan,
            "odds_ratio": math.nan,
            "or_ci_low": math.nan,
            "or_ci_high": math.nan,
            "rate_a": math.nan,
            "rate_b": math.nan,
            "rate_diff": math.nan,
            "rate_diff_ci_low": math.nan,
            "rate_diff_ci_high": math.nan,
            "n_cell_min": n_cell_min,
            "null_hypothesis": _null_hypothesis(metric),
            "alternative": _alternative_hypothesis(metric),
        }

    try:
        chi2, chi2_p, _, expected = chi2_contingency(table2x2, correction=False)
    except Exception as exc:
        expected = None
        chi2_p = math.nan
        chi2_error = str(exc)
    else:
        chi2_error = None

    use_fisher = expected is None or np.any(expected < CHI2_EXPECTED_MIN)
    if use_fisher:
        _, p_value = fisher_exact(table2x2.astype(int), alternative="two-sided")
        test_name = "fisher"
        test_stat = math.nan
    else:
        p_value = chi2_p
        test_name = "chi2"
        test_stat = float(chi2)

    or_value, or_low, or_high = _or_and_ci(table2x2)

    rate_a = g1_num / g1_den if g1_den > 0 else math.nan
    rate_b = g2_num / g2_den if g2_den > 0 else math.nan
    diff, diff_lo, diff_hi = _difference_ci(rate_a, g1_den, rate_b, g2_den)

    test_reason = (
        "expected_cells>=5" if test_name == "chi2" else "expected_cell_below_5"
    )
    if use_fisher and chi2_error is not None:
        test_reason = f"chi2_failed__{chi2_error}".replace("__", "_")

    return {
        "dimension": group_col,
        "metric": metric,
        "group_a": str(g1),
        "group_b": str(g2),
        "n_group_a": int(len(g1_df)),
        "n_group_b": int(len(g2_df)),
        "n1": g1_den,
        "n2": g2_den,
        "num_a": g1_num,
        "num_b": g2_num,
        "test": test_name,
        "test_reason": test_reason,
        "p_value_raw": float(p_value),
        "p_value_fdr": math.nan,
        "reject_fdr": False,
        "test_statistic": float(test_stat),
        "odds_ratio": or_value,
        "or_ci_low": or_low,
        "or_ci_high": or_high,
        "rate_a": rate_a,
        "rate_b": rate_b,
        "rate_diff": diff,
        "rate_diff_ci_low": diff_lo,
        "rate_diff_ci_high": diff_hi,
        "n_cell_min": n_cell_min,
        "null_hypothesis": _null_hypothesis(metric),
        "alternative": _alternative_hypothesis(metric),
    }


def _null_hypothesis(metric: str) -> str:
    if metric == "tpr":
        return "P(y_hat=1 | y=1, g=a) = P(y_hat=1 | y=1, g=b)"
    if metric == "ppv":
        return "P(y=1 | y_hat=1, g=a) = P(y=1 | y_hat=1, g=b)"
    raise ValueError(f"Unsupported metric: {metric}")


def _alternative_hypothesis(metric: str) -> str:
    if metric == "tpr":
        return "TPR differs between groups in the same comparison"
    if metric == "ppv":
        return "PPV differs between groups in the same comparison"
    raise ValueError(f"Unsupported metric: {metric}")


def _analyze_fairness(
    df: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dimensions = ["nome_qtd_quartis", "bandeira", "faixa_idade", "sexo", "uf"]
    metrics = ["tpr", "ppv"]

    group_rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []

    for dim in dimensions:
        dim_series = df[dim].astype("string")
        dim_counts = dim_series.value_counts(dropna=False)
        eligible_values = sorted(
            str(group)
            for group, n_rows in dim_counts.items()
            if pd.notna(group)
            and str(group) != "MISSING"
            and int(n_rows) >= config.n_min_cell
        )
        if len(eligible_values) < 1:
            continue

        dim_df = df.loc[dim_series.isin(eligible_values)].copy()

        for metric in metrics:
            group_rows.append(_build_group_metrics(dim_df, dim, metric))

            if len(eligible_values) >= 2:
                for g1, g2 in combinations(eligible_values, 2):
                    tests.append(
                        _build_pairwise_test(
                            df=dim_df,
                            group_col=dim,
                            metric=metric,
                            g1=g1,
                            g2=g2,
                            alpha=config.alpha,
                        )
                    )

    group_metrics = (
        pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    )
    tests_df = pd.DataFrame(tests)

    # Benjamini-Hochberg over all completed tests.
    pvals = tests_df["p_value_raw"].tolist()
    pvals_adj = _benjamini_hochberg(
        [float(v) if pd.notna(v) else math.nan for v in pvals], alpha=config.alpha
    )
    tests_df["p_value_fdr"] = pvals_adj
    tests_df["reject_fdr"] = tests_df["p_value_fdr"].le(config.alpha)

    # Keep one stable column order for easier review.
    ordered_cols = [
        "dimension",
        "metric",
        "group_a",
        "group_b",
        "test",
        "test_reason",
        "null_hypothesis",
        "alternative",
        "n_group_a",
        "n_group_b",
        "n1",
        "n2",
        "num_a",
        "num_b",
        "n_cell_min",
        "test_statistic",
        "p_value_raw",
        "p_value_fdr",
        "reject_fdr",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
        "rate_a",
        "rate_b",
        "rate_diff",
        "rate_diff_ci_low",
        "rate_diff_ci_high",
    ]
    tests_df = tests_df[[c for c in ordered_cols if c in tests_df.columns]]

    stability_rows: list[dict[str, Any]] = []
    for band in ["0-6", "6-11", "11-25", ">25"]:
        if band == "0-6":
            band_mask = df["score_band"] == "0-6"
        elif band == "6-11":
            band_mask = df["score_band"] == "6-11"
        elif band == "11-25":
            band_mask = df["score_band"] == "11-25"
        else:
            band_mask = df["score_band"] == ">25"

        band_df = df.loc[band_mask]
        if len(band_df) == 0:
            stability_rows.append(
                {
                    "score_band": band,
                    "n_rows": 0,
                    "n_true": 0,
                    "n_pred_pos": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "tpr": math.nan,
                    "tpr_ci_low": math.nan,
                    "tpr_ci_high": math.nan,
                    "ppv": math.nan,
                    "ppv_ci_low": math.nan,
                    "ppv_ci_high": math.nan,
                }
            )
            continue

        y_true = band_df["y_true"].to_numpy()
        y_pred = band_df["y_pred"].to_numpy()
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else math.nan
        tpr_lo, tpr_hi = _wilson_ci(tp, tp + fn)
        ppv_lo, ppv_hi = _wilson_ci(tp, tp + fp)

        stability_rows.append(
            {
                "score_band": band,
                "n_rows": int(len(band_df)),
                "n_true": int((band_df["y_true"] == 1).sum()),
                "n_pred_pos": int((band_df["y_pred"] == 1).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tpr": tpr,
                "tpr_ci_low": tpr_lo,
                "tpr_ci_high": tpr_hi,
                "ppv": ppv,
                "ppv_ci_low": ppv_lo,
                "ppv_ci_high": ppv_hi,
            }
        )

    stability_df = pd.DataFrame(stability_rows)
    return group_metrics, tests_df, stability_df


def _write_artifacts(
    df: pd.DataFrame,
    group_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    manifest: dict[str, Any],
    extra: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "qw3_analysis_frame.csv", index=False)
    group_df.to_csv(output_dir / "qw3_group_metrics.csv", index=False)
    tests_df.to_csv(output_dir / "qw3_pairwise_tests.csv", index=False)
    stability_df.to_csv(output_dir / "qw3_score_band_stability.csv", index=False)

    manifest_path = output_dir / "qw3_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    extra_path = output_dir / "qw3_notes.json"
    extra_path.write_text(
        json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_memo(
    cfg: Config,
    tests_df: pd.DataFrame,
    group_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    total_tests = len(tests_df)
    completed = int(tests_df["test"].ne("skipped").sum())
    skipped = total_tests - completed

    adjusted = tests_df.copy()
    sig = adjusted[adjusted["reject_fdr"] == True]

    lines = [
        "# Memorando de Interpretação (Quick Win 3)",
        "",
        "## Método resumido",
        "A análise usa como critério de decisão binário a coluna de score informada",
        "e avalia paridade de TPR e PPV entre fatias de estratificação,",
        "com correção de múltiplas comparações por Benjamini-Hochberg (FDR=0.05).",
        "",
        f"- Arquivo de decisão: ``{cfg.score_col}``, limiar={cfg.threshold:.4g}",
        f"- Arquivo de score para faixa de estabilidade: ``{cfg.score_band_col}``",
        "- Semente global (reprodutibilidade): " + str(cfg.seed),
        f"- Limite de agregação em UF: n >= {cfg.n_min_cell}",
        "",
        "## Resultado de aderência à framework",
        f"- Testes realizados: {completed} de {total_tests} planejados.",
        f"- Testes pulados por condições de teste (ex.: grupo vazio ou denominador zero): {skipped}.",
        f"- UF com contagem < {cfg.n_min_cell} foram agregadas por região (regra aplicada na etapa de preparação).",
    ]

    if sig.empty:
        lines.extend(
            [
                "",
                "## Achados (pós-FDR)",
                "- Nenhum par de grupo apresentou evidência forte de desequilíbrio após FDR.",
                "- A leitura para discussão é de ausência de violação robusta de paridade nas comparações avaliadas com dados disponíveis.",
            ]
        )
    else:
        lines.extend(["", "## Achados (pós-FDR)"])
        for _, row in sig.sort_values("p_value_fdr").head(10).iterrows():
            lines.append(
                (
                    f"- {row['dimension']} | {row['metric'].upper()} | "
                    f"{row['group_a']} x {row['group_b']}: p_FDR={row['p_value_fdr']:.4g}, "
                    f"diferença={row['rate_diff']:.3f} [{row['rate_diff_ci_low']:.3f}, {row['rate_diff_ci_high']:.3f}], "
                    f"teste={row['test']}"
                )
            )

    # Group-level snapshot useful for thesis writing (parcial por dimensão).
    for metric in ["tpr", "ppv"]:
        subset = group_df[group_df["metric"] == metric]
        if subset.empty:
            continue

        lines.extend(
            [
                "",
                f"## Resumo por dimensão ({metric.upper()})",
            ]
        )
        for dimension in sorted(subset["dimension"].unique()):
            dim_rows = subset[subset["dimension"] == dimension]
            span = dim_rows[
                [
                    "group",
                    "rate",
                    "wilson_low",
                    "wilson_high",
                    "n_total",
                    "n_true",
                    "n_pred_pos",
                ]
            ]
            lines.append(f"### {dimension}")
            for _, r in dim_rows.iterrows():
                lines.append(
                    (
                        f"- {r['group']}: taxa={r['rate']:.4f} "
                        f"[{r['wilson_low']:.4f}, {r['wilson_high']:.4f}], "
                        f"n={r['n_total']}"
                    )
                )

    memo_path = output_dir / "qw3_interpretation_memo.md"
    memo_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick Win 3 fairness analysis for record linkage outputs."
    )
    parser.add_argument(
        "--input-csv",
        default="data/COMPARADORSEMIDENT.csv",
        help="Input dataset CSV.",
    )
    parser.add_argument(
        "--output-dir", default="data/qw3", help="Output directory for results."
    )
    parser.add_argument("--sep", default=";", help="CSV separator used by input file.")
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--target-col", default="PAR", help="Ground-truth label column."
    )
    parser.add_argument(
        "--target-pos-values",
        default="1,2",
        help="Comma-separated labels for target=positive.",
    )
    parser.add_argument(
        "--score-col",
        default="nota final",
        help="Score/probability used to derive binary prediction.",
    )
    parser.add_argument(
        "--prediction-col",
        default="",
        help="Optional direct binary prediction column (if already available).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="Threshold to binarize the score when --prediction-col is not set.",
    )
    parser.add_argument(
        "--score-band-col",
        default="SCORE,C,15,0",
        help="Score column used for stability bands.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_DEFAULT,
        help="Test alpha and CI level complement.",
    )
    parser.add_argument(
        "--n-min-cell",
        type=int,
        default=N_MIN_CELL_DEFAULT,
        help="Minimum number of observations per 2x2 cell.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    target_pos_values = tuple(
        int(v.strip()) for v in args.target_pos_values.split(",") if v.strip()
    )
    if not target_pos_values:
        raise ValueError("--target-pos-values must contain at least one integer")

    cfg = Config(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        sep=args.sep,
        seed=args.seed,
        target_col=args.target_col,
        target_pos_values=target_pos_values,
        score_col=args.score_col,
        threshold=float(args.threshold),
        prediction_col=args.prediction_col or None,
        score_band_col=args.score_band_col,
        alpha=float(args.alpha),
        n_min_cell=int(args.n_min_cell),
    )

    df, prep = _prepare_fairness_frame(cfg)
    group_df, tests_df, stability_df = _analyze_fairness(df, cfg)

    manifest: dict[str, Any] = {
        "analysis_name": "quick_win_3_fairness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "seed": cfg.seed,
        "rng_state_probe": float(rng.random()),
        "platform": platform.platform(),
        "input_csv": str(cfg.input_csv),
        "sep": cfg.sep,
        "target_col": cfg.target_col,
        "target_pos_values": list(cfg.target_pos_values),
        "score_col": cfg.score_col,
        "prediction_col": cfg.prediction_col,
        "threshold": cfg.threshold,
        "score_band_col": cfg.score_band_col,
        "alpha": cfg.alpha,
        "n_min_cell": cfg.n_min_cell,
        "rows_input": int(len(df)),
        "rows_with_target": int(df["y_true"].notna().sum()),
        "rows_with_prediction": int(df["y_pred"].notna().sum()),
        "git": {"sha": _safe_git_sha()},
        "preparation": prep,
    }

    artifact_summary: dict[str, Any] = {
        "files": {
            "analysis_frame": "qw3_analysis_frame.csv",
            "group_metrics": "qw3_group_metrics.csv",
            "pairwise_tests": "qw3_pairwise_tests.csv",
            "score_band_stability": "qw3_score_band_stability.csv",
            "interpretation": "qw3_interpretation_memo.md",
            "manifest": "qw3_manifest.json",
            "notes": "qw3_notes.json",
        },
        "notes": [
            "TPR parity and PPV parity are reported with explicit null/alternative hypotheses.",
            "Chi² is used when expected counts in all cells are >=5; otherwise Fisher exact is used.",
            "FDR uses Benjamini-Hochberg over all executed 2x2 tests.",
            "UFs with fewer than 30 rows are mapped to regions through BR state mapping.",
        ],
        "acceptance": {
            "quartis_nome_qtd_frag_iguais": "Q1-Q4 via pd.qcut",
            "bandeira_as_binary": "C_SITUENCE in {3,4}",
            "faixa_idade": "derived from R_DTOBITO - R_DTNASC",
            "seed_set": True,
            "fdr_documented": True,
        },
    }

    _write_artifacts(
        df, group_df, tests_df, stability_df, manifest, artifact_summary, cfg.output_dir
    )
    _build_memo(cfg, tests_df, group_df, cfg.output_dir)


if __name__ == "__main__":
    main()
