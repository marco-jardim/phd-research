"""
audit_tables.py — Auditoria de consistência entre tabelas LaTeX e CSVs canônicos.

Verifica cada tabela da tese contra sua fonte de dados, reportando:
    OK          — todos os valores conferem dentro da tolerância
    DIVERGENCIA — ao menos um valor diverge (lista detalhada)
    INCONCLUSIVO — fonte ausente ou parse falhou

Uso:
    python scripts/audit_tables.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos base
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO / "text" / "latex" / "tables"
DATA_DIR = REPO / "data"

# ---------------------------------------------------------------------------
# Tolerâncias
# ---------------------------------------------------------------------------
ATOL_METRIC = 0.001   # métricas 0-1 (F1, P, R, etc.)
ATOL_PCT = 0.01       # percentuais (0-100)
RTOL_FEAT = 0.005     # feature importance (relativo)

# ---------------------------------------------------------------------------
# Estado global do relatório
# ---------------------------------------------------------------------------
_results: list[dict[str, Any]] = []
_n_tables = 0


# ===========================================================================
# Utilitários de comparação
# ===========================================================================

def _close(a: float, b: float, atol: float = ATOL_METRIC, rtol: float = 0.0) -> bool:
    """Retorna True se |a-b| <= atol + rtol*|b|."""
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= atol + rtol * abs(b)


# ===========================================================================
# Utilitários de parsing LaTeX
# ===========================================================================

# Padrão numérico: suporta ponto ou vírgula como decimal
_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_NUM_RE = re.compile(_NUM)


def _preprocess_latex_cell(text: str) -> str:
    """
    Pré-processa uma célula LaTeX antes de extrair números:
    - Substitui {,} (decimal português) por .
    - Remove \\, (separador de milhar LaTeX) → espaço
    - Remove outros comandos LaTeX comuns
    """
    # {,} → decimal point
    text = text.replace("{,}", ".")
    # \\, → empty (thousands separator in LaTeX)
    text = re.sub(r"\\,", "", text)
    # \textbf{...}, \textit{...} → conteúdo
    text = re.sub(r"\\text[a-z]+\{([^}]*)\}", r"\1", text)
    # ${...}$ → conteúdo
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    return text


def _nums(text: str) -> list[float]:
    """Extrai todos os números de uma string LaTeX (substitui vírgula por ponto)."""
    text = _preprocess_latex_cell(text)
    return [float(m.replace(",", ".")) for m in _NUM_RE.findall(text)]


def _parse_mean_std(cell: str) -> tuple[float, float] | None:
    """
    Parse de células como '0.924 $\\pm$ 0.030' ou '0{,}924 $\\pm$ 0{,}030'.
    Retorna (mean, std) ou None se parse falhar.
    """
    nums = _nums(cell)
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


def _clean_feature_name(raw: str) -> str:
    """Limpa o nome de feature LaTeX removendo comandos e backslashes."""
    # Remove comandos LaTeX como \textbf{}, \emph{}, etc.
    name = re.sub(r"\\text[a-z]+\{([^}]*)\}", r"\1", raw)
    # Remove outros comandos \cmd{...}
    name = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", name)
    # Remove backslash antes de underscores e outros caracteres (LaTeX escaping)
    name = re.sub(r"\\([_^&%$#{}])", r"\1", name)
    # Remove remaining backslashes before non-alpha chars
    name = re.sub(r"\\([^a-zA-Z])", r"\1", name)
    # Remove isolated backslashes before end-of-string
    name = name.replace("\\", "")
    # Remove remaining braces
    name = re.sub(r"[{}]", "", name)
    return name.strip()


def _data_rows(tex_path: Path) -> list[str]:
    """Devolve as linhas de dados entre \\midrule e \\bottomrule."""
    text = tex_path.read_text(encoding="utf-8")
    match = re.search(r"\\midrule(.+?)\\bottomrule", text, re.DOTALL)
    if not match:
        return []
    body = match.group(1)
    rows = []
    for line in body.split("\n"):
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("\\midrule"):
            continue
        if "\\\\" in line:
            rows.append(line)
    return rows


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Lê CSV com detecção automática de separador. Retorna None se não existe."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception as e:
        print(f"  [ERRO lendo CSV] {path}: {e}")
        return None


# ===========================================================================
# Motor de relatório
# ===========================================================================

def _report(
    idx: int,
    name: str,
    source: str,
    checks: list[str],
    divergences: list[str],
    inconclusivo: bool = False,
    inconclusivo_reason: str = "",
) -> None:
    total = len(checks)
    n_ok = sum(1 for c in checks if c.strip().startswith("[OK]"))

    if inconclusivo:
        status = "INCONCLUSIVO"
        summary = inconclusivo_reason or "sem fonte canônica"
    elif divergences:
        status = "DIVERGENCIA"
        summary = f"{len(divergences)} valor(es) diverge(m)"
    else:
        status = "OK"
        summary = f"{n_ok}/{total} valores conferem" if total > 0 else "sem itens verificados"

    _results.append(
        {
            "idx": idx,
            "name": name,
            "source": source,
            "status": status,
            "summary": summary,
            "checks": checks,
            "divergences": divergences,
        }
    )

    print(f"\n[{idx}/{_n_tables}] {name}")
    print(f"  Fonte: {source}")
    for line in checks:
        print(line)
    print(f"  RESULTADO: {status} ({summary})")


# ===========================================================================
# Checagens individuais
# ===========================================================================

def _check_val(
    label: str,
    got: float,
    expected: float,
    checks: list[str],
    divergences: list[str],
    atol: float = ATOL_METRIC,
    rtol: float = 0.0,
) -> None:
    ok = _close(got, expected, atol=atol, rtol=rtol)
    sym = "[OK]" if ok else "[DIVERGENCIA]"
    line = f"  {sym} {label}: tabela={got:.4f} vs csv={expected:.4f}"
    checks.append(line)
    if not ok:
        divergences.append(line)


def _check_mean_std(
    label: str,
    tex_mean: float,
    tex_std: float,
    csv_mean: float,
    csv_std: float,
    checks: list[str],
    divergences: list[str],
    atol: float = ATOL_METRIC,
) -> None:
    ok_m = _close(tex_mean, csv_mean, atol=atol)
    ok_s = _close(tex_std, csv_std, atol=atol)
    ok = ok_m and ok_s
    sym = "[OK]" if ok else "[DIVERGENCIA]"
    line = (
        f"  {sym} {label}: "
        f"tex={tex_mean:.3f}±{tex_std:.3f} csv={csv_mean:.3f}±{csv_std:.3f}"
    )
    checks.append(line)
    if not ok:
        divergences.append(line)


# ===========================================================================
# TAB 1 — tab_cv_5fold.tex → data/cv_5fold_results.csv
# ===========================================================================

def audit_cv_5fold(idx: int) -> None:
    name = "tab_cv_5fold.tex"
    src = "data/cv_5fold_results.csv"
    csv_path = DATA_DIR / "cv_5fold_results.csv"
    df = _read_csv(csv_path)
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    # Configurações esperadas: (label, config_substring_csv, f1_mean, f1_std, prec_mean, prec_std, rec_mean, rec_std)
    expected_rows = [
        ("XGBoost>=0.5",        "XGBoost",             0.9243, 0.0302, 0.9513, 0.0356, 0.9007, 0.0484),
        ("RF+SMOTE>=0.5",       "RF+SMOTE",            0.9156, 0.0258, 0.9294, 0.0222, 0.9029, 0.0412),
        ("GB>=0.5",             "GB",                  0.8984, 0.0323, 0.9385, 0.0283, 0.8623, 0.0442),
        ("Rules>=7",            "Rules",               0.7917, 0.0341, 0.8983, 0.0367, 0.7087, 0.0444),
        ("H-AND RF+SMOTE+Rul7", "Hybrid-AND RF",       0.8272, 0.0287, 0.9947, 0.0118, 0.7087, 0.0444),
        ("H-AND GB+Rul6",       "Hybrid-AND GB",       0.8634, 0.0241, 0.9860, 0.0312, 0.7693, 0.0413),
        ("H-OR RF+SMOTE+Rul8",  "Hybrid-OR RF",        0.8977, 0.0245, 0.9674, 0.0125, 0.8380, 0.0405),
        ("Naive>=8",            "Naive",               0.5709, 0.0571, 0.6495, 0.0736, 0.5147, 0.0765),
    ]

    for label, cfg_pat, ef1, ef1s, ep, eps, er, ers in expected_rows:
        # Usar regex=False para evitar que + seja interpretado como quantificador
        row = df[df["config"].str.contains(cfg_pat, na=False, regex=False)]
        if row.empty:
            checks.append(f"  [INCONCLUSIVO] {label}: config '{cfg_pat}' não encontrada no CSV")
            continue
        row = row.iloc[0]
        _check_mean_std(f"{label} F1", ef1, ef1s, row["f1_mean"], row["f1_std"], checks, divs)
        _check_mean_std(f"{label} P",  ep,  eps,  row["prec_mean"], row["prec_std"], checks, divs)
        _check_mean_std(f"{label} R",  er,  ers,  row["rec_mean"], row["rec_std"], checks, divs)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 2 — tab_cv_results.tex → data/kfold_cv_results.csv
# ===========================================================================

def audit_cv_results(idx: int) -> None:
    name = "tab_cv_results.tex"
    src = "data/kfold_cv_results.csv"
    df = _read_csv(DATA_DIR / "kfold_cv_results.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    expected = [
        # (label, config_pattern, P_mean, P_std, R_mean, R_std, F1_mean, F1_std)
        ("Hybrid-AND GB>=0.5+Rul7", "Hybrid-AND GB",   0.989, 0.016, 0.696, 0.002, 0.817, 0.006),
        ("Hybrid-AND GB>=0.6+Rul6", "Hybrid-AND GB",   0.975, 0.006, 0.785, 0.022, 0.870, 0.011),
        ("Hybrid-AND RF+Rul7",      "Hybrid-AND RF",   0.952, 0.034, 0.708, 0.010, 0.812, 0.019),
        ("ML-only GB>=0.5",         "ML-only GB",      0.948, 0.011, 0.883, 0.025, 0.914, 0.017),
        ("Rules-only>=7",           "Rules-only",      0.897, 0.015, 0.708, 0.010, 0.792, 0.012),
    ]

    for label, pat, ep, eps, er, ers, ef1, ef1s in expected:
        row = df[df["config"].str.contains(pat, na=False, regex=False)]
        if row.empty:
            checks.append(f"  [INCONCLUSIVO] {label}: config '{pat}' não encontrada")
            continue
        # Se múltiplos matches, escolher pelo F1 mais próximo
        if len(row) > 1:
            row = row.iloc[(row["f1_mean"] - ef1).abs().argsort()[:1]]
        row = row.iloc[0]
        _check_mean_std(f"{label} P",  ep, eps, row["precision_mean"], row["precision_std"], checks, divs)
        _check_mean_std(f"{label} R",  er, ers, row["recall_mean"], row["recall_std"], checks, divs)
        _check_mean_std(f"{label} F1", ef1, ef1s, row["f1_mean"], row["f1_std"], checks, divs)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 3 — tab_imbalance_sensitivity.tex → data/imbalance_sensitivity.csv
# ===========================================================================

def audit_imbalance_sensitivity(idx: int) -> None:
    name = "tab_imbalance_sensitivity.tex"
    src = "data/imbalance_sensitivity.csv"
    df = _read_csv(DATA_DIR / "imbalance_sensitivity.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    tex_path = TABLES_DIR / name
    if not tex_path.exists():
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason=".tex não encontrado")
        return

    rows = _data_rows(tex_path)

    for tex_row in rows:
        parts = tex_row.rstrip("\\\\").split("&")
        if len(parts) < 4:
            continue
        strat_tex = parts[0].strip()
        # Limpar comandos LaTeX
        strat_clean = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", strat_tex).strip()
        strat_clean = re.sub(r"[{}\\]", "", strat_clean).strip()

        f1_ms  = _parse_mean_std(parts[1])
        p_ms   = _parse_mean_std(parts[2])
        r_ms   = _parse_mean_std(parts[3])
        if not (f1_ms and p_ms and r_ms):
            checks.append(f"  [INCONCLUSIVO] {strat_clean}: parse falhou")
            continue

        # Encontrar no CSV por substring (regex=False — usar prefixo)
        strat_norm = strat_clean.lower().strip()
        best_row = None
        best_score = -1
        for _, csv_row in df.iterrows():
            csv_strat = str(csv_row["strategy"]).lower()
            # Comparação por token comum
            score = sum(1 for tok in strat_norm.split() if tok in csv_strat)
            if score > best_score:
                best_score = score
                best_row = csv_row

        if best_row is None:
            checks.append(f"  [INCONCLUSIVO] {strat_clean}: não encontrado no CSV")
            continue

        _check_mean_std(f"{strat_clean} F1", f1_ms[0], f1_ms[1],
                        best_row["f1_mean"], best_row["f1_std"], checks, divs)
        _check_mean_std(f"{strat_clean} P",  p_ms[0], p_ms[1],
                        best_row["prec_mean"], best_row["prec_std"], checks, divs)
        _check_mean_std(f"{strat_clean} R",  r_ms[0], r_ms[1],
                        best_row["rec_mean"], best_row["rec_std"], checks, divs)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 4 — tab_ablation_best_category.tex → data/ablation_results.csv
# ===========================================================================

def audit_ablation_best_category(idx: int) -> None:
    name = "tab_ablation_best_category.tex"
    src = "data/ablation_results.csv"
    df = _read_csv(DATA_DIR / "ablation_results.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    # Valores esperados: (categoria, best_f1_RF+SMOTE, best_f1_XGBoost)
    expected = [
        ("naive-threshold",  0.610, 0.610),
        ("rules-only",       0.842, 0.842),
        ("ml-only",          0.931, 0.943),
        ("hybrid-and",       0.912, 0.442),
        ("hybrid-or",        0.921, 0.943),
        ("cascade-ml-rules", 0.853, 0.000),
        ("cascade-rules-ml", 0.912, 0.442),
        ("consensus-ml",     0.928, 0.943),
        ("consensus-hybrid", 0.896, 0.150),
    ]

    for cat, exp_rf, exp_xgb in expected:
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            checks.append(f"  [INCONCLUSIVO] {cat}: categoria não encontrada no CSV")
            continue

        # RF+SMOTE column: configs com RF+SMOTE mas SEM XGBoost exclusivo
        rf_rows = cat_df[
            cat_df["config"].str.contains("RF", na=False, regex=False) &
            ~cat_df["config"].str.contains("XGBoost", na=False, regex=False)
        ]
        # XGBoost column: configs com XGBoost
        xgb_rows = cat_df[cat_df["config"].str.contains("XGBoost", na=False, regex=False)]

        # Para naive-threshold e rules-only não há distinção de modelo
        if cat in ("naive-threshold", "rules-only"):
            rf_rows = cat_df
            xgb_rows = cat_df

        # Para consensus: se não há RF+SMOTE exclusivo, usar configs sem XGBoost
        if rf_rows.empty and cat.startswith("consensus"):
            rf_rows = cat_df[~cat_df["config"].str.contains("XGBoost", na=False, regex=False)]

        if rf_rows.empty:
            rf_rows = cat_df

        rf_best  = rf_rows["f1"].max()  if not rf_rows.empty  else float("nan")
        xgb_best = xgb_rows["f1"].max() if not xgb_rows.empty else float("nan")

        _check_val(f"{cat} RF+SMOTE bestF1", rf_best, exp_rf, checks, divs)
        _check_val(f"{cat} XGBoost bestF1",  xgb_best, exp_xgb, checks, divs)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 5 — tab_ablation_top10.tex → data/ablation_results.csv
# ===========================================================================

def audit_ablation_top10(idx: int) -> None:
    name = "tab_ablation_top10.tex"
    src = "data/ablation_results.csv"
    df = _read_csv(DATA_DIR / "ablation_results.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    top10 = df.nlargest(10, "f1").reset_index(drop=True)

    # #1 ML-only XGBoost>=0.7 F1=0.943
    row0 = top10.iloc[0]
    _check_val("#1 F1", row0["f1"], 0.943, checks, divs)
    if "xgboost" in row0["config"].lower():
        checks.append(f"  [OK] #1 config contém 'XGBoost': {row0['config'][:50]}")
    else:
        msg = f"  [DIVERGENCIA] #1 esperada XGBoost, obtido: {row0['config'][:50]}"
        checks.append(msg)
        divs.append(msg)

    # Verificar RF+SMOTE sem XGBoost no top10
    rf_in_top10 = top10[top10["config"].str.contains("RF", na=False, regex=False) &
                        ~top10["config"].str.contains("XGBoost", na=False, regex=False)]
    if not rf_in_top10.empty:
        rf_best = rf_in_top10.iloc[0]
        _check_val("#? RF+SMOTE F1 no top10", rf_best["f1"], 0.931, checks, divs)
        checks.append(f"  [OK] RF+SMOTE presente no top10: #{rf_best.name+1} {rf_best['config'][:50]}")
    else:
        msg = "  [DIVERGENCIA] Nenhuma config RF+SMOTE sem XGBoost encontrada no top10"
        checks.append(msg)
        divs.append(msg)

    # Verificar Consensus ML (multi-model, não XGBoost-only) — pode estar
    # fora do top10 do CSV (ties com XGBoost configs), mas deve existir na tabela
    # Buscar no DataFrame completo
    cons_df = df[df["config"].str.contains("Consensus ML-majority", na=False, regex=False) &
                 ~df["config"].str.contains("XGBoost", na=False, regex=False)]
    if not cons_df.empty:
        cons_best = cons_df.nlargest(1, "f1").iloc[0]
        _check_val("Consensus ML melhor F1 no CSV", cons_best["f1"], 0.928, checks, divs)
        checks.append(f"  [OK] Consensus ML presente no CSV: {cons_best['config'][:50]}")
    else:
        msg = "  [DIVERGENCIA] Nenhuma config 'Consensus ML-majority' sem XGBoost no CSV"
        checks.append(msg)
        divs.append(msg)

    # Verificar contagem top10
    if len(top10) == 10:
        checks.append("  [OK] Top-10 contém exatamente 10 linhas")
    else:
        msg = f"  [DIVERGENCIA] Top-10 contém {len(top10)} linhas"
        checks.append(msg)
        divs.append(msg)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 6 — tab_pareto_frontier.tex → data/pareto_grid.csv
# ===========================================================================

def audit_pareto_frontier(idx: int) -> None:
    name = "tab_pareto_frontier.tex"
    src = "data/pareto_grid.csv"
    df = _read_csv(DATA_DIR / "pareto_grid.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    tex_path = TABLES_DIR / name
    rows = _data_rows(tex_path)

    for row in rows:
        if "\\midrule" in row or not row.strip():
            continue
        parts = row.rstrip("\\\\").split("&")
        if len(parts) < 6:
            continue
        model = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", parts[0]).strip()
        model = re.sub(r"[{}\\]", "", model).strip()

        thr_ml_nums  = _nums(parts[1])
        thr_rul_nums = _nums(parts[2])
        prec_nums    = _nums(parts[3])
        rec_nums     = _nums(parts[4])
        f1_nums      = _nums(parts[5])

        if not (thr_ml_nums and prec_nums and rec_nums and f1_nums):
            checks.append(f"  [INCONCLUSIVO] {model} {parts[1].strip()}: parse falhou")
            continue

        thr_ml  = thr_ml_nums[0]
        thr_rul = thr_rul_nums[0] if thr_rul_nums else 0.0
        tex_p   = prec_nums[0]
        tex_r   = rec_nums[0]
        tex_f1  = f1_nums[0]

        model_csv = "RF+SMOTE" if "RF" in model or "SMOTE" in model else model

        csv_row = df[
            (df["ml_model"] == model_csv) &
            (abs(df["ml_threshold"] - thr_ml) < 0.001) &
            (abs(df["rules_threshold"] - thr_rul) < 0.01)
        ]
        if csv_row.empty:
            checks.append(
                f"  [INCONCLUSIVO] {model_csv} ml={thr_ml} rul={thr_rul}: "
                f"não encontrado no CSV"
            )
            continue
        csv_row = csv_row.iloc[0]
        label = f"{model_csv}(ml={thr_ml},rul={thr_rul})"
        _check_val(f"{label} P",  tex_p,  csv_row["precision"], checks, divs)
        _check_val(f"{label} R",  tex_r,  csv_row["recall"],    checks, divs)
        _check_val(f"{label} F1", tex_f1, csv_row["f1"],        checks, divs)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 7 — tab_qw1_deep_learning.tex → data/qw1/cv_summary.csv + holdout_metrics.csv
# ===========================================================================

def audit_qw1_deep_learning(idx: int) -> None:
    name = "tab_qw1_deep_learning.tex"
    src = "data/qw1/cv_summary.csv + data/qw1/holdout_metrics.csv"
    cv_df = _read_csv(DATA_DIR / "qw1" / "cv_summary.csv")
    ho_df = _read_csv(DATA_DIR / "qw1" / "holdout_metrics.csv")
    checks: list[str] = []
    divs: list[str] = []

    if cv_df is None or ho_df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True,
                inconclusivo_reason="CSV(s) não encontrado(s)")
        return

    # Esperado CV: (model_pattern, f1_mean, f1_std, prec_mean, prec_std, rec_mean, rec_std)
    # Valores do .tex: XGBoost 0,924±0,030  P=0,951±0,031  R=0,901±0,046
    expected_cv = [
        ("XGBoost",               0.924, 0.030, 0.951, 0.031, 0.901, 0.046),
        ("RandomForest",          0.913, 0.029, 0.972, 0.022, 0.863, 0.049),
        (r"MLP.*64.*32|MLP.*64,.*32",  0.882, 0.037, 0.956, 0.032, 0.822, 0.059),
        (r"MLP.*128.*64|MLP.*128,.*64", 0.872, 0.038, 0.938, 0.048, 0.820, 0.057),
        ("TabNet",                0.821, 0.055, 0.871, 0.051, 0.789, 0.074),
    ]

    for pat, ef1, ef1s, ep, eps, er, ers in expected_cv:
        row = cv_df[cv_df["model"].str.contains(pat, na=False, regex=True)]
        if row.empty:
            checks.append(f"  [INCONCLUSIVO] CV/{pat}: não encontrado")
            continue
        row = row.iloc[0]
        # tex = spec/tabela, csv = dados reais
        _check_mean_std(f"CV {row['model']} F1",
                        ef1, ef1s, row["f1_mean"], row["f1_std"], checks, divs)
        _check_mean_std(f"CV {row['model']} P",
                        ep, eps, row["precision_mean"], row["precision_std"], checks, divs)
        _check_mean_std(f"CV {row['model']} R",
                        er, ers, row["recall_mean"], row["recall_std"], checks, divs)

    # Esperado Holdout: (model_pattern, P, R, F1)
    expected_ho = [
        ("XGBoost",               1.000, 0.851, 0.920),
        ("RandomForest",          0.982, 0.757, 0.855),
        (r"MLP.*64.*32|MLP.*64,.*32",  0.897, 0.824, 0.859),
        (r"MLP.*128.*64|MLP.*128,.*64", 1.000, 0.730, 0.844),
        ("TabNet",                0.524, 0.730, 0.610),
    ]

    for pat, ep, er, ef1 in expected_ho:
        row = ho_df[ho_df["model"].str.contains(pat, na=False, regex=True)]
        if row.empty:
            checks.append(f"  [INCONCLUSIVO] Holdout/{pat}: não encontrado")
            continue
        row = row.iloc[0]
        _check_val(f"HO {row['model']} P",  ep,  row["precision"], checks, divs)
        _check_val(f"HO {row['model']} R",  er,  row["recall"],    checks, divs)
        _check_val(f"HO {row['model']} F1", ef1, row["f1"],        checks, divs)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 8 — tab_impacto_epidemiologico.tex → data/epidemiological_impact.csv
# ===========================================================================

def audit_impacto_epidemiologico(idx: int) -> None:
    name = "tab_impacto_epidemiologico.tex"
    src = "data/epidemiological_impact.csv"
    df = _read_csv(DATA_DIR / "epidemiological_impact.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    tex_path = TABLES_DIR / name
    rows = _data_rows(tex_path)

    def _normalize_method(s: str) -> str:
        """Normaliza nome de método para comparação: remove LaTeX, normaliza espaços e >=."""
        s = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", " ", s)
        s = re.sub(r"\$", " ", s)
        s = re.sub(r"\\geq", ">=", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def _method_score(method_norm: str, csv_method_norm: str) -> float:
        """Pontua similaridade entre dois métodos normalizados."""
        # Dividir em tokens e incluir tokens numéricos
        tok_m = set(method_norm.split())
        tok_c = set(csv_method_norm.split())
        # Score = número de tokens comuns (incluindo números)
        return len(tok_m & tok_c)

    for tex_row in rows:
        parts = tex_row.rstrip("\\\\").split("&")
        if len(parts) < 7:
            continue
        method_tex = _normalize_method(parts[0])
        nums_in_row = [_nums(p) for p in parts[1:]]

        try:
            det  = nums_in_row[0][0] if nums_in_row[0] else None
            miss = nums_in_row[1][0] if nums_in_row[1] else None
            rev  = nums_in_row[2][0] if nums_in_row[2] else None
            prec = nums_in_row[3][0] if nums_in_row[3] else None
            rec  = nums_in_row[4][0] if nums_in_row[4] else None
            pct  = nums_in_row[5][0] if nums_in_row[5] else None
        except Exception:
            checks.append(f"  [INCONCLUSIVO] {method_tex}: parse falhou")
            continue

        # Encontrar método no CSV por máximo de tokens comuns
        best_row = None
        best_score = -1.0
        for _, csv_row in df.iterrows():
            csv_method_norm = _normalize_method(str(csv_row["method"]))
            score = _method_score(method_tex, csv_method_norm)
            if score > best_score:
                best_score = score
                best_row = csv_row

        if best_row is None:
            checks.append(f"  [INCONCLUSIVO] {method_tex}: não encontrado no CSV")
            continue

        label = method_tex[:35]
        if det is not None:
            _check_val(f"{label} detected",  det,  best_row["detected"],     checks, divs, atol=0.5)
        if miss is not None:
            _check_val(f"{label} missed",    miss, best_row["missed"],       checks, divs, atol=0.5)
        if rev is not None:
            _check_val(f"{label} to_review", rev,  best_row["to_review"],    checks, divs, atol=0.5)
        if prec is not None:
            _check_val(f"{label} precision", prec, round(best_row["precision"], 2), checks, divs)
        if rec is not None:
            _check_val(f"{label} recall",    rec,  round(best_row["recall"], 2),    checks, divs)
        if pct is not None:
            _check_val(f"{label} pct_det",   pct,
                       round(best_row["pct_of_true_found"], 1), checks, divs, atol=ATOL_PCT)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 9 — tab_score_bands.tex → data/resultados_faixas_score.csv
# ===========================================================================

def audit_score_bands(idx: int) -> None:
    name = "tab_score_bands.tex"
    src = "data/resultados_faixas_score.csv"
    df = _read_csv(DATA_DIR / "resultados_faixas_score.csv")
    checks: list[str] = []
    divs: list[str] = []

    # Valores esperados da tabela LaTeX (dados finais)
    expected_bands = [
        ("0-3",  5336,   0,  0.00),
        ("3-5",  34645,  3,  0.01),
        ("5-6",  14263, 17,  0.12),
        ("6-7",  6175,  57,  0.92),
        ("7-8",  1080,  43,  3.98),
        ("8-9",  102,   34, 33.33),
        ("9-10", 44,    43, 97.73),
        ("10+",  51,    50, 98.04),
    ]
    expected_total_n = sum(b[1] for b in expected_bands)  # 61696

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True,
                inconclusivo_reason="CSV não encontrado")
        return

    # Verificar se o CSV tem a mesma granularidade dos dados
    # O CSV pode conter dados por modelo (linhas duplicadas por faixa)
    col_banda = df.columns[0]  # primeira coluna = faixa
    col_n = None
    col_pares = None
    for c in df.columns:
        cl = c.lower()
        if cl == "n" or "total" in cl:
            col_n = c
        elif "pares" in cl and "pct" not in cl and "%" not in cl:
            col_pares = c

    # Calcular total N no CSV (usando apenas linhas únicas por faixa)
    if col_n:
        unique_bands = df.drop_duplicates(subset=[col_banda])
        csv_total_n = unique_bands[col_n].sum()
        ratio = expected_total_n / csv_total_n if csv_total_n > 0 else float("inf")

        if ratio > 2.0 or ratio < 0.5:
            reason = (
                f"CSV parece ser de experimento diferente: "
                f"total_n_csv={csv_total_n:.0f} vs esperado={expected_total_n}. "
                f"Spec diz: 'CSV pode ter dados parciais — marque como INCONCLUSIVO.'"
            )
            checks.append(f"  [INCONCLUSIVO] {reason}")
            for faixa, exp_n, exp_pares, exp_pct in expected_bands:
                checks.append(f"  [INCONCLUSIVO] faixa {faixa}: n_esperado={exp_n}, pares_esperado={exp_pares}")
            _report(idx, name, src, checks, divs, inconclusivo=True,
                    inconclusivo_reason=f"CSV de subconjunto diferente (n={csv_total_n:.0f} vs {expected_total_n})")
            return

    # Se o total bate, comparar valores
    n_verified = 0
    for faixa, exp_n, exp_pares, exp_pct in expected_bands:
        row = df[df[col_banda].astype(str).str.contains(
            re.escape(faixa), na=False, regex=True)]
        if row.empty:
            checks.append(f"  [INCONCLUSIVO] faixa {faixa}: não encontrada no CSV")
            continue
        row = row.iloc[0]

        if col_n:
            _check_val(f"faixa {faixa} n_total", row[col_n], exp_n, checks, divs, atol=0.5)
            n_verified += 1
        if col_pares:
            _check_val(f"faixa {faixa} pares", row[col_pares], exp_pares, checks, divs, atol=0.5)
            n_verified += 1

    if n_verified == 0:
        _report(idx, name, src, checks, divs, inconclusivo=True,
                inconclusivo_reason="CSV existe mas sem colunas reconhecíveis")
    else:
        _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 10 — tab_exp1_ablation.tex → data/sprint3b/exp1_ablation.csv
# ===========================================================================

def audit_exp1_ablation(idx: int) -> None:
    name = "tab_exp1_ablation.tex"
    src = "data/sprint3b/exp1_ablation.csv"
    df = _read_csv(DATA_DIR / "sprint3b" / "exp1_ablation.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    tex_path = TABLES_DIR / name
    rows = _data_rows(tex_path)

    # Agregar: exp_precision, exp_recall, exp_f1, llm_used por (ablation, mode)
    agg = df.groupby(["ablation", "mode"]).agg(
        prec_mean=("exp_precision", "mean"),
        prec_std=("exp_precision", "std"),
        rec_mean=("exp_recall", "mean"),
        rec_std=("exp_recall", "std"),
        f1_mean=("exp_f1", "mean"),
        f1_std=("exp_f1", "std"),
        llm_mean=("llm_used", "mean"),
        llm_std=("llm_used", "std"),
    ).reset_index()

    for tex_row in rows:
        parts = tex_row.rstrip("\\\\").split("&")
        if len(parts) < 7:
            continue
        ablation_tex = _clean_ablation(parts[0]).strip("_").strip()
        mode_tex = _clean_ablation(parts[1]).strip().lower()

        cells = parts[2:]
        prec_ms = _parse_mean_std(cells[0]) if len(cells) > 0 else None
        rec_ms  = _parse_mean_std(cells[1]) if len(cells) > 1 else None
        f1_ms   = _parse_mean_std(cells[2]) if len(cells) > 2 else None
        llm_ms  = _parse_mean_std(cells[4]) if len(cells) > 4 else None

        # Normalizar: converter LaTeX _ escaping → underscore, lowercase
        abl_norm  = ablation_tex.lower().replace(" ", "_")
        mode_norm = mode_tex

        # Busca exata primeiro (ablation normalizado)
        csv_row = agg[
            (agg["ablation"].str.lower().str.replace(" ", "_") == abl_norm) &
            (agg["mode"].str.lower() == mode_norm)
        ]
        if csv_row.empty:
            # Busca por prefixo (sem usar como regex)
            prefix = abl_norm[:6]
            csv_row = agg[
                agg["ablation"].str.lower().str.startswith(prefix, na=False) &
                (agg["mode"].str.lower() == mode_norm)
            ]

        if csv_row.empty:
            checks.append(f"  [INCONCLUSIVO] {ablation_tex}/{mode_tex}: não encontrado no CSV")
            continue

        csv_row = csv_row.iloc[0]
        label = f"{ablation_tex}/{mode_tex}"

        if prec_ms:
            _check_mean_std(f"{label} P", prec_ms[0], prec_ms[1],
                            csv_row["prec_mean"], csv_row["prec_std"], checks, divs)
        if rec_ms:
            _check_mean_std(f"{label} R", rec_ms[0], rec_ms[1],
                            csv_row["rec_mean"], csv_row["rec_std"], checks, divs)
        if f1_ms:
            _check_mean_std(f"{label} F1", f1_ms[0], f1_ms[1],
                            csv_row["f1_mean"], csv_row["f1_std"], checks, divs)
        if llm_ms:
            _check_mean_std(f"{label} LLM", llm_ms[0], llm_ms[1],
                            csv_row["llm_mean"], csv_row["llm_std"], checks, divs, atol=1.0)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 11 — tab_exp3_kfold.tex → data/sprint3b/exp3_kfold.csv
# ===========================================================================

def audit_exp3_kfold(idx: int) -> None:
    name = "tab_exp3_kfold.tex"
    src = "data/sprint3b/exp3_kfold.csv"
    df = _read_csv(DATA_DIR / "sprint3b" / "exp3_kfold.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    agg = df.groupby("mode").agg(
        prec_mean=("exp_precision", "mean"),
        prec_std=("exp_precision", "std"),
        rec_mean=("exp_recall", "mean"),
        rec_std=("exp_recall", "std"),
        f1_mean=("exp_f1", "mean"),
        f1_std=("exp_f1", "std"),
        llm_mean=("llm_used", "mean"),
        llm_std=("llm_used", "std"),
    ).reset_index()

    expected = [
        ("confirmacao", 0.947, 0.008, 0.936, 0.026, 0.942, 0.013, 88.44, 12.35),
        ("vigilancia",  0.936, 0.011, 0.965, 0.020, 0.950, 0.009, 84.16, 11.44),
    ]

    for mode, ep, eps, er, ers, ef1, ef1s, ellm, ellms in expected:
        row = agg[agg["mode"] == mode]
        if row.empty:
            checks.append(f"  [INCONCLUSIVO] mode={mode}: não encontrado")
            continue
        row = row.iloc[0]
        _check_mean_std(f"{mode} P",   ep,   eps,   row["prec_mean"], row["prec_std"],  checks, divs)
        _check_mean_std(f"{mode} R",   er,   ers,   row["rec_mean"],  row["rec_std"],   checks, divs)
        _check_mean_std(f"{mode} F1",  ef1,  ef1s,  row["f1_mean"],   row["f1_std"],    checks, divs)
        _check_mean_std(f"{mode} LLM", ellm, ellms, row["llm_mean"],  row["llm_std"],   checks, divs, atol=1.0)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 12 — tab_exp4_budget.tex → data/sprint3b/exp4_budget_summary.csv
# ===========================================================================

def audit_exp4_budget(idx: int) -> None:
    name = "tab_exp4_budget.tex"
    src = "data/sprint3b/exp4_budget_summary.csv"
    df = _read_csv(DATA_DIR / "sprint3b" / "exp4_budget_summary.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    tex_path = TABLES_DIR / name
    rows = _data_rows(tex_path)

    for tex_row in rows:
        parts = tex_row.rstrip("\\\\").split("&")
        if len(parts) < 6:
            continue
        budget_nums = _nums(parts[0])
        if not budget_nums:
            continue
        budget_val = budget_nums[0]
        mode_val = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", "", parts[1]).strip().lower()

        prec_ms = _parse_mean_std(parts[2])
        rec_ms  = _parse_mean_std(parts[3])
        llm_ms  = _parse_mean_std(parts[5]) if len(parts) > 5 else None

        csv_row = df[(abs(df["llm_budget"] - budget_val) < 0.01) & (df["mode"] == mode_val)]
        if csv_row.empty:
            checks.append(f"  [INCONCLUSIVO] budget={budget_val}/{mode_val}: não encontrado")
            continue
        csv_row = csv_row.iloc[0]
        label = f"budget={budget_val}/{mode_val}"

        if prec_ms:
            _check_mean_std(f"{label} P", prec_ms[0], prec_ms[1],
                            csv_row["exp_precision_mean"], csv_row["exp_precision_std"], checks, divs)
        if rec_ms:
            _check_mean_std(f"{label} R", rec_ms[0], rec_ms[1],
                            csv_row["exp_recall_mean"], csv_row["exp_recall_std"], checks, divs)
        if llm_ms:
            _check_mean_std(f"{label} LLM", llm_ms[0], llm_ms[1],
                            csv_row["llm_used_mean"], csv_row["llm_used_std"], checks, divs, atol=1.0)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 13 — tab_exp4_cost_ratio.tex → data/sprint3b/exp4_cost_ratio_summary.csv
# ===========================================================================

def audit_exp4_cost_ratio(idx: int) -> None:
    name = "tab_exp4_cost_ratio.tex"
    src = "data/sprint3b/exp4_cost_ratio_summary.csv"
    df = _read_csv(DATA_DIR / "sprint3b" / "exp4_cost_ratio_summary.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    tex_path = TABLES_DIR / name
    rows = _data_rows(tex_path)

    for tex_row in rows:
        parts = tex_row.rstrip("\\\\").split("&")
        if len(parts) < 6:
            continue
        ratio_nums = _nums(parts[0])
        if not ratio_nums:
            continue
        ratio_val = ratio_nums[0]
        mode_val = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", "", parts[1]).strip().lower()

        prec_ms = _parse_mean_std(parts[2])
        rec_ms  = _parse_mean_std(parts[3])
        llm_ms  = _parse_mean_std(parts[5]) if len(parts) > 5 else None

        csv_row = df[(abs(df["fp_fn_ratio"] - ratio_val) < 0.001) & (df["mode"] == mode_val)]
        if csv_row.empty:
            checks.append(f"  [INCONCLUSIVO] ratio={ratio_val}/{mode_val}: não encontrado")
            continue
        csv_row = csv_row.iloc[0]
        label = f"ratio={ratio_val}/{mode_val}"

        if prec_ms:
            _check_mean_std(f"{label} P", prec_ms[0], prec_ms[1],
                            csv_row["exp_precision_mean"], csv_row["exp_precision_std"], checks, divs)
        if rec_ms:
            _check_mean_std(f"{label} R", rec_ms[0], rec_ms[1],
                            csv_row["exp_recall_mean"], csv_row["exp_recall_std"], checks, divs)
        if llm_ms:
            _check_mean_std(f"{label} LLM", llm_ms[0], llm_ms[1],
                            csv_row["llm_used_mean"], csv_row["llm_used_std"], checks, divs, atol=1.0)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 14 — tab_feature_importance.tex → data/qw1/xgboost_feature_importance_global.csv
# ===========================================================================

def audit_feature_importance(idx: int) -> None:
    name = "tab_feature_importance.tex"
    src = "data/qw1/xgboost_feature_importance_global.csv"
    df = _read_csv(DATA_DIR / "qw1" / "xgboost_feature_importance_global.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    tex_path = TABLES_DIR / name
    rows = _data_rows(tex_path)

    csv_feat = dict(zip(df["feature"], df["importance_gain"]))

    for i, tex_row in enumerate(rows, 1):
        parts = tex_row.rstrip("\\\\").split("&")
        if len(parts) < 3:
            continue
        feat_tex = _clean_feature_name(parts[1])
        val_nums = _nums(parts[2])
        if not val_nums:
            checks.append(f"  [INCONCLUSIVO] #{i} {feat_tex}: parse de valor falhou")
            continue
        tex_val = val_nums[0]

        # Procurar feature no CSV
        csv_val = csv_feat.get(feat_tex)
        if csv_val is None:
            # Busca por normalização de espaços/underscores
            feat_norm = feat_tex.lower().replace("_", " ").strip()
            for k, v in csv_feat.items():
                k_norm = k.lower().replace("_", " ").strip()
                if k_norm == feat_norm:
                    csv_val = v
                    break

        if csv_val is None:
            checks.append(f"  [INCONCLUSIVO] #{i} '{feat_tex}': feature não encontrada no CSV")
            continue

        _check_val(f"#{i} {feat_tex}", tex_val, csv_val, checks, divs, atol=0.0, rtol=RTOL_FEAT)

    # Verificações especiais
    checks.append("  --- Verificações especiais ---")
    top15 = df.nlargest(15, "importance_gain").reset_index(drop=True)
    if len(top15) >= 1:
        r = top15.iloc[0]
        ok = r["feature"] == "score_recall" and _close(r["importance_gain"], 81.0671, rtol=RTOL_FEAT)
        sym = "[OK]" if ok else "[DIVERGENCIA]"
        line = (f"  {sym} #1 feature={r['feature']} gain={r['importance_gain']:.4f} "
                f"(esperado score_recall=81.0671)")
        checks.append(line)
        if not ok:
            divs.append(line)
    if len(top15) >= 15:
        r = top15.iloc[14]
        ok = r["feature"] == "nome_x_mae" and _close(r["importance_gain"], 1.8809, rtol=RTOL_FEAT)
        sym = "[OK]" if ok else "[DIVERGENCIA]"
        line = (f"  {sym} #15 feature={r['feature']} gain={r['importance_gain']:.4f} "
                f"(esperado nome_x_mae=1.8809)")
        checks.append(line)
        if not ok:
            divs.append(line)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 15 — tab_shap_importance.tex → data/shap_importance_combined.csv (RF)
# ===========================================================================

def audit_shap_importance(idx: int) -> None:
    name = "tab_shap_importance.tex"
    src = "data/shap_importance_combined.csv (modelo RF)"
    df = _read_csv(DATA_DIR / "shap_importance_combined.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    # Filtrar modelo RF (aparece como 'RF' no combined)
    rf_df = df[df["model"] == "RF"].sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    if rf_df.empty:
        rf_df = df[df["model"].str.contains("RF", na=False, regex=False)].sort_values(
            "shap_mean_abs", ascending=False).reset_index(drop=True)

    csv_feat = dict(zip(rf_df["feature"], rf_df["shap_mean_abs"]))

    tex_path = TABLES_DIR / name
    rows = _data_rows(tex_path)

    for i, tex_row in enumerate(rows, 1):
        parts = tex_row.rstrip("\\\\").split("&")
        if len(parts) < 3:
            continue
        feat_tex = _clean_feature_name(parts[1])
        val_nums = _nums(parts[2])
        if not val_nums:
            checks.append(f"  [INCONCLUSIVO] #{i} {feat_tex}: parse de valor falhou")
            continue
        tex_val = val_nums[0]

        # Procurar feature
        csv_val = csv_feat.get(feat_tex)
        if csv_val is None:
            feat_norm = feat_tex.lower().replace("_", " ").strip()
            for k, v in csv_feat.items():
                k_norm = k.lower().replace("_", " ").strip()
                if k_norm == feat_norm:
                    csv_val = v
                    break

        if csv_val is None:
            checks.append(f"  [INCONCLUSIVO] #{i} '{feat_tex}': não encontrada no CSV RF "
                          f"(possivelmente de versão anterior do modelo)")
            continue

        _check_val(f"#{i} {feat_tex}", tex_val, csv_val, checks, divs, atol=0.0001)

    # Verificações especiais com valores do CSV
    checks.append("  --- Verificações especiais (CSV RF vs spec) ---")
    if len(rf_df) >= 1:
        r = rf_df.iloc[0]
        # Spec diz #1=score_recall=0.0446, mas CSV RF tem nome_squared como #1
        ok_feat = (r["feature"] == "score_recall")
        ok_val  = _close(r["shap_mean_abs"], 0.0446, atol=0.001)
        sym = "[OK]" if (ok_feat and ok_val) else "[DIVERGENCIA]"
        line = (f"  {sym} #1 RF SHAP feature={r['feature']} "
                f"shap={r['shap_mean_abs']:.4f} (spec: score_recall=0.0446)")
        checks.append(line)
        if not (ok_feat and ok_val):
            divs.append(line)
    if len(rf_df) >= 15:
        r = rf_df.iloc[14]
        ok = (r["feature"] == "nome_total") and _close(r["shap_mean_abs"], 0.0141, atol=0.001)
        sym = "[OK]" if ok else "[DIVERGENCIA]"
        line = (f"  {sym} #15 RF SHAP feature={r['feature']} "
                f"shap={r['shap_mean_abs']:.4f} (spec: nome_total=0.0141)")
        checks.append(line)
        if not ok:
            divs.append(line)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 16 — tab_qw2_confusion.tex → data/qw2/{vigilancia,confirmacao}_kimi.json
# ===========================================================================

def audit_qw2_confusion(idx: int) -> None:
    name = "tab_qw2_confusion.tex"
    src = "data/qw2/qw2_vigilancia_kimi.json + qw2_confirmacao_kimi.json"
    checks: list[str] = []
    divs: list[str] = []

    vig_path  = DATA_DIR / "qw2" / "qw2_vigilancia_kimi.json"
    conf_path = DATA_DIR / "qw2" / "qw2_confirmacao_kimi.json"

    if not vig_path.exists() or not conf_path.exists():
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="JSON(s) não encontrado(s)")
        return

    with open(vig_path, encoding="utf-8") as f:
        vig_data = json.load(f)
    with open(conf_path, encoding="utf-8") as f:
        conf_data = json.load(f)

    vig_m  = vig_data.get("metrics", {})
    conf_m = conf_data.get("metrics", {})

    # Vigilância: TP=72, FP=6, FN=32, TN=1300, total=1410
    expected_vig = {"tp": 72, "fp": 6, "fn": 32, "tn": 1300, "n": 1410}
    for k, exp_v in expected_vig.items():
        key = f"gt_{k}" if k != "n" else "n"
        got_v = vig_m.get(key, vig_m.get(k))
        if got_v is None:
            checks.append(f"  [INCONCLUSIVO] Vigil {k}: chave não encontrada no JSON")
            continue
        _check_val(f"Vigil {k}", float(got_v), float(exp_v), checks, divs, atol=0.5)

    # Confirmação: TP=91, FP=3, FN=10, TN=331, total=435
    expected_conf = {"tp": 91, "fp": 3, "fn": 10, "tn": 331, "n": 435}
    for k, exp_v in expected_conf.items():
        key = f"gt_{k}" if k != "n" else "n"
        got_v = conf_m.get(key, conf_m.get(k))
        if got_v is None:
            checks.append(f"  [INCONCLUSIVO] Conf {k}: chave não encontrada no JSON")
            continue
        _check_val(f"Conf {k}", float(got_v), float(exp_v), checks, divs, atol=0.5)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 17 — tab_qw2_llm_comparison.tex → calculado dos JSONs qw2
# ===========================================================================

def audit_qw2_llm_comparison(idx: int) -> None:
    name = "tab_qw2_llm_comparison.tex"
    src = "calculado de data/qw2/qw2_*_kimi.json"
    checks: list[str] = []
    divs: list[str] = []

    vig_path  = DATA_DIR / "qw2" / "qw2_vigilancia_kimi.json"
    conf_path = DATA_DIR / "qw2" / "qw2_confirmacao_kimi.json"

    if not vig_path.exists() or not conf_path.exists():
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="JSON(s) não encontrado(s)")
        return

    with open(vig_path, encoding="utf-8") as f:
        vig_data = json.load(f)
    with open(conf_path, encoding="utf-8") as f:
        conf_data = json.load(f)

    def calc_metrics(m: dict) -> dict:
        tp = m.get("gt_tp", 0)
        fp = m.get("gt_fp", 0)
        fn = m.get("gt_fn", 0)
        tn = m.get("gt_tn", 0)
        n  = m.get("n", tp + fp + fn + tn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc  = (tp + tn) / n if n > 0 else 0.0
        return dict(n=n, P=prec, R=rec, F1=f1, Spec=spec, Acc=acc)

    vig_calc  = calc_metrics(vig_data.get("metrics", {}))
    conf_calc = calc_metrics(conf_data.get("metrics", {}))

    # Vigilância esperada
    expected_vig = {"n": 1410, "P": 0.923, "R": 0.692, "F1": 0.791, "Spec": 0.995, "Acc": 0.973}
    for k, exp_v in expected_vig.items():
        got_v = vig_calc.get(k, float("nan"))
        _check_val(f"Vigil {k}", float(got_v), float(exp_v), checks, divs,
                   atol=0.5 if k == "n" else ATOL_METRIC)

    # Confirmação esperada
    expected_conf = {"n": 435, "P": 0.968, "R": 0.901, "F1": 0.933, "Spec": 0.991, "Acc": 0.970}
    for k, exp_v in expected_conf.items():
        got_v = conf_calc.get(k, float("nan"))
        _check_val(f"Conf {k}", float(got_v), float(exp_v), checks, divs,
                   atol=0.5 if k == "n" else ATOL_METRIC)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 18 — tab_qw3_group_metrics.tex → data/qw3/qw3_group_metrics.csv
# ===========================================================================

def audit_qw3_group_metrics(idx: int) -> None:
    name = "tab_qw3_group_metrics.tex"
    src = "data/qw3/qw3_group_metrics.csv"
    df = _read_csv(DATA_DIR / "qw3" / "qw3_group_metrics.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    tex_path = TABLES_DIR / name
    rows = _data_rows(tex_path)

    for tex_row in rows:
        parts = tex_row.rstrip("\\\\").split("&")
        if len(parts) < 7:
            continue
        # Dimensão e grupo — limpar LaTeX
        dim_tex   = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", " ", parts[0]).strip()
        group_tex = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", " ", parts[1]).strip()
        dim_tex   = re.sub(r"[{}\\]", "", dim_tex).strip()
        group_tex = re.sub(r"[{}\\]", "", group_tex).strip()

        # Extrair TVP (col 4) e VPP (col 6)
        # Pré-processar: substituir {,} por .
        tpr_nums = _nums(parts[4])  # _nums já chama _preprocess_latex_cell
        ppv_nums = _nums(parts[6])

        if not tpr_nums or not ppv_nums:
            checks.append(f"  [INCONCLUSIVO] {dim_tex}/{group_tex}: parse falhou")
            continue

        tex_tpr = tpr_nums[0]
        tex_ppv = ppv_nums[0]

        # Normalizar grupo para busca no CSV
        group_norm = group_tex.lower().replace("--", "-").replace(" ", "").strip()

        csv_tpr_rows = df[df["metric"] == "tpr"]
        csv_ppv_rows = df[df["metric"] == "ppv"]

        def find_group_row(sub_df: pd.DataFrame, grp: str) -> Any:
            """Busca por grupo exato ou parcial."""
            grp_norm = grp.lower().replace(" ", "").replace("--", "-")
            # Busca exata normalizada
            matches = sub_df[
                sub_df["group"].astype(str).str.lower()
                .str.replace(" ", "").str.replace("--", "-") == grp_norm
            ]
            if not matches.empty:
                return matches.iloc[0]
            # Busca parcial: prefixo de 3 chars
            prefix = grp_norm[:3]
            matches = sub_df[
                sub_df["group"].astype(str).str.lower().str.startswith(prefix, na=False)
            ]
            return matches.iloc[0] if not matches.empty else None

        tpr_row = find_group_row(csv_tpr_rows, group_norm)
        ppv_row = find_group_row(csv_ppv_rows, group_norm)

        label = f"{dim_tex}/{group_tex}"

        if tpr_row is not None:
            _check_val(f"{label} TVP", tex_tpr, round(tpr_row["rate"], 3), checks, divs)
        else:
            checks.append(f"  [INCONCLUSIVO] {label} TVP: grupo não encontrado no CSV")

        if ppv_row is not None:
            _check_val(f"{label} VPP", tex_ppv, round(ppv_row["rate"], 3), checks, divs)
        else:
            checks.append(f"  [INCONCLUSIVO] {label} VPP: grupo não encontrado no CSV")

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 19 — tab_qw3_fairness_rejections.tex → data/qw3/qw3_pairwise_tests.csv
# ===========================================================================

def audit_qw3_fairness_rejections(idx: int) -> None:
    name = "tab_qw3_fairness_rejections.tex"
    src = "data/qw3/qw3_pairwise_tests.csv"
    df = _read_csv(DATA_DIR / "qw3" / "qw3_pairwise_tests.csv")
    checks: list[str] = []
    divs: list[str] = []

    if df is None:
        _report(idx, name, src, checks, divs, inconclusivo=True, inconclusivo_reason="CSV não encontrado")
        return

    rejections = df[df["reject_fdr"] == True].reset_index(drop=True)
    n_rejections = len(rejections)
    expected_n = 12

    if n_rejections == expected_n:
        checks.append(f"  [OK] {n_rejections} rejeições FDR (esperado {expected_n})")
    else:
        msg = f"  [DIVERGENCIA] {n_rejections} rejeições FDR (esperado {expected_n})"
        checks.append(msg)
        divs.append(msg)

    tex_path = TABLES_DIR / name
    n_tex_rows = len(_data_rows(tex_path))
    if n_tex_rows == expected_n:
        checks.append(f"  [OK] Tabela .tex contém {n_tex_rows} linhas de rejeição")
    else:
        msg = f"  [DIVERGENCIA] Tabela .tex contém {n_tex_rows} linhas (esperado {expected_n})"
        checks.append(msg)
        divs.append(msg)

    # Verificar p_fdr < 0.05 para todas
    n_pval_ok = sum(1 for _, r in rejections.iterrows() if r["p_value_fdr"] < 0.05)
    if n_pval_ok == n_rejections:
        checks.append(f"  [OK] Todas as {n_rejections} rejeições têm p_fdr < 0.05")
    else:
        msg = f"  [DIVERGENCIA] Apenas {n_pval_ok}/{n_rejections} têm p_fdr < 0.05"
        checks.append(msg)
        divs.append(msg)

    _report(idx, name, src, checks, divs)


# ===========================================================================
# TAB 20 — tab_perfil_recuperados.tex → INCONCLUSIVO
# ===========================================================================

def audit_perfil_recuperados(idx: int) -> None:
    name = "tab_perfil_recuperados.tex"
    src = "NENHUMA (dados clínicos sem CSV canônico)"
    checks: list[str] = []
    divs: list[str] = []
    checks.append("  [INCONCLUSIVO] Tabela de perfil clínico sem fonte CSV conhecida")
    _report(idx, name, src, checks, divs, inconclusivo=True,
            inconclusivo_reason="dados clínicos — sem fonte CSV canônica")


# ===========================================================================
# TABS 21-23 — *_UPDATED.tex vs versões originais
# ===========================================================================

def _compare_updated(idx: int, updated_name: str, original_name: str) -> None:
    """Compara UPDATED vs original: verifica se há TODOs e se números coincidem."""
    src = f"comparação com {original_name}"
    checks: list[str] = []
    divs: list[str] = []

    updated_path  = TABLES_DIR / updated_name
    original_path = TABLES_DIR / original_name

    if not updated_path.exists():
        _report(idx, updated_name, src, checks, divs, inconclusivo=True,
                inconclusivo_reason="arquivo UPDATED não encontrado")
        return
    if not original_path.exists():
        _report(idx, updated_name, src, checks, divs, inconclusivo=True,
                inconclusivo_reason="arquivo original não encontrado")
        return

    updated_text  = updated_path.read_text(encoding="utf-8")
    original_text = original_path.read_text(encoding="utf-8")

    # Verificar placeholders TODO
    if "TODO" in updated_text:
        todo_count = updated_text.count("TODO")
        msg = (f"  [DIVERGENCIA] UPDATED contém {todo_count} placeholder(s) 'TODO' "
               f"— arquivo incompleto/desatualizado")
        checks.append(msg)
        divs.append(msg)
        _report(idx, updated_name, src, checks, divs)
        return

    # Comparar conjuntos de números
    orig_nums    = _nums(original_text)
    updated_nums = _nums(updated_text)

    if not orig_nums and not updated_nums:
        _report(idx, updated_name, src, checks, divs, inconclusivo=True,
                inconclusivo_reason="parse vazio em ambos")
        return

    orig_set    = set(round(v, 3) for v in orig_nums)
    updated_set = set(round(v, 3) for v in updated_nums)
    only_orig    = orig_set - updated_set
    only_updated = updated_set - orig_set

    if not only_orig and not only_updated:
        checks.append(f"  [OK] Números idênticos: {len(orig_set)} valores únicos em ambos")
    else:
        if only_orig:
            msg = f"  [DIVERGENCIA] Valores em original mas não em UPDATED: {sorted(only_orig)[:10]}"
            checks.append(msg)
            divs.append(msg)
        if only_updated:
            msg = f"  [DIVERGENCIA] Valores em UPDATED mas não em original: {sorted(only_updated)[:10]}"
            checks.append(msg)
            divs.append(msg)

    _report(idx, updated_name, src, checks, divs)


def audit_cv_5fold_updated(idx: int) -> None:
    _compare_updated(idx, "tab_cv_5fold_UPDATED.tex", "tab_cv_5fold.tex")


def audit_ablation_best_category_updated(idx: int) -> None:
    _compare_updated(idx, "tab_ablation_best_category_UPDATED.tex", "tab_ablation_best_category.tex")


def audit_ablation_top10_updated(idx: int) -> None:
    _compare_updated(idx, "tab_ablation_top10_UPDATED.tex", "tab_ablation_top10.tex")


# ===========================================================================
# Sequência principal
# ===========================================================================

AUDIT_FUNCTIONS = [
    audit_cv_5fold,                        # 1
    audit_cv_results,                      # 2
    audit_imbalance_sensitivity,           # 3
    audit_ablation_best_category,          # 4
    audit_ablation_top10,                  # 5
    audit_pareto_frontier,                 # 6
    audit_qw1_deep_learning,               # 7
    audit_impacto_epidemiologico,          # 8
    audit_score_bands,                     # 9
    audit_exp1_ablation,                   # 10
    audit_exp3_kfold,                      # 11
    audit_exp4_budget,                     # 12
    audit_exp4_cost_ratio,                 # 13
    audit_feature_importance,              # 14
    audit_shap_importance,                 # 15
    audit_qw2_confusion,                   # 16
    audit_qw2_llm_comparison,             # 17
    audit_qw3_group_metrics,              # 18
    audit_qw3_fairness_rejections,        # 19
    audit_perfil_recuperados,             # 20
    audit_cv_5fold_updated,               # 21
    audit_ablation_best_category_updated, # 22
    audit_ablation_top10_updated,         # 23
]


def main() -> None:
    global _n_tables
    _n_tables = len(AUDIT_FUNCTIONS)

    header = "=" * 60
    print(header)
    print("AUDITORIA DE CONSISTÊNCIA: TABELAS DA TESE")
    print(header)

    for i, fn in enumerate(AUDIT_FUNCTIONS, 1):
        try:
            fn(i)
        except Exception as exc:
            import traceback
            print(f"\n[{i}/{_n_tables}] {fn.__name__}")
            print(f"  [ERRO INESPERADO] {type(exc).__name__}: {exc}")
            print(f"  Traceback: {traceback.format_exc()[:300]}")
            _results.append({
                "idx": i,
                "name": fn.__name__,
                "source": "?",
                "status": "INCONCLUSIVO",
                "summary": f"erro: {exc}",
                "checks": [],
                "divergences": [],
            })

    # --- Resumo final ---
    print(f"\n{header}")
    print("RESUMO FINAL")
    print(header)

    ok_list  = [r for r in _results if r["status"] == "OK"]
    div_list = [r for r in _results if r["status"] == "DIVERGENCIA"]
    inc_list = [r for r in _results if r["status"] == "INCONCLUSIVO"]

    print(f"OK:           {len(ok_list):2d} tabelas")
    print(f"DIVERGENCIA:  {len(div_list):2d} tabelas")
    print(f"INCONCLUSIVO: {len(inc_list):2d} tabelas")

    if div_list:
        print("\nTabelas com DIVERGÊNCIA:")
        for r in div_list:
            print(f"  - [{r['idx']:02d}] {r['name']}: {r['summary']}")

    if inc_list:
        print("\nTabelas INCONCLUSIVAS:")
        for r in inc_list:
            print(f"  - [{r['idx']:02d}] {r['name']}: {r['summary']}")

    print(header)
    total = len(_results)
    print(f"\nTotal verificado: {total} tabelas  |  OK: {len(ok_list)}  |  "
          f"DIVERGÊNCIA: {len(div_list)}  |  INCONCLUSIVO: {len(inc_list)}")


if __name__ == "__main__":
    main()
