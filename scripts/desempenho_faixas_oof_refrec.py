"""Desempenho por faixa da nota final do estimador OOF por REFREC do GZ-CMD (Estudo 4).

Cruza ``data/pcal_ml_oof_refrec.csv`` (``p_cal_ml``: XGBoost com calibracao
isotonica aninhada, particao agrupada por REFREC) com o rotulo do extrato
``data/COMPARADORSEMIDENT.csv`` (nao versionado) por ``pair_id`` =
sha256("COMPREC|REFREC|PASSO")[:16], e agrega por faixa da nota final:
n, vinculos, AUC, AP, Brier e VP/FP/FN, sensibilidade e precisao a
``p_cal_ml >= 0,5``. O limiar 0,5 e declarado a priori e nao e otimizado.
Grava somente agregados (sem ``pair_id`` nem valores de campos).

Saidas:
  - data/desempenho_faixas_oof_refrec.csv
  - data/desempenho_faixas_oof_refrec_meta.json

Uso: python scripts/desempenho_faixas_oof_refrec.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SCRIPT_VERSION = "1.0"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTRATO = DATA / "COMPARADORSEMIDENT.csv"
OOF = DATA / "pcal_ml_oof_refrec.csv"
GRUPOS = DATA / "pcal_ml_oof_grupos_metricas.csv"
OUT_CSV = DATA / "desempenho_faixas_oof_refrec.csv"
OUT_META = DATA / "desempenho_faixas_oof_refrec_meta.json"

THRESHOLD = 0.5
N_ESPERADO = 61696
VINCULOS_ESPERADO = 247
TOL = 0.0005

# (rotulo, inferior, superior): [inferior, superior); mesmas bandas do GZ-CMD.
FAIXAS: list[tuple[str, float, float]] = [
    ("<5", -np.inf, 5.0),
    ("5-6", 5.0, 6.0),
    ("6-7", 6.0, 7.0),
    ("7-8", 7.0, 8.0),
    ("8-9", 8.0, 9.0),
    (">=9", 9.0, np.inf),
]
COLUNAS = [
    "faixa", "n", "vinculos", "auc", "ap", "brier", "vp", "fp", "fn", "sensibilidade", "precisao",
]


def pair_id(comprec: str, refrec: str, passo: str) -> str:
    raw = f"{comprec}|{refrec}|{passo}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def atribuir_faixa(nota: float) -> str:
    """Rotulo da faixa da nota final (intervalo [inferior, superior))."""
    for rotulo, inf, sup in FAIXAS:
        if inf <= nota < sup:
            return rotulo
    raise ValueError(f"nota fora das faixas: {nota!r}")


def _razao(num: int, den: int) -> float:
    return float(num) / den if den > 0 else float("nan")


def metricas(y: np.ndarray, p: np.ndarray, limiar: float = THRESHOLD) -> dict:
    """Metricas de um recorte; AUC e AP ficam vazias sem as duas classes."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = p >= limiar
    vp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    duas = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "vinculos": int(y.sum()),
        "auc": float(roc_auc_score(y, p)) if duas else float("nan"),
        "ap": float(average_precision_score(y, p)) if duas else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "vp": vp,
        "fp": fp,
        "fn": fn,
        "sensibilidade": _razao(vp, vp + fn),
        "precisao": _razao(vp, vp + fp),
    }


def agregar(df: pd.DataFrame, limiar: float = THRESHOLD) -> pd.DataFrame:
    """Agrega por faixa, area cinza [5,9) e total. Exige colunas nota, y, p_cal_ml."""
    faixa = df["nota"].map(atribuir_faixa)
    cortes = {
        **{r: faixa == r for r, _, _ in FAIXAS},
        "area_cinza": faixa.isin(["5-6", "6-7", "7-8", "8-9"]),
        "total": pd.Series(True, index=df.index),
    }
    linhas = []
    for rotulo in [f[0] for f in FAIXAS] + ["area_cinza", "total"]:
        sub = df[cortes[rotulo]]
        linhas.append({"faixa": rotulo, **metricas(sub["y"].to_numpy(), sub["p_cal_ml"].to_numpy(), limiar)})
    return pd.DataFrame(linhas, columns=COLUNAS)


def carregar_cruzado() -> pd.DataFrame:
    ext = pd.read_csv(EXTRATO, sep=";", dtype=str)
    ext.columns = [c.split(",")[0] for c in ext.columns]
    ext["pair_id"] = [pair_id(c, r, p) for c, r, p in zip(ext["COMPREC"], ext["REFREC"], ext["PASSO"])]
    ext["y"] = pd.to_numeric(ext["PAR"], errors="coerce").isin([1, 2]).astype(int)
    assert ext["pair_id"].is_unique, "pair_id duplicado no extrato"
    oof = pd.read_csv(OOF)
    m = oof.merge(ext[["pair_id", "y"]], on="pair_id", how="inner", validate="one_to_one")
    return m[["nota", "p_cal_ml", "y"]]


def verificar(m: pd.DataFrame, res: pd.DataFrame) -> dict:
    assert len(m) == N_ESPERADO, f"linhas casadas {len(m)}, esperado {N_ESPERADO}"
    assert int(m["y"].sum()) == VINCULOS_ESPERADO, f"vinculos {int(m['y'].sum())}, esperado {VINCULOS_ESPERADO}"
    ref = pd.read_csv(GRUPOS)
    ref = ref[(ref.split_by == "refrec") & (ref.recorte == "area_cinza")].iloc[0]
    g = res[res.faixa == "area_cinza"].iloc[0]
    assert int(g.n) == int(ref.n) == 21620, f"n area cinza {int(g.n)}"
    assert int(g.vinculos) == int(ref.vinculos) == 151, f"vinculos area cinza {int(g.vinculos)}"
    assert abs(g.auc - 0.9951) <= TOL, f"AUC area cinza {g.auc:.4f}"
    assert abs(g.ap - 0.8623) <= TOL, f"AP area cinza {g.ap:.4f}"
    return {
        "n": int(g.n), "vinculos": int(g.vinculos), "auc": round(float(g.auc), 4), "ap": round(float(g.ap), 4),
        "referencia": "data/pcal_ml_oof_grupos_metricas.csv (refrec, area_cinza)", "tolerancia": TOL,
    }


def main() -> None:
    m = carregar_cruzado()
    res = agregar(m)
    controle = verificar(m, res)
    res.to_csv(OUT_CSV, index=False, float_format="%.6f")
    meta = {
        "script": "scripts/desempenho_faixas_oof_refrec.py",
        "versao_script": SCRIPT_VERSION,
        "data_execucao": date.today().isoformat(),
        "entrada_oof": "data/pcal_ml_oof_refrec.csv",
        "estimador": "XGBoost 300 arvores, prof. 6, taxa 0,05, isotonica aninhada, split_by refrec, semente 42 (gzcmd/ml_oof.py)",
        "limiar": THRESHOLD,
        "limiar_otimizado": False,
        "n_linhas_casadas": int(len(m)),
        "n_vinculos": int(m["y"].sum()),
        "faixas": [f[0] for f in FAIXAS],
        "area_cinza": "[5,9)",
        "controle_area_cinza": controle,
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(res.to_string(index=False))
    print(f"[OK] {OUT_CSV.name}, {OUT_META.name}")


if __name__ == "__main__":
    main()
