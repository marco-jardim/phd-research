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
import ast
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SCRIPT_VERSION = "1.1"
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
    "faixa", "n", "vinculos", "auc", "ap", "brier", "vp", "fp", "fn", "sensibilidade", "precisao", "minoritaria_n",
]


def configurar() -> None:
    sys.path.insert(0, str(ROOT))
    if hasattr(pd.options.future, "infer_string"):
        pd.options.future.infer_string = False


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
        "minoritaria_n": int(min(y.sum(), len(y) - y.sum())),
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
    from gzcmd.llm_dossier import make_pair_id

    ext = pd.read_csv(EXTRATO, sep=";", dtype=str)
    ext.columns = [c.split(",")[0] for c in ext.columns]
    ext["pair_id"] = [make_pair_id(c.strip(), r.strip(), p.strip()) for c, r, p in zip(ext["COMPREC"], ext["REFREC"], ext["PASSO"])]
    ext["y"] = pd.to_numeric(ext["PAR"], errors="coerce").isin([1, 2]).astype(int)
    if not ext["pair_id"].is_unique:
        raise ValueError("pair_id duplicado no extrato")
    oof = pd.read_csv(OOF, dtype={"pair_id": str})
    m = oof.merge(ext[["pair_id", "y"]], on="pair_id", how="inner", validate="one_to_one")
    if len(m) != len(oof) or len(m) != len(ext):
        raise ValueError("OOF e extrato nao possuem os mesmos pares")
    return m[["nota", "p_cal_ml", "y"]]


def verificar(m: pd.DataFrame, res: pd.DataFrame) -> dict:
    if len(m) != N_ESPERADO:
        raise ValueError(f"linhas casadas {len(m)}, esperado {N_ESPERADO}")
    if int(m["y"].sum()) != VINCULOS_ESPERADO:
        raise ValueError(f"vinculos {int(m['y'].sum())}, esperado {VINCULOS_ESPERADO}")
    ref = pd.read_csv(GRUPOS)
    ref = ref[(ref.split_by == "refrec") & (ref.recorte == "area_cinza")].iloc[0]
    g = res[res.faixa == "area_cinza"].iloc[0]
    for campo in ("n", "vinculos", "auc", "ap"):
        tolerancia = TOL if campo in ("auc", "ap") else 0
        if not abs(g[campo] - ref[campo]) <= tolerancia:
            raise ValueError(f"{campo} area cinza {g[campo]}, esperado {ref[campo]}")
    return {
        "n": int(g.n), "vinculos": int(g.vinculos), "auc": round(float(g.auc), 4), "ap": round(float(g.ap), 4),
        "referencia": "data/pcal_ml_oof_grupos_metricas.csv (refrec, area_cinza)", "tolerancia": TOL,
    }


def proveniencia() -> dict:
    """Copia o metadado original; fallback explicito para caches sem parametros."""
    original = json.loads(OOF.with_suffix(".json").read_text(encoding="utf-8"))
    csv_sha256 = hashlib.sha256(OOF.read_bytes()).hexdigest()
    if csv_sha256 != original["csv_sha256"]:
        raise ValueError("SHA-256 do CSV OOF diverge do metadado original")
    resultado = {"csv_sha256": csv_sha256, "fingerprint": original["fingerprint"]}
    if "hiperparametros" in original:
        resultado["hiperparametros"] = original["hiperparametros"]
        resultado["origem_hiperparametros"] = "data/pcal_ml_oof_refrec.json"
    else:
        codigo = ROOT / "gzcmd" / "ml_oof.py"
        if hashlib.sha256(codigo.read_bytes()).hexdigest() != original["fingerprint"]["code_sha256"]:
            raise ValueError("Codigo OOF diverge do fingerprint; nao e possivel recuperar parametros")
        arvore = ast.parse(codigo.read_text(encoding="utf-8"))
        funcao = next(n for n in arvore.body if isinstance(n, ast.FunctionDef) and n.name == "_classifier")
        chamada = next(n.value for n in funcao.body if isinstance(n, ast.Return))
        parametros = {}
        for kw in chamada.keywords:
            if isinstance(kw.value, ast.Constant):
                parametros[kw.arg] = ast.literal_eval(kw.value)
            elif isinstance(kw.value, ast.Name) and kw.value.id == "seed":
                parametros[kw.arg] = original["fingerprint"]["seed"]
            else:
                parametros[kw.arg] = ast.unparse(kw.value)
        resultado["hiperparametros"] = parametros
        resultado["origem_hiperparametros"] = "gzcmd/ml_oof.py::_classifier, SHA-256 conferido; ausentes do JSON original"
        resultado["scale_pos_weight_nota"] = "Razao negativos/positivos recalculada no subconjunto de treino de cada ajuste externo e interno; y na expressao corresponde apenas a esse subconjunto."
    return resultado


def main() -> None:
    configurar()
    import sklearn
    import xgboost

    m = carregar_cruzado()
    res = agregar(m)
    controle = verificar(m, res)
    res.to_csv(OUT_CSV, index=False, float_format="%.6f")
    meta = {
        "script": "scripts/desempenho_faixas_oof_refrec.py",
        "versao_script": SCRIPT_VERSION,
        "data_execucao": date.today().isoformat(),
        "entrada_oof": "data/pcal_ml_oof_refrec.csv",
        **proveniencia(),
        "versoes": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "xgboost": xgboost.__version__},
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
