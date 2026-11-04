"""Desempenho por faixa da nota final na configuracao exata da ablacao (Estudo 2).

Reproduz o protocolo de ``ablation_study.py`` e ``epidemiological_impact.py``:
58 atributos de ``feature_engineering.py``, divisao 70/30 estratificada
(semente 42), SMOTE (razao 0,3, k=3, semente 42) somente no treino para o
RF+SMOTE (150 arvores, profundidade 12), Gradient Boosting (100 estimadores,
profundidade 5, taxa 0,1) sem balanceamento e limiar de decisao 0,5, fixado a
priori. Agrega VP/FP/FN, sensibilidade e precisao por faixa da nota final no
conjunto de teste. Grava somente agregados (sem pares nem valores de campos).

Saidas:
  - data/desempenho_faixas_ablacao.csv
  - data/desempenho_faixas_ablacao_meta.json

Uso: python scripts/desempenho_faixas_ablacao.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import engineer_features, load_data, split_data  # noqa: E402

warnings.filterwarnings("ignore")

# pandas >= 3 usa dtype string do Arrow por padrao, o que quebra engineer_features
try:
    pd.options.future.infer_string = False
except (AttributeError, KeyError, ValueError):
    pass

SCRIPT_VERSION = "1.0"
ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "desempenho_faixas_ablacao.csv"
OUT_META = ROOT / "data" / "desempenho_faixas_ablacao_meta.json"

RANDOM_STATE = 42
THRESHOLD = 0.5
RF_PARAMS = {"n_estimators": 150, "max_depth": 12, "random_state": RANDOM_STATE, "n_jobs": -1}
GB_PARAMS = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": RANDOM_STATE}
SMOTE_PARAMS = {"sampling_strategy": 0.3, "k_neighbors": 3, "random_state": RANDOM_STATE}

# Faixas fechadas a esquerda e abertas a direita; a ultima inclui notas >= 10.
FAIXAS: list[tuple[str, float, float]] = [
    ("0-3", -np.inf, 3.0),
    ("3-5", 3.0, 5.0),
    ("5-6", 5.0, 6.0),
    ("6-7", 6.0, 7.0),
    ("7-8", 7.0, 8.0),
    ("8-9", 8.0, 9.0),
    ("9-10", 9.0, 10.0),
    ("10+", 10.0, np.inf),
]
AREA_CINZA = ("5-6", "6-7", "7-8", "8-9")
COLUNAS = ["modelo", "faixa", "n", "vinculos", "vp", "fp", "fn", "sensibilidade", "precisao"]

# Valores de controle: ablation_results.csv e epidemiological_profile.json.
ESPERADO_RF = {"vp": 67, "fp": 3}
ESPERADO_GB = {"vp": 60, "fp": 4}
ESPERADO_VP_FAIXAS_RF = {"5-6": 3, "6-7": 10, "7-8": 14, "8-9": 9}


def atribuir_faixa(nota: float) -> str:
    """Rotulo da faixa da nota final (intervalo [inferior, superior))."""
    for rotulo, inf, sup in FAIXAS:
        if inf <= nota < sup:
            return rotulo
    raise ValueError(f"nota fora das faixas: {nota!r}")


def _razao(num: int, den: int) -> float:
    return float(num) / den if den > 0 else float("nan")


def agregar(nota: pd.Series, y: np.ndarray, pred: np.ndarray, modelo: str) -> pd.DataFrame:
    """Contagens e razoes por faixa e total; razao com denominador zero fica vazia."""
    faixa = nota.map(atribuir_faixa).to_numpy()
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    linhas = []
    for rotulo in [f[0] for f in FAIXAS] + ["total"]:
        m = np.ones(len(y), dtype=bool) if rotulo == "total" else faixa == rotulo
        vp = int(((pred == 1) & (y == 1) & m).sum())
        fp = int(((pred == 1) & (y == 0) & m).sum())
        fn = int(((pred == 0) & (y == 1) & m).sum())
        linhas.append(
            {
                "modelo": modelo,
                "faixa": rotulo,
                "n": int(m.sum()),
                "vinculos": int((y[m] == 1).sum()),
                "vp": vp,
                "fp": fp,
                "fn": fn,
                "sensibilidade": _razao(vp, vp + fn),
                "precisao": _razao(vp, vp + fp),
            }
        )
    return pd.DataFrame(linhas, columns=COLUNAS)


def verificar(res: pd.DataFrame) -> None:
    def tot(modelo: str) -> pd.Series:
        return res[(res.modelo == modelo) & (res.faixa == "total")].iloc[0]

    rf, gb = tot("RF+SMOTE"), tot("GB")
    assert (int(rf.vp), int(rf.fp)) == (ESPERADO_RF["vp"], ESPERADO_RF["fp"]), (
        f"RF+SMOTE total VP/FP = {int(rf.vp)}/{int(rf.fp)}, esperado 67/3"
    )
    assert (int(gb.vp), int(gb.fp)) == (ESPERADO_GB["vp"], ESPERADO_GB["fp"]), (
        f"GB total VP/FP = {int(gb.vp)}/{int(gb.fp)}, esperado 60/4"
    )
    r = res[res.modelo == "RF+SMOTE"].set_index("faixa")
    obtido = {k: int(r.loc[k, "vp"]) for k in ESPERADO_VP_FAIXAS_RF}
    assert obtido == ESPERADO_VP_FAIXAS_RF, f"VP por faixa {obtido}, esperado {ESPERADO_VP_FAIXAS_RF}"


def main() -> None:
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    df = load_data()
    X, y = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=RANDOM_STATE)

    X_res, y_res = SMOTE(**SMOTE_PARAMS).fit_resample(X_train, y_train)
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_res, y_res)
    gb = GradientBoostingClassifier(**GB_PARAMS).fit(X_train, y_train)

    nota = X_test["nota final"]
    pred_rf = rf.predict_proba(X_test)[:, 1] >= THRESHOLD
    pred_gb = gb.predict_proba(X_test)[:, 1] >= THRESHOLD

    res = pd.concat(
        [
            agregar(nota, y_test.to_numpy(), pred_rf, "RF+SMOTE"),
            agregar(nota, y_test.to_numpy(), pred_gb, "GB"),
        ],
        ignore_index=True,
    )
    verificar(res)

    res.to_csv(OUT_CSV, index=False, float_format="%.6f")
    meta = {
        "script": "scripts/desempenho_faixas_ablacao.py",
        "versao_script": SCRIPT_VERSION,
        "data_execucao": date.today().isoformat(),
        "semente": RANDOM_STATE,
        "limiar": THRESHOLD,
        "limiar_otimizado": False,
        "n_atributos": int(X.shape[1]),
        "split": {"test_size": 0.3, "estratificado": True, "random_state": RANDOM_STATE},
        "n_total": int(len(X)),
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
        "vinculos_treino": int(y_train.sum()),
        "vinculos_teste": int(y_test.sum()),
        "smote": SMOTE_PARAMS,
        "rf": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        "gb": GB_PARAMS,
        "faixas": [f[0] for f in FAIXAS],
        "area_cinza": list(AREA_CINZA),
        "controles": {"rf_total": ESPERADO_RF, "gb_total": ESPERADO_GB, "rf_vp_faixas": ESPERADO_VP_FAIXAS_RF},
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(res.to_string(index=False))
    print(f"[OK] {OUT_CSV.name}, {OUT_META.name}")


if __name__ == "__main__":
    main()
