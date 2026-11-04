"""Desempenho por faixa da nota final na configuracao exata da ablacao (Estudo 2).

Reproduz o protocolo de ``ablation_study.py`` e ``epidemiological_impact.py``:
58 atributos de ``feature_engineering.py``, divisao 70/30 estratificada
(semente 42), SMOTE (razao 0,3, k=3, semente 42) somente no treino para o
RF+SMOTE (150 arvores, profundidade 12), Gradient Boosting (100 estimadores,
profundidade 5, taxa 0,1) sem balanceamento e limiar de decisao 0,5,
selecionado no teste da ablacao entre 0,3, 0,5 e 0,7. Agrega VP/FP/FN, sensibilidade e precisao por faixa da nota final no
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

SCRIPT_VERSION = "1.1"
ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "desempenho_faixas_ablacao.csv"
OUT_META = ROOT / "data" / "desempenho_faixas_ablacao_meta.json"

RANDOM_STATE = 42
THRESHOLD = 0.5
# Origem: scripts/ablation_study.py:155-184; igualdade verificada em teste.
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

CONTROLE = ROOT / "data" / "ablation_results.csv"


def configurar() -> None:
    sys.path.insert(0, str(ROOT))
    warnings.filterwarnings("ignore")
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


def verificar(res: pd.DataFrame, controle: Path = CONTROLE) -> dict:
    referencia = pd.read_csv(controle).set_index("config")
    controles = {}
    for modelo in ("RF+SMOTE", "GB"):
        ref = referencia.loc[f"ML-only {modelo} (≥0.5)"]
        esperado = {c: int(ref[k]) for c, k in (("vp", "tp"), ("fp", "fp"), ("fn", "fn"))}
        total = res[(res.modelo == modelo) & (res.faixa == "total")].iloc[0]
        obtido = {c: int(total[c]) for c in esperado}
        if obtido != esperado:
            raise ValueError(f"{modelo} total VP/FP/FN = {obtido}, esperado {esperado}")
        controles[modelo] = esperado
    return controles


def main() -> None:
    configurar()
    import imblearn
    import sklearn
    from scripts.feature_engineering import engineer_features, load_data, split_data
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
    controles = verificar(res)

    res.to_csv(OUT_CSV, index=False, float_format="%.6f")
    meta = {
        "script": "scripts/desempenho_faixas_ablacao.py",
        "versao_script": SCRIPT_VERSION,
        "data_execucao": date.today().isoformat(),
        "semente": RANDOM_STATE,
        "limiar": THRESHOLD,
        "limiar_otimizado": "selecionado_no_teste_da_ablacao",
        "versoes": {"sklearn": sklearn.__version__, "imblearn": imblearn.__version__, "pandas": pd.__version__},
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
        "controles": controles,
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(res.to_string(index=False))
    print(f"[OK] {OUT_CSV.name}, {OUT_META.name}")


if __name__ == "__main__":
    main()
