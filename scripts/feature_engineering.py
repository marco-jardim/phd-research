"""
Módulo compartilhado de engenharia de atributos para os 3 notebooks da tese.

Garante que TODOS os notebooks utilizem o mesmo conjunto de 58 features
(29 base + 29 engenharia), eliminando inconsistências entre estratégias.

Uso:
    from scripts.feature_engineering import load_data, engineer_features, SCORE_COLS, ENGINEERED_COLS

    df = load_data()
    X, y = engineer_features(df)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 29 colunas-base do comparador de registros (cols 30-59 do CSV, exceto col 50)
# ---------------------------------------------------------------------------
SCORE_COLS: list[str] = [
    "NOME prim frag igual",
    "NOME ult frag igual",
    "NOME qtd frag iguais",
    "NOME qtd frag raros",
    "NOME qtd frag comuns",
    "NOME qtd frag muito parec",
    "NOME qtd frag abrev",
    "NOMEMAE prim frag igual",
    "NOMEMAE ult frag igual",
    "NOMEMAE qtd frag iguais",
    "NOMEMAE qtd frag raros",
    "NOMEMAE qtd frag comuns",
    "NOMEMAE qtd frag muito parec",
    "NOMEMAE qtd frag abrev",
    "DTNASC dt iguais",
    "DTNASC dt ap 1digi",
    "DTNASC dt inv dia",
    "DTNASC dt inv mes",
    "DTNASC dt inv ano",
    "CODMUNRES uf igual",
    "CODMUNRES local igual",
    "CODMUNRES local prox",
    "ENDERECO via igual",
    "ENDERECO via prox",
    "ENDERECO numero igual",
    "ENDERECO compl prox",
    "ENDERECO texto prox",
    "ENDERECO tokens jacc",
    "nota final",
]

# ---------------------------------------------------------------------------
# Nomes das 29 features engenheiradas (para referência externa)
# ---------------------------------------------------------------------------
ENGINEERED_COLS: list[str] = [
    # Agregados (5)
    "nome_total",
    "mae_total",
    "dtnasc_total",
    "endereco_total",
    "local_total",
    # Escore ponderado (1)
    "score_recall",
    # Interações (5)
    "nome_x_dtnasc",
    "nome_x_mae",
    "nome_x_local",
    "dtnasc_x_local",
    "nome_x_endereco",
    # Flags binários (6)
    "nome_bom",
    "nome_prim_ok",
    "dtnasc_ok",
    "dtnasc_parcial",
    "mae_presente",
    "local_ok",
    # Contagem (1)
    "evidencias_positivas",
    # Min/Max (2)
    "min_score_nome",
    "max_score_dtnasc",
    # Razões (2)
    "ratio_nome_mae",
    "ratio_nome_nota",
    # Contexto (2)
    "obito_sinan",
    "obito_tb",
    # Diferença temporal (2)
    "diff_ano_nasc",
    "diff_ano_ok",
    # Polinômios (3)
    "nome_squared",
    "dtnasc_squared",
    "nota_squared",
]

ALL_FEATURE_COLS: list[str] = SCORE_COLS + ENGINEERED_COLS


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def _clean_col(col: str) -> str:
    """Remove sufixos DBF (ex: 'COMPREC,C,12,0' -> 'COMPREC')."""
    return col.split(",")[0] if "," in col else col


def _find_data_path(filename: str = "COMPARADORSEMIDENT.csv") -> Path:
    """Busca o CSV subindo a árvore de diretórios a partir de cwd."""
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "data" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Não foi possível localizar data/{filename}. CWD={Path.cwd()}"
    )


# ---------------------------------------------------------------------------
# Carga e limpeza
# ---------------------------------------------------------------------------


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Carrega o CSV, limpa nomes de colunas e converte tipos numéricos.

    Parameters
    ----------
    path : Path ou str, opcional
        Caminho explícito para o CSV. Se None, busca automaticamente.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas limpas e valores numéricos.
    """
    if path is None:
        path = _find_data_path()
    else:
        path = Path(path)

    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = [_clean_col(c) for c in df.columns]

    # Converter colunas object com vírgula decimal para float
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."), errors="coerce"
                )
            except Exception:
                pass

    logger.info("CSV carregado: %d linhas, %d colunas", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Engenharia de atributos
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica a engenharia canônica de 58 features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame retornado por ``load_data()``.

    Returns
    -------
    X : pd.DataFrame
        Matriz com 58 features (29 base + 29 engenharia).
    y : pd.Series
        Variável alvo binária (1 = par verdadeiro, 0 = não par).
    """
    # Selecionar colunas-base disponíveis
    available = [c for c in SCORE_COLS if c in df.columns]
    missing = set(SCORE_COLS) - set(available)
    if missing:
        logger.warning("Colunas ausentes no CSV (preenchidas com 0): %s", missing)

    X = df[available].fillna(0).copy()

    # Preencher colunas ausentes com 0
    for col in missing:
        X[col] = 0

    # --- 1. Escores agregados (5) ---
    X["nome_total"] = (
        X["NOME prim frag igual"]
        + X["NOME ult frag igual"]
        + X["NOME qtd frag iguais"]
        + X["NOME qtd frag muito parec"]
    )
    X["mae_total"] = (
        X["NOMEMAE prim frag igual"]
        + X["NOMEMAE ult frag igual"]
        + X["NOMEMAE qtd frag iguais"]
    )
    X["dtnasc_total"] = (
        X["DTNASC dt iguais"]
        + X["DTNASC dt ap 1digi"] * 0.8
        + X["DTNASC dt inv dia"] * 0.5
    )
    X["endereco_total"] = (
        X["ENDERECO via igual"]
        + X["ENDERECO via prox"] * 0.7
        + X["ENDERECO texto prox"]
    )
    X["local_total"] = X["CODMUNRES local igual"] + X["CODMUNRES local prox"] * 0.5

    # --- 2. Escore ponderado para recall (1) ---
    X["score_recall"] = (
        X["NOME qtd frag iguais"] * 3.0
        + X["NOME prim frag igual"] * 2.0
        + X["DTNASC dt iguais"] * 2.5
        + X["DTNASC dt ap 1digi"] * 2.0
        + X["NOMEMAE qtd frag iguais"] * 1.5
        + X["CODMUNRES local igual"] * 1.0
        + X["ENDERECO via igual"] * 0.5
    )

    # --- 3. Interações multiplicativas (5) ---
    X["nome_x_dtnasc"] = X["NOME qtd frag iguais"] * (
        X["DTNASC dt iguais"] + X["DTNASC dt ap 1digi"]
    )
    X["nome_x_mae"] = X["NOME qtd frag iguais"] * X["NOMEMAE qtd frag iguais"]
    X["nome_x_local"] = X["NOME qtd frag iguais"] * X["CODMUNRES local igual"]
    X["dtnasc_x_local"] = X["dtnasc_total"] * X["local_total"]
    X["nome_x_endereco"] = X["nome_total"] * X["endereco_total"]

    # --- 4. Flags binários relaxados (6) ---
    X["nome_bom"] = (X["NOME qtd frag iguais"] >= 0.7).astype(int)
    X["nome_prim_ok"] = (X["NOME prim frag igual"] >= 0.8).astype(int)
    X["dtnasc_ok"] = (
        (X["DTNASC dt iguais"] == 1) | (X["DTNASC dt ap 1digi"] == 1)
    ).astype(int)
    X["dtnasc_parcial"] = (
        (X["DTNASC dt inv dia"] == 1) | (X["DTNASC dt inv mes"] == 1)
    ).astype(int)
    X["mae_presente"] = (X["NOMEMAE qtd frag iguais"] > 0.3).astype(int)
    X["local_ok"] = (X["CODMUNRES local igual"] >= 0.8).astype(int)

    # --- 5. Contagem de evidências positivas (1) ---
    X["evidencias_positivas"] = (
        X["nome_bom"]
        + X["nome_prim_ok"]
        + X["dtnasc_ok"]
        + X["dtnasc_parcial"]
        + X["mae_presente"]
        + X["local_ok"]
    )

    # --- 6. Min/Max (2) ---
    X["min_score_nome"] = X[
        ["NOME prim frag igual", "NOME ult frag igual", "NOME qtd frag iguais"]
    ].min(axis=1)
    X["max_score_dtnasc"] = X[["DTNASC dt iguais", "DTNASC dt ap 1digi"]].max(axis=1)

    # --- 7. Razões (2) ---
    X["ratio_nome_mae"] = X["nome_total"] / (X["mae_total"] + 0.1)
    X["ratio_nome_nota"] = X["nome_total"] / (X["nota final"] + 0.1)

    # --- 8. Contexto — requer colunas externas (2) ---
    if "C_SITUENCE" in df.columns:
        X["obito_sinan"] = df["C_SITUENCE"].isin([3, 4]).astype(int)
        X["obito_tb"] = (df["C_SITUENCE"] == 3).astype(int)
    else:
        logger.warning("C_SITUENCE ausente — obito_sinan e obito_tb = 0")
        X["obito_sinan"] = 0
        X["obito_tb"] = 0

    # --- 9. Diferença temporal (2) ---
    if "R_DTNASC" in df.columns and "C_DTNASC" in df.columns:
        r_ano = pd.to_numeric(df["R_DTNASC"].astype(str).str[:4], errors="coerce")
        c_ano = pd.to_numeric(df["C_DTNASC"].astype(str).str[:4], errors="coerce")
        X["diff_ano_nasc"] = np.abs(r_ano - c_ano).fillna(99)
        X["diff_ano_ok"] = (X["diff_ano_nasc"] <= 5).astype(int)
    else:
        logger.warning("R_DTNASC/C_DTNASC ausentes — diff_ano features = 0")
        X["diff_ano_nasc"] = 99
        X["diff_ano_ok"] = 0

    # --- 10. Polinômios (3) ---
    X["nome_squared"] = X["NOME qtd frag iguais"] ** 2
    X["dtnasc_squared"] = X["DTNASC dt iguais"] ** 2
    X["nota_squared"] = X["nota final"] ** 2

    # --- Variável alvo ---
    y = df["PAR"].isin([1, 2]).astype(int)

    # Validação final
    assert X.shape[1] == 58, (
        f"Esperado 58 features, obtido {X.shape[1]}. "
        f"Colunas: {sorted(set(X.columns) - set(ALL_FEATURE_COLS))}"
    )
    logger.info(
        "Features: %d base + %d eng = %d total. Target: %d positivos / %d total",
        len(available),
        len(ENGINEERED_COLS),
        X.shape[1],
        y.sum(),
        len(y),
    )

    return X, y


# ---------------------------------------------------------------------------
# Split estratificado canônico
# ---------------------------------------------------------------------------


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divisão estratificada reprodutível.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
