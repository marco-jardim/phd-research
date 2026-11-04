"""Sensibilidade 2x2 da ablacao: somente agregados, sem chamadas externas.

Uso: python scripts/sensibilidade_sem_obito_refrec.py
Mantem os limiares exploratorios originais, sem nova selecao validada.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gzcmd.splitting import SplitSpec, split_train_test_indices
from scripts.ablation_study import (
    compute_rules_scores,
    eval_binary,
    evaluate_configurations,
    train_ablation_models,
)
from scripts.feature_engineering import engineer_features, split_data

SEED = 42
OBITO = ["obito_tb", "obito_sinan"]
EXTRATO = ROOT / "data/COMPARADORSEMIDENT.csv"
OUT = ROOT / "data/sensibilidade_sem_obito_refrec.csv"
META = OUT.with_name(OUT.stem + "_meta.json")
XGB_PARAMS = dict(
    max_depth=6, learning_rate=0.1, n_estimators=100,
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    device="cpu", random_state=SEED,
)
log = logging.getLogger(__name__)


def carregar(path: Path) -> pd.DataFrame:
    """Preserva REFREC textual; converte explicitamente os campos numericos."""
    pd.options.future.infer_string = False
    df = pd.read_csv(path, sep=";", dtype=str)
    df.columns = [c.split(",")[0] for c in df.columns]
    for col in df.columns:
        if col not in ("REFREC", "COMPREC"):
            df[col] = pd.to_numeric(
                df[col].str.replace(",", ".", regex=False), errors="coerce"
            )
    return df


def remover_obito(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=OBITO)


def montar_cenarios(df, X, y):
    """A/B compartilham indices canonicos; C/D compartilham grupos REFREC."""
    treino, teste, _, _ = split_data(X, y, test_size=0.3, random_state=SEED)
    tr, te = split_train_test_indices(
        df, y, spec=SplitSpec(split_by="refrec", test_size=0.3, seed=SEED,
                             group_stratify=True),
    )
    assert set(df.iloc[tr].REFREC).isdisjoint(df.iloc[te].REFREC)
    reduzida = remover_obito(X)
    return {
        "A": ("linha", X, treino.index, teste.index),
        "B": ("linha", reduzida, treino.index, teste.index),
        "C": ("REFREC", X, X.index[tr], X.index[te]),
        "D": ("REFREC", reduzida, X.index[tr], X.index[te]),
    }


def impacto(y, proba, nota):
    verdadeiro = np.asarray(y) == 1
    ml = (np.asarray(proba) >= 0.5) & verdadeiro
    regra = (np.asarray(nota) >= 8) & verdadeiro
    base = int(regra.sum())
    liquido = int(ml.sum()) - base
    return {
        "recuperados": int((ml & ~regra).sum()),
        "detectados_ambos": int((ml & regra).sum()),
        "detectados_nota": base,
        "detectados_ml": int(ml.sum()),
        "perdidos_vs_nota": int((regra & ~ml).sum()),
        "liquido": liquido,
        "incremento_pct": 100 * liquido / base if base else None,
    }


def conferir_controle(configs, resumo):
    """Compara cada configuracao original por nome, nao pela ordenacao de F1."""
    referencia = pd.read_csv(ROOT / "data/ablation_results.csv").set_index("config")
    obtido = configs.set_index("config")
    assert len(obtido) == 70 and obtido.index.is_unique
    for nome, linha in obtido.iterrows():
        assert nome in referencia.index, nome
        for col in ("tp", "fp", "fn"):
            assert int(linha[col]) == int(referencia.loc[nome, col]), (nome, col)
    assert tuple(resumo[f"rf_smote_{c}"] for c in ("tp", "fp", "fn")) == (67, 3, 7)
    assert tuple(resumo[f"gb_{c}"] for c in ("tp", "fp", "fn")) == (60, 4, 14)
    assert tuple(resumo[c] for c in ("recuperados", "detectados_ambos", "liquido")) == (27, 40, 24)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limpar_nan(obj):
    """Normaliza escalares e ausencias recursivamente para JSON estrito."""
    if isinstance(obj, dict):
        return {k: limpar_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [limpar_nan(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def executar():
    inicio = perf_counter()
    data_inicio = datetime.now(timezone.utc).isoformat()
    df = carregar(EXTRATO)
    X, y = engineer_features(df)
    assert X.shape[1] == 58 and remover_obito(X).shape[1] == 56
    linhas, detalhes = [], {}
    for cenario, (particao, matriz, tr, te) in montar_cenarios(df, X, y).items():
        t0 = perf_counter()
        log.info("Cenario %s: %s/%d", cenario, particao, matriz.shape[1])
        Xtr, Xte, ytr, yte = matriz.loc[tr], matriz.loc[te], y.loc[tr], y.loc[te]
        modelos, probas = train_ablation_models(Xtr, ytr, Xte)
        configs = evaluate_configurations(
            yte, Xte["nota final"].to_numpy(), compute_rules_scores(Xte), probas
        )
        assert len(configs) == 70
        xgb = XGBClassifier(**XGB_PARAMS).fit(Xtr, ytr)
        modelos["XGBoost local (CPU)"] = xgb
        probas["XGBoost local (CPU)"] = xgb.predict_proba(Xte)[:, 1]
        linha = dict(
            cenario=cenario, particao=particao, n_atributos=matriz.shape[1],
            n_treino=len(tr), n_teste=len(te), vinculos_teste=int(yte.sum()),
            n_REFREC_teste=int(df.loc[te, "REFREC"].nunique()),
            n_REFREC_sobrepostos=int(len(set(df.loc[tr, "REFREC"]) & set(df.loc[te, "REFREC"]))),
        )
        for nome, prefixo in (("RF", "rf"), ("RF+SMOTE", "rf_smote"),
                              ("GB", "gb"), ("XGBoost local (CPU)", "xgb_local")):
            metricas = eval_binary(yte, (probas[nome] >= 0.5).astype(int))
            linha.update({f"{prefixo}_{k}": v for k, v in metricas.items()})
        linha.update(impacto(yte, probas["RF+SMOTE"], Xte["nota final"]))
        melhores = configs.loc[configs.groupby("category")["f1"].idxmax()]
        linha.update({"melhor_f1_" + r.category: r.f1 for r in melhores.itertuples()})
        if cenario == "A":
            conferir_controle(configs, linha)
        detalhe = {
            "tempo_segundos": perf_counter() - t0,
            "configuracoes_70": configs.to_dict(orient="records"),
            "melhor_por_categoria": melhores.to_dict(orient="records"),
            "hiperparametros": {n: m.get_params() for n, m in modelos.items()},
        }
        if cenario in ("A", "C"):
            # Ganho medio por divisao, nao SHAP; desempate pela ordem dos atributos.
            ganhos = xgb.feature_importances_
            ordem = np.argsort(-ganhos, kind="stable")
            for atributo in OBITO:
                j = matriz.columns.get_loc(atributo)
                linha[f"ganho_posicao_{atributo}"] = int(np.flatnonzero(ordem == j)[0] + 1)
                linha[f"ganho_{atributo}"] = float(ganhos[j])
            detalhe["importancia"] = "ganho medio normalizado; ordem decrescente; desempate pela ordem das colunas"
        detalhes[cenario] = detalhe
        linhas.append(linha)
        log.info("Cenario %s concluido em %.1f s: F1 RF+SMOTE %.6f; impacto %d/%d",
                 cenario, detalhe["tempo_segundos"], linha["rf_smote_f1"],
                 linha["recuperados"], linha["liquido"])
    res = pd.DataFrame(linhas)
    res.to_csv(OUT, index=False, float_format="%.10g")
    from scripts.gerar_tabela_sensibilidade import gerar
    gerar(res)
    fontes = [
        "scripts/sensibilidade_sem_obito_refrec.py", "scripts/gerar_tabela_sensibilidade.py",
        "scripts/feature_engineering.py", "scripts/ablation_study.py",
        "scripts/epidemiological_impact.py", "scripts/qw1_modal_unified_pipeline.py",
        "gzcmd/splitting.py", "data/ablation_results.csv",
    ]
    meta = {
        "versao_script": "1.0", "inicio_utc": data_inicio,
        "fim_utc": datetime.now(timezone.utc).isoformat(),
        "tempo_total_segundos": perf_counter() - inicio, "semente": SEED,
        "versoes": {p: importlib.metadata.version(p) for p in
                    ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost")},
        "python": platform.python_version(), "plataforma": platform.platform(),
        "entrada_sha256": sha256(EXTRATO), "csv_sha256": sha256(OUT),
        "fontes_sha256": {p: sha256(ROOT / p) for p in fontes},
        "split_linha": {"implementacao": "feature_engineering.split_data (sklearn)",
                        "test_size": 0.3, "seed": SEED, "stratify": "y"},
        "split_REFREC": {"implementacao": "gzcmd.splitting.split_train_test_indices",
                         "test_size": 0.3, "seed": SEED, "group_stratify": True},
        "smote": {"sampling_strategy": 0.3, "k_neighbors": 3, "random_state": SEED,
                  "aplicacao": "somente treino RF+SMOTE"},
        "limiar_ml": 0.5, "limiar_nota": 8,
        "limiares": "herdados da ablacao, selecionados no teste; melhores F1 por categoria sao exploratorios",
        "impacto": "recuperados=ML e nao nota entre vinculos; liquido=VP_ML-VP_nota; incremento_pct=100*liquido/VP_nota",
        "xgboost": "XGBoost local (CPU); parametros do fluxo Modal, sem equivalencia numerica presumida",
        "controle_A": "70 configuracoes comparadas individualmente por config/tp/fp/fn com ablation_results.csv",
        "cenarios": detalhes,
    }
    with META.open("w", encoding="utf-8") as arquivo:
        json.dump(limpar_nan(meta), arquivo, ensure_ascii=False, indent=2, allow_nan=False)
        arquivo.write("\n")
    print(res[["cenario", "n_teste", "vinculos_teste", "rf_smote_f1", "gb_f1",
               "xgb_local_f1", "recuperados", "liquido"]].to_string(index=False))
    return res


if __name__ == "__main__":
    executar()
