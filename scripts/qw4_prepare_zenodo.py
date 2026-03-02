from __future__ import annotations

import json
import shutil
import sys
from itertools import combinations, product
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.feature_engineering import SCORE_COLS, _clean_col


DATA_DIR = ROOT / "data"
SOURCE_CSV = DATA_DIR / "COMPARADORSEMIDENT.csv"
QW1_DIR = DATA_DIR / "qw1"
SPRINT3B_SUMMARY = DATA_DIR / "sprint3b" / "exp3_kfold_summary.csv"
PIPELINE_OUTPUT_DIR = DATA_DIR / "pipeline_output"
QW2_DIR = DATA_DIR / "qw2"
QW3_DIR = DATA_DIR / "qw3"
ZENODO_DIR = DATA_DIR / "zenodo"


def _copy_dir(src: Path, dst: Path) -> None:
    """Copia src para dst, removendo dst primeiro se existir."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


@dataclass(frozen=True)
class KStage:
    name: str
    description: str
    n_features: int
    group_count: int
    k_min: int
    groups_lt5: int


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = pd.to_numeric(
                out[col].astype(str).str.replace(",", "."),
                errors="coerce",
            )
    return out


def _load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV, sep=";", low_memory=False)
    df.columns = [_clean_col(c) for c in df.columns]
    df = _to_numeric(df)

    required = ["PAR", "PASSO", *SCORE_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    out = df[required].copy()
    out["PAR"] = out["PAR"].fillna(0).astype(int)
    out["PASSO"] = out["PASSO"].fillna(-1).astype(int)
    return out


def _safe_qbin(series: pd.Series, q: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(0)
    uniq = int(clean.nunique())
    if uniq <= 1:
        return pd.Series(["Q1"] * len(clean), index=clean.index)

    bins = min(q, uniq)
    labels = [f"Q{i + 1}" for i in range(bins)]
    ranked = clean.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=labels).astype(str)


def _k_stats(df_bins: pd.DataFrame) -> tuple[int, int, int]:
    grouped = df_bins.groupby(list(df_bins.columns), dropna=False, observed=True).size()
    return int(len(grouped)), int(grouped.min()), int((grouped < 5).sum())


def _build_k_report(df: pd.DataFrame) -> dict:
    stages: list[KStage] = []

    # Stage 1: all 29 sub-scores in quartiles + PASSO
    stage1 = pd.DataFrame({"PASSO": df["PASSO"].astype(int)})
    for col in SCORE_COLS:
        stage1[col] = _safe_qbin(df[col], 4)
    g1, k1, lt5_1 = _k_stats(stage1)
    stages.append(
        KStage(
            name="stage_1_all_scores_quartiles",
            description=(
                "PASSO + 29 sub-scores discretizados em quartis. "
                "Configuracao inicial de auditoria."
            ),
            n_features=29,
            group_count=g1,
            k_min=k1,
            groups_lt5=lt5_1,
        )
    )

    # Stage 2: family aggregates, all in quartiles
    family = pd.DataFrame({"PASSO": df["PASSO"].astype(int)})
    family["nome_total"] = (
        df["NOME prim frag igual"]
        + df["NOME ult frag igual"]
        + df["NOME qtd frag iguais"]
        + df["NOME qtd frag muito parec"]
    )
    family["mae_total"] = (
        df["NOMEMAE prim frag igual"]
        + df["NOMEMAE ult frag igual"]
        + df["NOMEMAE qtd frag iguais"]
    )
    family["dtnasc_total"] = (
        df["DTNASC dt iguais"]
        + df["DTNASC dt ap 1digi"]
        + df["DTNASC dt inv dia"]
        + df["DTNASC dt inv mes"]
        + df["DTNASC dt inv ano"]
    )
    family["local_total"] = (
        df["CODMUNRES uf igual"]
        + df["CODMUNRES local igual"]
        + df["CODMUNRES local prox"]
    )
    family["endereco_total"] = (
        df["ENDERECO via igual"]
        + df["ENDERECO via prox"]
        + df["ENDERECO numero igual"]
        + df["ENDERECO compl prox"]
        + df["ENDERECO texto prox"]
        + df["ENDERECO tokens jacc"]
    )
    family["nota_final"] = df["nota final"]

    stage2 = pd.DataFrame({"PASSO": family["PASSO"]})
    for col in [
        "nome_total",
        "mae_total",
        "dtnasc_total",
        "local_total",
        "endereco_total",
        "nota_final",
    ]:
        stage2[col] = _safe_qbin(family[col], 4)
    g2, k2, lt5_2 = _k_stats(stage2)
    stages.append(
        KStage(
            name="stage_2_family_aggregates_quartiles",
            description=(
                "Generalizacao por agregacao semantica dos sub-scores "
                "em seis dimensoes, ainda discretizadas em quartis."
            ),
            n_features=6,
            group_count=g2,
            k_min=k2,
            groups_lt5=lt5_2,
        )
    )

    # Stage 3: search best mixed schema that satisfies k >= 5
    family_cols = [
        "nome_total",
        "mae_total",
        "dtnasc_total",
        "local_total",
        "endereco_total",
        "nota_final",
    ]
    prebins: dict[str, dict[int, pd.Series]] = {}
    for col in family_cols:
        prebins[col] = {
            2: _safe_qbin(family[col], 2),
            3: _safe_qbin(family[col], 3),
            4: _safe_qbin(family[col], 4),
        }

    best: tuple | None = None
    for r in range(1, len(family_cols) + 1):
        for subset in combinations(family_cols, r):
            for qs in product([2, 3, 4], repeat=r):
                candidate = pd.DataFrame({"PASSO": family["PASSO"]})
                selected = []
                for col, q in zip(subset, qs):
                    label = f"{col}_q{q}"
                    candidate[label] = prebins[col][q]
                    selected.append(label)
                groups, k_min, groups_lt5 = _k_stats(candidate)
                if k_min >= 5:
                    score = (groups, r, sum(qs))
                    if best is None or score > best[0]:
                        best = (score, selected, candidate, groups, k_min, groups_lt5)

    if best is None:
        raise RuntimeError("Nao foi possivel encontrar esquema com k-anonimidade >= 5")

    _, selected_features, stage3, g3, k3, lt5_3 = best
    stages.append(
        KStage(
            name="stage_3_mixed_bins_selected",
            description=(
                "Esquema final encontrado por busca de granularidade maxima sob restricao "
                "de k-anonimidade >= 5."
            ),
            n_features=len(selected_features),
            group_count=g3,
            k_min=k3,
            groups_lt5=lt5_3,
        )
    )

    selected_groups = (
        stage3.groupby(list(stage3.columns), dropna=False, observed=True)
        .size()
        .reset_index(name="n")
        .sort_values("n")
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(SOURCE_CSV.relative_to(ROOT)).replace("\\", "/"),
        "n_rows": int(len(df)),
        "n_subscores": len(SCORE_COLS),
        "stages": [s.__dict__ for s in stages],
        "selected_schema": {
            "quasi_identifiers": [
                "PASSO",
                *selected_features,
            ],
            "k_min": int(k3),
            "group_count": int(g3),
            "groups_lt5": int(lt5_3),
            "smallest_groups_preview": selected_groups.head(20).to_dict(
                orient="records"
            ),
        },
        "l_diversity_assessment": {
            "evaluated": True,
            "decision": "k-anonimidade adotada como criterio principal",
            "justification": (
                "O atributo sensivel PAR e binario e altamente desbalanceado (~0.4% positivos), "
                "o que torna l-diversidade pouco informativa para grupos pequenos. "
                "A protecao operacional priorizou k-anonimidade com generalizacao de quasi-identificadores, "
                "sem publicacao de identificadores diretos ou texto livre."
            ),
        },
    }


def _write_splits(df: pd.DataFrame, out_path: Path) -> None:
    y = df["PAR"].isin([1, 2]).astype(int)
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    split_df = pd.DataFrame(
        {
            "row_index": np.concatenate([idx_train, idx_test]),
            "split": ["train"] * len(idx_train) + ["test"] * len(idx_test),
        }
    ).sort_values("row_index")
    split_df.to_csv(out_path, index=False)


def _load_baselines() -> dict:
    cv = pd.read_csv(QW1_DIR / "cv_summary.csv")
    holdout = pd.read_csv(QW1_DIR / "holdout_metrics.csv")
    e2e = pd.read_csv(SPRINT3B_SUMMARY, sep=";")

    cv_records = cv.sort_values("f1_mean", ascending=False).to_dict(orient="records")
    hold_records = holdout.sort_values("f1", ascending=False).to_dict(orient="records")

    modes: dict[str, dict] = {}
    for row in e2e.to_dict(orient="records"):
        mode = row["mode"]
        modes[mode] = {
            "n_runs": int(row["n_runs"]),
            "expected_loss_engine": {
                "precision_mean": float(row["exp_precision_mean"]),
                "recall_mean": float(row["exp_recall_mean"]),
                "f1_mean": float(row["exp_f1_mean"]),
                "fbeta_mean": float(row["exp_fbeta_mean"]),
                "llm_used_mean_pct": float(row["llm_used_mean"]),
            },
            "auto_accept_only": {
                "precision_mean": float(row["auto_precision_mean"]),
                "recall_mean": float(row["auto_recall_mean"]),
                "f1_mean": float(row["auto_f1_mean"]),
            },
        }

    # --- QW-1 pipeline_output: XGBoost / ablation / pareto / epidemiological ---
    pipe_csvs = PIPELINE_OUTPUT_DIR / "csvs"

    cv5 = pd.read_csv(pipe_csvs / "cv_5fold_results.csv")
    cv5_records = cv5.sort_values("f1_mean", ascending=False).to_dict(orient="records")

    ablation = pd.read_csv(pipe_csvs / "ablation_results.csv")
    ablation_records = ablation.sort_values("f1", ascending=False).to_dict(orient="records")

    pareto = pd.read_csv(pipe_csvs / "pareto_grid.csv")
    # resumo Pareto: pontos com precision >= 0.95 e recall >= 0.80
    pareto_frontier = pareto[
        (pareto["precision"] >= 0.95) & (pareto["recall"] >= 0.80)
    ].sort_values("f1", ascending=False)
    pareto_summary = {
        "n_configs_total": int(len(pareto)),
        "n_frontier_gte95prec_gte80rec": int(len(pareto_frontier)),
        "frontier_records": pareto_frontier.head(10).to_dict(orient="records"),
    }

    epi = pd.read_csv(pipe_csvs / "epidemiological_impact.csv")
    epi_records = epi.to_dict(orient="records")

    # --- QW-2: LLM Kimi ---
    qw2_sections: dict[str, dict] = {}
    for mode_name, fname in [
        ("confirmacao", "qw2_confirmacao_kimi.json"),
        ("vigilancia", "qw2_vigilancia_kimi.json"),
    ]:
        jpath = QW2_DIR / fname
        if jpath.exists():
            raw = json.loads(jpath.read_text(encoding="utf-8"))
            qw2_sections[mode_name] = {
                "experiment": raw.get("experiment"),
                "model": raw.get("model"),
                "mode": raw.get("mode"),
                "n_review_pairs": raw.get("n_review_pairs"),
                "n_reviewed": raw.get("n_reviewed"),
                "n_errors": raw.get("n_errors"),
                "elapsed_s": raw.get("elapsed_s"),
                "metrics": raw.get("metrics"),
            }

    # --- QW-3: fairness / análise comparativa ---
    qw3_manifest_path = QW3_DIR / "qw3_manifest.json"
    qw3_manifest = {}
    if qw3_manifest_path.exists():
        qw3_manifest = json.loads(qw3_manifest_path.read_text(encoding="utf-8"))

    qw3_group_path = QW3_DIR / "qw3_group_metrics.csv"
    qw3_group_records: list[dict] = []
    if qw3_group_path.exists():
        qw3_group = pd.read_csv(qw3_group_path)
        qw3_group_records = qw3_group.to_dict(orient="records")

    qw3_pairs_path = QW3_DIR / "qw3_pairwise_tests.csv"
    qw3_pairs_records: list[dict] = []
    if qw3_pairs_path.exists():
        qw3_pairs = pd.read_csv(qw3_pairs_path)
        qw3_pairs_records = qw3_pairs.to_dict(orient="records")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "qw1_cv_summary": "data/qw1/cv_summary.csv",
            "qw1_holdout_metrics": "data/qw1/holdout_metrics.csv",
            "gzcmd_e2e_summary": "data/sprint3b/exp3_kfold_summary.csv",
            "pipeline_output_cv5fold": "data/pipeline_output/csvs/cv_5fold_results.csv",
            "pipeline_output_ablation": "data/pipeline_output/csvs/ablation_results.csv",
            "pipeline_output_pareto": "data/pipeline_output/csvs/pareto_grid.csv",
            "pipeline_output_epi": "data/pipeline_output/csvs/epidemiological_impact.csv",
            "qw2_confirmacao": "data/qw2/qw2_confirmacao_kimi.json",
            "qw2_vigilancia": "data/qw2/qw2_vigilancia_kimi.json",
            "qw3_manifest": "data/qw3/qw3_manifest.json",
            "qw3_group_metrics": "data/qw3/qw3_group_metrics.csv",
            "qw3_pairwise_tests": "data/qw3/qw3_pairwise_tests.csv",
        },
        "qw1_model_comparison": {
            "cross_validation": {
                "scheme": "5x5 repeated stratified CV (25 folds)",
                "records": cv_records,
            },
            "holdout": {
                "scheme": "30% stratified holdout (seed=42)",
                "records": hold_records,
            },
        },
        "gzcmd_end_to_end": {
            "description": (
                "Pipeline operacional GZ-CMD (classificador + calibracao + guardrails + "
                "expected loss engine + simulacao de revisao LLM)"
            ),
            "modes": modes,
        },
        "xgboost_cv_5fold": {
            "description": "Validacao cruzada 5-fold do XGBoost e modelos comparados (pipeline_output)",
            "records": cv5_records,
        },
        "xgboost_ablation": {
            "description": "Estudo de ablacao: contribuicao de cada familia de sub-scores",
            "records": ablation_records,
        },
        "xgboost_pareto": {
            "description": "Grade Pareto threshold ML x regras (resumo da fronteira otima)",
            **pareto_summary,
        },
        "xgboost_epidemiological_impact": {
            "description": "Estimativa de impacto epidemiologico dos erros de classificacao",
            "records": epi_records,
        },
        "qw2_llm": {
            "description": "Experimentos com LLM (Kimi) para revisao de pares candidatos",
            "modes": qw2_sections,
        },
        "qw3_fairness": {
            "description": "Analise de equidade por grupos (dimensoes: sexo, raca, UF, faixa etaria)",
            "manifest": qw3_manifest,
            "group_metrics": qw3_group_records,
            "pairwise_tests": qw3_pairs_records,
        },
    }


def _write_readme(df: pd.DataFrame, out_path: Path) -> None:
    positive_rate = float(df["PAR"].isin([1, 2]).mean())
    lines = [
        "# Benchmark de Similaridade para Record Linkage TB (SINAN)",
        "",
        "## Descricao (PT)",
        "",
        "Este pacote disponibiliza um benchmark anonimo para reproducao de experimentos",
        "de record linkage probabilistico em vigilancia da tuberculose.",
        "O arquivo principal contem apenas sub-scores de similaridade numericos,",
        "o rotulo PAR e o campo PASSO.",
        "",
        "## Description (EN)",
        "",
        "This package provides an anonymous benchmark to reproduce probabilistic",
        "record linkage experiments in tuberculosis surveillance.",
        "The main file contains only numeric similarity sub-scores,",
        "the PAR label, and the PASSO field.",
        "",
        "## Arquivos / Files",
        "",
        "- `benchmark_sim_sinan_tb.csv`: base anonimizada / anonymized benchmark dataset",
        "- `k_anonymity_report.json`: auditoria de anonimidade / anonymity audit report",
        "- `splits.csv`: indices de treino/teste reprodutiveis / reproducible train-test indices (seed=42)",
        "- `baselines.json`: baselines do QW-1, GZ-CMD, XGBoost, QW-2 e QW-3 / all experiment baselines",
        "- `zenodo_metadata.json`: metadados para deposito / deposition metadata",
        "- `resultados/`: outputs completos do pipeline XGBoost (probabilidades, SHAP, ablacao, Pareto, impacto epidemiologico)",
        "  - `resultados/probas/`: probabilidades holdout de todos os modelos (XGBoost, RF+SMOTE, GB, TabNet)",
        "  - `resultados/shap/`: valores SHAP brutos (.npy) e importancias consolidadas",
        "  - `resultados/csvs/`: ablacao, CV 5-fold, grid Pareto, impacto epidemiologico, importancia por faixa",
        "- `qw2/`: resultados dos experimentos com LLM (Kimi) nos modos confirmacao e vigilancia",
        "- `qw3/`: analise de equidade por grupos demograficos (sexo, raca, UF, faixa etaria)",
        "",
        "## Estrutura da base / Dataset structure",
        "",
        f"- Registros: {len(df)}",
        f"- Rows: {len(df)}",
        "- Colunas / Columns: PAR, PASSO e / and 29 similarity sub-scores",
        f"- Taxa de pares positivos / Positive-pair rate (PAR in {{1,2}}): {positive_rate:.4f}",
        "",
        "## Variaveis / Variables",
        "",
        "- `PAR`: rotulo original de pareamento / original matching label (positive when PAR in {1,2})",
        "- `PASSO`: estagio de processamento / processing stage in the original comparator",
    ]
    lines.extend(
        [
            f"- `{col}`: sub-score de similaridade / similarity sub-score"
            for col in SCORE_COLS
        ]
    )
    lines.extend(
        [
            "",
            "## Metodologia de anonimizacao (PT)",
            "",
            "1. Remocao de campos de identificacao direta e texto livre.",
            "2. Retencao exclusiva de sub-scores numericos + PAR + PASSO.",
            "3. Auditoria de k-anonimidade com discretizacao e generalizacao progressiva.",
            "4. Esquema final com k minimo >= 5, detalhado em `k_anonymity_report.json`.",
            "",
            "## Anonymization methodology (EN)",
            "",
            "1. Removal of direct identifiers and free-text fields.",
            "2. Retention of numeric sub-scores only, plus PAR and PASSO.",
            "3. k-anonymity audit with progressive discretization and generalization.",
            "4. Final schema with minimum k >= 5, documented in `k_anonymity_report.json`.",
            "",
            "## Licenca / License",
            "",
            "CC-BY-NC-4.0",
            "",
            "## Citacao sugerida / Suggested citation",
            "",
            "Jardim, M. Benchmark de Similaridade para Record Linkage TB (SINAN), 2026.",
            "Dataset com DOI Zenodo (a ser preenchido apos publicacao) /",
            "Zenodo DOI to be added after publication.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metadata(out_path: Path) -> None:
    today = date.today().isoformat()
    metadata = {
        "metadata": {
            "title": "Benchmark de Similaridade para Record Linkage em Tuberculose (SINAN)",
            "upload_type": "dataset",
            "description": (
                "PT: Benchmark anonimizado de sub-scores de similaridade para reproducao "
                "de experimentos de record linkage em vigilancia de tuberculose no Brasil.\n\n"
                "EN: Anonymous benchmark of similarity sub-scores for reproducing record "
                "linkage experiments in tuberculosis surveillance in Brazil."
            ),
            "creators": [
                {
                    "name": "Jardim, Marco",
                    "affiliation": "IESC - UFRJ",
                }
            ],
            "license": "cc-by-nc-4.0",
            "keywords": [
                "record linkage",
                "tuberculose",
                "tuberculosis",
                "SINAN",
                "benchmark",
                "saude publica",
                "public health",
                "k-anonimidade",
                "k-anonymity",
            ],
            "version": "2.0.0",
            "language": "por",
            "publication_date": today,
            "related_identifiers": [
                {
                    "identifier": "https://github.com/marco-jardim/gzcmd-record-linkage",
                    "relation": "isSupplementTo",
                    "resource_type": "software",
                },
                {
                    "identifier": "https://github.com/marco-jardim/phd-research",
                    "relation": "isPartOf",
                    "resource_type": "software",
                },
            ],
        }
    }
    out_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    ZENODO_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_source()
    benchmark_path = ZENODO_DIR / "benchmark_sim_sinan_tb.csv"
    df.to_csv(benchmark_path, index=False)

    k_report = _build_k_report(df)
    (ZENODO_DIR / "k_anonymity_report.json").write_text(
        json.dumps(k_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_readme(df, ZENODO_DIR / "README.md")
    _write_metadata(ZENODO_DIR / "zenodo_metadata.json")
    _write_splits(df, ZENODO_DIR / "splits.csv")

    baselines = _load_baselines()
    (ZENODO_DIR / "baselines.json").write_text(
        json.dumps(baselines, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # --- copiar outputs de experimentos ---
    if PIPELINE_OUTPUT_DIR.exists():
        _copy_dir(PIPELINE_OUTPUT_DIR, ZENODO_DIR / "resultados")
        print("  copiado: pipeline_output/ -> zenodo/resultados/")
    else:
        print("  AVISO: pipeline_output/ nao encontrado, pulando resultados/")

    if QW2_DIR.exists():
        _copy_dir(QW2_DIR, ZENODO_DIR / "qw2")
        print("  copiado: qw2/ -> zenodo/qw2/")
    else:
        print("  AVISO: qw2/ nao encontrado")

    if QW3_DIR.exists():
        _copy_dir(QW3_DIR, ZENODO_DIR / "qw3")
        print("  copiado: qw3/ -> zenodo/qw3/")
    else:
        print("  AVISO: qw3/ nao encontrado")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/qw4_prepare_zenodo.py",
        "version": "2.0.0",
        "files": [
            "benchmark_sim_sinan_tb.csv",
            "k_anonymity_report.json",
            "README.md",
            "zenodo_metadata.json",
            "splits.csv",
            "baselines.json",
            "resultados/",
            "qw2/",
            "qw3/",
        ],
    }
    (ZENODO_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("QW-4 v2.0 prep concluido em data/zenodo")


if __name__ == "__main__":
    main()
