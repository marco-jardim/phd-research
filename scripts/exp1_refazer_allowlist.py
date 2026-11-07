"""Reproduz o histórico antes de executar a ablação opt-in, sem exportar pares."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.classifier import GZCMDClassifier, get_feature_columns
from gzcmd.config import load_config
from gzcmd.loader import LoadConfig, load_comparador_csv
from gzcmd.splitting import SplitSpec, split_train_test_indices
from scripts.exp1_testes_t import comparar

SEEDS = [42, 43, 44, 45, 46]
ARMS = {"macd": True, "nomacd": False}
METRICS = ["exp_precision", "exp_recall", "exp_f1", "exp_f2", "review_selected"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_f2(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    denominator = 5 * out.exp_tp + 4 * out.exp_fn + out.exp_fp
    out["exp_f2"] = np.divide(
        5 * out.exp_tp, denominator,
        out=np.zeros(len(out), dtype=float), where=denominator.to_numpy() != 0,
    )
    return out


def compare_reproduction(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, float]:
    keys = ["mode", "seed"]
    a = old.sort_values(keys).reset_index(drop=True)
    b = new.sort_values(keys).reset_index(drop=True)
    # O CSV histórico antecede esta coluna de metadados. Aceitar somente
    # a adição conhecida e somente se os guardrails continuarem habilitados.
    if "guardrails_enabled" not in a and "guardrails_enabled" in b:
        if not b["guardrails_enabled"].eq(True).all():
            raise ValueError("Reprodução exige guardrails habilitados")
        b = b.drop(columns="guardrails_enabled")
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise ValueError("Esquema ou tamanho da reprodução diverge")
    numeric = a.select_dtypes(include="number").columns
    differences = {c: float((a[c] - b[c]).abs().max()) for c in numeric}
    for column in a.columns.difference(numeric):
        if not a[column].equals(b[column]):
            raise ValueError(f"Coluna não numérica divergente: {column}")
    return differences


def comparison_table(
    historical: dict[str, pd.DataFrame], clean: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for mode in sorted(historical["macd"]["mode"].unique()):
        for arm in ARMS:
            a = add_f2(historical[arm].query("mode == @mode"))
            b = add_f2(clean[arm].query("mode == @mode"))
            for metric in METRICS:
                rows.append({
                    "mode": mode, "arm": arm, "metric": metric,
                    "historical_mean": a[metric].mean(),
                    "historical_sd": a[metric].std(ddof=1),
                    "allowlist_mean": b[metric].mean(),
                    "allowlist_sd": b[metric].std(ddof=1),
                    "difference_allowlist_minus_historical": b[metric].mean() - a[metric].mean(),
                })
        reduction = {"mode": mode, "arm": "on_vs_off", "metric": "review_reduction_percent"}
        for label, frames in [("historical", historical), ("allowlist", clean)]:
            on = frames["macd"].query("mode == @mode").set_index("seed").sort_index()
            off = frames["nomacd"].query("mode == @mode").set_index("seed").sort_index()
            if not on.index.equals(off.index):
                raise ValueError("Sementes não coincidem para redução de revisões")
            per_seed = 100 * (1 - on.review_selected / off.review_selected)
            # Razão entre médias reproduz o percentual histórico citado.
            reduction[f"{label}_mean"] = 100 * (1 - on.review_selected.mean() / off.review_selected.mean())
            reduction[f"{label}_sd"] = per_seed.std(ddof=1)
            reduction[f"{label}_paired_p"] = float(stats.ttest_rel(on.review_selected, off.review_selected).pvalue)
        reduction["difference_allowlist_minus_historical"] = reduction["allowlist_mean"] - reduction["historical_mean"]
        rows.append(reduction)
    return pd.DataFrame(rows)


def historical_importances() -> pd.DataFrame:
    rows = []
    cfg = load_config(ROOT / "gzcmd/gzcmd_v3_config.yaml")
    for arm, enabled in ARMS.items():
        frame = load_comparador_csv(ROOT / "data/COMPARADORSEMIDENT.csv", cfg=LoadConfig(macd_enabled=enabled))
        seed_importances = []
        for seed in SEEDS:
            spec = SplitSpec(split_by="comprec", test_size=0.3, seed=seed,
                             group_stratify=cfg.evaluation.group_stratify)
            train_idx, _ = split_train_test_indices(frame, frame.TARGET, spec=spec)
            clf = GZCMDClassifier().fit(frame.iloc[train_idx])
            calibrated = clf.pipeline_.named_steps["classifier"]
            seed_importances.append(np.mean([
                fold.estimator.feature_importances_ for fold in calibrated.calibrated_classifiers_
            ], axis=0))
        means = np.mean(seed_importances, axis=0)
        for name, value in zip(clf.features_, means, strict=True):
            rows.append({"arm": arm, "feature": name, "mean_gini_importance": float(value)})
    return pd.DataFrame(rows).sort_values(["arm", "mean_gini_importance"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-dir", type=Path, default=Path(os.environ["TEMP"]) / "exp1_allowlist_20261107")
    args = parser.parse_args()
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, "-m", "gzcmd", "eval", "--input", "data/COMPARADORSEMIDENT.csv",
            "--config", "gzcmd/gzcmd_v3_config.yaml", "--split-by", "comprec", "--test-size", "0.3",
            "--seeds", *map(str, SEEDS), "--calibration", "ml_rf", "--policy-version", "v3"]
    commands = []
    historical = {}
    reproduction = {}
    versions = {name: importlib.metadata.version(name) for name in ["scikit-learn", "xgboost", "pandas", "numpy", "scipy"]}
    versions["python"] = platform.python_version()
    for arm, enabled in ARMS.items():
        output = args.temp_dir / f"ablation_comprec_{arm}_reproduced.csv"
        command = base + ([] if enabled else ["--no-macd"]) + ["--metrics-out", str(output)]
        commands.append(command)
        print("COMMAND", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        historical[arm] = pd.read_csv(ROOT / f"data/ablation_comprec_{arm}.csv", sep=";")
        reproduction[arm] = compare_reproduction(historical[arm], pd.read_csv(output, sep=";"))
        print("MAX_DIFF", arm, json.dumps(reproduction[arm], sort_keys=True), flush=True)
    proof = {"commands": commands, "versions": versions, "max_abs_differences": reproduction,
             "schema_note": "CSV atual acrescenta guardrails_enabled=True; todas as colunas históricas comparadas, sem excluir nenhuma métrica",
             "tolerance": 0.0005, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    (args.temp_dir / "historical_reproduction.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    if any(value > proof["tolerance"] or not np.isfinite(value)
           for columns in reproduction.values() for value in columns.values()):
        print("REPRODUCTION_FAILED: rodada saneada não executada", flush=True)
        raise SystemExit(2)
    print("REPRODUCTION_PASSED", flush=True)

    clean = {}
    features = {}
    for arm, enabled in ARMS.items():
        output = f"data/ablation_comprec_{arm}_allowlist.csv"
        command = base + ([] if enabled else ["--no-macd"]) + [
            "--feature-allowlist", "--metrics-out", output,
            "--summary-out", f"data/ablation_comprec_{arm}_allowlist_summary.csv"]
        commands.append(command)
        print("COMMAND", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        clean[arm] = pd.read_csv(ROOT / output, sep=";")
        frame = load_comparador_csv(ROOT / "data/COMPARADORSEMIDENT.csv", cfg=LoadConfig(macd_enabled=enabled))
        names = get_feature_columns(frame, feature_set="allowlist", macd_enabled=enabled)
        features[arm] = {"names": names, "n_features": len(names)}
    tests = comparar(on_path=ROOT / "data/ablation_comprec_macd_allowlist.csv",
                     off_path=ROOT / "data/ablation_comprec_nomacd_allowlist.csv")
    tests.to_csv(ROOT / "data/exp1_testes_t_allowlist.csv", index=False, float_format="%.6g")
    comparison = comparison_table(historical, clean)
    comparison.to_csv(ROOT / "data/exp1_vazamento_comparacao.csv", index=False)
    importance = historical_importances()
    importance.to_csv(ROOT / "data/exp1_historical_rf_importances.csv", index=False)
    paths = ["gzcmd/classifier.py", "gzcmd/eval.py", "gzcmd/cli.py", "gzcmd/loader.py",
             "gzcmd/gzcmd_v3_config.yaml", "data/COMPARADORSEMIDENT.csv",
             "scripts/exp1_refazer_allowlist.py", "scripts/exp1_testes_t.py"]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "commands_argv": commands,
        "commands_powershell": [subprocess.list2cmdline(command) for command in commands],
        "seeds": SEEDS, "rf_random_state": 42, "versions": versions,
        "policy_version": "v3 (override explícito; guardrails 3/10 e custo LLM 0,25, como runner histórico)",
        "sha256": {path: sha256(ROOT / path) for path in paths}, "features": features,
        "historical_reproduction": proof,
        "loader": "gzcmd.loader.load_comparador_csv (histórico, inferência de tipos preservada)",
        "error_rates_semantics": "evaluate_v3_dataframe usa cfg.llm_review.error_rates_by_band, taxas inline da config v3, não estimated_rates_path; preservado para isolar seleção de atributos",
        "evaluation": "mesmos splits COMPREC; teste externo intocado no fit; calibração sigmoid CV=5 no treino; sem LLM",
        "reduction_percent": "100*(1-media_on/media_off); DP das reduções por seed; p t pareado bilateral on versus off",
        "importance": "Gini, média das cinco RFs de calibração, depois média das cinco seeds; modelos iguais entre modos; diagnóstico descritivo, não efeito causal",
    }
    (ROOT / "data/ablation_comprec_allowlist_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "docs/plans/RELATORIO_E1_SANEADO_2026-11-07.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Experimento 1: seleção por allowlist\n\n"
        + "Data UTC: " + metadata["created_utc"] + "\n\n## Comandos efetivamente executados\n\n```text\n"
        + "\n".join(metadata["commands_powershell"]) + "\n```\n\n"
        + "## Reprodução histórica\n\nDiferença máxima absoluta por coluna numérica:\n\n```json\n"
        + json.dumps(reproduction, indent=2) + "\n```\n\n"
        + proof["schema_note"] + ".\n\n"
        + "Atributos por braço: " + json.dumps(features, ensure_ascii=False) + ".\n\n"
        + "## Histórico versus saneado\n\n```text\n" + comparison.to_string(index=False) + "\n```\n\n"
        + "F2 calculado de TP, FP e FN esperados em ambos os modos, não confundido com Fbeta=0,5 na confirmação. "
        + metadata["reduction_percent"] + ".\n\n## Testes t saneados\n\n```text\n"
        + tests.to_string(index=False) + "\n```\n\n## Importâncias históricas (15 maiores por braço)\n\n```text\n"
        + importance.groupby("arm", sort=False).head(15).to_string(index=False) + "\n```\n\n"
        + metadata["importance"] + ". A importância de Gini é enviesada por cardinalidade e por correlação; "
        + "não prova causalidade. Lista integral agregada em data/exp1_historical_rf_importances.csv.\n\n"
        + "## Semântica e limites\n\n" + metadata["error_rates_semantics"] + ". "
        + "O comando original sem --policy-version falhou com ValueError: v3.1 ALWAYS_NONMATCH is allowed only below 5. "
        + "O eval passou a expor o override existente no runner; --policy-version v3 restaura os limiares 3/10 e a política histórica, sem alterar defaults. "
        + "O RF mantém random_state=42; 42 a 46 são as seeds dos splits. "
        + "A retirada de SCORE e identificadores não prova ausência universal de vazamento, "
        + "pois nota final e subescores permanecem e o gabarito tem relação com o comparador.\n\n"
        + "## Arquivos e verificação\n\nArtefatos e hashes no metadado. Resultados dos testes serão anexados após execução.\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False), flush=True)
    print(tests.to_string(index=False), flush=True)
    print("EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
