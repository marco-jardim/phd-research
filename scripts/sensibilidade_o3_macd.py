"""Sensibilidade post hoc de O3 com MACD, CPU, sem LLM e sem saída por par."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256
from gzcmd.loader import LoadConfig, feature_engineer, resolve_prefixed_column
from gzcmd.ml_oof import N_SPLITS, SEED, _classifier, nested_oof_probabilities, numeric_features, pair_ids
from scripts.comparador_mesma_carga import align_original, cumulative_counts, load_raw

TARGETS = (9, 23, 36)
TOP_K = (17, 58, 96, 440, 2522)
VARIANTS = ("O1_SCORE", "O3_controle", "O3_MACD")
METRICS = tuple(f"revisoes_{n}" for n in TARGETS) + tuple(
    f"par2_top_{k}" for k in TOP_K
) + ("auc_inteira", "ap_inteira", "auc_abaixo", "ap_abaixo")
PREFIX = "sensibilidade_o3_macd"


def prepare(data_root):
    source = Path(data_root) / "data/COMPARADORSEMIDENT.csv"
    raw = load_raw(source)  # dtype=str, conversão explícita, sem arrays Arrow.
    below = below_cutoff(raw)
    original_path = Path(data_root) / "data/pcal_ml_oof.csv"
    original = align_original(raw, original_path, source)
    original_meta = json.loads(original_path.with_suffix(".json").read_text(encoding="utf-8"))
    x = numeric_features(raw)
    if list(x) != original_meta["features"] or len(x.columns) != 30:
        raise ValueError("Allowlist original de 30 atributos divergiu")
    # O loader espera subscores numéricos, inclusive os originalmente decimais.
    # Preservar datas/identificadores como strings e converter só a allowlist.
    loader_input = raw.copy()
    loader_input[x.columns] = x
    enriched = feature_engineer(loader_input, cfg=LoadConfig(macd_enabled=True))
    ablation_path = ROOT / "data/ablation_comprec_allowlist_metadata.json"
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    if ablation["sha256"]["data/COMPARADORSEMIDENT.csv"] != sha256(source):
        raise ValueError("Extrato diferente do Experimento 1 saneado")
    macd_names = [c for c in ablation["features"]["macd"]["names"] if c.startswith("macd_")]
    if len(macd_names) != 9 or set(macd_names) != {c for c in enriched if c.startswith("macd_")}:
        raise ValueError("MACD divergem do Experimento 1 saneado")
    # Mesma imputação constante das MACD no classificador RF saneado.
    macd = enriched[macd_names].apply(pd.to_numeric, errors="raise").astype("float64").fillna(-1.0)
    if not enriched.index.equals(raw.index) or not np.isfinite(macd.to_numpy()).all():
        raise ValueError("MACD não finitas ou desalinhadas")
    y = pd.to_numeric(raw.PAR, errors="raise").isin([1, 2]).astype(int).to_numpy()
    group_col = resolve_prefixed_column(raw, "REFREC")
    groups = raw[group_col].str.strip().to_numpy(dtype=str)
    if pd.isna(raw[group_col]).any() or (groups == "").any():
        raise ValueError("REFREC ausente")
    return {"source": source, "raw": raw, "below": below, "original": original,
            "original_meta": original_meta, "y": y, "groups": groups,
            "ids": pair_ids(raw).to_numpy(),
            "x": {"O3_controle": x, "O3_MACD": pd.concat([x, macd], axis=1)}}


def estimate_minutes(data):
    """Dois fits-piloto reais; 60 fits totais, fator conservador 3."""
    elapsed = []
    for x in data["x"].values():
        outer = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
        train, _ = next(outer.split(x, data["y"], data["groups"]))
        inner = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + 1)
        fit, pred = next(inner.split(x.iloc[train], data["y"][train], data["groups"][train]))
        start = time.perf_counter()
        model = _classifier(data["y"][train[fit]], SEED)
        model.fit(x.iloc[train[fit]], data["y"][train[fit]])
        model.predict_proba(x.iloc[train[pred]])
        elapsed.append(time.perf_counter() - start)
    return {"pilot_seconds": elapsed, "model_fits": 60, "pilot_fits": 2,
            "safety_factor": 3, "estimated_minutes": max(elapsed) * 60 * 3 / 60}


def queue(ids, scores, below):
    return pd.DataFrame({"id": ids[below], "score": scores[below]}).sort_values(
        ["score", "id"], ascending=[False, True]
    ).id.to_numpy()


def check_control(data, p, folds):
    old = data["original"].p_cal_ml.to_numpy()
    result = {
        "correlation": float(np.corrcoef(p, old)[0, 1]),
        "max_abs_difference": float(np.max(np.abs(p - old))),
        "folds_identical": bool(np.array_equal(folds, data["original"].fold.to_numpy())),
        "queues_identical": bool(np.array_equal(queue(data["ids"], p, data["below"]),
                                                queue(data["ids"], old, data["below"]))),
        "tolerance": 1e-6,
    }
    print("Reprodução do controle: " + json.dumps(result), flush=True)
    if result["max_abs_difference"] > result["tolerance"] or not result["folds_identical"] or not result["queues_identical"]:
        raise RuntimeError("Controle não reproduziu O3; interrompido antes de ajustar MACD: " + json.dumps(result))
    return result


def fit_control(data):
    p, folds = nested_oof_probabilities(data["x"]["O3_controle"], data["y"],
                                       split_by="refrec", groups=data["groups"])
    return p, folds, check_control(data, p, folds)


def summarize(data, scores):
    rows, curves = [], []
    par2 = pd.to_numeric(data["raw"].PAR, errors="raise").eq(2).to_numpy()
    below = data["below"]
    for variant, p in scores.items():
        counts = cumulative_counts(data["ids"][below], p[below], par2[below])
        if len(counts) != 61485 or counts[-1] != 36:
            raise ValueError("Universo abaixo do corte divergiu")
        values = {f"revisoes_{n}": int(np.searchsorted(counts, n) + 1) for n in TARGETS}
        values.update({f"par2_top_{k}": int(counts[k - 1]) for k in TOP_K})
        for scope, mask in (("inteira", np.ones(len(p), dtype=bool)), ("abaixo", below)):
            values[f"auc_{scope}"] = float(roc_auc_score(data["y"][mask], p[mask]))
            values[f"ap_{scope}"] = float(average_precision_score(data["y"][mask], p[mask]))
        rows.extend({"variante": variant, "metrica": key, "valor": value} for key, value in values.items())
        curves.extend({"variante": variant, "k": k, "par2_acumulados": int(n)}
                      for k, n in enumerate(counts, 1))
    summary = pd.DataFrame(rows)
    o1 = summary.loc[summary.variante.eq("O1_SCORE")].set_index("metrica").valor
    if [int(o1[f"revisoes_{n}"]) for n in TARGETS] != [51, 501, 2521]:
        raise ValueError("O1 não reproduziu 51/501/2521")
    return summary, pd.DataFrame(curves)


def write_report(summary, metadata, path):
    wide = summary.pivot(index="variante", columns="metrica", values="valor").loc[list(VARIANTS), list(METRICS)]
    lines = ["# Sensibilidade post hoc de O3 com MACD", "", "## Delimitação", "",
             "Análise exploratória post hoc, exclusivamente em CPU, sem LLM. Não substitui O3 na hipótese fixada (O3 versus O1, margem de 25%). Revisor perfeito; fila abaixo do corte com 61.485 potenciais pares e 36 PAR=2. Desempate pelo hash lexical, sem expansão de empates. O1 usa SCORE, não uma probabilidade OOF; suas AUC/AP são referências de ordenação.", "",
             "## Resultados", "", "| Variante | " + " | ".join(METRICS) + " |",
             "|---|" + "---:|" * len(METRICS)]
    for variant, row in wide.iterrows():
        cells = [f"{row[m]:.6f}".replace(".", ",") if m.startswith(("auc", "ap"))
                 else f"{int(row[m]):,}".replace(",", ".") for m in METRICS]
        lines.append("| " + variant + " | " + " | ".join(cells) + " |")
    control = wide.loc["O3_controle"]
    macd = wide.loc["O3_MACD"]
    lines += ["", "## Resposta", ""]
    before = f"{int(control.revisoes_36):,}".replace(",", ".")
    after = f"{int(macd.revisoes_36):,}".replace(",", ".")
    reduction = f"{100 * (1 - macd.revisoes_36 / control.revisoes_36):.1f}".replace(".", ",")
    lines.append(f"Para recuperar os 36 PAR=2, a cauda passa de {before} para {after} revisões (redução de {reduction}%).")
    lines += ["Os valores para 9 e 23 PAR=2 e os top-k constam da tabela; não se pressupõe melhora uniforme. Os PAR=2 decorrem da revisão do serviço, presumivelmente ordenada por SCORE: há viés de verificação favorável a O1. Não se trata de validação independente de sensibilidade.", "",
              "## Reprodução do controle", "", "```json", json.dumps(metadata["control_reproduction"], indent=2), "```", "",
              "Mesmas partições aninhadas 5×5 por REFREC, seed externa/modelo 42 e seeds internas 43 a 47, XGBoost com 300 estimadores e calibração isotônica. Funções originais importadas, sem alterar a implementação nem o cache. O controle precisa reproduzir as probabilidades a 1e-6, os folds e a fila inteira antes de ajustar a variante MACD.", "",
              "## Atributos equivalentes nos Estudos 1 a 3", "",
              "Em scripts/feature_engineering.py, a engenharia canônica dos três estudos contém diff_ano_nasc (diferença absoluta entre anos) e diff_ano_ok (diferença até cinco anos). Também contém DTNASC dt iguais, DTNASC dt ap 1digi, DTNASC dt inv dia, DTNASC dt inv mes, DTNASC dt inv ano, dtnasc_total, dtnasc_ok, dtnasc_parcial, max_score_dtnasc, dtnasc_squared, nome_x_dtnasc e dtnasc_x_local. São sinais temporais ou derivados de similaridade, não equivalentes exatos às nove MACD. Não há indicadores separados de coincidência de dia, mês e ano nem proximidade em sete/trinta dias na allowlist canônica de 58 atributos. A implementação de diff_ano_nasc usa os quatro primeiros caracteres da data: não equivale à diferença contínua calculada com datas completas pelo loader.", "",
              "O loader gzcmd.loader.feature_engineer também produz diff_ano e dtnasc_score_total, além das nove macd_* quando habilitadas. A presença no loader não implica inclusão no classificador: a allowlist saneada do Experimento 1 contém 30 atributos sem MACD e 39 com MACD. O3 original recebe somente os 30 atributos de numeric_features, sem diff_ano, engenharia canônica de 58 atributos ou macd_*.", "",
              "As chamadas de engineer_features foram confirmadas em scripts/qw1_modal_unified_pipeline.py, scripts/qw1_deep_learning_comparison.py e scripts/ablation_study.py. Nos notebooks históricos, há ainda dtnasc_score_total e dtnasc_perfeito (Estudo 1), diff_ano_nasc e diff_ano_ok (Estudo 2), diff_ano, ano_consistente, dtnasc_exato e dtnasc_quase (Estudo 3). Esses sinais não adicionam os indicadores separados de dia/mês/ano das MACD. A análise nova usa exclusivamente os 30 atributos originais de O3, acrescidos das nove MACD; ausências nas MACD são imputadas com -1, como no classificador do Experimento 1 saneado, enquanto a imputação original dos 30 atributos permanece zero.", "",
              "## Proveniência", "", "Comando: `" + metadata["command"] + "`", "",
              "Artefatos: data/sensibilidade_o3_macd.csv, data/sensibilidade_o3_macd_curva.csv e data/sensibilidade_o3_macd_metadata.json. As saídas contêm somente métricas e contagens agregadas, sem identificadores de pares. Hashes, versões, atributos, estimativa e tempos reais constam dos metadados."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(data_root=ROOT, out=ROOT / "data", *, estimate_only=False):
    import sklearn
    import xgboost

    start = time.perf_counter()
    data = prepare(data_root)
    expected = data["original_meta"]["fingerprint"]
    if sklearn.__version__ != expected["sklearn"] or xgboost.__version__ != expected["xgboost"]:
        raise RuntimeError("Versões sklearn/XGBoost diferentes do O3 original")
    estimate = estimate_minutes(data)
    print("Estimativa antes da execução completa: " + json.dumps(estimate), flush=True)
    if estimate["estimated_minutes"] > 60:
        raise RuntimeError("Estimativa superior a 60 minutos; execução interrompida")
    if estimate_only:
        return None
    p, folds, reproduction = fit_control(data)
    pm, fm = nested_oof_probabilities(data["x"]["O3_MACD"], data["y"], split_by="refrec", groups=data["groups"])
    if not np.array_equal(folds, fm):
        raise ValueError("Partições das variantes divergem")
    score = pd.to_numeric(data["raw"].SCORE.str.replace(",", ".", regex=False), errors="raise").to_numpy()
    summary, curve = summarize(data, dict(zip(VARIANTS, (score, p, pm))))
    code_names = ("scripts/sensibilidade_o3_macd.py", "scripts/comparador_mesma_carga.py",
                  "scripts/feature_engineering.py", "gzcmd/ml_oof.py", "gzcmd/loader.py", "gzcmd/arranjo_corte.py")
    metadata = {
        "analysis": "sensibilidade_post_hoc", "registered": False,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]), "cpu_only": True, "llm_calls": 0,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "seed": SEED, "inner_seeds": list(range(SEED + 1, SEED + N_SPLITS + 1)),
        "folds": N_SPLITS, "split_by": "refrec", "calibration": "nested isotonic 5x5",
        "classifier_parameters": {name: ("NaN" if isinstance(value, float) and np.isnan(value) else value)
                                  for name, value in _classifier(data["y"], SEED).get_params().items()},
        "scale_pos_weight": "recalculado em cada conjunto de treinamento pela função original",
        "features": {name: list(x) for name, x in data["x"].items()},
        "missing_values": {"original_30": 0.0, "macd": -1.0},
        "source_sha256": sha256(data["source"]),
        "input_sha256": {**{name: sha256(Path(data_root) / name) for name in (
            "data/pcal_ml_oof.json", "data/pcal_ml_oof.csv")},
            "data/ablation_comprec_allowlist_metadata.json": sha256(ROOT / "data/ablation_comprec_allowlist_metadata.json")},
        "code_sha256": {name: sha256(ROOT / name) for name in code_names},
        "original_fingerprint": expected, "control_reproduction": reproduction,
        "estimate": estimate, "elapsed_seconds": time.perf_counter() - start,
        "below_cutoff": 61485, "par2": 36, "targets": list(TARGETS), "top_k": list(TOP_K),
        "ordering": "score decrescente, hash lexical crescente, exatamente k potenciais pares",
        "limitation": "Viés de verificação do serviço; post hoc, não altera hipótese O3 vs O1 com margem de 25%.",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"{PREFIX}.csv", index=False, lineterminator="\n")
    curve.to_csv(out / f"{PREFIX}_curva.csv", index=False, lineterminator="\n")
    metadata["output_sha256"] = {f"{PREFIX}{suffix}.csv": sha256(out / f"{PREFIX}{suffix}.csv") for suffix in ("", "_curva")}
    (out / f"{PREFIX}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary, metadata, ROOT / "docs/plans/RELATORIO_O3_MACD_2026-11-07.md")
    print(summary.pivot(index="variante", columns="metrica", values="valor").to_string(), flush=True)
    print(f"Concluído em {metadata['elapsed_seconds']:.1f} s", flush=True)
    return summary, curve, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    generate(args.data_root, args.out, estimate_only=args.estimate_only)
