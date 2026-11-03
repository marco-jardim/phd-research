"""Nested group OOF sensitivity, existing-response replay only; seed 42, no API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.config import load_config
from gzcmd.final_funnel import acceptance_metrics, cascade_cells
from gzcmd.ml_oof import fit_xgb_oof, pair_ids
from gzcmd.policy_engine import Costs, HumanError
from gzcmd.reliability import load_band_rates
from gzcmd.runner import run_v3
from scripts.auditar_funil_adversarial import endpoint_projection
from scripts.funil_gzcmd_v2 import HAIKU_V4, load_responses, replay, safe_funnel


def main():
    source = ROOT / "data/COMPARADORSEMIDENT.csv"
    cfg_path = ROOT / "gzcmd/gzcmd_v3_config.yaml"
    rates_path = ROOT / "gzcmd/error_rates_by_band_v4.yaml"
    cfg = load_config(cfg_path)
    raw = pd.read_csv(source, sep=";", dtype=str, low_memory=False)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    ids = pair_ids(raw)
    labels = pd.Series(pd.to_numeric(raw.PAR, errors="coerce").isin([1, 2]).astype(int).to_numpy(), index=ids)
    metadata = {"seed": 42, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "default": cfg.evaluation.split_by,
                "group_counts": {name: int(raw[name.upper()].str.strip().nunique()) for name in ("refrec", "comprec")}}
    metric_rows, cascade_rows = [], []
    for split in ("row", "refrec", "comprec"):
        output = ROOT / f"data/pcal_ml_oof_{split}.csv"
        previous = pd.read_csv(output).set_index("pair_id") if split == "row" and output.exists() else None
        oof = fit_xgb_oof(source, output_path=output, reuse=True, split_by=split).set_index("pair_id")
        if previous is not None:
            metadata["row_max_abs_difference_previous"] = float((oof.p_cal_ml - previous.p_cal_ml.reindex(oof.index)).abs().max())
        group = raw[split.upper()].str.strip().to_numpy() if split != "row" else np.arange(len(raw))
        folds = pd.Series(ids.map(oof.fold).to_numpy())
        audit = pd.DataFrame({"group": group, "fold": folds})
        assert audit.groupby("group").fold.nunique().max() == 1
        metadata[split] = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        for region, mask in (("total", np.ones(len(oof), dtype=bool)),
                             ("area_cinza", oof.nota.ge(5) & oof.nota.lt(9))):
            part = oof.loc[mask]
            y = labels.reindex(part.index)
            row = {"split_by": split, "recorte": region, "n": len(part), "vinculos": int(y.sum()),
                   "auc": float(roc_auc_score(y, part.p_cal_ml)),
                   "ap": float(average_precision_score(y, part.p_cal_ml)),
                   "brier": float(brier_score_loss(y, part.p_cal_ml)),
                   "mediana_p_vinculos": float(part.loc[y.eq(1), "p_cal_ml"].median())}
            metric_rows.append(row)
            print("OOF_METRICS", json.dumps(row), flush=True)
        if split == "comprec":
            continue
        for mode in ("vigilancia", "confirmacao"):
            result, _ = run_v3(input_csv=source, config_path=cfg_path, mode=mode, p_cal="ml_xgb", policy_version="v3.1",
                                ml_oof_path=output, reuse_ml_oof=True, split_by=split, policy_kind="cost", rates_yaml=rates_path)
            frame = safe_funnel(result)
            responses, _ = load_responses(mode, HAIKU_V4, include_v31=True)
            evaluated = replay(frame, responses, set(frame.index))
            c = cfg.decision_policy.modes[mode]
            rates = load_band_rates(cfg.llm_review.error_rates_by_band, mode=mode, source=rates_path)
            metrics = endpoint_projection(evaluated, Costs(c.false_positive_cost, c.false_negative_cost, c.llm_review_cost,
                                                          c.human_review_cost), rates, HumanError(c.human_e_fp, c.human_e_fn))
            complete = not evaluated.final.eq("SEM_RESPOSTA_LLM").any()
            if complete:
                metrics.update(acceptance_metrics(evaluated))
            summary = {"tipo_linha": "metricas", "split_by": split, "modo": mode, "n": len(frame), "vinculos": int(frame.y.sum()),
                       "LLM": int(frame.action.eq("LLM_REVIEW").sum()), "inteiramente_observado": complete,
                       "AUTO_MATCH": int(frame.decisao_motor.eq("AUTO_MATCH").sum()),
                       "AUTO_MATCH_FP": int((frame.decisao_motor.eq("AUTO_MATCH") & frame.y.eq(0)).sum()),
                       "ALWAYS_MATCH": int(frame.decisao_motor.eq("ALWAYS_MATCH").sum()),
                       "vinculos_auto_nonmatch": int((frame.action.eq("NONMATCH") & frame.y.eq(1)).sum()),
                       "vinculos_auto_nonmatch_cinza": int((frame.action.eq("NONMATCH") & frame.y.eq(1) & frame.grey_zone).sum()),
                       "pendentes_vinculos": int((evaluated.final.eq("SEM_RESPOSTA_LLM") & evaluated.y.eq(1)).sum()),
                       **metrics}
            summary["sens_automatica_pendentes_superior"] = (summary["TP"] + summary["pendentes_vinculos"]) / int(frame.y.sum())
            cascade_rows.append(summary)
            cells = cascade_cells(evaluated, mode, "v3.1")
            cascade_rows.extend({**cell, "split_by": split} for cell in cells)
            for band, part in evaluated.loc[evaluated.final.eq("SEM_RESPOSTA_LLM")].groupby("faixa"):
                cascade_rows.append({"tipo_linha": "cascata", "split_by": split, "modo": mode, "versao": "v3.1",
                                     "nivel": 3, "faixa": band, "etapa": "LLM_SEM_RESPOSTA", "n": len(part), "vinculos": int(part.y.sum())})
            print("CASCADE", json.dumps(summary), flush=True)
    table = pd.DataFrame(metric_rows)
    table.to_csv(ROOT / "data/pcal_ml_oof_grupos_metricas.csv", index=False)
    cascades = pd.DataFrame(cascade_rows)
    cascades.to_csv(ROOT / "data/funil_gzcmd_oof_grupo.csv", index=False)
    summaries = cascades.loc[cascades.tipo_linha.eq("metricas")].copy()
    material = []
    for row in metric_rows:
        if row["split_by"] != "row" and row["auc"] < .98:
            material.append(f"AUC <0,98: {row['split_by']}/{row['recorte']} = {row['auc']:.6f}")
    for mode in ("vigilancia", "confirmacao"):
        pair = summaries.loc[summaries.modo.eq(mode)].set_index("split_by")
        delta = float(pair.loc["refrec", "sens_observada"] - pair.loc["row", "sens_observada"])
        optimistic_delta = float(pair.loc["refrec", "sens_automatica_pendentes_superior"] - pair.loc["row", "sens_observada"])
        if abs(delta) > .02:
            material.append(f"Diferença material de sensibilidade automática conhecida em {mode}: {100 * delta:.4f} p.p.; pendências não são respostas observadas.")
        if optimistic_delta < -.02:
            material.append(f"Mesmo aceitando corretamente todos os vínculos LLM pendentes, a sensibilidade automática REFREC em {mode} fica {100 * optimistic_delta:.4f} p.p. abaixo da por linha, mantendo encaminhamentos humanos separados.")
    metadata["material"] = material
    (ROOT / "data/funil_gzcmd_oof_grupo_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# OOF aninhado por grupo", "", "Seed 42; cinco folds externos e cinco internos; XGBoost e calibração isotônica.",
              "StratifiedGroupKFold mantém grupos disjuntos em ambos os níveis; nenhum identificador de grupo foi persistido.",
              "evaluation.split_by é honrado pelo runner; este script escolhe cada perfil explicitamente em arquivos separados, sem substituir o cache padrão.",
              "REFREC e COMPREC são análises separadas: nenhum deles garante independência simultânea dos dois lados do pareamento.",
              "As métricas são calculadas de novo, sem interpretar 0,993/0,907/0,982 como uma sequência inequívoca de AUC/AP/Brier.",
              "AUC/AP/Brier são OOF. p_cal_platt permanece apenas comparador histórico ajustado globalmente, não é a predição usada nesta avaliação.",
              "A cascata usa taxas Haiku v4 e custos aprovados. Respostas existentes são reutilizadas, pendentes explicitados; sens_observada é a fração de todos os vínculos já aceita automaticamente, portanto limite inferior quando há pendências.",
              "sens_esperada e espec_aceitacao_esperada completam pendências por expectativas, não por classificações observadas. UNSURE é encaminhamento humano, cujo erro hipotético de 5% permanece separado.",
              "sens_automatica_pendentes_superior aceita todos os vínculos LLM pendentes, sem converter encaminhamentos humanos em aceitação automática. Já sens_superior herdada inclui também encaminhamentos humanos, não sendo o limite específico das pendências LLM.",
              "## Métricas", "```", table.to_string(index=False), "```", "## Cascatas lado a lado", "```", summaries.to_string(index=False), "```",
              "## Materialidade", *(material or ["Nenhum critério de materialidade atingido."]),
              f"O padrão configurado é {cfg.evaluation.split_by}; esta comparação não o altera."]
    (ROOT / "data/funil_gzcmd_oof_grupo_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("GROUP_OOF_COMPLETE", json.dumps(material, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
