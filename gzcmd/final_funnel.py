"""Observed acceptance endpoint and disjoint final cascade for table generation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from .intervals import wilson95

from .llm_usage_cost import PRICES_USD_PER_MILLION, selected_cost


def acceptance_metrics(frame: pd.DataFrame) -> dict:
    if frame.final.eq("SEM_RESPOSTA_LLM").any():
        raise ValueError("Final observed endpoint requires complete LLM responses")
    positive, accepted = frame.y.eq(1), frame.final.eq("MATCH")
    human = frame.final.isin(["DEFERRED", "UNSURE"])
    vp, fp = int((positive & accepted).sum()), int((~positive & accepted).sum())
    fn, vn = int((positive & ~accepted).sum()), int((~positive & ~accepted).sum())
    return {"VP_cadeia": vp, "FP_cadeia": fp, "FN_cadeia": fn, "VN_cadeia": vn,
            "VP_resolvidos": vp, "FP_resolvidos": fp,
            "FN_resolvidos": int((positive & frame.final.eq("NONMATCH")).sum()),
            "VN_resolvidos": int((~positive & frame.final.eq("NONMATCH")).sum()),
            "FN_nonmatch_observado": int((positive & frame.final.eq("NONMATCH")).sum()),
            "VN_nonmatch_observado": int((~positive & frame.final.eq("NONMATCH")).sum()),
            "sens_cadeia_observada": vp / (vp + fn) if vp + fn else np.nan,
            "espec_cadeia_observada": vn / (vn + fp) if vn + fp else np.nan,
            "pares_ao_humano": int(human.sum()), "vinculos_ao_humano": int((human & positive).sum()),
            "nao_vinculos_ao_humano": int((human & ~positive).sum()),
            "humano_direto": int(frame.final.eq("DEFERRED").sum()),
            "humano_unsure": int(frame.final.eq("UNSURE").sum()),
            "pares_ao_llm": int(frame.action.eq("LLM_REVIEW").sum()),
            "llm_sem_resposta": 0, "tipo_metrica": "observada_aceitacao_automatica"}


def cascade_cells(frame: pd.DataFrame, mode: str, version: str) -> list[dict]:
    rows = []

    def add(level, band, stage, subset):
        rows.append({"modo": mode, "versao": version, "tipo_linha": "cascata", "nivel": level,
                     "faixa": band, "etapa": stage, "n": len(subset), "vinculos": int(subset.y.sum())})

    add(0, "TODAS", "pares_revisados", frame)
    for band in ("<5", "[5,9)", ">=9"):
        b = frame.loc[frame.faixa.eq(band)]
        add(1, band, "faixa", b)
        # Level 2 is a partition: sum(n)=band size, including the LLM parent.
        for action in ("ALWAYS_MATCH", "ALWAYS_NONMATCH", "AUTO_MATCH", "AUTO_NONMATCH", "LLM", "DEFERRED"):
            subset = b.loc[b.decisao_motor.eq(action)]
            add(2, band, "HUMANO_DIRETO" if action == "DEFERRED" else action, subset)
        reviewed = b.loc[b.action.eq("LLM_REVIEW")]
        for decision in ("MATCH", "UNSURE", "NONMATCH"):
            add(3, band, "LLM_" + decision, reviewed.loc[reviewed.llm_decision.eq(decision)])
        # Level 4 is a rollup, NOT an additional disjoint branch.
        add(4, band, "HUMANO_TOTAL", b.loc[b.final.isin(["UNSURE", "DEFERRED"])])
    return rows


def write_final_funnel(root: Path):
    # Local import avoids importing any LLM executor or creating a module cycle.
    from scripts.funil_gzcmd_v2 import HAIKU_V4, artifact_paths

    metric_source = pd.read_csv(root / "data/funil_gzcmd_metricas.csv")
    rows, summaries, usd_rows, metadata = [], [], [], []
    for mode in ("vigilancia", "confirmacao"):
        for version, name in (("v3", f"funil_gzcmd_antes_{mode}.csv"), ("v3.1", f"funil_gzcmd_{mode}.csv")):
            frame = pd.read_csv(root / "data" / name, low_memory=False)
            if frame.pair_id.duplicated().any():
                raise ValueError("Duplicate pair_id in final funnel")
            metrics = acceptance_metrics(frame)
            low, high = wilson95(metrics["VP_cadeia"], metrics["VP_cadeia"] + metrics["FN_cadeia"])
            metrics.update({"sens_ci95_low": low, "sens_ci95_high": high})
            prior = metric_source.loc[metric_source.modo.eq(mode) & metric_source.versao.eq(version)].iloc[0]
            metrics.update({"modo": mode, "versao": version,
                            "sens_esperada_pre_lote31": prior.sens_esperada_pre_lote31,
                            "espec_esperada_aceitacao_pre_lote31": prior.espec_esperada_aceitacao_pre_lote31,
                            "sens_esperada_com_humano": prior.sens_esperada_com_humano,
                            "espec_esperada_com_humano": prior.espec_esperada_com_humano,
                            "hipotese_erro_humano": .05 if version == "v3.1" else 0.,
                            "perda_unidades_humanas": prior.perda,
                            "perda_erro_humano_esperada": prior.perda_erro_humano_esperada})
            ids = set(frame.loc[frame.action.eq("LLM_REVIEW"), "pair_id"])
            costs, source_details = selected_cost(artifact_paths(mode, HAIKU_V4, include_v31=version == "v3.1"), ids)
            usd = float(sum(costs.values()))
            metrics.update({"custo_llm_usd": usd, "custo_llm_usd_por_par": usd / len(ids) if ids else 0.,
                            "custo_usd_tipo": "estimativa_tarifaria_tokens_observados"})
            incremental = sum(s["custo_usd_selecionados"] for s in source_details if "anthropic_sem_mo_v4_lote31" in s["source"])
            metrics["custo_llm_usd_lote31"] = incremental
            summaries.append(metrics)
            usd_rows.append({"modo": mode, "versao": version, "pares_llm": len(ids), "custo_llm_usd": usd,
                             "custo_llm_usd_por_par": metrics["custo_llm_usd_por_par"],
                             "custo_usd_lote_antigo_alocado": usd - incremental,
                             "custo_usd_lote31_alocado": incremental,
                             "tipo": metrics["custo_usd_tipo"]})
            metadata.append({"modo": mode, "versao": version, "sources": source_details})
            rows.extend(cascade_cells(frame, mode, version))
            rows.append({"modo": mode, "versao": version, "tipo_linha": "metricas", "nivel": 5,
                         "faixa": "TODAS", "etapa": "metricas_finais", "n": len(frame),
                         "vinculos": int(frame.y.sum()), **metrics})
    table, metrics, usd = pd.DataFrame(rows), pd.DataFrame(summaries), pd.DataFrame(usd_rows)
    table.to_csv(root / "data/funil_gzcmd_final.csv", index=False)
    (root / "data/funil_gzcmd_final_metadata.json").write_text(json.dumps({
        "pricing_usd_per_million_tokens": PRICES_USD_PER_MILLION, "sources": metadata,
        "cost_note": "Recorded tokens priced at the runs' historical tariff; not an invoice",
        "matrix_endpoint": "MATCH accepted automatically; all other states not automatically accepted",
        "human_outcomes_observed": False, "human_error_scenario_v31": .05,
    }, indent=2), encoding="utf-8")
    return table, metrics, usd
