"""Agregados do probe local; não grava IDs nem valores de campos em data/."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_frontier_preparar import OUT, MODELS
from run_exp2_batch_anthropic import read, save

DECISIONS = ("MATCH", "NONMATCH", "UNSURE")


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z*z/n
    middle = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return middle - half, middle + half


def agreement(left, right, weights=None):
    pairs = [(a, b, w) for a, b, w in zip(left, right, weights if weights is not None else np.ones(len(left)))
             if a in DECISIONS and b in DECISIONS]
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None}
    total = sum(w for _, _, w in pairs)
    observed = sum(w for a, b, w in pairs if a == b) / total
    expected = sum(sum(w for a, _, w in pairs if a == c) * sum(w for _, b, w in pairs if b == c)
                   for c in DECISIONS) / total**2
    return {"n": len(pairs), "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 - 1e-12 else None}


def counters(frame, weighted=False):
    weights = frame.peso if weighted else pd.Series(1., index=frame.index)
    count = lambda mask: float(weights.loc[mask].sum()) if weighted else int(mask.sum())
    dec, par = frame.decision, frame.PAR
    vp = count(dec.eq("MATCH") & par.isin([1, 2]))
    fp = count(dec.eq("MATCH") & par.eq(0))
    n = float(weights.sum()) if weighted else len(frame)
    matches = count(dec.eq("MATCH"))
    lo, hi = wilson(vp, matches) if not weighted else (None, None)
    return {"n": n, **{d: count(dec.eq(d)) for d in DECISIONS}, "VP": vp, "FP": fp,
            "PAR2_recuperados": count(par.eq(2) & dec.eq("MATCH")),
            "PAR2_descartados": count(par.eq(2) & dec.eq("NONMATCH")),
            "PAR2_unsure": count(par.eq(2) & dec.eq("UNSURE")),
            "VPP": vp/(vp + fp) if vp + fp else None, "VPP_IC95_inf": lo, "VPP_IC95_sup": hi,
            "proporcao_NONMATCH": count(dec.eq("NONMATCH"))/n if n else None,
            "proporcao_UNSURE": count(dec.eq("UNSURE"))/n if n else None,
            "abstencao": count(dec.eq("UNSURE"))/n if n else None}


def analyse():
    sample = pd.read_csv(OUT / "estratos.csv")
    baseline = read(OUT / "baseline.json")
    ledger = read(OUT / "custo.json")
    rows, run_status, frames = [], [], {}
    runs = []
    for mode in ("vigilancia", "confirmacao"):
        runs.append({"model": "claude-haiku-5-5", "mode": mode, "passada": "historico",
                     "results": [{"pair_id": pid, "decision": b["haiku"]} for pid, b in baseline[mode].items()],
                     "complete": True, "elapsed_s": None, "cost_estimate_usd": None})
        runs.append({"model": "claude-sonnet-5-5", "mode": mode, "passada": "historico",
                     "results": [{"pair_id": pid, "decision": b["sonnet"]} for pid, b in baseline[mode].items() if b["sonnet"]],
                     "complete": False, "elapsed_s": None, "cost_estimate_usd": None})
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or folder.name.endswith("_piloto"):
            continue
        for path in folder.glob("exp2_llm_sync_*.json"):
            run = read(path)
            if run:
                runs.append(run)
    plan = read(OUT / "plano.json") or {"jobs": []}
    for run in runs:
        model, mode, passed = run["model"], run["mode"], run["passada"]
        record = {"model": model, "modo": mode, "passada": passed, "complete": run["complete"],
                  "elapsed_s": run.get("elapsed_s"), "cost_usd": run.get("cost_estimate_usd")}
        sample_ids = set(sample.loc[sample.modo.eq(mode), "pair_id"])
        record["n_expected"] = run.get("n_review_pairs", len(sample_ids))
        record["n_analisavel"] = sum(r["pair_id"] in sample_ids for r in run["results"])
        record["n_sem_resultado"] = record["n_expected"] - record["n_analisavel"]
        run_status.append(record)
        if not run["results"]:
            continue
        frame = sample.loc[sample.modo.eq(mode)].merge(pd.DataFrame(run["results"]), on="pair_id", validate="one_to_one")
        for field in ("PAR", "haiku", "sonnet"):
            frame[field] = frame.pair_id.map(lambda p: baseline[mode][p][field])
        half = any(j["model"] == model and j["mode"] == mode and j["passada"] == passed and j["s4_half"] for j in plan["jobs"])
        if half:
            frame.loc[frame.estrato.eq("S4"), "peso"] *= 2
        frames[(model, mode, passed)] = frame
        for label, group in list(frame.groupby("estrato")) + [("TOTAL_AMOSTRA", frame), ("TOTAL_PONDERADO", frame)]:
            weighted = label == "TOTAL_PONDERADO"
            # Réplica dirigida não estima a população inteira, nem Sonnet parcial.
            population_estimate = (label == "TOTAL_PONDERADO" and passed != "replica"
                                   and len(frame) == len(sample.loc[sample.modo.eq(mode)])
                                   and bool(run.get("complete")))
            if weighted and not population_estimate and not (half and passed == "principal" and run["complete"]):
                continue
            result = {"model": model, "modo": mode, "estrato": label, "passada": passed,
                      "n_observado": len(group), "ponderado": weighted, **counters(group, weighted)}
            for comparator in ("haiku", "sonnet"):
                value = agreement(group.decision.tolist(), group[comparator].tolist(), group.peso.tolist() if weighted else None)
                result.update({f"{key}_{comparator}": v for key, v in value.items()})
            selected_ids = set(run.get("selected_pair_ids", frame.pair_id))
            call_ids = (set(sample.loc[sample.modo.eq(mode) & sample.estrato.eq(label), "pair_id"]) & selected_ids
                        if not label.startswith("TOTAL") else selected_ids)
            calls = [v for k, v in ledger["calls"].items()
                     if k.startswith(f"{model}_{passed}:{mode}:") and any(p in k for p in call_ids)]
            result["n_sem_resultado"] = len(call_ids) - len(group)
            result["truncadas"] = sum(v["stop_reason"] == "max_tokens" for v in calls)
            rows.append(result)
    comparisons = []
    keys = sorted(frames)
    for i, leftkey in enumerate(keys):
        for rightkey in keys[i+1:]:
            if leftkey[1] != rightkey[1]:
                continue
            common = frames[leftkey].merge(frames[rightkey][["pair_id", "decision"]], on="pair_id", suffixes=("_a", "_b"))
            for label, group in list(common.groupby("estrato")) + [("TOTAL_COMUM", common)]:
                comparisons.append({"model_a": leftkey[0], "passada_a": leftkey[2], "model_b": rightkey[0],
                                    "passada_b": rightkey[2], "modo": leftkey[1], "estrato": label,
                                    **agreement(group.decision_a.tolist(), group.decision_b.tolist())})
    # Attach within-model replica agreement to the requested main aggregate table.
    for row in rows:
        for comparison in comparisons:
            if (comparison["model_a"] == row["model"] == comparison["model_b"]
                    and comparison["modo"] == row["modo"]
                    and any(p.startswith("replica") for p in (comparison["passada_a"], comparison["passada_b"]))
                    and comparison["estrato"] == ("TOTAL_COMUM" if row["estrato"].startswith("TOTAL") else row["estrato"])):
                row.update({f"{key}_replica": comparison[key] for key in ("n", "agreement", "kappa")})
    pd.DataFrame(rows).to_csv(OUT / "resultados_agregados.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "concordancias_agregadas.csv", index=False)
    save(OUT / "execucao_agregada.json", {"runs": run_status, "spent_usd": ledger["spent_usd"],
                                         "reservations_usd": sum(ledger["reservations"].values())})
    report(rows, comparisons, frames, run_status, ledger)
    print(json.dumps({"aggregate_rows": len(rows), "comparisons": len(comparisons), "spent_usd": ledger["spent_usd"],
                      "outstanding_reservations_usd": sum(ledger["reservations"].values())}), flush=True)


def report(rows, comparisons, frames, statuses, ledger):
    sampling = read(OUT / "amostragem.json")
    lines = ["# Probe exploratório de modelos de fronteira", "", "Data nominal do plano: 2026-11-07.", "",
             "## Desenho e execução", "", "Pré-registro: `.local/probe_frontier/PREREG.md`. Semente 20261107.",
             "Políticas P congeladas, mesmos dossiês sem saídas do motor e prompt v4 do Haiku.",
             "S1-S3 e confirmação censitários; S4 amostra aleatória simples, inferência ponderada por inclusão.", ""]
    for mode, a in sampling["modes"].items():
        lines.append(f"- {mode}: população {a['population']}, amostra {a['selected']}, estratos {a['strata']}; "
                     f"Sonnet disponível em {a['sonnet_coverage']}/{a['population']}.")
    lines += ["", "Uso observado multiplicado pelo preço oficial, não fatura. Reservas de custo máximo não são gasto real.",
              f"Custo observado acumulado: USD {ledger['spent_usd']:.6f}; reservas pendentes: USD {sum(ledger['reservations'].values()):.6f}.",
              "", "| Modelo | Chamadas | Entrada | Saída | Custo USD | Truncadas |", "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        calls = [v for v in ledger["calls"].values() if v["model"] == model]
        lines.append(f"| {model} | {len(calls)} | {sum(v['usage'].get('input_tokens',0) for v in calls)} | "
                     f"{sum(v['usage'].get('output_tokens',0) for v in calls)} | {sum(v['cost_usd'] for v in calls):.6f} | "
                     f"{sum(v['stop_reason']=='max_tokens' for v in calls)} |")
    lines += ["", "### Falhas e denominadores", ""]
    for status in statuses:
        if status["n_sem_resultado"]:
            lines.append(f"- {status['model']}, {status['modo']}, {status['passada']}: "
                         f"{status['n_analisavel']}/{status['n_expected']} decisões analisáveis; "
                         f"{status['n_sem_resultado']} sem resultado. "
                         + ("Cobertura parcial do histórico Sonnet." if status['model'] == 'claude-sonnet-5-5'
                            else "Falhas permanecem ausentes, sem imputação ou retry pago automático."))
    lines += ["Truncadas não são classificadas como UNSURE. Concordâncias usam apenas decisões completas em comum; "
              "a ausência pode ser seletiva e limita a comparação de réplicas do Haiku. "
              "Não se publica estimativa ponderada da população para uma passada incompleta."]
    lines += ["", "## Comparação", "", "| Modelo | Modo | Passada | Estrato | n | MATCH | NONMATCH | UNSURE | VP | FP | VPP |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["estrato"] not in ("S1", "S2", "C", "TOTAL_AMOSTRA", "TOTAL_PONDERADO"):
            continue
        values = [r[k] for k in ("model", "modo", "passada", "estrato", "n", "MATCH", "NONMATCH", "UNSURE", "VP", "FP", "VPP")]
        lines.append("| " + " | ".join("NA" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v) for v in values) + " |")
    lines += ["", "IC de Wilson somente nos contadores não ponderados, disponíveis no CSV. "
              "Totais ponderados são estimativas de Horvitz-Thompson; VPP ponderado é uma razão de totais, sem IC de Wilson ingênuo.",
              "Confirmação inteira não representa a vigilância. Sonnet histórico tem cobertura parcial, explicitada nas concordâncias.",
              "", "## Leitura dirigida", ""]
    baseline = read(OUT / "baseline.json")
    for model in MODELS:
        passed = "replica_historico" if model == MODELS[2] else "principal"
        frame = frames.get((model, "vigilancia", passed))
        if frame is None:
            lines.append(f"- {model}: passada principal ainda sem resultados analisáveis.")
            continue
        fp = frame.loc[frame.haiku.eq("MATCH") & frame.PAR.eq(0)]
        dropped = frame.loc[frame.PAR.eq(2) & frame.haiku.ne("MATCH")]
        lines.append(f"- {model}: dos {len(fp)} FP do Haiku observados em S1, "
                     f"{int(fp.decision.eq('MATCH').sum())} permanecem MATCH, {int(fp.decision.eq('NONMATCH').sum())} "
                     f"tornam-se NONMATCH e {int(fp.decision.eq('UNSURE').sum())} UNSURE. Dos {len(dropped)} PAR=2 "
                     f"não aceitos pelo Haiku, {int(dropped.decision.eq('MATCH').sum())} são recuperados, "
                     f"{int(dropped.decision.eq('NONMATCH').sum())} descartados e {int(dropped.decision.eq('UNSURE').sum())} permanecem incertos.")
    opus = frames.get((MODELS[0], "vigilancia", "principal"))
    fable = frames.get((MODELS[1], "vigilancia", "principal"))
    if opus is not None and fable is not None:
        both = opus.merge(fable[["pair_id", "decision"]], on="pair_id", suffixes=("_opus", "_fable"))
        fp = both.loc[both.haiku.eq("MATCH") & both.PAR.eq(0)]
        agree_match = int((fp.decision_opus.eq("MATCH") & fp.decision_fable.eq("MATCH")).sum())
        lines.append(f"- Opus e Fable mantêm MATCH simultaneamente em {agree_match}/{len(fp)} FP do Haiku. "
                     "Isso é apenas sinal para nova revisão humana: erros dos modelos podem ser correlacionados.")
    for mode in ("vigilancia", "confirmacao"):
        selected = [(m, p, frames.get((m, mode, p))) for m, p in
                    ((MODELS[0], "principal"), (MODELS[0], "replica"),
                     (MODELS[1], "principal"), (MODELS[1], "replica"))]
        present = [(m, p, f) for m, p, f in selected if f is not None]
        if len(present) != 4:
            lines.append(f"- {mode}: comparação completa do ruído entre réplicas Opus/Fable indisponível "
                         "enquanto não houver as quatro passadas, ou se a réplica Fable tiver sido retirada por custo.")
            continue
        common_ids = set.intersection(*(set(f.pair_id) for _, _, f in present))
        if not common_ids:
            continue
        aligned = {(m, p): f.set_index("pair_id").loc[sorted(common_ids), "decision"] for m, p, f in present}
        between = float((aligned[(MODELS[0], "principal")] != aligned[(MODELS[1], "principal")]).mean())
        within_opus = float((aligned[(MODELS[0], "principal")] != aligned[(MODELS[0], "replica")]).mean())
        within_fable = float((aligned[(MODELS[1], "principal")] != aligned[(MODELS[1], "replica")]).mean())
        exceeds = between > max(within_opus, within_fable)
        lines.append(f"- {mode}, mesmos {len(common_ids)} potenciais pares nas quatro passadas: discordância entre "
                     f"Opus/Fable {between:.1%}, entre réplicas Opus {within_opus:.1%} e Fable {within_fable:.1%}. "
                     + ("A diferença entre modelos excede descritivamente as duas taxas de variação entre réplicas. " if exceeds
                        else "A diferença entre modelos não excede descritivamente ambas as taxas de variação entre réplicas. ")
                     + "Não é um teste de superioridade; apenas uma réplica por modelo e estratos dirigidos.")
    lines += ["", "### Variação entre réplicas e entre modelos", "",
              "| Modelo A | Passada A | Modelo B | Passada B | Modo | Estrato | n comum | Concordância | Kappa |",
              "|---|---|---|---|---|---|---:|---:|---:|"]
    for c in comparisons:
        if c["estrato"] not in ("S1", "TOTAL_COMUM"):
            continue
        vals = [c[k] for k in ("model_a", "passada_a", "model_b", "passada_b", "modo", "estrato", "n", "agreement", "kappa")]
        lines.append("| " + " | ".join("NA" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals) + " |")
    lines += ["", "Discordância entre modelos deve ser comparada à discordância entre réplicas nos mesmos estratos, "
              "não em amostras diferentes. Sem replicações completas de S4, não é possível atribuir diferenças do total "
              "somente à identidade do modelo. Kappa indefinido em decisões constantes é NA, não zero.",
              "PAR=0 não é demonstração de ausência de vínculo, pois potenciais pares podem não ter sido examinados. "
              "VP/FP são rótulos operacionais relativos a PAR, não adjudicação independente de verdade.",
              "Este probe é exploratório e não substitui resultados principais da tese.",
              "", "## Completude e tempos", "", "```json", json.dumps(statuses, ensure_ascii=False, indent=2), "```",
              "", "Artefatos: `.local/probe_frontier/resultados_agregados.csv`, `concordancias_agregadas.csv`, "
              "`execucao_agregada.json`, `custo.json` e checkpoints por lote/run. Saídas individuais permanecem locais."]
    target = ROOT / "docs/plans/RELATORIO_PROBE_FRONTIER_2026-11-07.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    analyse()
