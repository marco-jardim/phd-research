"""Offline A1/A3/A5/A6 audit. No LLM calls, no pair-field output, seed 42 OOF reused.

python scripts/auditar_funil_adversarial.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.config import load_config
from gzcmd.cost_evaluation import expected_funnel_metrics
from gzcmd.final_funnel import acceptance_metrics
from gzcmd.intervals import wilson95
from gzcmd.policy_engine import Budget, Costs, HumanBudget, HumanError, LLMError, PolicyEngineV3
from gzcmd.reliability import load_band_rates
from scripts.fig_varredura_custos import load_candidates
from scripts.funil_gzcmd_v2 import HAIKU_V4, artifact_paths, load_responses, loss_metrics, replay

MODES = ("vigilancia", "confirmacao")


def estimate_rates(frame, responses, fixed, excluded):
    """Same M3 mapping: FP/all negatives, FN/all positives, u/all; n<10 fixed."""
    training = frame.loc[frame.index.isin(set(responses) - set(excluded))].copy()
    training["decision"] = training.index.map(responses)
    rates, rows = {}, []
    for band, fallback in fixed.items():
        part = training.loc[training.band.eq(band)]
        n, positives = len(part), int(part.y.sum())
        negatives = n - positives
        fp = int((part.y.eq(0) & part.decision.eq("MATCH")).sum())
        fn = int((part.y.eq(1) & part.decision.eq("NONMATCH")).sum())
        unsure = int(part.decision.eq("UNSURE").sum())
        rate = LLMError(fp / negatives if n >= 10 and negatives else fallback["e_fp"],
                        fn / positives if n >= 10 and positives else fallback["e_fn"],
                        unsure / n if n >= 10 else fallback.get("e_unsure", 0.))
        rates[band] = rate
        rows.append({"band": band, "n": n, "positivos": positives, "negativos": negatives,
                     "FP": fp, "FN": fn, "UNSURE": unsure, "e_fp": rate.e_fp,
                     "e_fn": rate.e_fn, "u": rate.e_unsure,
                     "fallback_n_lt10": n < 10, "fallback_fp": negatives == 0 or n < 10,
                     "fallback_fn": positives == 0 or n < 10})
    return rates, rows


def stage_metrics(frame):
    y, d = frame.y, frame.llm_decision
    tp = int((y.eq(1) & d.eq("MATCH")).sum())
    fp = int((y.eq(0) & d.eq("MATCH")).sum())
    fn = int((y.eq(1) & d.eq("NONMATCH")).sum())
    tn = int((y.eq(0) & d.eq("NONMATCH")).sum())
    ratio = lambda a, b: a / b if b else np.nan
    return {"n": len(frame), "VP": tp, "FP": fp, "FN": fn, "VN": tn,
            "UNSURE": int(d.eq("UNSURE").sum()),
            "UNSURE_pos": int((y.eq(1) & d.eq("UNSURE")).sum()),
            "UNSURE_neg": int((y.eq(0) & d.eq("UNSURE")).sum()),
            "sens": ratio(tp, tp + fn), "espec": ratio(tn, tn + fp),
            "sens_denominador": tp + fn, "espec_denominador": tn + fp,
            "sens_todos_vinculos": ratio(tp, int(y.sum()))}


def endpoint_projection(frame, costs, rates, human):
    """Acceptance specificity, including unanswered negatives without fake decisions."""
    metrics = loss_metrics(frame, costs, rates, human)
    negatives = frame.y.eq(0)
    pending = negatives & frame.final.eq("SEM_RESPOSTA_LLM")
    observed_fp = int((negatives & frame.final.eq("MATCH")).sum())
    expected_fp = observed_fp + float(frame.band.map({b: r.e_fp for b, r in rates.items()})[pending].sum())
    total = int(negatives.sum())
    metrics.update({"FP_aceitacao_esperados": expected_fp,
                    "espec_aceitacao_esperada": 1 - expected_fp / total,
                    "espec_aceitacao_inferior": 1 - (observed_fp + int(pending.sum())) / total,
                    "espec_aceitacao_superior": 1 - observed_fp / total})
    return metrics


def engine_run(candidates, config, mode, rates, *, cost=None, human_cap="configured"):
    c = config.decision_policy.modes[mode]
    costs = Costs(c.false_positive_cost, c.false_negative_cost,
                  c.llm_review_cost if cost is None else cost, c.human_review_cost)
    human = HumanError(c.human_e_fp, c.human_e_fn)
    cap = c.human_max_pairs_per_window if human_cap == "configured" else human_cap
    out = PolicyEngineV3(costs, rates, Budget(c.llm_max_calls_per_window), policy_version="v3.1",
                         human_error=human, human_budget=HumanBudget(cap)).triage(candidates)
    out["y"] = out.TARGET.astype(int)
    return out, costs, human


def pct(value):
    return f"{100 * value:.1f}".replace(".", ",")


def pct_ci(value, low, high):
    return f"{pct(value)} ({pct(low)}; {pct(high)})"


def update_tables():
    table_dir = ROOT / "text/latex/tables"
    metrics_path = ROOT / "data/funil_gzcmd_final_metricas.csv"
    metrics = pd.read_csv(metrics_path)
    for index, row in metrics.iterrows():
        low, high = wilson95(int(row.VP_cadeia), int(row.VP_cadeia + row.FN_cadeia))
        metrics.loc[index, ["sens_ci95_low", "sens_ci95_high"]] = [low, high]
    metrics.to_csv(metrics_path, index=False)
    path = table_dir / "tab_cap8_cascata_v31.tex"
    text = path.read_text(encoding="utf-8")
    values = []
    for mode in MODES:
        for version in ("v3", "v3.1"):
            row = metrics.loc[metrics.modo.eq(mode) & metrics.versao.eq(version)].iloc[0]
            values.append(pct_ci(row.sens_cadeia_observada, row.sens_ci95_low, row.sens_ci95_high))
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Sensibilidade automática"):
            lines[index] = r"Sensibilidade automática (\%, IC 95\%) & " + " & ".join(values) + r" \\"
    text = "\n".join(lines) + "\n"
    if "Intervalos de Wilson" not in text:
        text = text.replace("A sensibilidade automática inclui", "Intervalos de Wilson de 95\\% são descritivos, sem ajuste por dependência entre potenciais pares. A sensibilidade automática inclui")
    path.write_text(text, encoding="utf-8")
    summary = pd.read_csv(ROOT / "data/exp2_reruns_resumo.csv")
    path = table_dir / "tab_cap8_reruns.tex"
    lines = path.read_text(encoding="utf-8").splitlines()
    names = {"Kimi K2.5": "kimi-k2.5", "Kimi K3": "kimi-k3", "Claude Haiku 5.5": "claude-haiku-5-5",
             "DeepSeek V4.1 Flash": "deepseek-v4.1-flash", "Levanto Sage 2": "levanto-sage-2"}
    mode = None
    for index, line in enumerate(lines):
        if r"\textbf{Vigilância}" in line:
            mode = "vigilancia"
        elif r"\textbf{Confirmação}" in line:
            mode = "confirmacao"
        if " & " not in line or not any(line.startswith(name) for name in names):
            continue
        cells = line.split(" & ")
        name = cells[0].split("\\")[0]
        dossier, prompt_label = cells[1].split("/")
        prompt = prompt_label.split(" ", 1)[0]
        part = summary.loc[summary["mode"].eq(mode) & summary.model.eq(names[name])
                           & summary.dossie.eq("completo" if dossier == "Completo" else "sem_mo")
                           & summary.prompt.eq(prompt)]
        if "cohort" in summary:
            cohort = ("lote31_refrec" if "lote REFREC" in prompt_label else
                      "lote31_row" if "lote por linha" in prompt_label else None)
            part = part.loc[part.cohort.eq(cohort)] if cohort else part.loc[part.cohort.isna()]
        if len(part) != 1:
            raise ValueError(f"Ambiguous rerun table source: {mode}/{name}/{dossier}/{prompt}")
        row = part.iloc[0]
        for col, field in ((6, "sens"), (7, "espec")):
            cells[col] = "NA" if pd.isna(row[field]) else pct_ci(row[field], row[field + "_ci95_low"], row[field + "_ci95_high"])
        lines[index] = " & ".join(cells)
    text = "\n".join(lines) + "\n"
    text = text.replace("IC de 95\\% de Wilson e análises pareadas constam nos CSVs agregados.",
                        "Sensibilidade e especificidade apresentam IC de 95\\% de Wilson entre parênteses; análises pareadas constam nos CSVs agregados.")
    path.write_text(text, encoding="utf-8")
    path = table_dir / "tab_qw2_confusion.tex"
    path.write_text(re.sub(r"(?<=\d)\\,(?=\d{3}\b)", ".", path.read_text(encoding="utf-8")), encoding="utf-8")
    return metrics


def score_evidence():
    source = ROOT / "scripts/generate_missing_figures.py"
    raw = pd.read_csv(ROOT / "data/COMPARADORSEMIDENT.csv", sep=";", dtype=str, usecols=["nota final", "PAR"])
    nota = pd.to_numeric(raw["nota final"].str.split(",").str[0], errors="raise")
    y = pd.to_numeric(raw.PAR, errors="coerce").isin([1, 2]).astype(int)
    labels = ["0-3", "3-5", "5-6", "6-7", "7-8", "8-9", "9-10", "10+"]
    bands = pd.cut(nota, [0, 3, 5, 6, 7, 8, 9, 10, np.inf], labels=labels, right=False)
    counts = pd.DataFrame({"band": bands, "y": y}).groupby("band", observed=False).agg(n=("y", "size"), vinculos=("y", "sum"))
    assert int(counts.n.sum()) == len(raw)
    for mask, expected in ((nota.lt(5), 39981), (nota.ge(5) & nota.lt(9), 21620), (nota.ge(9), 95)):
        assert int(mask.sum()) == expected
    counts.to_csv(ROOT / "data/tab_score_bands_proveniencia.csv")
    path = ROOT / "text/latex/tables/tab_score_bands.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace("Distribuição dos pares candidatos por faixa do escore do \\textit{linkage} probabilístico", "Distribuição dos potenciais pares por nota final do comparador")
    text = text.replace("Faixa do escore prob.", "Nota final do comparador")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if any(line.strip().startswith(label + " &") for label in labels):
            label = line.strip().split(" &")[0]
            row = counts.loc[label]
            n, positives = int(row.n), int(row.vinculos)
            total_pt = f"{n:,}".replace(",", ".")
            percent_pt = f"{100 * positives / n:.2f}".replace(".", ",")
            lines[index] = f"    {label} & {total_pt} & {positives} & {percent_pt}" + r"\% \\"
        if line.startswith(r"{\footnotesize\textit{Nota}:"):
            lines[index] = r"{\footnotesize\textit{Nota}: a tabela utiliza a coluna \texttt{nota final} do comparador, e não o escore probabilístico do OpenRecLink. As faixas são fechadas à esquerda e abertas à direita; a última inclui notas a partir de 10. Fonte: extrato anonimizado, coluna \texttt{nota final}, e \texttt{scripts/generate\_missing\_figures.py}.}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"generator": source.relative_to(ROOT).as_posix(), "column": "nota final", "coarse_counts": [39981, 21620, 95],
            "generator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bands": counts.reset_index().to_dict("records")}


def main():
    cfg = load_config(ROOT / "gzcmd/gzcmd_v3_config.yaml")
    rates_path = ROOT / "gzcmd/error_rates_by_band_v4.yaml"
    fixed = cfg.llm_review.error_rates_by_band
    frames = {mode: pd.read_csv(ROOT / f"data/funil_gzcmd_{mode}.csv", low_memory=False).set_index("pair_id") for mode in MODES}
    historic = {mode: load_responses(mode, HAIKU_V4)[0] for mode in MODES}
    profile = yaml.safe_load(rates_path.read_text(encoding="utf-8"))
    old = {mode: {} for mode in MODES}
    for source in profile["sources"]:
        if source["model"] != "claude-haiku-5-5":
            continue
        path = ROOT / source["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Reliability source changed after estimation")
        for result in json.loads(path.read_text(encoding="utf-8"))["results"]:
            if result["pair_id"] in old[source["mode"]]:
                raise ValueError("Overlapping calibration sources")
            old[source["mode"]][result["pair_id"]] = result["decision"]
    reused = {mode: set(f.index[f.action.eq("LLM_REVIEW")]) & set(old[mode]) for mode, f in frames.items()}
    candidates = load_candidates(ROOT / "data/COMPARADORSEMIDENT.csv", ROOT / "data/pcal_ml_oof.csv", cfg)
    rows, rate_rows, stage_rows, reconciliation, sources = [], [], [], [], []
    for mode in MODES:
        frame = frames[mode]
        responses, _ = load_responses(mode, HAIKU_V4, include_v31=True)
        supplement_ids = set(responses) - set(historic[mode])
        new_frame = frame.loc[frame.index.isin(supplement_ids) & frame.action.eq("LLM_REVIEW")]
        assert new_frame.action.eq("LLM_REVIEW").all() and new_frame.llm_decision.notna().all()
        stage_rows.append({"modo": mode, "coorte": "somente_novos", **stage_metrics(new_frame)})
        rates = load_band_rates(fixed, mode=mode, source=rates_path)
        reproduced, original_details = estimate_rates(frame, old[mode], fixed, set())
        for band, rate in rates.items():
            np.testing.assert_allclose([rate.e_fp, rate.e_fn, rate.e_unsure],
                                       [reproduced[band].e_fp, reproduced[band].e_fn, reproduced[band].e_unsure])
        leave_rates, details = estimate_rates(frame, old[mode], fixed, reused[mode])
        rate_rows.extend({"modo": mode, "cenario": "original", **detail} for detail in original_details)
        rate_rows.extend({"modo": mode, "cenario": "leaveout", **detail} for detail in details)
        baseline, costs, human = engine_run(candidates, cfg, mode, rates)
        assert baseline.action.equals(frame.action.reindex(baseline.index))
        changed, _, _ = engine_run(candidates, cfg, mode, leave_rates)
        baseline_ids = set(frame.index[frame.action.eq("LLM_REVIEW")])
        leave_ids = set(changed.index[changed.action.eq("LLM_REVIEW")])
        for scenario, out, used_rates in (("original_observado", baseline, rates), ("leaveout", changed, leave_rates)):
            evaluated = replay(out, responses, set(out.index))
            metrics = endpoint_projection(evaluated, costs, used_rates, human)
            complete = not evaluated.final.eq("SEM_RESPOSTA_LLM").any()
            row = {"modo": mode, "cenario": scenario, "n": len(out), "llm": int(out.action.eq("LLM_REVIEW").sum()),
                   "reutilizados_original": len(reused[mode]), "excluidos_taxas": len(reused[mode]) if scenario == "leaveout" else 0,
                   "llm_ainda_nas_taxas": len(set(out.index[out.action.eq("LLM_REVIEW")])
                                              & (set(old[mode]) - (reused[mode] if scenario == "leaveout" else set()))),
                   "llm_adicionados": len(leave_ids - baseline_ids) if scenario == "leaveout" else 0,
                   "llm_removidos": len(baseline_ids - leave_ids) if scenario == "leaveout" else 0,
                   "acoes_alteradas": int(out.action.ne(baseline.action).sum()), "inteiramente_observado": complete, **metrics}
            if complete:
                row.update(acceptance_metrics(evaluated))
            rows.append(row)
        substituted = replay(baseline, {pid: d for pid, d in responses.items() if pid not in reused[mode]}, set(baseline.index))
        rows.append({"modo": mode, "cenario": "reutilizados_substituidos_expectativa", "llm": len(baseline_ids),
                     "reutilizados_original": len(reused[mode]), "inteiramente_observado": False,
                     **endpoint_projection(substituted, costs, rates, human)})
        prior = loss_metrics(replay(baseline, historic[mode], set(baseline.index)), costs, rates, human)
        prospective = expected_funnel_metrics(baseline, rates, human)
        sweep_out, _, _ = engine_run(candidates, cfg, mode, rates, human_cap=None)
        sweep_expected = expected_funnel_metrics(sweep_out, rates, human)
        exact_out, _, _ = engine_run(candidates, cfg, mode, rates, cost=1 / 240, human_cap=None)
        exact_expected = expected_funnel_metrics(exact_out, rates, human)
        observed = loss_metrics(replay(baseline, responses, set(baseline.index)), costs, rates, human)
        saved = pd.read_csv(ROOT / f"data/varredura_custos_{mode}.csv")
        saved = saved.loc[np.isclose(saved.C_llm, .005) & np.isclose(saved.e_fp_h, .05) & np.isinf(saved.orcamento_humano)].iloc[0]
        for field, prior_field in (("sens_esperada_automatica", "sens_esperada"), ("sens_esperada_com_humano", "sens_esperada_com_humano")):
            np.testing.assert_allclose(sweep_expected[field], saved[field])
            reuse_effect = prospective[field] - prior[prior_field]
            budget_effect = sweep_expected[field] - prospective[field]
            difference = sweep_expected[field] - prior[prior_field]
            np.testing.assert_allclose(difference, reuse_effect + budget_effect, atol=1e-14)
            reconciliation.append({"modo": mode, "metrica": field, "cascata_pre_lote": prior[prior_field],
                                   "cascata_pos_lote": observed[prior_field], "varredura": sweep_expected[field],
                                   "efeito_reutilizacao_pp": 100 * reuse_effect,
                                   "efeito_orcamento_pp": 100 * budget_effect,
                                   "diferenca_pre_lote_pp": 100 * difference,
                                   "efeito_respostas_novas_pp": 100 * (observed[prior_field] - prior[prior_field]),
                                   "efeito_custo_exato_vs_arredondado_pp": 100 * (exact_expected[field] - sweep_expected[field]),
                                   "erro_exibicao_1_casa_varredura_pp": round(100 * sweep_expected[field], 1) - 100 * sweep_expected[field],
                                   "llm_orcamento_diferencas": int(sweep_out.action.ne(baseline.action).sum()),
                                   "llm_custo_exato_diferencas": int(exact_out.action.ne(sweep_out.action).sum())})
        for path in artifact_paths(mode, HAIKU_V4, include_v31=True):
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    outputs = {"funil_gzcmd_leaveout.csv": pd.DataFrame(rows), "funil_gzcmd_leaveout_taxas.csv": pd.DataFrame(rate_rows),
               "funil_gzcmd_novos_metricas.csv": pd.DataFrame(stage_rows), "varredura_vs_cascata.csv": pd.DataFrame(reconciliation)}
    for name, table in outputs.items():
        table.to_csv(ROOT / "data" / name, index=False)
    score = score_evidence()
    final = update_tables()
    metadata = {"seed": 42, "sources": sources, "rates_sha256": hashlib.sha256(rates_path.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256((ROOT / "gzcmd/gzcmd_v3_config.yaml").read_bytes()).hexdigest(),
                "oof_sha256": hashlib.sha256((ROOT / "data/pcal_ml_oof.csv").read_bytes()).hexdigest(),
                "leaveout": "mode-specific selected pair_ids excluded from all Haiku v4 reliability sources listed in YAML; no pooling",
                "u_limitation": "marginal UNSURE approximates each class, clipped to 1-e_fn/1-e_fp",
                "score_evidence": score}
    (ROOT / "data/funil_gzcmd_adversarial_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# Auditoria A1/A3/A5/A6", "", "Análise offline das respostas existentes. Calibração OOF preservada. Taxas vigentes não são alteradas por esta auditoria.",
              "Leave-out exclui os identificadores selecionados pela política das fontes Haiku v4 listadas no YAML do respectivo modo, incluindo os suplementos utilizados na reestimação. Não mistura os modos.",
              "Sensibilidade/especificidade do estágio LLM excluem UNSURE; sens_todos_vinculos mantém todos os vínculos no denominador.",
              "UNSURE permanece encaminhamento humano. As projeções com humano pressupõem erro de 5%, não medido.",
              "A substituição dos reutilizados por expectativas é análise de sensibilidade, não IC nem avaliação independente completa.",
              "Mesmo o lote novo foi selecionado pela política informada por taxas antigas; leave-out testa a influência direta da reutilização, não elimina toda seleção adaptativa.",
              "O leave-out é uma perturbação única, não uma validação cruzada: novos selecionados podem pertencer à coorte restante usada na reestimação. A coluna llm_ainda_nas_taxas quantifica essa sobreposição residual; não se alega independência integral da variante.",
              "Se o leave-out selecionar pares sem resposta existente, a projeção e os limites são reportados, não acertos inventados.",
              "A5: cascata_pre_lote observa somente respostas da coorte histórica original; cascata_pos_lote usa também todos os suplementos disponíveis. Não são o mesmo momento.",
              "O efeito de arredondar o custo é um contrafactual separado: ambas as saídas publicadas usam 0,005, não se soma esse efeito à reconciliação.",
              "A1: scripts/generate_missing_figures.py, função fig_score_bands, lê df['nota final'] e pd.cut nessa coluna. Contagens confirmadas no extrato: 39.981/21.620/95.",
              "Figura cascata: escolhida a apresentação apenas dos ramos e contagens. A legenda do capítulo deve deixar de citar custo e sensibilidade com humano; esses indicadores permanecem na tabela. Capítulos não foram editados.",
              "ICs Wilson são descritivos, sem correção por dependência entre pares ou seleção da política."]
    report.extend(["\n## Arquivos alterados ou gerados nesta auditoria",
                   "Código: scripts/auditar_funil_adversarial.py; scripts/avaliar_reruns_exp2.py; scripts/fig_varredura_custos.py; scripts/fig_cascata_v31.py; scripts/generate_missing_figures.py; gzcmd/intervals.py; gzcmd/final_funnel.py; tests/test_auditoria_adversarial.py; gzcmd/M_auditoria_adversarial.md.",
                   "Dados: data/funil_gzcmd_leaveout.csv; data/funil_gzcmd_leaveout_taxas.csv; data/funil_gzcmd_novos_metricas.csv; data/varredura_vs_cascata.csv; data/tab_score_bands_proveniencia.csv; data/funil_gzcmd_final_metricas.csv; data/funil_gzcmd_adversarial_metadata.json; este relatório.",
                   "Tabelas: text/latex/tables/tab_cap8_reruns.tex; tab_cap8_cascata_v31.tex; tab_score_bands.tex; tab_qw2_confusion.tex.",
                   "Figuras PDF/PNG em text/latex/figures/: fig_cap8_varredura_custos; fig_cap8_cascata_v31; fig_cap8_np_confirmacao."])
    for mode in MODES:
        part = outputs["varredura_vs_cascata.csv"].query("modo == @mode")
        a, h = part.iloc[0], part.iloc[1]
        auto_effect = f"{a.efeito_reutilizacao_pp:.4f}".replace(".", ",")
        human_effect = f"{h.efeito_reutilizacao_pp:.4f}".replace(".", ",")
        historical_selected = len(set(frames[mode].index[frames[mode].action.eq("LLM_REVIEW")]) & set(historic[mode]))
        report.append(f"Frase para o capítulo ({mode}): A varredura prospectiva estima sensibilidade automática de {pct(a.varredura)}% e, sob erro humano hipotético de 5,0%, {pct(h.varredura)}%; a cascata pré-lote estimava {pct(a.cascata_pre_lote)}% e {pct(h.cascata_pre_lote)}%, respectivamente, pois incorporava respostas observadas de {historical_selected} pares históricos em vez de suas expectativas por banda, diferença de {auto_effect} e {human_effect} pontos percentuais, sem contribuição de orçamento ou arredondamento do custo na comparação do ponto aprovado.")
    for name, table in outputs.items():
        report.extend([f"\n## {name}", "```", table.to_string(index=False), "```"])
    report.extend(["\n## ICs finais", "```", final[["modo", "versao", "sens_cadeia_observada", "sens_ci95_low", "sens_ci95_high"]].to_string(index=False), "```"])
    (ROOT / "data/funil_gzcmd_adversarial_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    from scripts.fig_varredura_custos import draw_figure, plot_np_figure
    from scripts.fig_cascata_v31 import render
    figures = ROOT / "text/latex/figures"
    draw_figure({mode: pd.read_csv(ROOT / f"data/varredura_custos_{mode}.csv") for mode in MODES},
                figures / "fig_cap8_varredura_custos")
    plot_np_figure(pd.read_csv(ROOT / "data/varredura_np_confirmacao.csv"),
                   pd.read_csv(ROOT / "data/comparacao_np_confirmacao.csv"), figures / "fig_cap8_np_confirmacao")
    render(pd.read_csv(ROOT / "data/funil_gzcmd_final.csv"), figures / "fig_cap8_cascata_v31")


if __name__ == "__main__":
    main()
