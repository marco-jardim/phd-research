"""Offline Bayesian M1/M2/M6 funnel; no LLM calls or PII outputs.

python scripts/funil_gzcmd_v2.py --reuse-oof
Triaging uses M3 pooled rates. Model comparisons replay available responses and
use each reviewer's M3 expected loss for unanswered LLM pairs, not human costs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.calibration import get_nota_series
from gzcmd.config import load_config
from gzcmd.ml_oof import pair_ids
from gzcmd.np_evaluation import compare_confirmation, evaluate_birth_variant, evaluate_np_point
from gzcmd.policy_engine import Costs, HumanError, LLMError
from gzcmd.reliability import load_band_rates, reference_rates_model
from gzcmd.runner import run_v3

HAIKU_V4 = "Claude Haiku 5.5 v4 sem MO"
HAIKU_V3 = "Claude Haiku 5.5 v3 com MO"
MODELS = ("Kimi K2.5", HAIKU_V3, HAIKU_V4)
M3_MODELS = {"Kimi K2.5": "kimi-k2.5", "DeepSeek": "deepseek-v4.1-flash",
             "Haiku": "claude-haiku-5-5", "Kimi K3 parcial": "kimi-k3",
             HAIKU_V3: "claude-haiku-5-5", HAIKU_V4: "claude-haiku-5-5"}


def artifact_paths(mode: str, model: str, *, include_v31: bool = False) -> list[Path]:
    base = ROOT / "data/sprint3b"
    if model == HAIKU_V4:
        paths = [base / f"anthropic_sem_mo_v4/exp2_llm_sync_{mode}.json"]
        if include_v31:
            paths.append(base / f"anthropic_sem_mo_v4_lote31/exp2_llm_sync_{mode}.json")
        return paths
    if model == "Kimi K2.5":
        original = base / f"exp2_llm_sync_{mode}.json"
        if mode == "confirmacao" and not original.exists():
            return [ROOT / "data/qw2/qw2_confirmacao_kimi.json"]
        return [original]
    directory = {"DeepSeek": "opencode_go", "Haiku": "anthropic", HAIKU_V3: "anthropic", "Kimi K3 parcial": "opencode_go_kimi"}[model]
    # M3 uses the latest artifact, not the earlier partial336 snapshot.
    return [base / directory / f"exp2_llm_sync_{mode}.json"]


def load_responses(mode: str, model: str, *, include_v31: bool = False) -> tuple[dict[str, str], bool]:
    decisions: dict[str, str] = {}
    available = False
    prompt_sha = None
    for position, path in enumerate(artifact_paths(mode, model, include_v31=include_v31)):
        if not path.exists():
            if include_v31 and model == HAIKU_V4:
                raise FileNotFoundError(f"Required completed cohort missing: {path}")
            continue
        available = True
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("mode") != mode:
            raise ValueError(f"Wrong mode in {path}")
        removed = {"model_outputs." + name for name in ("p_cal", "base_choice", "band", "evr", "guardrail", "guardrail_reason")}
        if model == HAIKU_V4 and (run.get("prompt_version") != "v4" or not run.get("complete")
                                 or run.get("model") != "claude-haiku-5-5"
                                 or not removed.issubset(run.get("removed_fields", []))):
            raise ValueError(f"Invalid v4 reference provenance: {path}")
        if model == HAIKU_V4:
            if include_v31 and position > 0 and run.get("cohort") != "v31_lot":
                raise ValueError(f"Expected cohort v31_lot: {path}")
            if prompt_sha is not None and run.get("prompt_sha256") != prompt_sha:
                raise ValueError(f"Incompatible prompts across cohorts: {path}")
            prompt_sha = run.get("prompt_sha256")
            selected = run.get("selected_pair_ids")
            if selected is not None and set(selected) != {r["pair_id"] for r in run["results"]}:
                raise ValueError(f"Incomplete selected cohort: {path}")
        for result in run["results"]:
            decision = result["decision"].strip().upper()
            if decision not in {"MATCH", "NONMATCH", "UNSURE"}:
                raise ValueError(f"Invalid decision in {path}")
            pid = result["pair_id"]
            if pid in decisions:
                raise ValueError(f"Duplicate response in {path}")
            decisions[pid] = decision
    return decisions, available


def metrics_with_projection(frame, responses, historic_responses, historic_batch, costs, rates, human_error):
    """Keep the frozen pre-completion projection alongside the observed endpoint."""
    current = replay(frame, responses, historic_batch | set(frame.index[frame.action.eq("LLM_REVIEW")]))
    metrics = loss_metrics(current, costs, rates, human_error)
    before = replay(frame, historic_responses, historic_batch)
    prior = loss_metrics(before, costs, rates, human_error)
    negatives = before.y.eq(0)
    missing_negative = before.final.eq("SEM_RESPOSTA_LLM") & negatives
    expected_fp_prior = int((before.final.eq("MATCH") & negatives).sum())
    expected_fp_prior += float(before.band.map({b: r.e_fp for b, r in rates.items()})[missing_negative].sum())
    metrics.update({"sens_esperada_pre_lote31": prior["sens_esperada"],
                    "espec_esperada_pre_lote31": prior["espec_esperada"],
                    "sens_esperada_com_humano_pre_lote31": prior["sens_esperada_com_humano"],
                    "perda_antes_respostas_lote31": prior["perda"],
                    "espec_esperada_aceitacao_pre_lote31": 1 - expected_fp_prior / int(negatives.sum()) if negatives.any() else np.nan,
                    "metricas_classificacao_tipo": "observadas" if metrics["sem_resposta_LLM"] == 0 else "parciais",
                    "cobertura_llm_completa": metrics["sem_resposta_LLM"] == 0})
    return current, metrics


def export_v31_batch(out: pd.DataFrame, responses: dict[str, str], destination: Path, mode: str) -> dict:
    batch = out.loc[out.action.eq("LLM_REVIEW"), ["nota", "p_cal", "band", "decisao_motor"]].reset_index()
    batch = batch.rename(columns={"p_cal": "p_cal_ml", "band": "banda"})
    batch = batch[["pair_id", "nota", "p_cal_ml", "banda", "decisao_motor"]]
    pending = batch.loc[~batch.pair_id.isin(responses)]
    batch.to_csv(destination / f"lote_llm_v31_{mode}.csv", index=False)
    pending.to_csv(destination / f"lote_llm_v31_{mode}_pendentes.csv", index=False)
    counts = {"modo": mode, "lote_llm": len(batch), "com_resposta_haiku_v4": len(batch) - len(pending),
              "pendentes_haiku_v4": len(pending)}
    print("LOTE V3.1", counts)
    return counts


def safe_funnel(df: pd.DataFrame) -> pd.DataFrame:
    nota = pd.to_numeric(get_nota_series(df), errors="raise")
    out = pd.DataFrame({
        "pair_id": pair_ids(df), "nota": nota, "y": df.TARGET.astype(int),
        "faixa": np.where(nota < 5, "<5", np.where(nota < 9, "[5,9)", ">=9")),
        "band": df.band, "guardrail": df.guardrail.fillna("none"),
        "guardrail_reason": df.guardrail_reason.fillna("none"),
        "p_cal": df.p_cal, "evr": df.evr, "base_choice": df.base_choice, "action": df.action,
        "loss_match": df.loss_match, "loss_nonmatch": df.loss_nonmatch, "loss_llm": df.loss_llm,
    }).set_index("pair_id")
    for col in ("loss_human", "selected_loss", "decision_reason", "review_requested", "human_gain", "human_requested"):
        if col in df:
            out[col] = df[col].to_numpy()
    out["decisao_motor"] = np.where(
        out.action.eq("LLM_REVIEW"), "LLM",
        np.where(out.action.eq("DEFERRED"), "DEFERRED",
                 np.where(out.guardrail.eq("ALWAYS_MATCH"), "ALWAYS_MATCH",
                          np.where(out.guardrail.eq("ALWAYS_NONMATCH"), "ALWAYS_NONMATCH", "AUTO_" + out.action))),
    )
    out["grey_zone"] = out.nota.ge(5) & out.nota.lt(9)
    out["high_consistency_failed"] = out.nota.ge(9) & ~out.guardrail.eq("ALWAYS_MATCH")
    out["review_area"] = out.grey_zone | out.high_consistency_failed
    return out


def replay(out: pd.DataFrame, decisions: dict[str, str], old_batch: set[str]) -> pd.DataFrame:
    out = out.copy()
    eligible = {pid: d for pid, d in decisions.items() if pid in old_batch}
    out["llm_decision"] = pd.Series(out.index.map(eligible), index=out.index, dtype="string")
    out.loc[~out.action.eq("LLM_REVIEW"), "llm_decision"] = pd.NA
    out["sem_resposta_llm"] = out.action.eq("LLM_REVIEW") & out.llm_decision.isna()
    out["final"] = out.action.astype("string")
    mask = out.action.eq("LLM_REVIEW")
    out.loc[mask, "final"] = out.loc[mask, "llm_decision"].fillna("SEM_RESPOSTA_LLM")
    return out


def annotate_loss(out: pd.DataFrame, costs: Costs, rates: dict[str, LLMError] | None,
                  human_error: HumanError | None = None) -> pd.DataFrame:
    out = out.copy()
    human_error = human_error or HumanError()
    positive, negative = out.y.eq(1), out.y.eq(0)
    pending = out.final.eq("SEM_RESPOSTA_LLM")
    if pending.any() and rates is None:
        raise ValueError("Band rates required to estimate unanswered LLM decisions")
    out["perda_observada_par"] = (
        (out.final.eq("MATCH") & negative).astype(float) * costs.false_positive
        + (out.final.eq("NONMATCH") & positive).astype(float) * costs.false_negative
        + out.final.isin(["UNSURE", "DEFERRED"]).astype(float) * costs.human_review
        + (out.action.eq("LLM_REVIEW") & ~pending).astype(float) * costs.llm_review
    )
    referred = out.final.isin(["UNSURE", "DEFERRED"])
    error_loss = human_error.error_loss(out.p_cal, costs)
    out["perda_erro_humano_esperada_par"] = np.where(referred, error_loss, 0.0)
    out["perda_esperada_par"] = out.perda_erro_humano_esperada_par
    out["e_fp_revisor"] = 0.0
    out["e_fn_revisor"] = 0.0
    out["u_revisor"] = 0.0
    if rates is not None:
        for band, rate in rates.items():
            mask = out.band.eq(band)
            out.loc[mask, ["e_fp_revisor", "e_fn_revisor", "u_revisor"]] = [rate.e_fp, rate.e_fn, rate.e_unsure]
        if not set(out.band).issubset(rates):
            raise ValueError("Missing reliability band")
        p = out.p_cal
        estimate = (costs.llm_review + (1 - p) * out.e_fp_revisor * costs.false_positive
                    + p * out.e_fn_revisor * costs.false_negative + out.u_revisor * human_error.loss(p, costs))
        out.loc[pending, "perda_esperada_par"] = estimate[pending]
        out.loc[pending, "perda_erro_humano_esperada_par"] = (out.u_revisor * error_loss)[pending]
    out["tipo_perda"] = np.where(pending, "esperada", np.where(out.perda_esperada_par.gt(0), "mista", "observada"))
    return out


def loss_metrics(out: pd.DataFrame, costs: Costs, rates: dict[str, LLMError] | None = None,
                 human_error: HumanError | None = None) -> dict:
    human_error = human_error or HumanError()
    out = annotate_loss(out, costs, rates, human_error)
    positive, negative = out.y.eq(1), out.y.eq(0)
    match, non = out.final.eq("MATCH"), out.final.eq("NONMATCH")
    unsure, deferred = out.final.eq("UNSURE"), out.final.eq("DEFERRED")
    missing = out.final.eq("SEM_RESPOSTA_LLM")
    fp, fn = int((match & negative).sum()), int((non & positive).sum())
    tp, tn = int((match & positive).sum()), int((non & negative).sum())
    u, d, m = int(unsure.sum()), int(deferred.sum()), int(missing.sum())
    unresolved = unsure | deferred | missing
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    ratio = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    # M3 u is marginal, not class-conditional. Automatic expectations approximate
    # class u by marginal u and clip impossible complements; report this limitation.
    pos_pending, neg_pending = missing & positive, missing & negative
    expected_tp = tp + float((1 - out.e_fn_revisor - out.u_revisor).clip(0, 1)[pos_pending].sum())
    expected_tn = tn + float((1 - out.e_fp_revisor - out.u_revisor).clip(0, 1)[neg_pending].sum())
    u_pos = np.minimum(out.u_revisor, 1 - out.e_fn_revisor)
    u_neg = np.minimum(out.u_revisor, 1 - out.e_fp_revisor)
    expected_tp_human = (expected_tp + int(((unsure | deferred) & positive).sum()) * (1 - human_error.e_fn)
                         + float(u_pos[pos_pending].sum()) * (1 - human_error.e_fn))
    expected_tn_human = (expected_tn + int(((unsure | deferred) & negative).sum()) * (1 - human_error.e_fp)
                         + float(u_neg[neg_pending].sum()) * (1 - human_error.e_fp))
    observed_loss = float(out.perda_observada_par.sum())
    expected_loss = float(out.perda_esperada_par.sum())
    api_cost = int(out.action.eq("LLM_REVIEW").sum()) * costs.llm_review
    return {
        "FP": fp, "FN": fn, "TP": tp, "TN": tn, "UNSURE": u,
        "DEFERRED": d, "DEFERRED_motor": d, "sem_resposta_LLM": m,
        "esperada": m > 0 or expected_loss > 0, "tipo_perda": "mista" if m or expected_loss > 0 else "observada",
        "perda_observada": observed_loss, "perda_esperada": expected_loss,
        "perda": observed_loss + expected_loss, "custo_llm": api_cost,
        "perda_sem_custo_llm": observed_loss + expected_loss - api_cost,
        "perda_erro_humano_esperada": float(out.perda_erro_humano_esperada_par.sum()),
        "humanos_encaminhados_observados": u + d,
        "FP_esperados_pendentes": float(((1 - out.p_cal) * out.e_fp_revisor)[missing].sum()) if rates is not None else 0.0,
        "FN_esperados_pendentes": float((out.p_cal * out.e_fn_revisor)[missing].sum()) if rates is not None else 0.0,
        "UNSURE_esperados_pendentes": float(out.u_revisor[missing].sum()),
        "sens_observada": ratio(tp, n_pos), "sens_esperada": ratio(expected_tp, n_pos),
        "sens_esperada_com_humano": ratio(expected_tp_human, n_pos),
        "espec_observada": ratio(tn, n_neg), "espec_esperada": ratio(expected_tn, n_neg),
        "espec_esperada_com_humano": ratio(expected_tn_human, n_neg),
        "sens_inferior": ratio(tp, n_pos),
        "sens_superior": ratio(tp + int((unresolved & positive).sum()), n_pos),
        "espec_inferior": ratio(tn, n_neg),
        "espec_superior": ratio(tn + int((unresolved & negative).sum()), n_neg),
    }


def summarize(out: pd.DataFrame, mode: str, version: str) -> list[dict]:
    rows = []
    for band, b in [("TODAS", out), *[(fx, out[out.faixa.eq(fx)]) for fx in ("<5", "[5,9)", ">=9")]]:
        rows.append({"modo": mode, "versao": version, "faixa": band,
                     "etapa": "pares_revisados", "decisao_motor": "TODAS", "n": len(b), "vinc": int(b.y.sum())})
        for action, cell in b.groupby("decisao_motor", sort=True):
            rows.append({"modo": mode, "versao": version, "faixa": band,
                         "etapa": "decisao_motor", "decisao_motor": action, "n": len(cell), "vinc": int(cell.y.sum())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "gzcmd/gzcmd_v3_config.yaml")
    parser.add_argument("--reuse-oof", action="store_true")
    parser.add_argument("--policy-kind", choices=["cost", "np"], default=None)
    parser.add_argument("--rates-yaml", type=Path, default=None)
    parser.add_argument("--comparison-rates-yaml", type=Path, default=ROOT / "gzcmd/error_rates_by_band_estimadas.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rates_path = args.rates_yaml or Path(cfg.llm_review.estimated_rates_path)
    rates_model = reference_rates_model(rates_path)
    if rates_model == "claude-haiku-5-5":
        from gzcmd.artifacts import preserve_m3v3_artifacts
        archive = preserve_m3v3_artifacts(ROOT)
        print("ARQUIVOS M3V3 PRESERVADOS", len(archive["files"]))
    cfg = replace(cfg, llm_review=replace(cfg.llm_review, estimated_rates_path=str(rates_path.resolve())))
    kind = args.policy_kind or cfg.decision_policy.modes["confirmacao"].policy_kind
    if kind == "np":
        from scripts.fig_varredura_custos import load_candidates, run_np_analysis
        frame = load_candidates(args.input_csv, ROOT / "data/pcal_ml_oof.csv", cfg)
        run_np_analysis(frame, cfg, rates_path)
        rates = load_band_rates(cfg.llm_review.error_rates_by_band, mode="confirmacao", source=rates_path)
        row, out = evaluate_np_point(frame, cfg, rates, alpha=cfg.decision_policy.modes["confirmacao"].np_policy.max_fp_rate_auto)
        pd.DataFrame([row]).to_csv(ROOT / "data/funil_gzcmd_np_status.csv", index=False)
        if out is not None:
            out["guardrail_reason"] = np.where(out.guardrail.eq("ALWAYS_MATCH").fillna(False), "nota_high_probability_consistent", "none")
            safe_funnel(out).to_csv(ROOT / "data/funil_gzcmd_np_confirmacao.csv")
        else:
            stale = ROOT / "data/funil_gzcmd_np_confirmacao.csv"
            if stale.exists():
                stale.unlink()  # owned generated artifact; never expose a stale feasible run
            print("NP CONFIGURADO INVIAVEL: nenhum funil de decisoes foi emitido; consulte funil_gzcmd_np_status.csv")
        return
    summaries, losses, comparisons, references, report = [], [], [], [], []
    destination = ROOT / "data"
    report.extend([
        "# Reavaliação offline GZ-CMD — M1/M2/M6 bayesiano", "",
        "M1 preservado: PAR∈{1,2}; 5 folds externos × 5 internos, seed 42; isotônica somente no treino OOF interno.",
        "Guarda A-: nota<5. Guarda A+ relaxada: nota≥9 e p_cal_ml≥0,9, sem exigir nascimento/município exatos.",
        "Fora das guardas: argmin de (1-p)·C_fp, p·C_fn, C_llm+(1-p)·e_fp·C_fp+p·e_fn·C_fn+u·L_h, L_h.",
        "L_h=C_humano+p·e_fn_h·C_fn+(1-p)·e_fp_h·C_fp. Default e_fp_h=e_fn_h=0,05: hipótese de cenário, sem concordância inter-humana medida na tese.",
        "Unidade: revisões humanas equivalentes. C_humano=1; C_llm=0,005 (5 s / 20 min = 1/240 ≈0,00417, arredondado); C_fp/C_fn=10/50 e 100/20.",
        "LLM somente com ganho positivo sobre a melhor alternativa não LLM; orçamento alocado por ganho decrescente. Excesso executa a melhor alternativa não LLM.",
        "Orçamento humano limita encaminhamentos diretos, default ilimitado; UNSURE LLM é carga adicional. Ganho humano usa a melhor alternativa disponível; excesso vai a LLM com capacidade/ganho, senão AUTO_*.",
        "AUTO_MATCH/AUTO_NONMATCH são regras explícitas de argmin, não guardas. DEFERRED ocorre quando humano minimiza a perda e há orçamento.",
        f"Bloco de referência M3 por modo: {rates_model}; fonte={rates_path}; SHA256={hashlib.sha256(rates_path.read_bytes()).hexdigest()}.",
        "Com taxas v4, somente Claude Haiku 5.5, dossiê sem model_outputs e prompt v4, alimenta o motor. O pooled com DeepSeek V4.1 Flash não é usado.",
        "Para n<10 usa taxas fixas (u fixo default 0); denominador zero conserva fallback fixo do mapping M3.",
        "Perda por modelo: classificações conhecidas e tempo humano têm perda observada; erro humano e LLM pendente têm perda esperada (não há gabarito de erro humano).",
        "C_llm é somado para todas as chamadas, inclusive observadas; perda_sem_custo_llm permite comparar a definição residual M6 anterior.",
        "Pendente não é DEFERRED: colunas tipo_perda/esperada e perda_observada/perda_esperada discriminam a estimativa.",
        "Sensibilidade/especificidade observadas contam só decisões conhecidas; esperadas completam LLM pendentes pelas taxas condicionais de erro.",
        "M3 só fornece u marginal: aproximado por classe e limitado a 1-e_fn/1-e_fp. Colunas *_com_humano incluem o erro humano configurado, não um humano perfeito.",
        "As taxas M3 foram estimadas no lote antigo, não OOF, e não garantem generalização ao lote novo. Expectativas NÃO são resultados observados.",
        "Revisor de referência fixado em Claude Haiku 5.5 v4 sem MO. Comparadores históricos não substituem essa referência por terem menor perda aparente.",
        "O extrato inteiro (61.696 pares) é o universo; 1.410/435 eram lotes LLM. Validação OOF por par, não por pessoa.",
        "K2.5/confirmação: usa explicitamente data/qw2/qw2_confirmacao_kimi.json se o original estiver ausente (protocolo pode diferir; sem inferência causal).",
        "Nenhuma chamada LLM, PII ou alteração LaTeX. O perfil v3 + Platt preserva o resultado histórico.", "",
    ])
    for mode in ("vigilancia", "confirmacao"):
        c = cfg.decision_policy.modes[mode]
        costs = Costs(c.false_positive_cost, c.false_negative_cost, c.llm_review_cost, c.human_review_cost)
        old_costs = replace(costs, llm_review=.25)
        human_error = HumanError(c.human_e_fp, c.human_e_fn)
        old_df, old_summary = run_v3(input_csv=args.input_csv, config_path=args.config,
                                     mode=mode, p_cal="platt", policy_version="v3")
        new_df, new_summary = run_v3(input_csv=args.input_csv, config_path=args.config,
                                     mode=mode, p_cal="ml_xgb", policy_version="v3.1",
                                     ml_oof_path=destination / "pcal_ml_oof.csv",
                                     reuse_ml_oof=args.reuse_oof or mode == "confirmacao",
                                     policy_kind="cost", rates_yaml=rates_path)
        old, new = safe_funnel(old_df), safe_funnel(new_df)
        old_batch = set(old.index[old.action.eq("LLM_REVIEW")])
        new_batch = set(new.index[new.action.eq("LLM_REVIEW")])
        assert not (new.grey_zone & new.guardrail.eq("ALWAYS_NONMATCH")).any()
        assert not (new.guardrail.eq("ALWAYS_MATCH") & new.action.eq("LLM_REVIEW")).any()
        assert new_batch <= set(new.index[new.review_area]) <= set(new.index)
        new_ids = new.loc[new.index.isin(new_batch - old_batch), ["nota", "faixa", "y", "p_cal", "evr", "high_consistency_failed"]]
        new_ids.to_csv(destination / f"lote_llm_novos_{mode}.csv")
        summaries.extend(summarize(old, mode, "v3"))
        summaries.extend(summarize(new, mode, "v3.1"))
        loaded = {model: load_responses(mode, model) for model in MODELS}
        completed_reference, _ = load_responses(mode, HAIKU_V4, include_v31=True)
        if not new_batch.issubset(completed_reference):
            raise ValueError(f"Completed Haiku v4 cohorts do not cover the selected v3.1 batch: {mode}")
        rate_sources = {model: rates_path if model == HAIKU_V4 else args.comparison_rates_yaml for model in MODELS}
        rates_by_model = {model: load_band_rates(cfg.llm_review.error_rates_by_band, mode=mode,
                                               source=rate_sources[model], model=M3_MODELS[model]) for model in MODELS}
        for model in MODELS:
            if not loaded[model][1] or set(loaded[model][0]) != old_batch:
                raise ValueError(f"Historical cohort must have complete identical coverage: {mode}/{model}")
        batch_counts = export_v31_batch(new, completed_reference, destination, mode)
        report.extend(["## Lote v3.1", str(batch_counts)])
        available_models = [model for model in MODELS if loaded[model][1]]
        common = set.intersection(*[set(loaded[model][0]) & old_batch for model in available_models])
        for model in MODELS:
            responses, available = loaded[model]
            coverage = len(set(responses) & old_batch)
            for version, frame in (("v3", old), ("v3.1", new)):
                version_costs = old_costs if version == "v3" else costs
                version_human = HumanError(0, 0) if version == "v3" else human_error
                row = {"modo": mode, "modelo": model, "versao": version, "C_fp": version_costs.false_positive,
                       "C_fn": version_costs.false_negative, "C_llm": version_costs.llm_review, "C_humano": version_costs.human_review,
                       "e_fp_h": version_human.e_fp, "e_fn_h": version_human.e_fn,
                       "orcamento_humano": c.human_max_pairs_per_window,
                       "disponivel": available, "respostas_lote_antigo": coverage, "lote_antigo": len(old_batch),
                       "fonte_alternativa": any("qw2" in p.parts for p in artifact_paths(mode, model)),
                       "revisor_referencia": model == HAIKU_V4,
                       "comparador_com_mo": model != HAIKU_V4,
                       "rates_yaml": str(rate_sources[model]), "rates_model": M3_MODELS[model],
                       "rates_sha256": hashlib.sha256(rate_sources[model].read_bytes()).hexdigest(),
                       "cobertura_completa_antigo": available and coverage == len(old_batch)}
                if available:
                    active = completed_reference if model == HAIKU_V4 and version == "v3.1" else responses
                    replayed, observed = metrics_with_projection(frame, active, responses, old_batch,
                                                                  version_costs, rates_by_model[model], version_human)
                    row["respostas_lote_atual"] = int(replayed.llm_decision.notna().sum())
                    row.update(observed)
                    cohort = common & set(frame.index[frame.action.eq("LLM_REVIEW")])
                    row["n_comum"] = len(cohort)
                    row["perda_coorte_comum"] = loss_metrics(replayed.loc[replayed.index.isin(cohort)], version_costs, rates_by_model[model], version_human)["perda"]
                losses.append(row)
        candidates = [r for r in losses if r["modo"] == mode and r["versao"] == "v3.1" and r["cobertura_completa_antigo"]]
        best = next(r for r in candidates if r["modelo"] == HAIKU_V4)
        best_model = HAIKU_V4
        responses, rates = loaded[best_model][0], rates_by_model[best_model]
        for version, frame in (("v3", old), ("v3.1", new)):
            version_costs = old_costs if version == "v3" else costs
            version_human = HumanError(0, 0) if version == "v3" else human_error
            active = completed_reference if version == "v3.1" else responses
            replayed, observed = metrics_with_projection(frame, active, responses, old_batch,
                                                         version_costs, rates, version_human)
            replayed = annotate_loss(replayed, version_costs, rates, version_human)
            path = f"funil_gzcmd_{mode}.csv" if version == "v3.1" else f"funil_gzcmd_antes_{mode}.csv"
            replayed.to_csv(destination / path)
            grey_metric = loss_metrics(replayed.loc[replayed.grey_zone], version_costs, rates, version_human)
            comparisons.append({"modo": mode, "versao": version, "modelo": best_model,
                                "pares": len(frame), "vinc": int(frame.y.sum()),
                                "LLM": int(frame.action.eq("LLM_REVIEW").sum()),
                                "AUTO_MATCH": int(frame.decisao_motor.eq("AUTO_MATCH").sum()),
                                "AUTO_MATCH_FP": int((frame.decisao_motor.eq("AUTO_MATCH") & frame.y.eq(0)).sum()),
                                "AUTO_NONMATCH_vinc_cinza": int(frame.loc[frame.grey_zone & frame.action.eq("NONMATCH"), "y"].sum()),
                                "LLM_excecao_ge9": int((frame.nota.ge(9) & frame.action.eq("LLM_REVIEW")).sum()),
                                "cinza_sens_observada": grey_metric["sens_observada"],
                                "cinza_sens_esperada": grey_metric["sens_esperada"],
                                **observed})
        original_model = HAIKU_V4
        original = replay(old, loaded[original_model][0], old_batch)
        reference = loss_metrics(original.loc[original.grey_zone], old_costs, rates_by_model[original_model], HumanError(0, 0))
        references.append({"modo": mode, "modelo": original_model, "cinza_vinc": int(old.loc[old.grey_zone, "y"].sum()), **reference})
        lost = old.grey_zone & old.y.eq(1) & old.action.eq("NONMATCH")
        recovered = new.loc[lost]
        guards = new.guardrail.eq("ALWAYS_MATCH")
        grey = new.loc[new.grey_zone]
        lost_new = int(grey.loc[grey.action.eq("NONMATCH"), "y"].sum())
        auto_fp = int((grey.action.eq("MATCH") & grey.y.eq(0)).sum())
        high = new.loc[new.nota.ge(9)].groupby("decisao_motor").agg(n=("y", "size"), vinc=("y", "sum"))
        sanity = lost_new <= 15 and len(new_batch) <= 1500 and auto_fp <= 10
        lines = [f"## {mode}",
                 f"ANTES {old_summary.actions} -> DEPOIS {new_summary.actions}",
                 f"Vínculos cinzentos antes descartados={int(lost.sum())}; agora={recovered.decisao_motor.value_counts().to_dict()}.",
                 f"Guarda ≥9: {int(guards.sum())} pares, FP={int((guards & new.y.eq(0)).sum())}.",
                 f"Novos pair_ids no lote LLM={len(new_ids)}, vínculos={int(new_ids.y.sum())}.",
                 f"SANIDADE: perdidos cinza={lost_new} (≤15), LLM={len(new_batch)} (≤1500), FP AUTO_MATCH cinza={auto_fp} (≤10): {sanity}.",
                 f"Revisor de referência={best_model}, perda mista={best['perda']:.6f}; comparadores históricos identificados separadamente.",
                 "Decisões ≥9:", "```", high.to_string(), "```", ""]
        report.extend(lines)
        if mode == "confirmacao":
            policy_comparison = compare_confirmation(new_df, cfg, rates_path)
            birth_variant = evaluate_birth_variant(new_df)
            policy_comparison.to_csv(destination / "comparacao_np_confirmacao.csv", index=False)
            birth_variant.to_csv(destination / "variante_guarda_confirmacao.csv", index=False)
            report.extend(["## Custo versus NP — confirmação", "```", policy_comparison.to_string(index=False), "```",
                           "Guardas integram o limite. NA indica inviabilidade, não desempenho perfeito ou zero FP.",
                           "## Variante de nascimento", "```", birth_variant.to_string(index=False), "```",
                           "diff_ano>0 com ambos os anos conhecidos é conflito forte. p≥0,95 é subconjunto da guarda p≥0,9; recupera zero dos 17, logo não foi adotada."])
            print("\nCOMPARACAO NP\n", policy_comparison.to_string(index=False))
            print("\nVARIANTE GUARDA\n", birth_variant.to_string(index=False))
        print("\n" + "\n".join(lines))
        print("FUNIL ANTES/DEPOIS\n", pd.DataFrame([r for r in summaries if r["modo"] == mode]).to_string(index=False))
    loss_table = pd.DataFrame(losses)
    loss_table["selecionado"] = False
    loss_table["menor_perda_comparadores"] = False
    for mode in ("vigilancia", "confirmacao"):
        eligible = loss_table.modo.eq(mode) & loss_table.versao.eq("v3.1") & loss_table.cobertura_completa_antigo
        loss_table.loc[eligible & loss_table.revisor_referencia, "selecionado"] = True
        loss_table.loc[loss_table.loc[eligible, "perda"].idxmin(), "menor_perda_comparadores"] = True
    comparison_table, reference_table = pd.DataFrame(comparisons), pd.DataFrame(references)
    summary_table = pd.DataFrame(summaries)
    for filename, table in (("funil_gzcmd_resumo.csv", summary_table), ("perda_total_por_modelo.csv", loss_table),
                            ("funil_gzcmd_metricas.csv", comparison_table), ("funil_gzcmd_referencia.csv", reference_table)):
        table.to_csv(destination / filename, index=False)
    for title, table in (("FUNIL", summary_table), ("METRICAS", comparison_table), ("PERDAS", loss_table), ("REFERENCIA V3 COM HAIKU V4 NA AREA CINZA", reference_table)):
        report.extend([f"## {title}", "```", table.to_string(index=False), "```", ""])
        if title != "FUNIL":
            print(f"\n{title}\n", table.to_string(index=False))
    (destination / "funil_gzcmd_relatorio.md").write_text("\n".join(report), encoding="utf-8")
    from gzcmd.final_funnel import write_final_funnel
    final_table, final_metrics, cost_table = write_final_funnel(ROOT)
    final_metrics.to_csv(destination / "funil_gzcmd_final_metricas.csv", index=False)
    cost_table.to_csv(destination / "funil_gzcmd_custos_usd.csv", index=False)
    with (destination / "funil_gzcmd_relatorio.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Cascata final com cobertura LLM integral\n")
        handle.write("Todas as respostas Haiku v4 do lote v3.1 foram observadas. UNSURE e DEFERRED são encaminhamentos humanos, não erros humanos observados.\n")
        handle.write("A matriz da cadeia avalia aceitação automática: MATCH é positivo; os demais não foram aceitos automaticamente. As decisões NONMATCH incorretas e os encaminhamentos são discriminados.\n")
        handle.write("A especificidade da aceitação automática é 1-FP/não vínculos. A coluna anterior espec_observada mantém o limite conservador de decisões negativas já resolvidas.\n")
        handle.write("A hipótese de erro humano de 5% permanece somente no cenário esperado e no custo relativo. Não há decisões do revisor humano observadas.\n")
        handle.write("O custo US$ usa tokens registrados por par e tarifas históricas dos runs; é estimativa tarifária, não fatura. Inclui chamadas e tentativas do journal dos pares selecionados.\n")
        for title, table in (("Cascata final", final_table), ("Métricas finais observadas", final_metrics), ("Custo monetário", cost_table)):
            handle.write(f"\n### {title}\n```\n{table.to_string(index=False)}\n```\n")
    print("\nMETRICAS FINAIS OBSERVADAS\n", final_metrics.to_string(index=False))
    print("\nCUSTOS USD POR TOKENS OBSERVADOS\n", cost_table.to_string(index=False))


if __name__ == "__main__":
    main()
