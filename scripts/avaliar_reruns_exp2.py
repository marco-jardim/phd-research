"""Avaliação agregada dos reruns Exp2, inclusive checkpoints, sem chamadas a LLM.

Uso: python scripts/avaliar_reruns_exp2.py [--only-complete]
O filtro afeta somente a tabela LaTeX. CSVs e taxas v4 incluem os parciais.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import binomtest

try:
    from . import estimar_taxas_erro_revisor as rates
except ImportError:  # Execução direta, sem instalação de scripts como pacote.
    import estimar_taxas_erro_revisor as rates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gzcmd.provenance import sha256_file_normalized, sha256_normalized  # noqa: E402
# Manifesto explícito: não varrer backups, traces, resumes ou parciais obsoletos.
SOURCES = (
    ("kimi-k2.5", "completo", "v3", "data/sprint3b"),
    ("deepseek-v4.1-flash", "completo", "v3", "data/sprint3b/opencode_go"),
    ("claude-haiku-5-5", "completo", "v3", "data/sprint3b/anthropic"),
    ("kimi-k3", "completo", "v3", "data/sprint3b/opencode_go_kimi"),
    ("levanto-sage-2", "completo", "v3", "data/sprint3b/sage_com_mo"),
    ("claude-haiku-5-5", "sem_mo", "v3", "data/sprint3b/anthropic_sem_mo"),
    ("deepseek-v4.1-flash", "sem_mo", "v3", "data/sprint3b/opencode_go_sem_mo"),
    ("levanto-sage-2", "sem_mo", "v3", "data/sprint3b/sage_sem_mo"),
    ("claude-haiku-5-5", "sem_mo", "v4", "data/sprint3b/anthropic_sem_mo_v4"),
    ("deepseek-v4.1-flash", "sem_mo", "v4", "data/sprint3b/opencode_go_sem_mo_v4"),
)
MODEL_NAMES = {
    "kimi-k2.5": "Kimi K2.5", "kimi-k3": "Kimi K3",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "claude-haiku-5-5": "Claude Haiku 5.5", "levanto-sage-2": "Levanto Sage 2",
}
KEYS = ["mode", "model", "dossie", "prompt"]
CONDITIONS = (("completo", "v3"), ("sem_mo", "v3"), ("sem_mo", "v4"))


def parse_run(run: dict, expected_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Ausência de resposta não é UNSURE; validar o denominador da coorte."""
    frame = pd.DataFrame([
        {"pair_id": row["pair_id"], "decision": rates.normalize_decision(row["decision"])}
        for row in run["results"]
    ], columns=["pair_id", "decision"])
    if not frame.pair_id.is_unique:
        raise ValueError("pair_id duplicado no run")
    if not set(frame.pair_id).issubset(expected_ids):
        raise ValueError("Run contém pares fora da coorte original do modo")
    expected = len(expected_ids)
    if "n_review_pairs" in run and int(run["n_review_pairs"]) != expected:
        raise ValueError("n_review_pairs diverge da coorte original")
    selected = run.get("selected_pair_ids")
    if selected is not None and (len(selected) != expected or set(selected) != expected_ids):
        raise ValueError("selected_pair_ids diverge da coorte original")
    if "n_reviewed" in run and int(run["n_reviewed"]) != len(frame):
        raise ValueError("n_reviewed diverge das respostas persistidas")
    complete = len(frame) == expected
    if run.get("complete") is False or run.get("status") in {"running", "partial", "in_progress", "checkpoint"}:
        complete = False
    return frame, {"n_respondido": len(frame), "n_esperado": expected,
                   "n_sem_resposta": expected - len(frame), "completo": complete}


def load_runs(root: Path, gold: pd.DataFrame, definitions: dict) -> tuple[pd.DataFrame, list[dict]]:
    frames, sources, cohorts = [], [], {}
    for model, dossier, prompt, folder in SOURCES:
        for mode in rates.MODES:
            path = root / folder / f"exp2_llm_sync_{mode}.json"
            alternate = False
            if not path.exists() and model == "kimi-k2.5" and mode == "confirmacao":
                path = root / "data/qw2/qw2_confirmacao_kimi.json"
                alternate = True
            if not path.exists():
                # Há condições deliberadamente ausentes, como Sage completo na confirmação.
                print(f"AUSENTE {mode} {model} {dossier}/{prompt}")
                continue
            snapshot = path.read_bytes()
            run = json.loads(snapshot.decode("utf-8"))
            if rates.normalize_mode(run.get("mode")) != mode:
                raise ValueError(f"Modo inconsistente: {path}")
            if run.get("prompt_version", prompt) != prompt:
                raise ValueError(f"Prompt inconsistente: {path}")
            reported_model = run.get("model")
            accepted_models = {model}
            if model == "kimi-k2.5":
                accepted_models.add("accounts/fireworks/models/kimi-k2p5")
            if reported_model is not None and reported_model not in accepted_models:
                raise ValueError(f"Modelo inconsistente: {path}")
            if model == "kimi-k2.5":
                ids = {row["pair_id"] for row in run["results"]}
                if not ids or len(ids) != int(run.get("n_review_pairs", -1)):
                    raise ValueError("A fonte da coorte original deve estar completa")
                cohorts[mode] = ids
            if mode not in cohorts:
                raise ValueError(f"Falta coorte original para {mode}")
            frame, counts = parse_run(run, cohorts[mode])
            merged = frame.merge(gold, on="pair_id", how="left", validate="one_to_one")
            if merged.y.isna().any() or merged.nota_final.isna().any():
                raise ValueError(f"Pares sem gabarito/nota: {path}")
            merged["band"] = rates.assign_bands(merged.nota_final, definitions)
            if merged.band.isna().any():
                raise ValueError(f"Notas fora das bandas do YAML: {path}")
            identity = dict(zip(KEYS, (mode, model, dossier, prompt)))
            for key, value in {**identity, **counts}.items():
                merged[key] = value
            frames.append(merged)
            sources.append({**identity, **counts, "path": path.relative_to(root).as_posix(),
                            "sha256": sha256_normalized(snapshot),
                            "reported_model": reported_model, "alternate_source": alternate})
    if not frames:
        raise ValueError("Nenhum run disponível")
    return pd.concat(frames, ignore_index=True), sources


def metrics(part: pd.DataFrame) -> dict:
    d, y = part.decision, part.y
    vp = int((d.eq("MATCH") & y.eq(1)).sum())
    fp = int((d.eq("MATCH") & y.eq(0)).sum())
    fn = int((d.eq("NONMATCH") & y.eq(1)).sum())
    vn = int((d.eq("NONMATCH") & y.eq(0)).sum())
    u1 = int((d.eq("UNSURE") & y.eq(1)).sum())
    u0 = int((d.eq("UNSURE") & y.eq(0)).sum())
    row = {"n": len(part), "VP": vp, "FP": fp, "FN": fn, "VN": vn,
           "UNSURE": u1 + u0, "UNSURE_pos": u1, "UNSURE_neg": u0,
           "FN_unsure_nonmatch": fn + u1, "VN_unsure_nonmatch": vn + u0}
    for name, k, n in (
        ("sens", vp, vp + fn), ("espec", vn, vn + fp),
        ("sens_unsure_nonmatch", vp, vp + fn + u1),
        ("espec_unsure_nonmatch", vn + u0, vn + fp + u0),
        ("sens_incl_unsure", vp + u1, vp + fn + u1),
        ("fracao_unsure", u1 + u0, len(part)),
        ("acerto_estrito", vp + vn, len(part)),
    ):
        value, lo, hi = rates.wilson(k, n)
        row.update({name: value, f"{name}_ci95_low": lo, f"{name}_ci95_high": hi,
                    f"{name}_denominador": n})
    return row


def load_supplemental_runs(root: Path, gold: pd.DataFrame, definitions: dict):
    """Disjoint Haiku v4 supplements; never treat them as paired historical arms."""
    frames, sources = [], []
    for folder, cohort in (("anthropic_sem_mo_v4_lote31", "lote31_row"),
                           ("anthropic_sem_mo_v4_lote31_refrec", "lote31_refrec")):
        for mode in rates.MODES:
            path = root / "data/sprint3b" / folder / f"exp2_llm_sync_{mode}.json"
            if not path.exists():
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            original_path = root / f"data/sprint3b/anthropic_sem_mo_v4/exp2_llm_sync_{mode}.json"
            original = json.loads(original_path.read_text(encoding="utf-8"))
            removed = {"model_outputs." + key for key in ("p_cal", "band", "base_choice", "evr", "guardrail", "guardrail_reason")}
            if (run.get("mode") != mode or run.get("model") != "claude-haiku-5-5"
                    or run.get("prompt_version") != "v4" or run.get("cohort") != "v31_lot"
                    or run.get("prompt_sha256") != original.get("prompt_sha256")
                    or not run.get("complete") or not removed.issubset(run.get("removed_fields", []))):
                raise ValueError(f"Incompatible supplemental provenance: {path}")
            ids = set(run["selected_pair_ids"])
            part, counts = parse_run(run, ids)
            if not counts["completo"]:
                raise ValueError(f"Incomplete supplemental cohort: {path}")
            merged = part.merge(gold, on="pair_id", how="left", validate="one_to_one")
            if merged.y.isna().any() or merged.nota_final.isna().any():
                raise ValueError("Supplement contains an unknown pair hash")
            merged["band"] = rates.assign_bands(merged.nota_final, definitions)
            identity = dict(zip(KEYS, (mode, "claude-haiku-5-5", "sem_mo", "v4")))
            for key, value in {**identity, **counts, "cohort": cohort}.items():
                merged[key] = value
            frames.append(merged)
            sources.append({**identity, **counts, "cohort": cohort, "path": path.relative_to(root).as_posix(),
                            "sha256": sha256_file_normalized(path),
                            "reported_model": "claude-haiku-5-5", "alternate_source": False})
    return frames, sources


def mcnemar_exact(a_correct_b_wrong: int, a_wrong_b_correct: int) -> float:
    if min(a_correct_b_wrong, a_wrong_b_correct) < 0:
        raise ValueError("Discordâncias negativas")
    n = a_correct_b_wrong + a_wrong_b_correct
    return float(binomtest(a_correct_b_wrong, n, 0.5).pvalue) if n else 1.0


def paired_metrics(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    both = a[["pair_id", "y", "decision"]].merge(
        b[["pair_id", "y", "decision"]], on="pair_id", suffixes=("_a", "_b"), validate="one_to_one")
    if not both.y_a.equals(both.y_b):
        raise ValueError("Gabaritos divergentes na comparação pareada")
    da, db, y = both.decision_a, both.decision_b, both.y_a
    ca = (da.eq("MATCH") & y.eq(1)) | (da.eq("NONMATCH") & y.eq(0))
    cb = (db.eq("MATCH") & y.eq(1)) | (db.eq("NONMATCH") & y.eq(0))
    lost, gained = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = len(both)
    chance = sum(int(da.eq(label).sum()) * int(db.eq(label).sum())
                 for label in ("MATCH", "NONMATCH", "UNSURE")) / n**2 if n else None
    agreement = float(da.eq(db).mean()) if n else None
    kappa = (agreement - chance) / (1 - chance) if n and chance != 1 else None
    return {"n_comum": n, "n_so_a": len(a) - n, "n_so_b": len(b) - n,
            "FN_revertidos": int((y.eq(1) & da.eq("NONMATCH") & db.eq("MATCH")).sum()),
            "FN_para_UNSURE": int((y.eq(1) & da.eq("NONMATCH") & db.eq("UNSURE")).sum()),
            "FP_novos": int((y.eq(0) & ~da.eq("MATCH") & db.eq("MATCH")).sum()),
            "UNSURE_novos": int((~da.eq("UNSURE") & db.eq("UNSURE")).sum()),
            "acerto_a_erro_b": lost, "erro_a_acerto_b": gained,
            "mcnemar_p_exato": mcnemar_exact(lost, gained) if n else None,
            "concordancia": agreement, "kappa": kappa}


def summarize(data: pd.DataFrame, sources: list[dict], definitions: dict) -> tuple[list[dict], list[dict], list[dict]]:
    summary, bands, paired = [], [], []
    for source in sources:
        mask = pd.Series(True, index=data.index)
        for key in KEYS:
            mask &= data[key].eq(source[key])
        part = data[mask]
        summary.append({**source, **metrics(part)})
        for band in definitions:
            bands.append({**source, "estrato": "banda_yaml", "band": band,
                          **metrics(part[part.band.eq(band)])})
        for name, selection in (("s<9", part.nota_final.lt(9)), ("s>=9", part.nota_final.ge(9))):
            bands.append({**source, "estrato": "limiar_9", "band": name, **metrics(part[selection])})
    for (mode, model), part in data.groupby(["mode", "model"], sort=True):
        for (d_a, p_a), (d_b, p_b) in combinations(CONDITIONS, 2):
            a = part[part.dossie.eq(d_a) & part.prompt.eq(p_a)]
            b = part[part.dossie.eq(d_b) & part.prompt.eq(p_b)]
            if a.empty or b.empty:
                continue
            paired.append({"mode": mode, "model": model, "dossie_a": d_a, "prompt_a": p_a,
                           "dossie_b": d_b, "prompt_b": p_b, "completo_a": bool(a.completo.all()),
                           "completo_b": bool(b.completo.all()), **paired_metrics(a, b)})
    return summary, paired, bands


def make_v4_yaml(data: pd.DataFrame, sources: list[dict], definitions: dict, fixed: dict,
                 config_bytes: bytes) -> dict:
    """Reutiliza as taxas condicionais e fallbacks do estimador, incluindo checkpoints v4."""
    models = {"claude-haiku-5-5", "deepseek-v4.1-flash"}
    v4_sources = [s for s in sources if s["prompt"] == "v4" and s["model"] in models and s["dossie"] == "sem_mo"]
    v4 = data[data.prompt.eq("v4") & data.model.isin(models) & data.dossie.eq("sem_mo")]

    def profile(part: pd.DataFrame, selected: list[dict]) -> dict:
        partial = not selected or any(not s["completo"] for s in selected)
        rows = [rates.summarize(part[part.band.eq(band)], "all", "pooled", band, partial, fixed)
                for band in definitions]
        return rates.engine_profile(rows, fixed, partial)

    overall = profile(v4, v4_sources)
    by_mode = {}
    for mode in rates.MODES:
        subset = v4[v4["mode"].eq(mode)]
        selected = [s for s in v4_sources if s["mode"] == mode]
        pooled = profile(subset, selected)
        pooled["partial"] = pooled["partial"] or {s["model"] for s in selected} != models
        by_mode[mode] = {"models": {
            model: profile(subset[subset.model.eq(model)], [s for s in selected if s["model"] == model])
            for model in sorted({s["model"] for s in selected})}, "pooled": pooled}
    overall["partial"] = overall["partial"] or any(p["pooled"]["partial"] for p in by_mode.values())
    return {"schema_version": 1, "source_config": "gzcmd/gzcmd_v3_config.yaml",
            "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "method": "Wilson 95%; FP/negatives; FN/positives; UNSURE/n; explicit fixed fallback for zero denominators",
            "prompt": "v4", "dossie": "sem_mo", "pooled_excludes_partial": False,
            "provisional": overall["partial"],
            "limitations": ["Checkpoints are not a random sample; no extrapolation to unanswered pairs.",
                            "Pooled n counts decisions, not independent pairs; Wilson intervals are descriptive.",
                            "UNSURE is not FP/FN; human referral is not an observed correct decision."],
            "sources": v4_sources, "bands": definitions,
            "llm_review": {"reliability": {"error_rates_by_band": overall["error_rates_by_band"]}},
            "models": {model: profile(v4[v4.model.eq(model)], [s for s in v4_sources if s["model"] == model])
                       for model in sorted({s["model"] for s in v4_sources})},
            "pooled": overall, "by_mode": by_mode}


def make_latex(rows: list[dict], only_complete: bool = False) -> str:
    from gzcmd.intervals import wilson95

    selected = [r for r in rows if r["completo"] or not only_complete]

    def percent(value: float | None) -> str:
        return "NA" if value is None else f"{100 * value:.1f}".replace(".", ",")

    def interval(row, field):
        if row[field] is None:
            return "NA"
        if field + "_ci95_low" in row and field + "_ci95_high" in row:
            low, high = row[field + "_ci95_low"], row[field + "_ci95_high"]
        else:
            success = row["VP"] if field == "sens" else row["VN"]
            error = row["FN"] if field == "sens" else row["FP"]
            low, high = wilson95(success, success + error)
        return f"{percent(row[field])} ({percent(low)}; {percent(high)})"

    lines = ["% Gerada por scripts/avaliar_reruns_exp2.py; fonte: data/exp2_reruns_resumo.csv.",
             r"\begin{table}[!ht]", r"\centering", r"\footnotesize",
             r"\caption{Avaliação dos revisores no Exp2 segundo o dossiê e a versão do prompt.}",
             r"\label{tab:cap8-reruns}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Modelo & Dossiê/prompt & VP & FP & FN & UNSURE & Sens. (\%) & Espec. (\%) & UNSURE (\%) \\"]
    for mode, name in (("vigilancia", "Vigilância"), ("confirmacao", "Confirmação")):
        lines += [r"\midrule", rf"\multicolumn{{9}}{{l}}{{\textbf{{{name}}}}} \\"]
        for row in selected:
            if row["mode"] != mode:
                continue
            star = r"\textsuperscript{*}" if not row["completo"] else ""
            dossier = "Completo" if row["dossie"] == "completo" else "Sem MO"
            model = MODEL_NAMES[row["model"]]
            counts = " & ".join(f"{row[k]:,}".replace(",", ".") for k in ("VP", "FP", "FN", "UNSURE"))
            values = " & ".join(interval(row, k) for k in ("sens", "espec"))
            values += " & " + percent(row["fracao_unsure"])
            cohort_name = {"lote31_row": "lote por linha", "lote31_refrec": "lote REFREC"}.get(row.get("cohort"))
            suffix = f" ({cohort_name})" if cohort_name else ""
            lines.append(f"{model}{star} & {dossier}/{row['prompt']}{suffix} & {counts} & {values} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\par\smallskip",
              r"\parbox{\linewidth}{\footnotesize VP: verdadeiros positivos; FP: falsos positivos; FN: falsos negativos. "
              r"MO: bloco \texttt{model\_outputs}. Sensibilidade e especificidade excluem \texttt{UNSURE}; "
              r"a fração de \texttt{UNSURE} usa todas as respostas. NA indica denominador zero. "
              r"O encaminhamento ao revisor humano não constitui acerto observado. "
              r"Sensibilidade e especificidade apresentam IC de 95\% de Wilson entre parênteses; análises pareadas constam nos CSVs agregados. "
              r"A confirmação de Kimi K2.5 utiliza a fonte alternativa QW2, sem pressupor equivalência de protocolo. "
              r"As diferenças entre execuções não identificam isoladamente efeito causal do prompt. "
              r"Lotes complementares são apresentados separadamente, com seleção distinta da coorte histórica; não constituem braços pareados adicionais. "
              r"Fonte: artefatos do Exp2 \cite{Jardim2026phdrepo}.}"]
    partials = [r for r in selected if not r["completo"]]
    if partials:
        note = "; ".join(f"{MODEL_NAMES[r['model']]} ({r['mode']}, {r['dossie'].replace('_', ' ')}/{r['prompt']}): "
                         f"{r['n_respondido']}/{r['n_esperado']}" for r in partials)
        lines += [r"\par\smallskip", r"\parbox{\linewidth}{\footnotesize\textsuperscript{*} "
                  + "Execuções parciais, respostas disponíveis/coorte esperada: " + note + ". "
                  + "Pares sem resposta não são classificados como \\texttt{UNSURE}.}"]
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--only-complete", action="store_true", help="Excluir parciais somente da tabela LaTeX")
    args = parser.parse_args()
    root = args.root.resolve()
    config_bytes = (root / "gzcmd/gzcmd_v3_config.yaml").read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    definitions = config["bands"]["definitions"]
    fixed = config["llm_review"]["reliability"]["error_rates_by_band"]
    gold = rates.load_gold(root / "data/COMPARADORSEMIDENT.csv")
    data, sources = load_runs(root, gold, definitions)
    summary, paired, bands = summarize(data, sources, definitions)
    extra_frames, extra_sources = load_supplemental_runs(root, gold, definitions)
    if extra_frames:
        expanded = pd.concat([data, *extra_frames], ignore_index=True)
        haiku = expanded[expanded.model.eq("claude-haiku-5-5") & expanded.prompt.eq("v4")]
        if haiku.duplicated(["mode", "pair_id"]).any():
            raise ValueError("Overlapping Haiku reliability cohorts")
        for frame, source in zip(extra_frames, extra_sources):
            summary.append({**source, **metrics(frame)})
            for band in definitions:
                bands.append({**source, "estrato": "banda_yaml", "band": band, **metrics(frame[frame.band.eq(band)])})
        data, sources = expanded, [*sources, *extra_sources]
    profile = make_v4_yaml(data, sources, definitions, fixed, config_bytes)
    if extra_sources:
        profile["limitations"].append("Haiku includes adaptively selected supplemental cohorts; reliability estimates and policy replay are not independent validation.")
    outputs = {"data/exp2_reruns_resumo.csv": pd.DataFrame(summary),
               "data/exp2_reruns_pareado.csv": pd.DataFrame(paired),
               "data/exp2_reruns_por_banda.csv": pd.DataFrame(bands)}
    for name, frame in outputs.items():
        frame.to_csv(root / name, index=False, encoding="utf-8", float_format="%.10g")
    tex_path = root / "text/latex/tables/tab_cap8_reruns.tex"
    tex_path.write_text(make_latex(summary, args.only_complete), encoding="utf-8")
    yaml_path = root / "gzcmd/error_rates_by_band_v4.yaml"
    yaml_path.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for row in summary:
        print(f"RUN {row['mode']} {row['model']} {row['dossie']}/{row['prompt']}: "
              f"n={row['n_respondido']}/{row['n_esperado']} completo={row['completo']} "
              f"VP={row['VP']} FP={row['FP']} FN={row['FN']} VN={row['VN']} UNSURE={row['UNSURE']}")
    print(f"OK: {len(summary)} condições; {len(paired)} comparações pareadas; {len(bands)} estratos; "
          f"{len(data)} respostas; YAML v4 provisório={profile['provisional']}")
    for path in [*(root / name for name in outputs), tex_path, yaml_path]:
        print("SALVO", path.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
