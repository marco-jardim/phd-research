"""Estima erros do Exp2 por modo/modelo/banda, sem persistir dados por par.

Execute diretamente: python scripts/estimar_taxas_erro_revisor.py
Os artefatos finais são lidos; nenhum LLM é chamado e o config não é alterado.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from pathlib import Path
from statistics import NormalDist

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("kimi-k2.5", "data/sprint3b", False),
    ("deepseek-v4.1-flash", "data/sprint3b/opencode_go", False),
    ("claude-haiku-5-5", "data/sprint3b/anthropic", False),
    ("kimi-k3", "data/sprint3b/opencode_go_kimi", True),
)
MODES = ("vigilancia", "confirmacao")
FALLBACK_BANDS = {
    "grey_low": {"min": 5, "max": 6},
    "grey_mid": {"min": 6, "max": 7},
    "grey_high": {"min": 7, "max": 8},
    "near_high": {"min": 8, "max": 9},
    "high": {"min": 9, "max": math.inf},
}


def wilson(k: int, n: int) -> tuple[float | None, float | None, float | None]:
    """IC binomial Wilson bilateral, sem correção de continuidade; n=0 é NA."""
    if not 0 <= k <= n:
        raise ValueError("Contagens binomiais inválidas")
    if n == 0:
        return None, None, None
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return p, max(0.0, center - radius), min(1.0, center + radius)


def normalize_decision(value: object) -> str:
    # Os comparadores existentes usam as três decisões literais; não converter
    # erros/respostas desconhecidas silenciosamente em NONMATCH ou UNSURE.
    decision = str(value).strip().upper()
    if decision not in {"MATCH", "NONMATCH", "UNSURE"}:
        raise ValueError("Decisão fora de MATCH/NONMATCH/UNSURE")
    return decision


def normalize_mode(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def resolve_column(frame: pd.DataFrame, name: str) -> str:
    return next(c for c in frame.columns if c == name or c.startswith(name + ","))


def load_gold(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", dtype=str, decimal=",", low_memory=False)
    cols = [resolve_column(raw, name) for name in ("COMPREC", "REFREC", "PASSO")]
    ids = [hashlib.sha256(f"{a}|{b}|{c}".encode()).hexdigest()[:16]
           for a, b, c in zip(*(raw[col].str.strip() for col in cols))]
    labels = pd.to_numeric(raw["PAR"], errors="coerce").fillna(0).astype(int)
    scores = pd.to_numeric(raw[resolve_column(raw, "nota final")].str.replace(",", ".", regex=False), errors="coerce")
    gold = pd.DataFrame({"pair_id": ids, "y": labels.isin([1, 2]).astype(int), "nota_final": scores})
    if not gold.pair_id.is_unique:
        raise ValueError("pair_id duplicado no extrato")
    return gold


def assign_bands(scores: pd.Series, definitions: dict) -> pd.Series:
    bands = pd.Series(None, index=scores.index, dtype=object)
    for name, spec in definitions.items():
        upper = scores.le(float(spec["max"])) if spec.get("inclusive_max", False) else scores.lt(float(spec["max"]))
        mask = scores.ge(float(spec["min"])) & upper
        if (mask & bands.notna()).any():
            raise ValueError("Bandas sobrepostas no config")
        bands.loc[mask] = name
    return bands


def load_runs(root: Path, gold: pd.DataFrame, definitions: dict) -> tuple[pd.DataFrame, list[dict]]:
    frames, provenance = [], []
    for model, folder, partial in SOURCES:
        found = 0
        for mode in MODES:
            # Lista exata de arquivos: nunca inclui parcial336, resume ou traces.
            path = root / folder / f"exp2_llm_sync_{mode}.json"
            alternate = False
            if not path.exists() and model == "kimi-k2.5" and mode == "confirmacao":
                # O checkout não contém a confirmação no diretório sprint3b.
                # Usar o artefato completo existente, com proveniência explícita,
                # nunca o run sem modelo de 148 pares em opencode_server.
                path = root / "data/qw2/qw2_confirmacao_kimi.json"
                alternate = True
            if not path.exists():
                if partial:
                    continue
                raise FileNotFoundError(path)
            run = json.loads(path.read_text(encoding="utf-8"))
            if alternate and (run.get("model") != "accounts/fireworks/models/kimi-k2p5" or len(run["results"]) != 435):
                raise ValueError("Confirmação alternativa K2.5 precisa identificar o modelo e conter 435 pares")
            if normalize_mode(run.get("mode")) != mode:
                raise ValueError(f"Modo inconsistente em {path.name}")
            result = pd.DataFrame([
                {"pair_id": r["pair_id"], "decision": normalize_decision(r["decision"])}
                for r in run["results"]
            ])
            if result.empty or not result.pair_id.is_unique:
                raise ValueError(f"Run vazio ou duplicado: {path.name}")
            if model == "kimi-k3" and (mode != "confirmacao" or len(result) != 343):
                raise ValueError("K3: esperado somente artefato final parcial de 343 pares de confirmação")
            merged = result.merge(gold, on="pair_id", how="left", validate="one_to_one")
            if merged.y.isna().any() or merged.nota_final.isna().any():
                raise ValueError(f"Run com pares sem gabarito/nota: {path.name}")
            merged["band"] = assign_bands(merged.nota_final, definitions)
            if merged.band.isna().any():
                raise ValueError(f"Notas fora das bandas: {path.name}")
            merged["mode"], merged["model"], merged["partial"] = mode, model, partial
            frames.append(merged)
            provenance.append({"model": model, "reported_model": run.get("model"), "mode": mode,
                               "path": path.relative_to(root).as_posix(), "n": len(result),
                               "alternate_source": alternate,
                               "partial": partial, "n_expected": 435 if partial else len(result)})
            found += 1
        if not found:
            raise FileNotFoundError(f"Nenhum artefato final para {model}")
    complete = pd.concat(frames, ignore_index=True)
    for mode in MODES:
        cohorts = [set(part.pair_id) for _, part in complete[(complete["mode"] == mode) & ~complete.partial].groupby("model")]
        if not all(cohort == cohorts[0] for cohort in cohorts):
            raise ValueError(f"Coortes dos modelos completos diferem em {mode}; pooling requer revisão")
    return complete, provenance


def summarize(part: pd.DataFrame, mode: str, model: str, band: str, partial: bool, fixed: dict) -> dict:
    n = len(part)
    positives = int(part.y.eq(1).sum())
    negatives = n - positives
    fp = int((part.decision.eq("MATCH") & part.y.eq(0)).sum())
    fn = int((part.decision.eq("NONMATCH") & part.y.eq(1)).sum())
    unsure = int(part.decision.eq("UNSURE").sum())
    row = {"mode": mode, "model": model, "band": band, "partial": partial,
           "n": n, "n_vinculos": positives, "n_nao_vinculos": negatives,
           "fp": fp, "fn": fn, "n_unsure": unsure, "n_lt_10": n < 10,
           "denom_fp_lt_10": negatives < 10, "denom_fn_lt_10": positives < 10}
    for metric, k, denominator in (("e_fp", fp, negatives), ("e_fn", fn, positives), ("u", unsure, n)):
        rate, lower, upper = wilson(k, denominator)
        row.update({metric: rate, f"{metric}_ci95_low": lower, f"{metric}_ci95_high": upper})
        if metric != "u":
            baseline = fixed.get(band, {}).get(metric)
            row[f"{metric}_fixed"] = baseline
            row[f"{metric}_abs_diff"] = abs(rate - baseline) if rate is not None and baseline is not None else None
    return row


def estimate(data: pd.DataFrame, definitions: dict, fixed: dict) -> list[dict]:
    rows = []
    for (mode, model, partial), part in data.groupby(["mode", "model", "partial"], sort=True):
        for band in definitions:
            rows.append(summarize(part[part.band.eq(band)], mode, model, band, bool(partial), fixed))
    # Pooled por modo: mesmas coortes, logo soma das contagens equivale a
    # média das taxas por modelo ponderada por n (inclusive denominadores de classe).
    for mode in MODES:
        part = data[data["mode"].eq(mode) & ~data.partial]
        for band in definitions:
            rows.append(summarize(part[part.band.eq(band)], mode, "pooled", band, False, fixed))
    return rows


def engine_profile(rows: list[dict], fixed: dict, partial: bool = False) -> dict:
    rates, fallback = {}, {}
    for row in rows:
        band = row["band"]
        rates[band] = {}
        for metric in ("e_fp", "e_fn"):
            value = row[metric]
            if value is None:
                value = fixed.get(band, {}).get(metric)
                if value is None:
                    raise ValueError(f"Sem estimativa nem fallback fixo: {band}/{metric}")
                fallback.setdefault(band, []).append(metric)
            rates[band][metric] = value
    return {"partial": partial, "error_rates_by_band": rates, "fallback_fixed": fallback,
            "estimates_by_band": {r["band"]: {k: v for k, v in r.items() if k not in {"mode", "model", "band", "partial"}} for r in rows}}


def contradictions(data: pd.DataFrame, definitions: dict) -> list[dict]:
    output = []
    for mode in MODES:
        part = data[data["mode"].eq(mode)]
        a = part[part.model.eq("kimi-k2.5")]
        b = part[part.model.eq("deepseek-v4.1-flash")]
        common = a.merge(b[["pair_id", "decision"]], on="pair_id", suffixes=("_kimi", "_deepseek"), validate="one_to_one")
        mask = ((common.decision_kimi.eq("MATCH") & common.decision_deepseek.eq("NONMATCH")) |
                (common.decision_kimi.eq("NONMATCH") & common.decision_deepseek.eq("MATCH")))
        conflict = common[mask]
        for band in definitions:
            group = conflict[conflict.band.eq(band)]
            output.append({"mode": mode, "band": band, "n": len(group), "n_vinculos": int(group.y.eq(1).sum()),
                           "kimi_match_deepseek_nonmatch": int(group.decision_kimi.eq("MATCH").sum()),
                           "kimi_nonmatch_deepseek_match": int(group.decision_kimi.eq("NONMATCH").sum())})
    return output


def format_rate(row: dict, metric: str) -> str:
    if row[metric] is None:
        return "NA"
    return f"{row[metric]:.4f} [{row[metric + '_ci95_low']:.4f}, {row[metric + '_ci95_high']:.4f}]"


def make_report(rows: list[dict], sources: list[dict], conflicts: list[dict], fixed: dict) -> str:
    lines = ["# Taxas de erro do revisor LLM — Exp2", "",
             "## Método e limites", "",
             "Gabarito: PAR ∈ {1,2}; pair_id = SHA256(COMPREC|REFREC|PASSO)[:16], campos aparados. "
             "Extrato lido com dtype=str e conversão explícita to_numeric. Decisões aparadas, "
             "maiúsculas e validadas em MATCH/NONMATCH/UNSURE; não se usa confiança do LLM.", "",
             "e_fp = FP/n_não_vínculos; e_fn = FN/n_vínculos; u = UNSURE/n. "
             "UNSURE permanece nos denominadores de classe, não é contado como FP/FN. "
             "IC 95% Wilson bilateral (z=1,9599639845), sem correção. NA significa denominador zero, não erro zero.", "",
             "Pooled inclui apenas K2.5, DeepSeek e Haiku completos; K3 é reportado à parte. "
             "As coortes completas são verificadas idênticas por modo: pooling de contagens "
             "equivale à média das taxas de modelo ponderada por n por banda. Não há deduplicação "
             "entre revisores: n mede decisões, não pares independentes. IC pooled é descritivo, "
             "pois revisões do mesmo par são correlacionadas. Os modos também podem sobrepor pares.", "",
             "K3: seleção parcial por interrupção de quota, 343/435 de confirmação; não extrapolar "
             "para os 92 não revisados. Ignorados *.parcial336.json e resume_sem_erros.json. "
             "Haiku é estocástico (sem temperature); uma execução não estima variabilidade entre execuções. "
             "Amostra condicionada à triagem Exp2: não generalizar taxas para todos os pares.", "",
             "**Divergência de localização:** a confirmação K2.5 solicitada em "
             "`data/sprint3b/exp2_llm_sync_confirmacao.json` não existe neste checkout. "
             "Quando ausente, o script usa explicitamente `data/qw2/qw2_confirmacao_kimi.json` "
             "(435 pares, modelo accounts/fireworks/models/kimi-k2p5), valida a mesma coorte "
             "e registra `alternate_source: true`. Esse artefato QW2 pode ter protocolo/retries "
             "diferentes do run original: não interpretar as diferenças como efeito causal de modelo. "
             "O run opencode_server de 148 pares sem modelo não é utilizado. Se o artefato "
             "original for restaurado no caminho solicitado, ele terá precedência.", "",
             "## Artefatos incluídos", "", "| Modelo | Modo | n | Parcial | Arquivo |",
             "|---|---|---:|---|---|"]
    for source in sources:
        lines.append(f"| {source['model']} | {source['mode']} | {source['n']} | {source['partial']} | `{source['path']}` |")
    lines += ["", "## Referência fixa atual", "", "| Banda | e_fp fixo | e_fn fixo |", "|---|---:|---:|"]
    for band, rates in fixed.items():
        lines.append(f"| {band} | {rates['e_fp']:.4f} | {rates['e_fn']:.4f} |")
    lines += ["", "## Estimativas por modo × modelo × banda", "",
              "Taxas e IC em proporções; Δ é diferença absoluta em relação ao config, não diferença com sinal. "
              "⚠ sinaliza n<10; denominadores de classe pequenos também são sinalizados no CSV.", "",
              "| Modo | Modelo | Banda | n | Vínculos | e_fp [IC95%] | e_fn [IC95%] | u [IC95%] | Δ e_fp | Δ e_fn |",
              "|---|---|---|---:|---:|---|---|---|---:|---:|"]
    for row in rows:
        model = row["model"] + (" (parcial)" if row["partial"] else "")
        count = str(row["n"]) + (" ⚠" if row["n_lt_10"] else "")
        delta = ["NA" if row[f"{m}_abs_diff"] is None else f"{row[f'{m}_abs_diff']:.4f}" for m in ("e_fp", "e_fn")]
        lines.append(f"| {row['mode']} | {model} | {row['band']} | {count} | {row['n_vinculos']} | "
                     f"{format_rate(row, 'e_fp')} | {format_rate(row, 'e_fn')} | {format_rate(row, 'u')} | {delta[0]} | {delta[1]} |")
    lines += ["", "## Contradições MATCH ↔ NONMATCH: Kimi K2.5 × DeepSeek", "",
              "| Modo | Banda | Contradições | Vínculos | Kimi MATCH / DS NONMATCH | Kimi NONMATCH / DS MATCH |",
              "|---|---|---:|---:|---:|---:|"]
    for row in conflicts:
        lines.append(f"| {row['mode']} | {row['band']} | {row['n']} | {row['n_vinculos']} | "
                     f"{row['kimi_match_deepseek_nonmatch']} | {row['kimi_nonmatch_deepseek_match']} |")
    for mode in MODES:
        subset = [r for r in conflicts if r["mode"] == mode]
        lines.append(f"\n{mode}: {sum(r['n'] for r in subset)} contradições; {sum(r['n_vinculos'] for r in subset)} vínculos verdadeiros.")
    lines += ["", "## Uso pelo motor (sem alteração do config principal)", "",
              "A chave atual é `llm_review.reliability.error_rates_by_band`, um mapping "
              "`banda: {e_fp: float, e_fn: float}`; o motor utiliza esses dois números no custo esperado. "
              "Não há suporte implícito a caminho de arquivo nessa chave. Para adotar, carregar "
              "`gzcmd/error_rates_by_band_estimadas.yaml` externamente e atribuir o mapping de "
              "`by_mode.<modo>.models.<modelo>.error_rates_by_band` ou "
              "`by_mode.<modo>.pooled.error_rates_by_band` à chave atual antes de construir a política. "
              "O bloco `llm_review.reliability.error_rates_by_band` do arquivo novo fornece um default "
              "pooled dos modos completos; não substitua o mapping por uma string de caminho. "
              "Os blocos `models` agregam modos por modelo; `pooled` agrega somente modelos completos.", "",
              "Para denominador zero, somente os mappings destinados ao motor retêm o valor fixo "
              "do config (`fallback_fixed` identifica cada campo); estimativas/CSV mantêm NA e IC ausente. "
              "Isso evita vender falta de evidência como taxa zero. Bandas vazias não foram calibradas. "
              "Estes mappings de confiabilidade fornecem somente e_fp/e_fn; u é diagnóstico "
              "e não é embutido nessas duas taxas nem configura o custo humano de UNSURE.", "",
              "Artefatos persistidos são exclusivamente agregados, sem identificadores por par, evidências ou PII.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / "gzcmd/gzcmd_v3_config.yaml"
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    definitions = config.get("bands", {}).get("definitions") or FALLBACK_BANDS
    fixed = config["llm_review"]["reliability"]["error_rates_by_band"]
    gold = load_gold(root / "data/COMPARADORSEMIDENT.csv")
    data, sources = load_runs(root, gold, definitions)
    rows = estimate(data, definitions, fixed)
    conflicts = contradictions(data, definitions)
    complete = data[~data.partial]
    overall_models = {}
    for model, part in data.groupby("model", sort=True):
        overall_models[model] = engine_profile([
            summarize(part[part.band.eq(band)], "all", model, band, bool(part.partial.any()), fixed)
            for band in definitions], fixed, bool(part.partial.any()))
    overall = engine_profile([summarize(complete[complete.band.eq(band)], "all", "pooled", band, False, fixed)
                              for band in definitions], fixed)
    by_mode = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        by_mode[mode] = {"models": {
            model: engine_profile([r for r in mode_rows if r["model"] == model], fixed,
                                  any(r["partial"] for r in mode_rows if r["model"] == model))
            for model in sorted({r["model"] for r in mode_rows} - {"pooled"})},
            "pooled": engine_profile([r for r in mode_rows if r["model"] == "pooled"], fixed)}
    output = {"schema_version": 1, "source_config": "gzcmd/gzcmd_v3_config.yaml",
              "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "method": "Wilson 95%; class-conditional FP/FN; UNSURE/n; zero denominators use explicit fixed fallback only in engine mappings",
              "pooled_excludes_partial": True, "sources": sources, "bands": definitions,
              "llm_review": {"reliability": {"error_rates_by_band": overall["error_rates_by_band"]}},
              "models": overall_models, "pooled": overall, "by_mode": by_mode,
              "contradictions_kimi_deepseek": conflicts}
    csv_path = root / "data/taxas_erro_revisor_por_banda.csv"
    yaml_path = root / "gzcmd/error_rates_by_band_estimadas.yaml"
    report_path = root / "data/taxas_erro_revisor_por_banda.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8", float_format="%.10g")
    yaml_path.write_text(yaml.safe_dump(output, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report_path.write_text(make_report(rows, sources, conflicts, fixed), encoding="utf-8")
    for source in sources:
        print(f"RUN {source['mode']} {source['model']}: n={source['n']} partial={source['partial']}")
        if source["alternate_source"]:
            print("AVISO: confirmação K2.5 ausente em sprint3b; fonte alternativa explícita:", source["path"])
    print("\nmode model band n n_vinculos e_fp e_fn u (IC95% Wilson)")
    for row in rows:
        print(row["mode"], row["model"], row["band"], row["n"], row["n_vinculos"],
              *(format_rate(row, m) for m in ("e_fp", "e_fn", "u")),
              "n<10" if row["n_lt_10"] else "", "PARCIAL" if row["partial"] else "")
    print("\nCONTRADICOES", json.dumps(conflicts, ensure_ascii=True))
    print(f"OK: {len(rows)} linhas agregadas; {len(data)} decisões; pooled completo={len(complete)}")
    for path in (csv_path, yaml_path, report_path):
        print("SALVO", path.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
