"""Publica apenas contagens do probe e da coorte histórica; nunca chama APIs.

python scripts/probe_frontier_agregado.py --source-root D:/git/phd-research
As entradas individuais ficam locais. A regeneração integral exige essas entradas;
os testes dos artefatos publicados são portáveis. Nenhum runner é importado.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-fable-5-1": "Claude Fable 5.1",
}
DECISIONS = ("MATCH", "NONMATCH", "UNSURE")
SOURCES = ("resultados_agregados.csv", "concordancias_agregadas.csv",
           "execucao_agregada.json", "PREREG.md", "ENCERRAMENTO.md", "custo" + ".json")
ARTIFACTS = ("probe_frontier_agregado.csv", "probe_frontier_concordancias.csv",
             "probe_frontier_custos.csv", "probe_haiku_sonnet_coorte.csv")
REPORT = "RELATORIO_PROBE_4MODELOS_2026-11-07.md"
TEXT_VALUES = set(MODELS) | set(DECISIONS) | {
    "vigilancia", "confirmacao", "S1", "S2", "S3", "S4", "C",
    "TOTAL_AMOSTRA", "TOTAL_PONDERADO", "TOTAL_COMUM", "historico",
    "principal", "replica", "replica_historico", "True", "False",
    "probe", "coorte_historica", "original", "uniao_historica",
    "[5,7)", "[7,8)", "[8,9)", ">=9", "<5", "TOTAL",
    "batch", "direto", "batch+direto", "historico_reutilizado",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_public(rows):
    """Esquema fechado de valores: enums ou números, sem texto livre/campos.

    Além de rejeitar nomes identificadores, não permite IDs hexadecimais ou
    valores textuais arbitrários. Esta validação ocorre ANTES da publicação.
    """
    for row in rows:
        for column, value in row.items():
            if re.search(r"pair_id|comprec|refrec|field|campo", column, re.I):
                raise ValueError(f"Coluna proibida: {column}")
            text = str(value) if value is not None else ""
            if text in TEXT_VALUES or text == "":
                continue
            if re.search(r"pair_id|comprec|refrec", text, re.I):
                raise ValueError("Identificador em saída pública")
            if not math.isfinite(float(text)):
                raise ValueError("Número não finito na saída")


def write_csv(path, rows):
    validate_public(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wilson(vp, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p = vp / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center - half, center + half


def agreement(left, right):
    if not left or len(left) != len(right):
        raise ValueError("Concordância requer interseção não vazia e alinhada")
    n = len(left)
    same = sum(a == b for a, b in zip(left, right))
    a, b = Counter(left), Counter(right)
    expected_n2 = sum(a[d] * b[d] for d in DECISIONS)
    return same, same/n, ((same*n - expected_n2) / (n*n - expected_n2)
                          if expected_n2 != n*n else None)


def load_labels(path):
    labels = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        columns = {name.split(",")[0]: name for name in reader.fieldnames}
        for row in reader:
            identity = "|".join(row[columns[k]].strip()
                                for k in ("COMPREC", "REFREC", "PASSO"))
            pid = hashlib.sha256(identity.encode()).hexdigest()[:16]
            if pid in labels:
                raise ValueError("Identidade duplicada na entrada")
            labels[pid] = (int(row["PAR"]),
                           float(row["nota final"].strip().replace(",", ".")))
    return labels


def band(score):
    for low, high, label in ((5, 7, "[5,7)"), (7, 8, "[7,8)"), (8, 9, "[8,9)")):
        if low <= score < high:
            return label
    return ">=9" if score >= 9 else "<5"


def history(source_root, prompt_sha):
    """Coorte original e união dos suplementos lote31, sem intervenções tardias.

    Não agrega réplicas/adaptações arranjo, robustez ou filtro como novas
    observações. Custo usa transport/cost_estimate do run, não soma novamente
    journals ou execucao.json que resumem as mesmas chamadas.
    """
    runs, provenance, costs = [], [], []
    for model, prefix in (("claude-haiku-5-5", "anthropic_sem_mo_v4"),
                          ("claude-sonnet-5-5", "anthropic_sonnet_sem_mo_v4")):
        for folder in sorted((source_root / "data/sprint3b").glob(prefix + "*")):
            suffix = folder.name[len(prefix):]
            if suffix and not suffix.startswith("_lote31"):
                continue
            for mode in ("vigilancia", "confirmacao"):
                path = folder / f"exp2_llm_sync_{mode}.json"
                if not path.exists():
                    continue
                run = read_json(path)
                if (run["model"] != model or run["mode"] != mode
                        or run["prompt_version"] != "v4"
                        or run["prompt_sha256"] != prompt_sha
                        or not any(f.startswith("model_outputs.")
                                   for f in run.get("removed_fields", []))):
                    raise ValueError(f"Modelo/prompt/dossiê incompatível: {path}")
                audit_path = folder / f"dossier_field_audit_{mode}.json"
                audit = read_json(audit_path)
                if audit["forbidden_present"]:
                    raise ValueError("Dossiê histórico contém saídas do motor")
                provenance.append({"path": path.relative_to(source_root).as_posix(),
                                   "sha256": sha(path), "timestamp": run["timestamp"],
                                   "audit_sha256": sha(audit_path)})
                runs.append((model, mode, suffix, run))
        selected = [r for m, _, _, r in runs if m == model]
        transport = [r["transport"] for r in selected]
        cost = sum((Decimal(str(r["cost_estimate_usd"])) for r in selected), Decimal(0))
        reviewed = sum(len(r["results"]) for r in selected)
        costs.append(dict(model=model, escopo="coorte_historica",
                          chamadas=sum(t["calls"] for t in transport),
                          tokens_entrada=sum(t["input_tokens"] for t in transport),
                          tokens_saida=sum(t["output_tokens"] for t in transport),
                          tokens_raciocinio=None, usd_total=str(cost),
                          pares_revisados=reviewed, usd_por_par_revisado=str(cost/reviewed),
                          respostas_truncadas=None, chamadas_batch=0,
                          chamadas_direto=sum(t["calls"] for t in transport), transporte="direto"))
    return runs, provenance, costs


def cohort_rows(runs, labels):
    output, coverage = [], []
    for cohort in ("original", "uniao_historica"):
        for mode in ("vigilancia", "confirmacao"):
            decisions, fingerprints = {}, {}
            for model in ("claude-haiku-5-5", "claude-sonnet-5-5"):
                decisions[model], fingerprints[model] = {}, {}
                for m, md, suffix, run in runs:
                    if m != model or md != mode or (cohort == "original" and suffix):
                        continue
                    for row in run["results"]:
                        pid, decision = row["pair_id"], row["decision"]
                        if decision not in DECISIONS:
                            raise ValueError("Decisão histórica inválida")
                        fingerprint = run["dossier_sha256"][pid]
                        if pid in decisions[model] and (
                                decisions[model][pid] != decision
                                or fingerprints[model][pid] != fingerprint):
                            raise ValueError("Histórico conflitante; seleção explícita necessária")
                        decisions[model][pid] = decision
                        fingerprints[model][pid] = fingerprint
            haiku, sonnet = decisions.values()
            common = sorted(haiku.keys() & sonnet.keys())
            if not common or any(fingerprints["claude-haiku-5-5"][p]
                                 != fingerprints["claude-sonnet-5-5"][p] for p in common):
                raise ValueError("Interseção vazia ou dossiês diferentes")
            coverage.append(dict(coorte=cohort, modo=mode, n_haiku=len(haiku),
                                 n_sonnet=len(sonnet), n_comum=len(common)))
            groups = [(b, [p for p in common if band(labels[p][1]) == b])
                      for b in ("<5", "[5,7)", "[7,8)", "[8,9)", ">=9")]
            for faixa, ids in groups + [("TOTAL", common)]:
                if not ids:
                    continue
                same, agree, kappa = agreement([haiku[p] for p in ids], [sonnet[p] for p in ids])
                for model, dec in decisions.items():
                    counts = Counter(dec[p] for p in ids)
                    vp = sum(dec[p] == "MATCH" and labels[p][0] in (1, 2) for p in ids)
                    fp = sum(dec[p] == "MATCH" and labels[p][0] == 0 for p in ids)
                    lo, hi = wilson(vp, vp+fp)
                    output.append(dict(coorte=cohort, model=model, modo=mode, faixa=faixa,
                                       n=len(ids), **{d: counts[d] for d in DECISIONS},
                                       VP=vp, FP=fp,
                                       PAR2_total=sum(labels[p][0] == 2 for p in ids),
                                       PAR2_recuperados=sum(labels[p][0] == 2 and dec[p] == "MATCH" for p in ids),
                                       PAR2_descartados=sum(labels[p][0] == 2 and dec[p] == "NONMATCH" for p in ids),
                                       PAR2_unsure=sum(labels[p][0] == 2 and dec[p] == "UNSURE" for p in ids),
                                       VPP=vp/(vp+fp) if vp+fp else None,
                                       VPP_IC95_inf=lo, VPP_IC95_sup=hi,
                                       n_concordantes=same, agreement=agree, kappa=kappa))
    return output, coverage


def probe_costs(ledger, statuses):
    result = []
    for model in MODELS:
        calls = [v for v in ledger["calls"].values() if v["model"] == model]
        # Tarifas/usage do ledger têm até dez casas; remove ruído binário
        # presente em alguns floats serializados, sem arredondar a centavos.
        cost = sum((Decimal(str(v["cost_usd"])) for v in calls), Decimal(0)).quantize(
            Decimal("0.0000000001")).normalize()
        # Unidade: decisão final analisável por passada + piloto; não par único.
        reviewed = sum(r["n_analisavel"] for r in statuses if r["model"] == model
                       and r["passada"] != "historico") + (10 if model in (
                           "claude-opus-5-5", "claude-fable-5-1") else 0)
        thinking = [v["usage"].get("output_tokens_details", {}).get("thinking_tokens") for v in calls]
        batch = sum(v["batch"] for v in calls)
        result.append(dict(model=model, escopo="probe", chamadas=len(calls),
                           tokens_entrada=sum(v["usage"].get("input_tokens", 0) for v in calls),
                           tokens_saida=sum(v["usage"].get("output_tokens", 0) for v in calls),
                           tokens_raciocinio=sum(v for v in thinking if v is not None) if any(v is not None for v in thinking) else None,
                           usd_total=str(cost), pares_revisados=reviewed,
                           usd_por_par_revisado=str(cost/reviewed) if reviewed else None,
                           respostas_truncadas=sum(v["stop_reason"] == "max_tokens" for v in calls),
                           chamadas_batch=batch, chamadas_direto=len(calls)-batch,
                           transporte=("batch+direto" if batch and batch < len(calls) else
                                       "batch" if batch else "direto" if calls else "historico_reutilizado")))
    return result


def table(rows, columns):
    return ["| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)] + [
        "| " + " | ".join("NA" if r.get(c) in (None, "") else str(r[c]) for c in columns) + " |"
        for r in rows]


def report(out, aggregates, comparisons, costs, cohorts, metadata):
    lines = ["# Comparação dos quatro revisores Anthropic — 2026-11-07", "",
             "## Reprodução e limites", "",
             "`python scripts/probe_frontier_agregado.py --source-root D:/git/phd-research`", "",
             "Status: **exploratório post hoc** para a documentação; o probe possui pré-registro local anterior à inferência.",
             "Sem novas chamadas de API ou inferência. Apenas agregados são publicados. "
             "Os hashes de origem e os parâmetros constam de `data/probe_frontier_metadata.json`.",
             "Protocolo A+B independentes, árbitro no desacordo; prompt v4 e dossiê sem saídas do motor. "
             "max_tokens=2500, temperature ausente, thinking não configurado (raciocínio adaptativo padrão da API). "
             "Opus 5.5 e Fable 5.1 não permitem desligar o raciocínio.", "",
             "Vigilância: população P=344; S1=25, S2=13, S3=24, S4=88 (N4=282; peso=282/88). "
             "Confirmação: censo P=41. Réplicas Opus/Fable: S1–S3 (62) e confirmação (41). Seed 20261107.",
             "Wilson 95% só em contagens não ponderadas. TOTAL_PONDERADO é Horvitz–Thompson; "
             "não somar TOTAL_AMOSTRA, estratos e TOTAL_PONDERADO entre si. "
             "κ indefinido (decisões constantes) é NA, não zero.", "",
             "PAR=0 não prova ausência de vínculo. VP/FP são relativos ao PAR, sem adjudicação independente. "
             "Ausências não são UNSURE; concordância/κ usam casos completos em comum. "
             "Sem teste de superioridade, inferência causal ou substituição dos resultados principais.", "",
             "## Probe: contagens e VPP (valores sem arredondamento editorial)", ""]
    lines += table(aggregates, ["model", "modo", "passada", "estrato", "n", "MATCH", "NONMATCH", "UNSURE",
                                "VP", "FP", "PAR2_recuperados", "PAR2_descartados", "PAR2_unsure", "VPP",
                                "VPP_IC95_inf", "VPP_IC95_sup"])
    lines += ["", "S1 da vigilância: Haiku histórico 4/25=16%; Opus principal 4/21=19,047619…%; "
              "Fable principal 4/22=18,181818…%. Sonnet cobre só 139/150 e 33/41 neste probe, "
              "não confundir com a coorte histórica completa abaixo.", "", "## Concordância e réplicas", ""]
    lines += table([r for r in comparisons if r["estrato"] == "TOTAL_COMUM"], list(comparisons[0]))
    lines += ["", "## Completude do probe", ""]
    lines += table(metadata["runs"], ["model", "modo", "passada", "n_expected", "n_analisavel", "n_sem_resultado"])
    lines += ["", "Haiku réplica: 17 respostas truncadas afetaram 16 pares, sem retry pago; "
              "137/150 e 38/41 decisões válidas. Opus/Fable completos, zero truncadas. "
              "Comparações da réplica Haiku podem sofrer seleção por ausência.", "", "## Custos", "",
              f"Probe: US$ {metadata['spent_usd']} (US$ 44,44), teto US$ 90; inclui pilotos diretos, principais e réplicas Batch. "
              "São estimativas por usage/preço, não fatura. Sonnet histórico não gerou gasto novo no probe.",
              "Custo por par revisado = custo / decisões finais válidas por passada (inclui piloto). "
              "Réplicas contam novamente; não é custo por par único. Custos de chamadas falhas estão no numerador. "
              "Raciocínio já integra tokens de saída, não somar outra vez. NA = não registrado, não zero.", ""]
    lines += table(costs, list(costs[0]))
    lines += ["", "## Coorte histórica completa Haiku × Sonnet (sem custo novo)", "",
              "Original: runs `anthropic_sem_mo_v4` e `anthropic_sonnet_sem_mo_v4`. "
              "União histórica: acrescenta os suplementos lote31; decisões repetidas são deduplicadas "
              "e conflitos rejeitados. Arranjo, robustez e filtro são intervenções posteriores e ficam fora "
              "desta comparação/custo histórico. Não se misturam com a coorte original.",
              "Exigidos por código: mesmo prompt SHA-256, auditoria sem campos proibidos e fingerprint "
              "do dossiê idêntico entre modelos em cada par comum. Faixas usam `nota final` do comparador.", ""]
    lines += table(metadata["coorte_cobertura"], ["coorte", "modo", "n_haiku", "n_sonnet", "n_comum"])
    lines += [""] + table(cohorts, list(cohorts[0]))
    lines += ["", "Os custos históricos somam os transport/cost_estimate dos runs originais e lote31 uma única vez, "
              "sem somar journals e execucao.json de novo. Não são parte dos US$ 44,44 do probe.", ""]
    target = out / "docs/plans" / REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate(source_root, out=ROOT, probe=None):
    probe = probe or source_root / ".local/probe_frontier"
    aggregates = read_csv(probe / SOURCES[0])
    comparisons = read_csv(probe / SOURCES[1])
    execution = read_json(probe / SOURCES[2])
    ledger = read_json(probe / SOURCES[-1])
    prompt = out / "gzcmd/gzcmd_v4_llm_prompt.md"
    # Prompt is a text artifact explicitly LF; hash the frozen logical LF bytes.
    prompt_sha = hashlib.sha256(prompt.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    prereg = (probe / "PREREG.md").read_text(encoding="utf-8")
    if prompt_sha not in prereg:
        raise ValueError("Prompt atual não corresponde ao PREREG")
    for row in aggregates:
        # Recompute the Wilson interval from counts; never from weighted pseudo-n.
        if row["ponderado"] == "False":
            lo, hi = wilson(float(row["VP"]), float(row["MATCH"]))
            for column, expected in (("VPP_IC95_inf", lo), ("VPP_IC95_sup", hi)):
                if expected is not None and not math.isclose(float(row[column]), expected, abs_tol=1e-12):
                    raise ValueError("IC de origem incompatível com Wilson")
    runs, provenance, historical_costs = history(source_root, prompt_sha)
    label_path = source_root / "data/COMPARADORSEMIDENT.csv"
    cohorts, coverage = cohort_rows(runs, load_labels(label_path))
    costs = probe_costs(ledger, execution["runs"]) + historical_costs
    if not math.isclose(sum(float(r["usd_total"]) for r in costs if r["escopo"] == "probe"),
                        execution["spent_usd"], abs_tol=1e-10):
        raise ValueError("Ledger diverge do custo agregado")
    if ledger["reservations"] or not math.isclose(ledger["spent_usd"], execution["spent_usd"], abs_tol=1e-10):
        raise ValueError("Ledger incompleto/divergente")
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(ARTIFACTS, (aggregates, comparisons, costs, cohorts)):
        write_csv(data / name, rows)
    metadata = dict(schema_version=1, status="exploratório post hoc", models=MODELS,
                    data_nominal="2026-11-07", inicio_utc="2026-11-06T23:34:11.572322+00:00",
                    fim_usage_utc=ledger["updated_at"], prompt_file="gzcmd/gzcmd_v4_llm_prompt.md",
                    prompt_version="v4", prompt_sha256=prompt_sha,
                    parameters=dict(max_tokens=2500, temperature="não enviado", thinking="não configurado",
                                    reasoning="raciocínio adaptativo padrão da API",
                                    cannot_disable_thinking=["claude-opus-5-5", "claude-fable-5-1"]),
                    protocol="A+B independentes; consenso com maior confiança; árbitro no desacordo",
                    seed=20261107, strata=dict(S1=25, S2=13, S3=24, S4=88, C=41),
                    vigilancia_population=344, s4_population=282, s4_weight=282/88,
                    transport=["Batch API", "piloto direto"], budget_usd=90,
                    spent_usd=execution["spent_usd"], outstanding_reservations_usd=0,
                    source_sha256={name: sha(probe / name) for name in SOURCES},
                    historical_sources=provenance, labels_source="data/COMPARADORSEMIDENT.csv",
                    labels_sha256=sha(label_path), coorte_cobertura=coverage, runs=execution["runs"],
                    historical_selection="original + lote31; exclui intervenções arranjo/robustez/filtro",
                    privacy="CSV: somente enums fechados e números; sem identificadores ou valores de campo",
                    cost_denominator="decisões finais válidas por passada, incluindo pilotos; não pares únicos",
                    output_sha256={name: sha(data / name) for name in ARTIFACTS})
    (data / "probe_frontier_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    report(out, aggregates, comparisons, costs, cohorts, metadata)
    print(json.dumps(dict(rows=len(aggregates), concordancias=len(comparisons), coorte_rows=len(cohorts),
                          spent_usd=metadata["spent_usd"], cobertura=coverage), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    generate(args.source_root, args.output_root, args.probe)


if __name__ == "__main__":
    main()
