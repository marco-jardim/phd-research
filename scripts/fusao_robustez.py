"""Robustez exploratória de RRF: bootstrap REFREC, faixa e empates, só CPU."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fusao_filas import (  # noqa: E402
    DEFINITIONS, load_queues, metrics, numeric, rrf, sha256, verify_baselines,
)

SEED = 20261107
TARGETS = (9, 23, 36)
FRACTIONS = (0.25, 0.64, 1.0)
NAMES = ("O1", "O3", "F4", "F3")
COMPARISONS = (("F4", "O1"), ("F4", "O3"), ("F3", "O1"), ("F3", "O3"))


def fused_orders(queues):
    return {
        "O1": queues["O1"], "O3": queues["O3"],
        **{name: rrf([queues[a] for a in DEFINITIONS[name]["arms"]])
           for name in ("F4", "F3")},
    }


def resampled_queues(queues, multiplicity):
    """Cópias contíguas por hash; reranqueia cada braço antes de aplicar RRF.

    Cada cópia é uma observação da amostra bootstrap. A mesma cópia tem a
    mesma identidade nos três braços; não se deduplicam cópias bootstrap.
    """
    original = np.repeat(np.arange(len(multiplicity)), multiplicity)
    result = {}
    for name, q in queues.items():
        ranks = np.empty(len(q), dtype=np.int64)
        ranks[q] = np.arange(len(q))
        result[name] = np.argsort(ranks[original], kind="stable")
    return result, original


def reviews(orders, labels, targets):
    if not targets or min(targets) < 1 or max(targets) > int(labels.sum()):
        raise ValueError("Meta não alcançável")
    return {
        name: np.searchsorted(labels[q].astype(int).cumsum(), targets) + 1
        for name, q in orders.items()
    }


def bootstrap_once(queues, labels, group_codes, group_count, rng):
    draws = rng.integers(0, group_count, size=group_count)
    counts = np.bincount(draws, minlength=group_count)
    sampled, original = resampled_queues(queues, counts[group_codes])
    sampled_labels = labels[original]
    total = int(sampled_labels.sum())
    if not total:
        return None
    targets = TARGETS if total >= 36 else tuple(
        int(np.ceil(f * total)) for f in FRACTIONS
    )
    effort = reviews(fused_orders(sampled), sampled_labels, targets)
    return effort, total, len(original)


def summarize_bootstrap(runs):
    rows = []
    for fused, base in COMPARISONS:
        fv = np.array([r[0][fused] for r in runs])
        bv = np.array([r[0][base] for r in runs])
        diff = fv - bv
        reduction = 100 * (bv - fv) / bv
        for j, target in enumerate(TARGETS):
            row = {"comparacao": f"{fused}-{base}", "meta_nominal": target,
                   "fracao_se_menos_36": FRACTIONS[j], "replicas_validas": len(runs),
                   "proporcao_fusao_menor": float((diff[:, j] < 0).mean())}
            for name, values in (("diferenca_revisoes", diff[:, j]),
                                 ("reducao_percentual", reduction[:, j])):
                lo, median, hi = np.quantile(values, [0.025, 0.5, 0.975], method="linear")
                row.update({f"{name}_mediana": float(median),
                            f"{name}_ic95_inf": float(lo),
                            f"{name}_ic95_sup": float(hi)})
            rows.append(row)
    # Probabilidade conjunta por meta, além das probabilidades marginais acima.
    joint = np.mean([
        (r[0]["F4"] < r[0]["O1"]) & (r[0]["F4"] < r[0]["O3"])
        for r in runs
    ], axis=0)
    table = pd.DataFrame(rows)
    table["proporcao_F4_menor_ambas"] = table.meta_nominal.map(dict(zip(TARGETS, joint)))
    return table


def verify_reference(queues, labels):
    verify_baselines({n: labels[q].cumsum() for n, q in queues.items()})
    reference = pd.read_csv(ROOT / "data/fusao_filas.csv")
    for name, q in fused_orders(queues).items():
        expected = reference.loc[reference.fila.eq(name)].set_index("metrica").media
        for metric, value in metrics(labels[q].cumsum()).items():
            if not np.isclose(value, expected[metric], rtol=0, atol=1e-12):
                raise ValueError(f"Referência fusao_filas divergente: {name}/{metric}")
    return {"passed": True, "queues": list(NAMES), "all_metrics": True}


def reviewable_mask(scores):
    # O corte informado é arredondado: o bloco limítrofe tem -5.64846544302.
    # O filtro literal >= -5.648 exclui 27 candidatos, inclusive cinco PAR=2.
    return np.round(scores, 3) >= -5.648


def faixa_table(queues, labels, scores):
    mask = reviewable_mask(scores)
    if int(mask.sum()) != 2522 or int(labels[mask].sum()) != 36:
        raise ValueError("Faixa não reproduziu 2.522/36")
    restricted, original = resampled_queues(queues, mask.astype(int))
    effort = reviews(fused_orders(restricted), labels[original], TARGETS)
    rows = []
    for name in NAMES:
        for j, target in enumerate(TARGETS):
            row = {"fila": name, "meta": target, "revisoes": int(effort[name][j])}
            for base in ("O1", "O3"):
                row[f"diferenca_{base}"] = int(effort[name][j] - effort[base][j])
                row[f"reducao_percentual_{base}"] = float(
                    100 * (1 - effort[name][j] / effort[base][j]))
            rows.append(row)
    return pd.DataFrame(rows)


def distribution(values):
    return dict(zip(("minimo", "mediana", "maximo"),
                    map(float, np.quantile(values, [0, 0.5, 1]))))


def tie_audit(scores, labels, top):
    """Agregados de empates, sem identificadores nem valores por potencial par."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positions = np.arange(len(scores))  # ordem lexical do hash, não ordem do extrato
    # Posição relativa do positivo no empate, 0 primeiro / 1 último.
    order = np.lexsort((positions, inverse))
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    within = np.empty(len(scores), dtype=int)
    within[order] = np.arange(len(scores)) - starts
    selected = labels & (counts[inverse] > 1)
    relative = within[selected] / (counts[inverse[selected]] - 1)
    return {"blocos_empatados": int((counts > 1).sum()),
            "pares_em_empates": int((counts[inverse] > 1).sum()),
            "par2_em_empates": int(selected.sum()),
            "top17_em_empates": int((counts[inverse[top]] > 1).sum()),
            "posicao_relativa_media_par2_em_empates":
                float(relative.mean()) if len(relative) else None}


def random_tied_orders(scores_by_name, rng):
    # Relabelar por uma permutação independente permite reutilizar rrf também
    # no desempate final; nunca se usa PAR para construir a prioridade.
    n = len(next(iter(scores_by_name.values())))
    original = rng.permutation(n)
    queues = {name: np.argsort(-score[original], kind="stable")
              for name, score in scores_by_name.items()}
    return fused_orders({**queues, "O3_MACD": queues["O3"]})["F4"], original


def top_audit(queues, labels, scores, ml_scores, notes, raw, below, aligned):
    orders = fused_orders(queues)
    top = orders["F4"][:17]
    if int(labels[top].sum()) != 16:
        raise ValueError("Topo F4 não reproduziu 16/17")
    result = {"n": 17, "par2": int(labels[top].sum()), "nota_CdR": distribution(notes[top]),
              "score_OpenRecLink": distribution(scores[top]), "filas": {}}
    for name in ("O1", "O3"):
        rank = np.empty(len(labels), dtype=int)
        rank[queues[name]] = np.arange(1, len(labels) + 1)
        result["filas"][name] = {"posicao_media": float(rank[top].mean()),
                                   "no_top17": int((rank[top] <= 17).sum())}
    result["empates"] = {n: tie_audit(s, labels, top)
                         for n, s in (("O1", scores), ("O3", ml_scores))}
    fused_score = np.zeros(len(labels))
    for arm in DEFINITIONS["F4"]["arms"]:
        fused_score[queues[arm]] += 1.0 / (60 + np.arange(1, len(labels) + 1))
    result["empates"]["F4_RRF"] = tie_audit(fused_score, labels, top)
    # Diagnóstico da ordem do extrato. Apenas correlações/indicadores agregados.
    par_raw = numeric(raw.PAR).to_numpy()
    comp = numeric(raw.COMPREC).to_numpy()
    index = np.arange(len(raw))
    aligned_index = index[below][aligned]
    result["ordem"] = {
        "desempate_efetivo": "hash lexical SHA-256, não índice de linha/PAR/COMPREC",
        "PAR_monotono_extrato": bool(np.all(np.diff(par_raw) >= 0) or np.all(np.diff(par_raw) <= 0)),
        "COMPREC_monotono_extrato": bool(np.all(np.diff(comp) >= 0) or np.all(np.diff(comp) <= 0)),
        "spearman_indice_PAR_extrato": float(pd.Series(index).corr(pd.Series(par_raw), method="spearman")),
        "spearman_indice_COMPREC_extrato": float(pd.Series(index).corr(pd.Series(comp), method="spearman")),
        "correlacao_indice_PAR2_abaixo": float(np.corrcoef(aligned_index, labels)[0, 1]),
        "correlacao_rank_hash_PAR2_abaixo": float(np.corrcoef(np.arange(len(labels)), labels)[0, 1]),
    }
    rng = np.random.default_rng(SEED + 1)
    trials = []
    for _ in range(200):
        q, original = random_tied_orders({"O1": scores, "O3": ml_scores}, rng)
        lab = labels[original]
        counts = lab[q].cumsum()
        trials.append([int(counts[16]), *map(int, np.searchsorted(counts, TARGETS) + 1)])
    values = np.array(trials)
    result["desempate_aleatorio"] = {
        "seed": SEED + 1, "repeticoes": 200,
        "regra": "mesma prioridade aleatória independente em empates O1/O3 e RRF; nenhum rótulo",
        "quantis": "P2,5, mediana, P97,5 lineares (sensibilidade, não IC)",
        "metricas": {name: dict(zip(("p025", "mediana", "p975"),
                                    map(float, np.quantile(values[:, j], [0.025, 0.5, 0.975]))))
                     for j, name in enumerate(("par2_top17", "revisoes_9", "revisoes_23", "revisoes_36"))},
        "proporcao_top17_menos_16": float((values[:, 0] < 16).mean()),
    }
    return result


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_report(table, faixa, topo, metadata):
    def fmt(x):
        return f"{x:.2f}".replace(".", ",")
    lines = ["# Robustez exploratória da fusão de filas", "",
             "CPU, sem LLM, revisor perfeito simulado por PAR=2. Análise post hoc, sem validação independente.", "",
             "## Bootstrap agrupado por REFREC", "",
             f"{metadata['replicas']} réplicas; seed {SEED}. IC percentil de 95%, quantis lineares.",
             "Amostram-se tantos REFREC quanto os distintos abaixo do corte, com reposição; todos os seus potenciais pares recebem a multiplicidade do óbito. Recalculam-se posições e RRF em cada amostra, sem reajustar classificadores.",
             "Metas 9/23/36 quando há pelo menos 36 PAR=2; caso contrário, teto de 25%/64%/100% dos PAR=2. Esta regra híbrida muda o estimando entre réplicas; não equivale a sensibilidade fixa. Cópias bootstrap contam como revisões. Réplicas sem PAR=2 são excluídas e contadas nos metadados.",
             "Diferença = fusão menos base (negativo favorece fusão); redução = 100 × (base menos fusão)/base (positivo favorece fusão).", "",
             "| Comparação | Meta | Δ revisões, mediana [IC95%] | Redução %, mediana [IC95%] | P(fusão < base) | P(F4 < ambas) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in table.itertuples():
        lines.append(f"| {r.comparacao} | {r.meta_nominal} | {fmt(r.diferenca_revisoes_mediana)} [{fmt(r.diferenca_revisoes_ic95_inf)}; {fmt(r.diferenca_revisoes_ic95_sup)}] | {fmt(r.reducao_percentual_mediana)} [{fmt(r.reducao_percentual_ic95_inf)}; {fmt(r.reducao_percentual_ic95_sup)}] | {fmt(r.proporcao_fusao_menor)} | {fmt(r.proporcao_F4_menor_ambas)} |")
    lines += ["", "## Faixa operacional: 2.522 candidatos e 36 PAR=2", "",
              "Correção de precisão: SCORE ≥ −5,648 literal seleciona 2.495 candidatos e 31 PAR=2. O bloco limítrofe contém 27 candidatos, cinco PAR=2, com escore exato −5,64846544302, exibido como −5,648 a três casas. A faixa de 2.522 corresponde ao escore arredondado a três casas ≥ −5,648 (limite efetivo observado −5,64846544302), não ao filtro literal.",
              "Filas e RRF recalculados somente no subconjunto, não apenas filtrados após a fusão.", "",
              "| Fila | Meta | Revisões | Δ O1 | Δ O3 |", "|---|---:|---:|---:|---:|"]
    for r in faixa.itertuples():
        lines.append(f"| {r.fila} | {r.meta} | {r.revisoes} | {r.diferenca_O1} | {r.diferenca_O3} |")
    f4 = table.loc[table.comparacao.str.startswith("F4-")]
    robust = f4.groupby("meta_nominal").diferenca_revisoes_ic95_sup.max()
    favorable = [str(int(t)) for t, upper in robust.items() if upper < 0]
    inside = faixa.loc[faixa.fila.eq("F4")]
    survives = [str(int(r.meta)) for r in inside.itertuples() if r.diferenca_O1 < 0 and r.diferenca_O3 < 0]
    random = topo["desempate_aleatorio"]["metricas"]
    tie_reading = (
        "A contagem do topo não muda nos quantis de sensibilidade avaliados"
        if random["par2_top17"]["p025"] == random["par2_top17"]["p975"] == 16
        else "A contagem de 16 no topo é sensível ao desempate, não devendo ser tratada como invariável"
    )
    lines += ["", "## Topo e empates", "", "```json", json.dumps(topo, indent=2, ensure_ascii=False), "```", "",
              "## Leitura em cinco linhas", "",
              "1. F4 tem IC95% inteiramente favorável contra ambas as bases nas metas: " + (", ".join(favorable) or "nenhuma") + "; não se presume vantagem uniforme.",
              "2. Dentro da faixa, F4 exige menos revisões que ambas nas metas: " + (", ".join(survives) or "nenhuma") + "; persistência indica ordenação interna, sem eliminar o viés de verificação.",
              f"3. No topo original há 16 PAR=2 em 17; com desempate aleatório a mediana é {fmt(random['par2_top17']['mediana'])}, intervalo de sensibilidade [{fmt(random['par2_top17']['p025'])}; {fmt(random['par2_top17']['p975'])}].",
              "4. " + tie_reading + "; o desempate efetivo é pelo hash lexical, não pela linha do extrato, sem prova de ausência de viés.",
              "5. O bootstrap é condicional a probabilidades OOF e rótulos fixos, não mede incerteza de treino, revisão ou seleção post hoc; a hipótese registrada permanece inalterada.", "",
              "Metadados e SHA-256: data/fusao_robustez_metadata.json. Nenhuma saída contém identificador ou valor individual."]
    path = ROOT / "docs/plans/RELATORIO_FUSAO_ROBUSTEZ_2026-11-07.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(data_root):
    import sklearn
    import xgboost

    started = time.perf_counter()
    print("Estimativa preliminar: fits importados ~2 min segundo execução anterior; bootstrap estimado por 10 pilotos antes das 2.000 réplicas.", flush=True)
    queues, labels, data, fit_estimate = load_queues(data_root)
    reproduction = verify_reference(queues, labels)
    print("Reprodução de todas as métricas O1/O3/F3/F4 confirmada.", flush=True)
    below = data["below"]
    aligned = np.argsort(data["ids"][below])
    scores = numeric(data["raw"].SCORE).to_numpy()[below][aligned]
    ml_scores = data["original"].p_cal_ml.to_numpy()[below][aligned]
    notes = numeric(data["raw"]["nota final"]).to_numpy()[below][aligned]
    group_codes, groups = pd.factorize(data["groups"][below][aligned], sort=True)
    rng = np.random.default_rng(SEED)
    pilot_start = time.perf_counter()
    runs = [bootstrap_once(queues, labels, group_codes, len(groups), rng) for _ in range(10)]
    pilot_seconds = time.perf_counter() - pilot_start
    estimate = pilot_seconds / 10 * 2000 * 1.5 / 60
    repetitions = 500 if estimate > 30 else 2000
    print(f"Bootstrap: piloto {pilot_seconds:.3f}s/10; estimativa conservadora {estimate:.2f} min; {repetitions} réplicas.", flush=True)
    for i in range(10, repetitions):
        runs.append(bootstrap_once(queues, labels, group_codes, len(groups), rng))
        if (i + 1) % 250 == 0:
            print(f"Bootstrap {i + 1}/{repetitions}", flush=True)
    valid = [r for r in runs if r is not None]
    if not valid:
        raise ValueError("Nenhuma réplica com PAR=2")
    table = summarize_bootstrap(valid)
    faixa = faixa_table(queues, labels, scores)
    topo = top_audit(queues, labels, scores, ml_scores, notes, data["raw"], below, aligned)
    out = ROOT / "data"
    table.to_csv(out / "fusao_robustez_bootstrap.csv", index=False, lineterminator="\n")
    faixa.to_csv(out / "fusao_robustez_faixa.csv", index=False, lineterminator="\n")
    write_json(out / "fusao_robustez_topo.json", topo)
    code_names = ("scripts/fusao_robustez.py", "tests/test_fusao_robustez.py", "scripts/fusao_filas.py", "scripts/sensibilidade_o3_macd.py", "gzcmd/ml_oof.py")
    input_paths = {"data/COMPARADORSEMIDENT.csv": data["source"],
                   **{name: Path(data_root) / name for name in ("data/pcal_ml_oof.csv", "data/pcal_ml_oof.json")},
                   **{name: ROOT / name for name in ("data/fusao_filas.csv", "data/fusao_filas_metadata.json")}}
    metadata = {
        "analysis": "exploratory_post_hoc", "cpu_only": True, "llm_calls": 0,
        "date_utc": datetime.now(UTC).isoformat(), "seed": SEED,
        "rng": "numpy.default_rng PCG64", "replicas": repetitions,
        "replicas_sem_PAR2": repetitions - len(valid), "grupos_REFREC": len(groups),
        "below_cutoff": len(labels), "PAR2": int(labels.sum()),
        "faixa": {
            "rule": "round(SCORE, 3) >= -5.648; includes entire boundary tie block",
            "literal_threshold": -5.648, "literal_n": int((scores >= -5.648).sum()),
            "literal_PAR2": int(labels[scores >= -5.648].sum()),
            "operational_n": int(reviewable_mask(scores).sum()),
            "operational_PAR2": int(labels[reviewable_mask(scores)].sum()),
            "operational_min_score": float(scores[reviewable_mask(scores)].min()),
            "limitation": "Dispatch threshold was rounded: literal threshold does not reproduce 2522/36",
        },
        "PAR2_por_replica": distribution([r[1] for r in valid]),
        "replicas_menos_36": sum(r[1] < 36 for r in valid),
        "pares_por_replica": distribution([r[2] for r in valid]),
        "targets": list(TARGETS), "fractions_if_less_36": list(FRACTIONS),
        "targets_rule": "fixed 9/23/36 if total>=36; otherwise ceil(25%/64%/100% of total); zero-positive replicates excluded",
        "bootstrap": "REFREC sampled with replacement among below-cutoff groups; multiplicities expanded, all arms reranked before imported RRF; no model refit",
        "confidence_interval": "percentile 2.5/97.5, linear quantiles; conditional on fixed OOF scores and labels",
        "baseline_reproduction": reproduction, "fit_estimate": fit_estimate,
        "pilot_seconds_10": pilot_seconds, "estimated_bootstrap_2000_minutes_safety_1_5": estimate,
        "reduction_rule": "500 if estimated 2000-replicate bootstrap exceeds 30 minutes",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "input_sha256": {n: sha256(p) for n, p in input_paths.items()},
        "code_sha256": {n: sha256(ROOT / n) for n in code_names},
        "output_sha256": {n: sha256(out / n) for n in ("fusao_robustez_bootstrap.csv", "fusao_robustez_faixa.csv", "fusao_robustez_topo.json")},
        "elapsed_seconds": time.perf_counter() - started,
        "command": " ".join([sys.executable, *sys.argv]),
    }
    write_json(out / "fusao_robustez_metadata.json", metadata)
    write_report(table, faixa, topo, metadata)
    print(table.to_string(index=False), flush=True)
    print(faixa.to_string(index=False), flush=True)
    print(f"Concluído em {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    generate(parser.parse_args().data_root)
