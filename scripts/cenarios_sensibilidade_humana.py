"""Cenários agregados de sensibilidade humana e conferência offline do filtro.

Sem treinamento ou chamadas de LLM. A geração dos cenários usa exclusivamente
CSVs versionados. A conferência opcional lê respostas privadas já existentes,
valida suas origens pelo coletor histórico e grava somente contagens.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = ("arranjo_corte_metricas.csv", "comparador_p_match_humano.csv",
           "comparador_mesma_carga_curva.csv", "filtro_llm.csv")
PARAMS = "cenarios_sensibilidade_humana_parametros.json"


def one(frame, message):
    if len(frame) != 1:
        raise ValueError(message)
    return frame.iloc[0]


def compare_order(table):
    """Compare strict order and ties separately, without claiming dominance."""
    rows = []
    for mode, subset in table.groupby("modo", sort=False):
        groups = list(subset.groupby(["estrategia", "k"], sort=False))
        for ((arm_a, k_a), a), ((arm_b, k_b), b) in combinations(groups, 2):
            values = a.set_index("s").sensibilidade_total.subtract(b.set_index("s").sensibilidade_total)
            signs = [0 if abs(v) < 1e-12 else 1 if v > 0 else -1 for v in values]
            rows.append({"modo": mode, "estrategia_a": arm_a, "k_a": k_a,
                         "estrategia_b": arm_b, "k_b": k_b,
                         "mesma_carga_humana": int(a.iloc[0].revisoes_humanas) == int(b.iloc[0].revisoes_humanas),
                         "inverte_ordem_estrita": -1 in signs and 1 in signs,
                         "empate_varia": 0 in signs and len(set(signs)) > 1,
                         "diferenca_s_min": float(values.loc[min(values.index)]),
                         "diferenca_s_max": float(values.loc[max(values.index)])})
    return pd.DataFrame(rows)


def generate(root=ROOT, output=None):
    output = output or root / "data"
    output.mkdir(parents=True, exist_ok=True)
    parameters = json.loads((root / "data" / PARAMS).read_text(encoding="utf-8"))
    levels = sorted(parameters["sensibilidades_humanas"])
    if len(set(levels)) != len(levels) or not levels or not all(0 <= s <= 1 for s in levels):
        raise ValueError("Sensibilidades de cenário inválidas")
    metrics, match_human, curve, filtered = [pd.read_csv(root / "data" / n) for n in SOURCES]
    configurations = []
    for mode in match_human.modo:
        full = metrics.loc[metrics.modo.eq(mode) & metrics.faixa.eq("TODAS")]
        cut = one(full.loc[full.braco.eq("corte_sozinho")], "Corte ausente ou ambíguo")
        integral = one(full.loc[full.braco.eq("humano_integral")], "Revisão integral ausente ou ambígua")
        p = one(full.loc[full.braco.eq("P")], "P ausente ou ambíguo")
        match = one(match_human.loc[match_human.modo.eq(mode)], "P com MATCH humano ausente ou ambíguo")
        denominator, above = int(cut.vinculos), int(cut.aceitos)
        below_links = denominator - above

        def add(arm, k, human_pairs, accepted, human_links):
            accepted, human_links = int(accepted), int(human_links)
            coverage_loss = denominator - accepted - human_links
            if min(coverage_loss, human_links, accepted) < 0 or human_links > human_pairs:
                raise ValueError("Partição de vínculos ou carga inválida")
            configurations.append({"modo": mode, "estrategia": arm, "k": int(k),
                                   "revisoes_humanas": int(human_pairs), "vinculos_total": denominator,
                                   "vinculos_aceitos_sem_humano": accepted, "vinculos_ao_humano": human_links,
                                   "vinculos_abaixo_corte": below_links,
                                   "vinculos_perdidos_cobertura": coverage_loss})

        add("humano_integral", integral.pares_humano, integral.pares_humano, integral.aceitos, integral.vinc_humano)
        add("P", p.pares_humano, p.pares_humano, p.aceitos, p.vinc_humano)
        add("P_MATCH_ao_humano", match.P_revisoes_humanas_total, match.P_revisoes_humanas_total,
            above, match.P_par2_recuperados)
        o3 = curve.loc[curve.ordenacao.eq("O3_p_cal_ml")]
        for k in parameters["O3_revisoes"]:
            ranked = one(o3.loc[o3.k.eq(k)], "Ponto O3 ausente ou ambíguo")
            add("O3", ranked.k, ranked.k, above, ranked.par2_topk)
        selected = filtered.loc[filtered.modo.eq(mode) & filtered.k.between(
            parameters["filtro_k_min"], parameters["filtro_k_max"])]
        if selected.empty or selected.k.duplicated().any():
            raise ValueError("Pontos do filtro ausentes ou duplicados")
        for row in selected.itertuples():
            add("filtro_LLM", row.k, row.H, above, row.par2_recuperados)
    rows, amplitudes = [], []
    for config in configurations:
        sensitivity = lambda s: (config["vinculos_aceitos_sem_humano"] + s * config["vinculos_ao_humano"]) / config["vinculos_total"]
        amplitude = sensitivity(max(levels)) - sensitivity(min(levels))
        amplitudes.append({**config, "s_min": min(levels), "s_max": max(levels),
                           "sensibilidade_min": sensitivity(min(levels)), "sensibilidade_max": sensitivity(max(levels)),
                           "amplitude_pp": amplitude * 100})
        for s in levels:
            rows.append({**config, "s": s, "sensibilidade_total": sensitivity(s),
                         "amplitude_pp": amplitude * 100,
                         "especificidade_humana_assumida": parameters["especificidade_humana_assumida"],
                         "fonte_s": parameters["fonte_s"]})
    table, amplitude = pd.DataFrame(rows), pd.DataFrame(amplitudes)
    order = compare_order(table)
    for name, frame in (("cenarios_sensibilidade_humana", table),
                        ("cenarios_sensibilidade_humana_amplitude", amplitude),
                        ("cenarios_sensibilidade_humana_ordem", order)):
        frame.to_csv(output / (name + ".csv"), index=False, lineterminator="\n")
    metadata = {"analysis": "exploratory_sensitivity_scenarios", "registered": False,
                "human_sensitivity_measured": False, "human_specificity_assumed": parameters["especificidade_humana_assumida"],
                "human_FP_modeled": False, "formula": "(accepted_without_human + s * links_sent_to_human) / total_links",
                "automatic_FP_note": "Human specificity is assumed perfect, not automatic-decision specificity. P's observed automatic false positives are not changed by varying human sensitivity.",
                "inputs_sha256": {"data/" + n: hashlib.sha256((root / "data" / n).read_bytes()).hexdigest() for n in (*SOURCES, PARAMS)},
                "max_amplitude_pp": float(amplitude.amplitude_pp.max()),
                "max_amplitude_excluding_integral_pp": float(amplitude.loc[amplitude.estrategia.ne("humano_integral"), "amplitude_pp"].max()),
                "strict_reversals": int(order.inverte_ordem_estrita.sum()), "ties_change": int(order.empate_varia.sum()),
                "same_human_workload_strict_reversals": int(order.loc[order.mesma_carga_humana, "inverte_ordem_estrita"].sum()),
                "same_human_workload_ties_change": int(order.loc[order.mesma_carga_humana, "empate_varia"].sum()),
                "ordering_note": "Strict reversals and changing ties are reported separately. Comparisons across unequal human workloads are not resource-adjusted dominance claims.",
                "filter_note": "Only versioned k points in the requested interval are used; no interpolation. k is the LLM queue size, not the number H of human reviews.",
                "api_calls": 0, "model_fits": 0}
    (output / "cenarios_sensibilidade_humana_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return table, amplitude, order


def aggregate_agreement(frame, ks):
    """Private rows in, count-only allowlisted aggregate out."""
    rows = []
    for k in ks:
        top = frame.iloc[:k]
        if len(top) != k:
            raise ValueError("Fila menor que a carga")
        for label, subset in (("TODOS", top), ("PAR2", top.loc[top.par.eq(2)]),
                              ("nao_vinculo", top.loc[top.par.eq(0)])):
            cells = {a + "_" + b: int((subset.vigilancia.eq(a) & subset.confirmacao.eq(b)).sum())
                     for a in ("MATCH", "NONMATCH", "UNSURE") for b in ("MATCH", "NONMATCH", "UNSURE")}
            if sum(cells.values()) != len(subset):
                raise ValueError("Decisão ausente ou desconhecida")
            same = int(subset.vigilancia.eq(subset.confirmacao).sum())
            rows.append({"k": k, "classe": label, "pares": len(subset), "coincidentes": same,
                         "diferentes": len(subset) - same, **cells})
    return pd.DataFrame(rows)


def audit_filter_modes(data_root, root=ROOT, output=None):
    from gzcmd.ml_oof import pair_ids
    from scripts.filtro_llm import collect, resolve

    output = output or root / "data"
    output.mkdir(parents=True, exist_ok=True)
    filtered = pd.read_csv(root / "data/filtro_llm.csv")
    params = json.loads((root / "data" / PARAMS).read_text(encoding="utf-8"))
    ks = sorted(filtered.loc[filtered.k.between(params["filtro_k_min"], params["filtro_k_max"]), "k"].unique())
    top, decisions, _, provenance = collect(data_root)
    top = top[:max(ks)]  # collect constructs ONE mode-independent O3 queue.
    historical = json.loads((root / "data/filtro_llm_metadata.json").read_text(encoding="utf-8"))
    sources, resolved = {}, {}
    for mode, audit in provenance.items():
        recorded = {s["source"]: s["sha256"] for s in historical["response_audit"][mode]["sources"]}
        current = {s["source"]: s["sha256"] for s in audit["sources"]}
        if recorded != current:
            raise ValueError("STOP: fontes de respostas divergem dos metadados históricos do filtro")
        sources[mode], resolved[mode] = [], set()
        for source in audit["sources"]:
            path = resolve(data_root, source["source"])
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("mode") != mode or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("STOP: origem ou modo de resposta divergente; não corrigir automaticamente")
            resolved[mode].add(path.resolve())
            sources[mode].append({"source": source["source"], "declared_mode": run["mode"], "sha256": source["sha256"]})
    if resolved["vigilancia"] & resolved["confirmacao"]:
        raise ValueError("STOP: mesmo arquivo de respostas nos dois modos")
    fields = {"PAR", "COMPREC", "REFREC", "PASSO"}
    raw = pd.read_csv(resolve(data_root, "data/COMPARADORSEMIDENT.csv"), sep=";", dtype=str,
                      usecols=lambda c: c.split(",")[0] in fields)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    labels = pd.Series(pd.to_numeric(raw.PAR, errors="raise").to_numpy(), index=pair_ids(raw))
    if not labels.index.is_unique:
        raise ValueError("Chaves privadas duplicadas")
    frame = pd.DataFrame({"par": labels.loc[top].to_numpy(),
                          "vigilancia": [decisions["vigilancia"][pid] for pid in top],
                          "confirmacao": [decisions["confirmacao"][pid] for pid in top]})
    counts = aggregate_agreement(frame, ks)
    counts.to_csv(output / "filtro_llm_concordancia_modos.csv", index=False, lineterminator="\n")
    for mode in ("vigilancia", "confirmacao"):
        for row in filtered.loc[filtered.modo.eq(mode) & filtered.k.isin(ks)].itertuples():
            part = frame.iloc[:row.k]
            if int(part[mode].ne("NONMATCH").sum()) != row.H or int((part.par.eq(2) & part[mode].ne("NONMATCH")).sum()) != row.par2_recuperados:
                raise ValueError("STOP: decisões não reproduzem os agregados do filtro")
    last = one(counts.loc[counts.k.eq(max(ks)) & counts.classe.eq("TODOS")], "Resumo da conferência ausente")
    positives = one(counts.loc[counts.k.eq(max(ks)) & counts.classe.eq("PAR2")], "Resumo de PAR2 ausente")
    metadata = {"analysis": "aggregate_offline_audit", "queue": "One common descending p_cal_ml queue with pair_id tie-break constructed outside the mode loop in the frozen collect function.",
                "queue_same_by_design": True, "top_pairs": len(top), "distinct_response_files": True,
                "declared_modes_validated": True, "sources": sources,
                "historical_response_hashes_reproduced": True,
                "coincident_top_decisions": int(last.coincidentes), "different_top_decisions": int(last.diferentes),
                "coincident_PAR2_decisions": int(positives.coincidentes), "different_PAR2_decisions": int(positives.diferentes),
                "explanation": f"Among top {len(top)}, {int(last.coincidentes)} decisions coincide and {int(last.diferentes)} differ; among PAR=2, {int(positives.coincidentes)} coincide and {int(positives.diferentes)} differ. Equal recovered-link totals do not imply shared response files. Cross-mode NONMATCH/UNSURE counts among nonlinks explain differences in human workload and VPN; opposite switches cancel in k=17 and k=29.",
                "historical_filter_metadata_edited": False, "api_calls": 0, "model_fits": 0,
                "note": "The frozen original filtro_llm_metadata.json is not edited. Pair IDs and per-pair decisions are never written by this audit."}
    (output / "filtro_llm_concordancia_modos_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--conferir-modos", action="store_true")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    _, amplitude, order = generate(args.root)
    print(amplitude.to_string(index=False))
    print("INVERSOES_ESTRITAS", int(order.inverte_ordem_estrita.sum()), "EMPATES_VARIAM", int(order.empate_varia.sum()))
    if args.conferir_modos:
        if args.data_root is None:
            parser.error("--conferir-modos exige --data-root")
        print(audit_filter_modes(args.data_root, args.root).to_string(index=False))
