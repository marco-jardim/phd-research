"""Comparação exploratória com todo MATCH do LLM de P revisto por humano.

Somente agregados versionados; não treina, não lê dados privados e não chama
LLMs. Revisor perfeito é hipótese em ambos os lados, não resultado observado.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("comparador_mesma_carga.csv", "comparador_mesma_carga_curva.csv",
           "arranjo_corte_vpp_llm.csv", "arranjo_corte_par2.csv")


def unique(frame, message):
    if len(frame) != 1:
        raise ValueError(message)
    return frame.iloc[0]


def generate(root=ROOT, output=None):
    output = output or root / "data"
    output.mkdir(parents=True, exist_ok=True)
    same, curve, precision, primary = [pd.read_csv(root / "data" / name) for name in SOURCES]
    o3 = curve.loc[curve.ordenacao.eq("O3_p_cal_ml")].sort_values("k")
    if o3.empty or o3.k.duplicated().any():
        raise ValueError("Curva O3 ausente ou com orçamento duplicado")
    rows = []
    for mode in ("vigilancia", "confirmacao"):
        p = unique(primary.loc[primary.modo.eq(mode) & primary.braco.eq("P")], "Partição P ausente ou ambígua")
        llm = unique(precision.loc[precision.modo.eq(mode) & precision.braco.eq("P")], "VPP P ausente ou ambíguo")
        load = unique(same.loc[same.modo.eq(mode) & same.ordenacao.eq("O3_p_cal_ml")
                               & same.carga.eq("K1_humana")], "Carga humana P ausente ou ambígua")
        if (p.aceitos != llm.VP or llm.VP + llm.FP != llm.aceitacoes_llm_abaixo
                or load.P_fp != llm.FP):
            raise ValueError("Aceitações de P não são integralmente cobertas pelo MATCH do LLM")
        recovered = int(p.aceitos + p.humano)
        if recovered != load.P_par2_recuperados:
            raise ValueError("Recuperação de P diverge entre agregados")
        reviews = int(load.P_pares_humano + llm.aceitacoes_llm_abaixo)
        matched = unique(o3.loc[o3.k.eq(reviews)], "Curva não contém a mesma carga de P")
        equivalent = o3.loc[o3.par2_topk.ge(recovered)]
        if equivalent.empty:
            raise ValueError("Curva não alcança a recuperação de P")
        minimum = int(equivalent.iloc[0].k)
        if minimum != load.k_min_equivalente_P:
            raise ValueError("k mínimo diverge entre curva e agregado de mesma carga")
        rows.append({"modo": mode, "P_revisoes_humanas_originais": int(load.P_pares_humano),
                     "P_MATCH_llm_ao_humano": int(llm.aceitacoes_llm_abaixo),
                     "P_revisoes_humanas_total": reviews, "P_par2_recuperados": recovered,
                     "P_FP_revisor_perfeito": 0, "O3_k_mesma_carga": int(matched.k),
                     "O3_par2_mesma_carga": int(matched.par2_topk), "O3_FP_revisor_perfeito": 0,
                     "O3_k_min_igualar_P": minimum,
                     "O3_par2_no_k_min": int(equivalent.iloc[0].par2_topk),
                     "diferenca_P_O3_PAR2_mesma_carga": recovered - int(matched.par2_topk),
                     "diferenca_O3_k_min_P_revisoes": minimum - reviews,
                     "exploratorio_post_hoc": True, "revisor_perfeito_ambos": True})
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparador_p_match_humano.csv", index=False, lineterminator="\n")
    # Materializa a projeção usada na coluna editorial: cálculo só no gerador,
    # não no renderer. É perda hipotética de VP humanos, sem modelo de FP humanos.
    projections = same[["modo", "ordenacao", "carga", "k", "par2_topk"]].copy()
    projections["erro_humano_fn"] = .05
    projections["par2_topk_erro05"] = projections.par2_topk * (1 - projections.erro_humano_fn)
    projections["modela_FP_humano"] = False
    projections.to_csv(output / "comparador_mesma_carga_erro05.csv", index=False, lineterminator="\n")
    metadata = {"analysis": "exploratory_post_hoc", "registered": False,
                "sources": {"data/" + name: hashlib.sha256((root / "data" / name).read_bytes()).hexdigest()
                            for name in SOURCES},
                "perfect_reviewer": "Hypothesis on BOTH sides: all P human referrals and LLM MATCH pairs, and all O3 top-k pairs, are correctly reviewed. Not observed human outcomes.",
                "FP_perfect": "Zero conditional on perfect review in both arms; automatic MATCH of the LLM below cutoff is no longer accepted without human review.",
                "error05": "Sensitivity scenario only: 5% false negatives among true links sent to humans. Does not model human false positives. It does not assert zero human FP under imperfect review.",
                "interpretation": "At the same workload, report both PAR=2 counts; minimum O3 k is a separate comparison, not a claim of exact equality at all workloads.",
                "api_calls": 0, "model_fits": 0}
    (output / "comparador_p_match_humano_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(generate(args.root).to_string(index=False))
