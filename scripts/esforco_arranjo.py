"""Esforço humano hipotético de P e O3, a partir dos agregados de mesma carga."""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ratio(value, denominator):
    return value / denominator if denominator else None


def generate(root=ROOT, output=None):
    parameters = json.loads((root / "data/esforco_arranjo_parametros.json").read_text(encoding="utf-8"))
    counts = pd.read_csv(root / "data/carga_abaixo_corte.csv")
    hours_source = pd.read_csv(root / "data/carga_abaixo_corte_horas.csv")
    count_row = counts.loc[counts.limiar.eq("por_passo") & counts.passo.eq("TOTAL")]
    hours_row = hours_source.loc[hours_source.limiar.eq("por_passo")]
    if len(count_row) != 1 or len(hours_row) != 1:
        raise ValueError("Cenário por passo ausente ou ambíguo nos CSVs")
    reference = int(count_row.iloc[0].pares_acima_limiar)
    minutes = float(hours_row.iloc[0].minutos_por_par)
    if int(hours_row.iloc[0].pares) != reference:
        raise ValueError("CSV de horas diverge da contagem de pares")
    if minutes <= 0 or reference <= 0 or parameters["cenario"] is not True:
        raise ValueError("Hipóteses explícitas de esforço inválidas")
    same_load = pd.read_csv(root / "data/comparador_mesma_carga.csv")
    selected = same_load.loc[same_load.ordenacao.eq("O3_p_cal_ml")
                             & same_load.carga.isin(["K1_humana", "K2_total"])]
    if len(selected) != 4 or selected.duplicated(["modo", "carga"]).any():
        raise ValueError("Esperadas duas cargas por modo para O3")
    rows = []
    for row in selected.itertuples():
        for arm in ("P", "O3"):
            reviews = int(row.P_pares_humano if arm == "P" else row.k)
            recovered = int(row.P_par2_recuperados if arm == "P" else row.par2_topk)
            recovered05 = float(row.P_par2_aceitos + .95 * row.P_par2_humano if arm == "P"
                                else .95 * row.par2_topk)
            fp = int(row.P_fp if arm == "P" else row.comparador_fp_revisor_perfeito)
            hours = reviews * minutes / 60
            saved_reviews = reference - reviews
            saved_hours = saved_reviews * minutes / 60
            lost = 36 - recovered
            rows.append({"modo": row.modo, "braco": arm, "carga": row.carga,
                         "orcamento_avaliacoes": int(row.k), "revisoes": reviews,
                         "pares_llm": int(row.P_pares_llm) if arm == "P" else 0,
                         "minutos_por_par": minutes, "horas": hours,
                         "par2_recuperados": recovered, "par2_nao_recuperados": lost, "FP": fp,
                         "par2_recuperados_erro05": recovered05,
                         "revisoes_referencia": reference, "revisoes_poupadas": saved_reviews,
                         "horas_poupadas": saved_hours, "horas_por_vinculo_perdido": ratio(hours, lost),
                         "horas_poupadas_por_vinculo_perdido": ratio(saved_hours, lost),
                         "horas_por_fp": ratio(hours, fp), "horas_poupadas_por_fp": ratio(saved_hours, fp),
                         "cenario": True, "exploratorio_post_hoc": True, "horas_medidas": False})
    table = pd.DataFrame(rows)
    table.to_csv(output or root / "data/esforco_arranjo.csv", index=False, lineterminator="\n")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(generate(args.root).to_string(index=False))
