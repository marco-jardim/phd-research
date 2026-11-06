"""Perda aplicada às medianas marginais existentes, não mediana da perda.

Sem treino, amostragem ou LLM. Custos por modo da política histórica.
"""
from pathlib import Path
import yaml

from scripts.banca_c_agregados import MODES, one, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]


def compute(data, config):
    source = read_csv(data / "robustez_arranjo_resumo.csv")
    rows = []
    for mode in MODES:
        costs = config["decision_policy"]["modes"][mode]["costs"]
        for block in ("N", "P"):
            paired_medians = {metric: float(one(source,
                tipo="diferenca_P_menos_Platt", modo=mode, bloco=block,
                faixa="TODAS", metrica=metric)["mediana"])
                for metric in ("fp", "perdidos", "pares_humano", "pares_llm")}
            weighted_paired_medians = sum(paired_medians[metric] * float(costs[cost])
                for metric, cost in (("fp", "false_positive"), ("perdidos", "false_negative"),
                                    ("pares_humano", "human_review"), ("pares_llm", "llm_review")))
            arms = []
            for arm in ("P", "P-Platt"):
                values = {metric: float(one(source, tipo="desfecho", modo=mode,
                          bloco=block, braco=arm, faixa="TODAS", metrica=metric)["mediana"])
                          for metric in ("fp", "perdidos", "pares_humano", "pares_llm")}
                loss = sum(values[metric] * float(costs[cost]) for metric, cost in (
                    ("fp", "false_positive"), ("perdidos", "false_negative"),
                    ("pares_humano", "human_review"), ("pares_llm", "llm_review")))
                stability = {}
                for metric in ("delta_p_max", "jaccard_llm"):
                    row = one(source, tipo="estabilidade", modo=mode, bloco=block,
                              braco=arm, metrica=metric)
                    for stat in ("mediana", "minimo", "maximo"):
                        stability[f"{metric}_{stat}"] = float(row[stat])
                arms.append(dict(modo=mode, bloco=block, braco=arm, **values,
                                 C_FP=costs["false_positive"], C_FN=costs["false_negative"],
                                 C_H=costs["human_review"], C_LLM=costs["llm_review"],
                                 perda_das_medianas=round(loss, 6), **stability))
            difference = round(arms[0]["perda_das_medianas"] - arms[1]["perda_das_medianas"], 6)
            for row in arms:
                row["diferenca_P_menos_Platt"] = difference
                row["soma_ponderada_medianas_diferencas_pareadas"] = round(weighted_paired_medians, 6)
                row["estimando"] = "perda_das_medianas_marginais_nao_mediana_da_perda"
                row["fonte"] = "robustez_arranjo_resumo.csv;gzcmd/gzcmd_v3_config.yaml"
                rows.append(row)
    return rows


def main():
    config = yaml.safe_load((ROOT / "gzcmd/gzcmd_v3_config.yaml").read_text(encoding="utf-8"))
    rows = compute(ROOT / "data", config)
    write_csv(ROOT / "data/robustez_perda_ponderada.csv", rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
