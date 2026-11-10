"""Agregados editoriais da fatia C, sem API, treinamento ou dados por par.

Uso: python scripts/banca_c_agregados.py [--qw2-dir CAMINHO_PRIVADO]
Sem o argumento, regenera custos e tabelas apenas dos agregados versionados.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODES = ("vigilancia", "confirmacao")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def one(rows, **criteria):
    selected = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    if len(selected) != 1:
        raise ValueError(f"Esperada uma linha: {criteria}, obtidas {len(selected)}")
    return selected[0]


def equivalent_costs(data, config):
    """Custos por modo, em recuperação equivalente e na mesma carga humana.

    O campo legado custo_fn_comum só é comum quando fn_igual_a_P é verdadeiro.
    Nos outros cenários inclui as perdas adicionais, sem escondê-las no subtotal.
    """
    arrangement = read_csv(data / "arranjo_corte_metricas.csv")
    rankings = read_csv(data / "comparador_mesma_carga.csv")
    human = read_csv(data / "comparador_p_match_humano.csv")
    output = []
    for mode in MODES:
        p = one(arrangement, modo=mode, braco="P", faixa="TODAS")
        o3 = one(rankings, modo=mode, ordenacao="O3_p_cal_ml",
                 carga="k_min_equivalente_P")
        match = one(human, modo=mode)
        recovered = int(match["P_par2_recuperados"])
        if recovered != int(o3["par2_topk"]):
            raise ValueError("A comparação exige recuperação igual")
        if int(p["pares_humano"]) != int(match["P_revisoes_humanas_originais"]):
            raise ValueError("Carga humana de P divergente")
        costs = config["decision_policy"]["modes"][mode]["costs"]
        lost = int(p["perdidos"])
        c_h, c_l, c_fp, c_fn = (float(costs[k]) for k in
                              ("human_review", "llm_review", "false_positive",
                               "false_negative"))
        scenarios = [
            ("P", int(p["pares_humano"]), int(p["pares_llm"]), int(p["fp"]), recovered),
            ("O3_recuperacao_equivalente", int(o3["k"]), 0,
             int(o3["comparador_fp_revisor_perfeito"]), recovered),
            ("P_MATCH_ao_humano", int(match["P_revisoes_humanas_total"]),
             int(p["pares_llm"]), int(match["P_FP_revisor_perfeito"]), recovered),
        ]
        for ranking in ("O2_CdR", "O3_p_cal_ml"):
            for load, suffix in (("K1_humana", "mesma_carga_humana_P"),
                                 ("k_min_equivalente_P", "recuperacao_equivalente")):
                if ranking == "O3_p_cal_ml" and load == "k_min_equivalente_P":
                    continue  # Linha O3 histórica já presente.
                row = one(rankings, modo=mode, ordenacao=ranking, carga=load)
                scenarios.append((ranking.split("_")[0] + "_" + suffix,
                                  int(row["k"]), 0,
                                  int(row["comparador_fp_revisor_perfeito"]),
                                  int(row["par2_topk"])))
        for arm, h, llm, fp, scenario_recovered in scenarios:
            scenario_lost = lost + recovered - scenario_recovered
            human_cost, llm_cost, fp_cost = h * c_h, llm * c_l, fp * c_fp
            subtotal = human_cost + llm_cost + fp_cost
            output.append(dict(
                modo=mode, braco=arm, par2_recuperados=scenario_recovered,
                par2_perdidos=scenario_lost, revisoes_humanas=h, pares_llm=llm, fp=fp,
                C_H=c_h, C_LLM=c_l, C_FP=c_fp, C_FN=c_fn,
                custo_humano=human_cost, custo_llm=llm_cost, custo_fp=fp_cost,
                subtotal_revisao_fp=round(subtotal, 6),
                custo_fn_comum=scenario_lost * c_fn,
                fn_igual_a_P=scenario_lost == lost,
                perda_total=round(subtotal + scenario_lost * c_fn, 6),
                unidade="revisoes_humanas_equivalentes",
                exploratorio_post_hoc=True, revisor_perfeito=True,
                fontes="arranjo_corte_metricas.csv;comparador_mesma_carga.csv;"
                       "comparador_p_match_humano.csv;gzcmd/gzcmd_v3_config.yaml",
            ))
    return output


def aggregate_qw2(source):
    """Lista fechada de campos; nunca copiar pair_id, respostas ou texto livre."""
    rows = []
    for mode in MODES:
        path = source / f"qw2_{mode}_kimi.json"
        raw = path.read_bytes()
        doc = json.loads(raw)
        metrics = doc["metrics"]
        tp, fp, fn, tn = (int(metrics[k]) for k in
                          ("gt_tp", "gt_fp", "gt_fn", "gt_tn"))
        n = int(doc["n_reviewed"])
        consensus = int(metrics["protocols"]["consensus"])
        arbiter = int(metrics["protocols"]["arbiter"])
        if doc["mode"] != mode or tp + fp + fn + tn != n:
            raise ValueError("Modo ou matriz QW2 inconsistente")
        if consensus + arbiter != n or int(doc["n_errors"]) != 0:
            raise ValueError("Protocolo QW2 incompleto")
        if metrics["decisions"] != {"MATCH": tp + fp, "NONMATCH": fn + tn}:
            raise ValueError("Decisões QW2 não reconciliadas com a matriz")
        rows.append(dict(
            modo=mode, fonte=f"data/qw2/{path.name}",
            fonte_sha256=hashlib.sha256(raw).hexdigest(),
            timestamp=doc["timestamp"], modelo=doc["model"], seed=int(doc["seed"]),
            n=n, vp=tp, fp=fp, fn=fn, vn=tn, consenso=consensus, arbitro=arbiter,
        ))
    return rows


def render_qw2(rows):
    names = {"vigilancia": r"Vigil\^ancia", "confirmacao": r"Confirma\c{c}\~ao"}
    matrices, metrics = [], []

    def integer(value):
        return f"{int(value):,}".replace(",", ".")

    def decimal(value):
        return f"{value:.3f}".replace(".", ",")

    for index, row in enumerate(rows):
        tp, fp, fn, tn, n, consensus, arbiter = (
            int(row[k]) for k in ("vp", "fp", "fn", "vn", "n", "consenso", "arbitro"))
        mode = names[row["modo"]]
        matrices.extend([
            f"{mode} & Par verdadeiro & {integer(tp)} & {integer(fn)} & {integer(tp + fn)}"
            f" & consenso: {integer(consensus)}; \\'arbitro: {integer(arbiter)} " + r"\\",
            f"{mode} & N\\~ao par & {integer(fp)} & {integer(tn)} & {integer(fp + tn)}"
            # Sem filete duplo: o \midrule só separa os modos; o último segue direto ao \bottomrule.
            + (r" & -- \\ \midrule" if index < len(rows) - 1 else r" & -- \\"),
        ])
        values = (tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn),
                  tn / (tn + fp), (tp + tn) / n)
        metrics.append(" & ".join([mode, integer(n)] + [decimal(v) for v in values])
                       + r" \\")
    note = (
        r"\par\smallskip\parbox{\linewidth}{\footnotesize Fonte: "
        r"\path{data/qw2_kimi_agregado.csv}, gerado de JSONs privados por "
        r"\path{scripts/banca_c_agregados.py}, com hash de cada fonte e sem dados por par. "
        r"A vigilância é execução anterior, distinta da matriz reconstruída da "
        r"Tabela~\ref{tab:cap8-revisao-desempenho}. Comparações entre revisores: "
        r"Tabela~\ref{tab:cap8-reruns}.}"
    )
    def table(caption, label, spec, header, lines):
        return ("% Gerada por scripts/banca_c_agregados.py; não editar manualmente.\n"
                "\\begin{table}[htbp]\n\\centering\n\\small\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
                f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header}\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
                + note + "\n\\end{table}\n")
    return {
        "tab_qw2_confusion.tex": table(
            "Matrizes de confusão da execução anterior de Kimi K2.5, por modo "
            "de operação (15 de fevereiro de 2026).", "tab:qw2-confusion", "llrrrl",
            r"Modo & Classe real & MATCH & NONMATCH & Total & Protocolo \\", matrices),
        "tab_qw2_llm_comparison.tex": table(
            "Desempenho de Kimi K2.5 na execução anterior da revisão assistida "
            "de pares residuais, por modo de operação.", "tab:qw2-llm-comparison", "lrrrrrr",
            r"Modo & $n$ & VPP & Sensibilidade & $F_1$ & Especif. & Acur\'acia \\",
            metrics),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qw2-dir", type=Path)
    args = parser.parse_args()
    data = ROOT / "data"
    config = yaml.safe_load((ROOT / "gzcmd/gzcmd_v3_config.yaml").read_text(encoding="utf-8"))
    write_csv(data / "custos_equivalentes_exploratorios.csv", equivalent_costs(data, config))
    aggregate = data / "qw2_kimi_agregado.csv"
    if args.qw2_dir:
        write_csv(aggregate, aggregate_qw2(args.qw2_dir))
    editorial = ROOT / "text/latex/tables/editorial"
    for name, text in render_qw2(read_csv(aggregate)).items():
        (editorial / name).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
