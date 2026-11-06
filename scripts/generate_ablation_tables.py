"""Tabelas da ablação, preservando a distinção entre as duas origens."""

from pathlib import Path

import pandas as pd

LABELS = {
    "naive-threshold": "Decisão de limiar", "rules-only": "Somente regras",
    "ml-only": "Somente ML", "hybrid-and": "Híbrido AND", "hybrid-or": "Híbrido OR",
    "cascade-ml-rules": r"Cascata ML$\rightarrow$Regras",
    "cascade-rules-ml": r"Cascata Regras$\rightarrow$ML",
    "consensus-ml": "Consenso ML", "consensus-hybrid": "Consenso+Regras",
}


def pt(value):
    return f"{value:.3f}".replace(".", ",")


def config_tex(value):
    for old, new in [("ML-only", "Somente ML"), ("Rules-only", "Somente regras"),
                     ("Naive score", "Nota final"), ("Hybrid-AND", "Híbrido AND"),
                     ("Hybrid-OR", "Híbrido OR"), ("Consensus+Rules", "Consenso+Regras"),
                     ("Consensus-ML", "Consenso ML"), ("Consensus ML-majority", "Consenso ML"),
                     ("Cascade ML→Rules", "Cascata"), ("Cascade Rules→ML", "Cascata"),
                     ("Cascade", "Cascata"), ("Rules", "Regras"), ("majority", "maioria"),
                     ("th=", "limiar="), ("models", "modelos")]:
        value = value.replace(old, new)
    value = value.replace(">=", "≥").replace("≥", r"$\geq$")
    value = value.replace("->", "→").replace("→", r"$\rightarrow$")
    for old, new in [("0.3", "0,3"), ("0.5", "0,5"), ("0.6", "0,6"),
                     ("0.7", "0,7"), ("0.9", "0,9")]:
        value = value.replace(old, new)
    return value


def generate(abl: pd.DataFrame, out: Path):
    is_xgb = abl.config.str.contains("XGBoost", regex=False)
    original, modal = abl.loc[~is_xgb], abl.loc[is_xgb]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Configurações selecionadas por categoria de classificação no estudo de ablação, com resultados de origens experimentais distintas.}",
             r"\label{tab:ablation-best-category}", r"\footnotesize",
             r"\begin{tabular}{>{\raggedright\arraybackslash}p{3cm}>{\raggedright\arraybackslash}p{3.2cm}c>{\raggedright\arraybackslash}p{3.2cm}c}",
             r"\toprule",
             r"\multirow{2}{*}{Categoria} & \multicolumn{2}{c}{Ablação original} & \multicolumn{2}{c}{XGBoost} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
             r" & Configuração & F1 & Configuração & F1 \\", r"\midrule"]
    for category, label in LABELS.items():
        left = original.loc[original.category.eq(category)].sort_values("f1", ascending=False, kind="stable").iloc[0]
        subset = modal.loc[modal.category.eq(category)]
        right = left if subset.empty else subset.sort_values("f1", ascending=False, kind="stable").iloc[0]
        lines.append(f"{label} & {config_tex(left['config'])} & {pt(left.f1)} & "
                     f"{config_tex(right['config'])} & {pt(right.f1)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
              r"\par\small Nota: A ablação original contém 70 configurações; o fluxo Modal acrescentou 33 com XGBoost, que não correspondem ao teste interno QW1. "
              r"Decisão de limiar e somente regras são referências da ablação original, repetidas nas duas colunas. "
              r"A pontuação de regras do fluxo Modal foi corrigida e os agregados foram recomputados localmente com as probabilidades persistidas, sem novo treinamento. "
              r"A função de regras é agora a mesma da ablação original. Os consensos originais utilizam RF, RF+SMOTE e GB; os acrescentados utilizam XGBoost, RF+SMOTE e GB. "
              r"Os testes usam proporção de 30\% e semente 42. A seleção do maior $F_1$ nos testes é exploratória, não avaliação independente da escolha dos limiares. "
              r"Fonte: \path{data/ablation_results.csv}; auditoria: \path{data/qw1_modal_recompute_meta.json}.",
              r"\end{table}", ""]
    (out / "tab_ablation_best_category.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    top = abl.sort_values("f1", ascending=False, kind="stable").head(10)
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Dez configurações do estudo de ablação com maior F1, em ordenação exploratória nos respectivos testes.}",
             r"\label{tab:ablation-top10}", r"\small", r"\begin{tabular}{cp{8cm}ccc}",
             r"\toprule", r"\# & Configuração & Precisão & Sensibilidade & F1 \\", r"\midrule"]
    for i, row in enumerate(top.itertuples(index=False), 1):
        lines.append(f"{i} & {config_tex(row.config)} & {pt(row.precision)} & {pt(row.recall)} & {pt(row.f1)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
              r"\par\small Nota: As configurações identificadas por XGBoost provêm dos acréscimos do fluxo Modal, recomputados localmente após correção da pontuação de regras; as demais provêm da ablação original. "
              r"Os resultados não são os do teste interno QW1. A tabela ordena as 103 configurações por $F_1$, com desempate pela ordem do artefato, e apresenta as dez primeiras. "
              r"Os limiares foram explorados nos respectivos testes, sem avaliação independente de sua escolha. Fonte: \path{data/ablation_results.csv}.",
              r"\end{table}", ""]
    (out / "tab_ablation_top10.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")
