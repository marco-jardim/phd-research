"""Tabelas editoriais dos Estudos 1 a 3, sem alterar geradores congelados."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import gerar_tabelas_nb03_faixas as nb03

OUT = ROOT / "text/latex/tables/editorial"


def d(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def dominated(frame: pd.DataFrame, row: pd.Series) -> bool:
    return bool((
        frame.precision.ge(row.precision) & frame.recall.ge(row.recall)
        & (frame.precision.gt(row.precision) | frame.recall.gt(row.recall))
    ).any())


def tabela_pareto(destino: Path | None = None) -> None:
    grid = pd.read_csv(ROOT / "data/pareto_grid.csv")
    rf = grid[grid.ml_model.eq("RF+SMOTE")]
    # Representantes da tabela historica, sem escolher novos pontos no teste.
    representatives = [(0.1, 2), (0.3, 0), (0.35, 0), (0.35, 3.5),
                       (0.4, 0.5), (0.5, 1), (0.35, 5)]
    rows = []
    for ml, rules in representatives:
        selected = rf[(rf.ml_threshold - ml).abs().lt(1e-8)
                      & (rf.rules_threshold - rules).abs().lt(1e-8)]
        if len(selected) != 1:
            raise ValueError(f"Representante RF ausente ou ambiguo: {ml}, {rules}")
        rows.append(selected.iloc[0])
    xgb = grid[grid.ml_model.eq("XGBoost")].sort_values("ml_threshold")
    if len(xgb) != 5:
        raise ValueError("Esperados cinco limiares historicos do XGBoost")
    points = pd.concat([pd.DataFrame(rows), xgb], ignore_index=True)
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Pontos exploratórios de RF+SMOTE e XGBoost no plano precisão $\times$ sensibilidade. "
             r"Na grade apresentada, são dez pontos não dominados dentro de cada modelo (sete de RF+SMOTE e três de XGBoost); "
             r"os dois pontos dominados do XGBoost são mantidos e assinalados para documentar a grade histórica. "
             r"Os protocolos diferem entre os modelos. A pontuação de regras do fluxo Modal foi corrigida "
             r"para reutilizar a função da ablação original; os pontos XGBoost desta grade não aplicam esse filtro. Os limiares foram "
             r"explorados nos respectivos testes, sem avaliação independente de sua escolha.}",
             r"\label{tab:pareto-frontier}", r"\small",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Modelo & $\theta_{\text{ML}}$ & $\theta_{\text{Regras}}$ & Precisão & Sensibilidade & F1 & Situação \\",
             r"\midrule"]
    count = 0
    for _, row in points.iterrows():
        family = points[points.ml_model.eq(row.ml_model)]
        is_dominated = dominated(family, row)
        count += not is_dominated
        label = "Dominado" if is_dominated else "Não dominado"
        lines.append(f"{row.ml_model} & {d(row.ml_threshold, 2)} & {d(row.rules_threshold, 1)} & "
                     f"{d(row.precision)} & {d(row.recall)} & {d(row.f1)} & {label} " + r"\\")
    if count != 10:
        raise ValueError(f"Contagem por modelo inesperada: {count}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\footnotesize "
              r"$\theta_{\text{ML}}$: probabilidade mínima; $\theta_{\text{Regras}}$: escore mínimo de regras, "
              r"com zero indicando ausência desse filtro. XGBoost 0,50 e 0,90 são dominados por 0,70. "
              r"A contagem é por modelo, não da união entre protocolos distintos. "
              r"A grade é assimétrica: cinco pontos XGBoost sem regras contra a exploração mais extensa de RF+SMOTE. "
              r"O limiar de regras 1,0 do XGBoost não restringe suas aceitações. "
              r"No conjunto ampliado da ablação, a união XGBoost com regras $\geq8$ domina também os pontos "
              r"XGBoost 0,30 e 0,70; não dominado nesta tabela não significa não dominado nesse conjunto. "
              r"Teste: 18.509 potenciais pares, 74 vínculos. Fonte: \path{data/pareto_grid.csv} e "
              r"\path{data/ablation_results.csv}.", r"\end{table}", ""]
    (destino or OUT / "tab_pareto_frontier.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def tabela_nb02() -> None:
    data = json.loads((ROOT / "data/nb02_results.json").read_text(encoding="utf-8"))
    labels = ["RF com pesos extremos", "RF + SMOTE", "RF + BorderlineSMOTE",
              "Gradient Boosting + SMOTE", "AdaBoost + SMOTE", "MLP + SMOTE", "Votação suave"]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Estratégia de sensibilidade (NB02): resultados canônicos do script uniforme, "
             r"em teste com 18.509 potenciais pares e 74 vínculos, com 58 atributos.}",
             r"\label{tab:nb02-recall}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Modelo & Limiar & Precisão & Sensibilidade & $F_1$ \\", r"\midrule"]
    for label, row in zip(labels, data["all_results"], strict=True):
        threshold = row["Modelo"].split("th=")[1].rstrip(")")
        precision = d(row['Precision'], 4 if label == "RF + BorderlineSMOTE" else 3)
        lines.append(f"{label} & {threshold.replace('.', ',')} & {precision} & "
                     f"{d(row['Recall'])} & {d(row['F1-Score'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\footnotesize Fonte: "
              r"\path{data/nb02_results.json}, gerado por \path{scripts/run_nb02_uniform.py}. "
              r"A diferença em relação ao caderno histórico não tem causa documentada; "
              r"adota-se aqui apenas o artefato do script uniforme.", r"\end{table}", ""]
    (OUT / "tab_nb02_recall.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def tabela_impacto() -> None:
    data = pd.read_csv(ROOT / "data/epidemiological_impact.csv")
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Comparação de métodos: vínculos detectados, perdidos, revisões e sensibilidade.}",
             r"\label{tab:impacto-epidemiologico}", r"\footnotesize",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Método & Det. & Perd. & Rev. & Prec. & Sensib. \\", r"\midrule"]
    for _, row in data.iterrows():
        method = str(row.method)
        for old, new in [("Naive threshold", "Decisão de limiar"), ("Hybrid-OR", "Híbrido OR"),
                         ("Hybrid-AND", "Híbrido AND"), ("Rules", "Regras"), ("ML XGBoost", "XGBoost")]:
            method = method.replace(old, new)
        method = method.replace(">=", r"$\geq$").replace("0.5", "0,5").replace("0.7", "0,7")
        lines.append(f"{method} & {int(row.detected)} & {int(row.missed)} & {int(row.to_review)} & "
                     f"{d(row.precision, 2)} & {d(row.recall, 2)} " + r"\\")
    ablation = pd.read_csv(ROOT / "data/ablation_results.csv")
    union = ablation.loc[ablation.category.eq("hybrid-or") & ablation.config.str.contains("XGBoost")]
    selected = union.sort_values("f1", ascending=False, kind="stable").iloc[0]
    detected = round(74 * selected.recall)
    reviews = round(detected / selected.precision)
    if (detected, reviews) != (68, 68):
        raise ValueError("O ponto editorial XGBoost OR deve reproduzir 68 VP e zero FP")
    lines.append(r"XGBoost OR Regras ($\geq8$) & "
                 f"{detected} & {74 - detected} & {reviews} & {d(selected.precision, 2)} & "
                 f"{d(selected.recall, 2)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\footnotesize Fonte: "
              r"\path{data/epidemiological_impact.csv}. Det.: detectados; Perd.: perdidos; Rev.: revisões; "
              r"Prec.: precisão; Sensib.: sensibilidade. Teste com 18.509 potenciais pares e 74 vínculos. "
              r"O custo por vínculo (inverso do VPP), representado na figura, não é coluna desta tabela. "
              r"As linhas XGBoost provêm do fluxo Modal, não da ablação original. A união com regras "
              r"usa o ponto de probabilidade 0,7 e escore de regras 8 da ablação, com 68 VP e zero FP "
              r"(\path{data/ablation_results.csv}); a escolha no teste não é validada. "
              r"OR é união; AND é interseção.", r"\end{table}", ""]
    (OUT / "tab_impacto_epidemiologico.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def tabela_perfil_recuperados() -> None:
    historical = ROOT / "text/latex/tables/tab_perfil_recuperados.tex"
    text = historical.read_text(encoding="utf-8").replace(
        "Perfil dos óbitos recuperados", "Perfil dos vínculos recuperados"
    )
    text = "\n".join(line for line in text.splitlines()
                     if not line.startswith(("Idade (", "Sexo masculino", "Nota final (média)"))) + "\n"
    text = text.replace(r"\end{table}", r"\par\footnotesize Fonte: \path{data/epidemiological_impact.csv} "
                        r"e \path{data/banca_b_recuperados_por_par.json}, produzidos por "
                        r"\path{scripts/epidemiological_impact.py} e \path{scripts/banca_b_recuperados_por_par.py}. "
                        r"Teste com 18.509 potenciais pares e 74 vínculos. As colunas descrevem 27 vínculos "
                        r"exclusivos do RF+SMOTE e 40 comuns à nota e ao modelo; os três exclusivos da nota "
                        r"não integram o perfil. As faixas da nota decorrem dessa definição, não explicam a recuperação. "
                        r"As médias de idade e da nota e a distribuição por sexo não são apresentadas porque "
                        r"não há agregado versionado que permita "
                        r"conferir esses resumos sem o extrato privado." + "\n" + r"\end{table}")
    (OUT / historical.name).write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tabela_pareto()
    tabela_nb02()
    tabela_impacto()
    tabela_perfil_recuperados()
    nb03.OUT = OUT
    nb03.tabela_nb03()
    table = OUT / "tab_nb03_precision.tex"
    text = table.read_text(encoding="utf-8").replace(
        "entre os modelos base do caderno 03", "entre os modelos base do script uniforme NB03"
    ).replace("pares verdadeiros", "vínculos").replace("pares candidatos", "potenciais pares")
    text = text.replace(r"N~=~número", r"Fonte canônica: \path{data/nb03_results.json}, "
                        r"gerado por \path{scripts/run_nb03_uniform.py}. "
                        r"A causa da diferença em relação ao caderno histórico não está documentada. N~=~número")
    table.write_text(text, encoding="utf-8", newline="\n")
    qw1 = ROOT / "text/latex/tables/tab_qw1_deep_learning.tex"
    note = (r"\par\footnotesize Fonte: \path{data/review2_qw1_resumo.csv}. "
            r"Teste interno por linha: 18.509 potenciais pares, 74 vínculos. "
            r"Resultados históricos do QW1. No controle A0, o RF passou de "
            r"$F_1=0{,}855$ para $0{,}891$, e o TabNet, de $0{,}610$ para $0{,}712$, "
            r"fora da tolerância absoluta de 0,01; a validação cruzada histórica não foi recalculada "
            r"(Apêndice~\ref{sec:sensibilidade-qw1}).")
    (OUT / qw1.name).write_text(qw1.read_text(encoding="utf-8").replace(
        "avaliação independente", "avaliação interna").replace(
        "Avaliação independente", "Avaliação interna").replace(
        r"\end{table}", note + "\n" + r"\end{table}"
    ), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
