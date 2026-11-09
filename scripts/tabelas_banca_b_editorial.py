"""Tabelas editoriais dos Estudos 1 a 3, sem alterar geradores congelados."""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import gerar_tabelas_nb03_faixas as nb03
from scripts.tabelas_persona_editorial import transform

OUT = ROOT / "text/latex/tables/editorial"


def d(value: float, digits: int = 3) -> str:
    # Arredondamento comercial (meia unidade para cima) sobre a representação decimal do artefato:
    # o formato binário de 0.8765 arredondaria para 0,876, embora 71/81 = 0,87654 arredonde para 0,877.
    q = Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{q:.{digits}f}".replace(".", ",")


def fracao(numerador: int, denominador: int, digits: int = 3) -> str:
    """Fração exata com `digits` casas, meia unidade para cima, vírgula decimal (aritmética inteira)."""
    escala = 10 ** digits
    arredondado = (2 * numerador * escala + denominador) // (2 * denominador)
    return f"{arredondado // escala},{arredondado % escala:0{digits}d}"


def metricas_exatas(row: dict, pos_test: int) -> tuple[str, str, str]:
    """VPP, sensibilidade e F1 a partir das contagens reconstruídas dos valores de 4 casas do artefato.

    O artefato guarda 0,9595 (= 71/74 = 0,95946); arredondar esse valor de novo daria 0,960, e o
    correto é 0,959. VP = sensibilidade x positivos do teste; positivos preditos = VP / VPP. Se as
    contagens não reproduzirem os valores do artefato (4 casas), usa-se o valor do artefato
    (caso do AdaBoost, cujo VPP e F1 registrados não admitem o mesmo número de positivos preditos).
    """
    vp = round(row["Recall"] * pos_test)
    n_pred = round(vp / row["Precision"]) if vp and row["Precision"] else 0
    consistente = (
        n_pred > 0
        and round(vp / pos_test, 4) == round(row["Recall"], 4)
        and round(vp / n_pred, 4) == round(row["Precision"], 4)
        and round(2 * vp / (n_pred + pos_test), 4) == round(row["F1-Score"], 4)
    )
    if not consistente:
        return d(row["Precision"]), d(row["Recall"]), d(row["F1-Score"])
    return fracao(vp, n_pred), fracao(vp, pos_test), fracao(2 * vp, n_pred + pos_test)


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
              r"O limiar de regras do XGBoost é 0 nessa grade, isto é, sem filtro de regras. "
              r"No conjunto ampliado da ablação, a união XGBoost com regras $\geq8$ domina também os pontos "
              r"XGBoost 0,30 e 0,70; não dominado nesta tabela não significa não dominado nesse conjunto. "
              r"Teste: 18.509 potenciais pares, 74 vínculos. Fonte: \path{data/pareto_grid.csv} e "
              r"\path{data/ablation_results.csv}.", r"\end{table}", ""]
    (destino or OUT / "tab_pareto_frontier.tex").write_text(transform("tab_pareto_frontier", "\n".join(lines)), encoding="utf-8", newline="\n")


def tabela_nb02() -> None:
    data = json.loads((ROOT / "data/nb02_results.json").read_text(encoding="utf-8"))
    labels = ["RF com pesos extremos", "RF + SMOTE", "RF + BorderlineSMOTE",
              "Gradient Boosting + SMOTE", "AdaBoost + SMOTE", "MLP + SMOTE", "Votação suave"]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Estratégia de vigilância (NB02): resultados canônicos do script uniforme, "
             r"em teste com 18.509 potenciais pares e 74 vínculos, com 58 atributos.}",
             r"\label{tab:nb02-recall}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Modelo & Limiar & Precisão & Sensibilidade & $F_1$ \\", r"\midrule"]
    for label, row in zip(labels, data["all_results"], strict=True):
        threshold = row["Modelo"].split("th=")[1].rstrip(")")
        precision, recall, f1 = metricas_exatas(row, data["pos_test"])
        lines.append(f"{label} & {threshold.replace('.', ',')} & {precision} & "
                     f"{recall} & {f1} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\footnotesize Fonte: "
              r"\path{data/nb02_results.json}, gerado por \path{scripts/run_nb02_uniform.py}. "
              r"A diferença em relação ao caderno histórico não tem causa documentada; "
              r"adota-se aqui apenas o artefato do script uniforme.", r"\end{table}", ""]
    (OUT / "tab_nb02_recall.tex").write_text(transform("tab_nb02_recall", "\n".join(lines)), encoding="utf-8", newline="\n")


def tabela_impacto() -> None:
    data = pd.read_csv(ROOT / "data/epidemiological_impact.csv")
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Comparação de métodos: vínculos detectados, perdidos, revisões e sensibilidade.}",
             r"\label{tab:impacto-epidemiologico}", r"\footnotesize",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Método & Det. & Perd. & Rev. & VPP & Sensib. \\", r"\midrule"]
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
              r"VPP: valor preditivo positivo; Sensib.: sensibilidade. Teste com 18.509 potenciais pares e 74 pares verdadeiros. "
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
    profile = pd.read_csv(ROOT / "data/perfil_obitos_agregado.csv").set_index("conjunto")
    recovered, found = profile.loc["recuperados"], profile.loc["ja_encontrados"]
    replacements = {
        "Idade (": "Idade (média $\\pm$ DP) & " + " & ".join(
            f"{d(row.idade_media, 1)} $\\pm$ {d(row.idade_dp, 1)}" for row in (recovered, found)) + r" \\",
        "Sexo masculino": "Sexo masculino (\\%) & " + " & ".join(
            f"{int(row.sexo_masculino_n)} ({d(row.sexo_masculino_pct, 1)}" + r"\%)"
            for row in (recovered, found)) + r" \\",
        "Nota final (média)": "Nota final (média) & " + " & ".join(
            d(row.nota_media, 2) for row in (recovered, found)) + r" \\",
    }
    text = "\n".join(next((value for prefix, value in replacements.items() if line.startswith(prefix)), line)
                     for line in text.splitlines()) + "\n"
    text = text.replace(r"\end{table}", r"\par\footnotesize Fonte: \path{data/epidemiological_impact.csv} "
                        r"e \path{data/banca_b_recuperados_por_par.json}, produzidos por "
                        r"\path{scripts/epidemiological_impact.py} e \path{scripts/banca_b_recuperados_por_par.py}. "
                        r"Teste com 18.509 potenciais pares e 74 vínculos. As colunas descrevem 27 vínculos "
                        r"exclusivos do RF+SMOTE e 40 comuns à nota e ao modelo; os três exclusivos da nota "
                        r"não integram o perfil. As faixas da nota decorrem dessa definição, não explicam a recuperação. "
                        r"Médias de idade e nota e distribuição por sexo: \path{data/perfil_obitos_agregado.csv}, "
                        r"linhas do teste importadas do agregado histórico versionado "
                        r"\path{data/epidemiological_profile.json}, produzido por "
                        r"\path{scripts/epidemiological_impact.py}. Idade derivada de R\_DTNASC e R\_DTOBITO; "
                        r"sexo lido de R\_SEXO. Idade disponível nos 27 e 40 vínculos; DP amostral. "
                        r"Valores históricos arredondados; medianas e DP da nota não foram registrados." + "\n" + r"\end{table}")
    (OUT / historical.name).write_text(transform(historical.name, text), encoding="utf-8", newline="\n")


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
    table.write_text(transform(table.name, text), encoding="utf-8", newline="\n")
    tabela_qw1()


QW1_LINHAS = [("XGBoost", "XGBoost"), ("RandomForest", "Random Forest"),
              ("MLP (64,32)", "MLP (64, 32)"), ("MLP (128,64)", "MLP (128, 64)"),
              ("TabNet (fallback=FT-Transformer)", "TabNet")]


def qw1_linhas() -> str:
    """Linhas numéricas de tab_qw1_deep_learning lidas de data/review2_qw1_resumo.csv.

    O corpo da tabela histórica foi digitado à mão e divergia do artefato em quatro células;
    aqui cada valor vem do CSV (média e DP da validação cruzada 5x5; métricas do teste interno).
    O maior valor de cada coluna, pelo valor exibido, aparece em negrito (empates incluídos).
    """
    frame = pd.read_csv(ROOT / "data/review2_qw1_resumo.csv").set_index("model")
    colunas_media = ["f1_mean", "precision_mean", "recall_mean"]
    colunas_teste = ["f1", "precision", "recall"]
    maior = {c: max(d(frame.loc[m, c]) for m, _ in QW1_LINHAS) for c in colunas_media + colunas_teste}

    def negrito(coluna: str, valor: float) -> str:
        texto = d(valor)
        return rf"\textbf{{{texto}}}" if texto == maior[coluna] else texto

    linhas = []
    for chave, rotulo in QW1_LINHAS:
        linha = frame.loc[chave]
        cv = " & ".join(
            f"{negrito(c, linha[c])} $\\pm$ {d(linha[c.replace('_mean', '_std')])}" for c in colunas_media
        )
        teste = " & ".join(negrito(c, linha[c]) for c in colunas_teste)
        linhas.append(f"{rotulo:<17} & {cv}\n{'':<18}&& {teste} " + r"\\")
    return "\n".join(linhas) + "\n"


def tabela_qw1() -> None:
    qw1 = ROOT / "text/latex/tables/tab_qw1_deep_learning.tex"
    note = (r"\par\footnotesize Fonte: \path{data/review2_qw1_resumo.csv}. "
            r"Teste interno por linha: 18.509 potenciais pares, 74 vínculos. "
            r"Resultados históricos do QW1. No controle A0, o RF passou de "
            r"$F_1=0{,}855$ para $0{,}891$, e o TabNet, de $0{,}610$ para $0{,}712$, "
            r"fora da tolerância absoluta de 0,01; a validação cruzada histórica não foi recalculada "
            r"(Apêndice~\ref{sec:sensibilidade-qw1}).")
    historical = qw1.read_text(encoding="utf-8")
    head, rest = historical.split(r"\midrule" + "\n", 1)
    _, tail = rest.split(r"\bottomrule", 1)
    text = (head + r"\midrule" + "\n" + qw1_linhas() + r"\bottomrule" + tail).replace(
        "O melhor resultado de cada coluna aparece em negrito.",
        "O maior valor de cada coluna aparece em negrito, sem indicar diferença demonstrada."
    ).replace(
        "avaliação independente", "avaliação interna").replace(
        "Avaliação independente", "Avaliação interna").replace(
        r"\end{table}", note + "\n" + r"\end{table}"
    )
    (OUT / qw1.name).write_text(transform(qw1.name, text), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
