"""Camada editorial: só apresentação; geradores e tabelas históricos intactos."""
from pathlib import Path
import sys
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import equidade_arranjo as equity
from scripts import tabela_referencia_arranjo as reference
from scripts.tabelas_persona_editorial import rotulos_braco, transform


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Substituição editorial exige count == 1: {old!r}, count={count}")
    return text.replace(old, new, 1)


def step_ppv_note(maximum, groups, tests):
    """Distingue o resumo de diferença máxima dos testes pareados."""
    parts = []
    for decision, label in equity.LABELS.items():
        row = maximum.loc[maximum.decisao.eq(decision) & maximum.dimension.eq("passo")
                          & maximum.metric.eq("ppv")].squeeze()
        subset = groups.loc[groups.decisao.eq(decision) & groups.dimension.eq("passo")]
        eligible = subset.loc[subset.n_pred_pos.ge(equity.legacy.N_MIN)]
        if len(subset) != row.n_grupos_elegiveis or len(eligible) != row.n_grupos_den_ge_30:
            raise ValueError("Denominadores do resumo de VPP não reproduzem o CSV")
        step = subset.loc[subset.group.eq("passo_11")].squeeze()
        paired = tests.loc[tests.decisao.eq(decision) & tests.dimension.eq("passo")
                           & tests.metric.eq("ppv")]
        if not (paired.group_a.eq("passo_11") | paired.group_b.eq("passo_11")).any():
            raise ValueError("O passo 11 não aparece nos testes pareados do CSV")
        status = "incluído" if int(step.n_pred_pos) >= equity.legacy.N_MIN else "excluído"
        # A chave de dados e LABELS históricos dizem P; a nota exibida diz ASR.
        parts.append(f"{rotulos_braco(label)}: {int(row.n_grupos_den_ge_30)} de {int(row.n_grupos_elegiveis)} grupos; "
                     f"passo 11 {status}, com {int(step.n_pred_pos)} preditos positivos")
    return (" A diferença máxima de VPP por passo exige pelo menos "
            f"{equity.legacy.N_MIN} preditos positivos por grupo. " + "; ".join(parts) +
            ". Essa restrição vale para o resumo de diferença máxima, não para os testes pareados de VPP, "
            "que incluem o passo 11 nas quatro decisões.")


def render_reference(referencia, arranjo, cascata, carga):
    text = reference.render(referencia, arranjo, cascata)
    totals = carga.loc[carga.passo.astype(str).eq("TOTAL")].set_index("limiar")
    lower = reference.inteiro(totals.loc["por_passo", "pares_acima_limiar"])
    global_lower = reference.inteiro(totals.loc["global", "pares_acima_limiar"])
    ceiling = reference.inteiro(totals.loc["por_passo", "total_abaixo_corte"])
    changes = [
        ("scripts/tabela_referencia_arranjo.py", "scripts/tabelas_arranjo_editorial.py"),
        # O arranjo é o braço do registro interno, não o resultado principal (decisão D2 da autora, 2026-11-08).
        (r"\multicolumn{8}{l}{\textit{Resultado principal}}", r"\multicolumn{8}{l}{\textit{Braço do registro interno}}"),
        ("arranjo sequencial (resultado principal)", "arranjo sequencial (braço do registro interno)"),
        ("Corte + revisão manual do serviço & ambos & n.d.",
         "Corte + revisão manual do serviço & ambos & cenário $\\geq " + lower + "$"),
        (f"Corte + revisão humana integral (cenário) & ambos & até {ceiling}",
         f"Corte + revisão humana integral (cenário) & ambos & {ceiling} (teto)"),
        ("O volume examinado e o limite inferior da revisão do serviço não estão documentados (n.d.).",
         "O volume examinado total e a ordem de revisão não foram registrados. Se a revisão tivesse seguido "
         "ordem decrescente do escore do OpenRecLink, o cenário exigiria pelo menos "
         f"{lower} potenciais pares, com limiar por passo, ou {global_lower} com limiar global. "
         r"A reconstrução é exploratória, \textit{post hoc} e não registrada, não carga histórica observada."),
        (f"A revisão integral abaixo do corte é cenário de até {ceiling} potenciais pares, não carga histórica medida.",
         f"A revisão integral abaixo do corte é cenário de revisão de todos os {ceiling} candidatos, "
         "não carga histórica medida."),
        # M3 da auditoria de 2026-11-09: a classe da referência é "par verdadeiro"; "vínculo" é só a decisão.
        ("o arranjo aceita 127 vínculos na vigilância e 127 na confirmação, incluindo 123 pelo corte em cada modo; "
         "a cascata sobre a base inteira aceita 113 e 65.",
         "o arranjo aceita 127 pares verdadeiros na vigilância e 127 na confirmação, incluindo 123 pelo corte em cada modo; "
         "a cascata sobre a base inteira aceita 113 e 65 pares verdadeiros."),
        ("CdR, revisão humana da área cinza &", "CdR, revisão humana da área cinza da nota &"),
        ("CdR, limiar da nota em 8 &", r"CdR, decisão de limiar (nota $\geq 8$) &"),
        ("A revisão da área cinza do CdR é cenário", "A revisão da área cinza da nota do CdR é cenário"),
        ("Zero FP acima do corte também decorre do rótulo.",
         r"Zero FP acima do corte também decorre do \textit{label}."),
        ("revisão do serviço incorporada ao rótulo", r"revisão do serviço incorporada ao \textit{label}"),
        ("n.a.: não se aplica.",
         "O limiar 8 da nota é referência histórica do Estudo~3, não limiar escolhido para ASR. n.a.: não se aplica."),
        ("registro prévio versionado no repositório, com emendas e suplemento, não a registro público",
         r"registro interno versionado antes da avaliação com os \textit{labels} (S3), com emendas e suplemento"),
        (r"Fontes: \path{data/referencia_comparador_pos_orl.csv}",
         r"Fontes: \path{data/carga_abaixo_corte.csv}, \path{data/referencia_comparador_pos_orl.csv}"),
    ]
    for old, new in changes:
        text = replace_once(text, old, new)
    text = replace_once(text,
        r"Todas as sensibilidades com humano, salvo a revisão do serviço incorporada ao \textit{label}, usam erro hipotético de 5,0\%, não medido.",
        r"As células com humano, salvo a revisão do serviço incorporada ao \textit{label}, exibem o cenário de sensibilidade humana 0,95, não medida. Os três cenários, 0,95, 0,99 e 1,00, estão na Tabela~\ref{tab:cap8-sensibilidade-humana} e em \path{data/cenarios_sensibilidade_humana.csv}; a especificidade humana é perfeita por hipótese.")
    text = text.replace("revisão manual do serviço", "revisão clerical de origem")
    text = text.replace("revisão do serviço", "revisão clerical de origem")
    return legible_reference_layout(transform("tab_cap8_referencia", text))


def legible_reference_layout(text):
    """Só apresentação: sem \\resizebox, coluna de estratégia que quebra linha e cabeçalhos curtos em duas linhas.

    O preâmbulo não carrega tabularx; colunas p{} com frações de \\linewidth (soma de 0,88 mais o
    preenchimento de 16 \\tabcolsep de 3 pt, cerca de 0,11 da largura útil). Células e ordem intactas.
    """
    right = r">{\raggedleft\arraybackslash}"
    left = r">{\raggedright\arraybackslash}"
    columns = (left + r"p{0.25\linewidth}" + left + r"p{0.115\linewidth}"
               + right + r"p{0.08\linewidth}" + right + r"p{0.065\linewidth}"
               + right + r"p{0.11\linewidth}" + right + r"p{0.065\linewidth}"
               + right + r"p{0.08\linewidth}" + right + r"p{0.115\linewidth}")
    text = replace_once(text, "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{llrrrrrr}",
                        "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{" + columns + "}")
    text = replace_once(text, "\\end{tabular}}\n", "\\end{tabular}\n")
    text = replace_once(
        text,
        r"Estratégia & Modo & Pares ao humano & Pares ao LLM & Sens. automática (\%) & FP automáticos"
        r" & Pares verdadeiros perdidos & Sens. com humano (\%) \\",
        r"Estratégia & Modo & \shortstack[r]{Humano\\(pares)} & \shortstack[r]{LLM\\(pares)}"
        r" & \shortstack[r]{Sens. auto.\\(\%)} & \shortstack[r]{FP\\auto.} & Perdidos"
        r" & \shortstack[r]{Sens. com\\humano (\%)} \\")
    text = replace_once(
        text,
        r"Pares verdadeiros perdidos são apenas os classificados como \texttt{NONMATCH}, excluindo encaminhamentos pendentes.",
        r"Perdidos são apenas os pares verdadeiros classificados como \texttt{NONMATCH}, excluindo encaminhamentos pendentes.")
    return replace_once(
        text,
        r"Faixa intermediária da nota final: nota final do CdR em $[5,9)$.",
        r"Faixa intermediária da nota final: nota final do CdR em $[5,9)$. "
        r"Humano e LLM: pares encaminhados ao revisor humano e ao LLM. "
        r"Sens.: sensibilidade; auto.: automática, isto é, sem revisão humana; "
        r"FP auto.: falsos positivos aceitos automaticamente.")


def render_arrangement(historical):
    """Mantém todas as células; explicita o teto e os termos na apresentação."""
    lines = []
    for line in historical.splitlines():
        if "scripts/arranjo_corte_s3_avaliar.py" in line:
            line = replace_once(line, "scripts/arranjo_corte_s3_avaliar.py", "scripts/tabelas_arranjo_editorial.py")
        if "até 61.485 (36)" in line:
            line = replace_once(line, "até 61.485 (36)", "61.485 (teto; 36)")
        if "21.497 (28)" in line:
            line = replace_once(line, "21.497 (28)", "21.497 (teto; 28)")
        if line.startswith("(i) Corte & 211 &"):
            line = replace_once(line, "85,4 [80,5; 89,3]", "85,4")
        if line.startswith("(i) Corte & 123 &"):
            line = replace_once(line, "81,5 [74,5; 86,8]", "81,5")
        if " & " in line and len(line.split(" & ")) == 10 and not line.startswith("Braço &"):
            cells = line.split(" & ")
            if re.fullmatch(r"\d+(?:,\d+)? \[[^]]+\]", cells[2]):
                cells[2] = cells[2].split(" [", 1)[0]
            line = " & ".join(cells)
        if "registro prévio" in line:
            line = replace_once(line, "registro prévio", "registro interno")
        if "A posteriori" in line:
            line = replace_once(line, "A posteriori", r"\textit{A posteriori}")
        if "O a posteriori reutiliza" in line:
            line = replace_once(line, "O a posteriori reutiliza", r"O \textit{a posteriori} reutiliza")
        if "área cinza" in line:
            line = replace_once(line, "área cinza", "área cinza da nota")
        if "as classes PAR~=~1 e PAR~=~2 também participam" in line:
            line = replace_once(line,
                "as classes PAR~=~1 e PAR~=~2 também participam do ajuste supervisionado, com viés de incorporação.",
                "as classes PAR~=~1 e PAR~=~2 não estabelecem, por si, viés de incorporação: este diz respeito ao uso das evidências na referência, não ao ajuste supervisionado em si.")
        if line.startswith(r"\caption{Arranjo sequencial"):
            line = replace_once(line, line,
                r"\caption{Arranjo sequencial OpenRecLink e GZ-CMD v3.1, Claude Haiku 5.5, instruções v4: decisões, revisão e custo. São 247 vínculos na base e 151 em $[5,9)$. PAR~=~1 é definido pelo corte; as classes PAR~=~1 e PAR~=~2 não comprovam incorporação, que depende das evidências usadas na referência. Predição fora da amostra não equivale a validação externa.}")
        if line.startswith("Braço &") and "$S_A$" in line:
            line = replace_once(line, r"$S_A$ (IC 95\%)", r"$S_A$ (\%)")
        lines.append(line)
    text = "\n".join(lines) + "\n"
    # M2 da auditoria de 2026-11-09: V_A conta pares verdadeiros aceitos (a classe), não decisões de vínculo.
    text = replace_once(text, "$V_A$: vínculos aceitos automaticamente", "$V_A$: pares verdadeiros aceitos automaticamente")
    # Mesma grandeza (FP aceitos automaticamente) com um nome só em todas as tabelas do arranjo.
    text = text.replace("FP (abaixo)", "FP automáticos (abaixo)")
    text = replace_once(text,
        r"IC de Wilson de 95\% descritivo, sem correção pela dependência entre potenciais pares ou pelos vieses do padrão-ouro. ", "")
    text = replace_once(text,
        r"$S_H=(V_A+0{,}95V_H)/n$ é cenário de erro humano de 5\%, não resultado humano observado.",
        r"$S_H=(V_A+0{,}95V_H)/n$ exibe o cenário de sensibilidade humana 0,95, não resultado humano observado. Os três cenários, 0,95, 0,99 e 1,00, constam da Tabela~\ref{tab:cap8-sensibilidade-humana} e de \path{data/cenarios_sensibilidade_humana.csv}; a especificidade humana é perfeita por hipótese.")
    # Legenda curta no corpo e nota explicativa completa, sem truncar a legenda.
    text = replace_once(text, r"\caption{Arranjo sequencial", r"\caption[Arranjo sequencial: decisões, revisão e custo]{Arranjo sequencial")
    text = replace_once(text, "As ressalvas sobre o padrão-ouro e a dependência entre pares permanecem.",
                        "As ressalvas sobre a referência operacional e a dependência entre pares permanecem.")
    return transform("tab_cap8_arranjo_corte", text)


def render_anchors(historical):
    """Mantém as células; troca apenas o termo da legenda."""
    return transform("tab_cap8_ancoras", replace_once(historical, "Validação dos conjuntos âncora contra o padrão-ouro",
                        "Validação dos conjuntos âncora contra a referência operacional").replace("vínculos verdadeiros", "vínculos"))


def generate(data_dir: Path, tables_dir: Path) -> None:
    if tables_dir.resolve() == (ROOT / "text/latex/tables").resolve():
        raise ValueError("Use tables/editorial; as tabelas históricas são imutáveis")
    tables_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(data_dir / f"equidade_arranjo_{name}.csv")
              for name in ("global", "maxdiff", "pairwise_tests", "group_metrics")}
    target = tables_dir / "tab_equidade_arranjo.tex"
    equity.table_summary(frames["global"], frames["maxdiff"], frames["pairwise_tests"], target)
    text = target.read_text(encoding="utf-8")
    for decision, label in equity.LABELS.items():
        row = frames["global"].loc[frames["global"].decisao.eq(decision)].iloc[0]
        original = next(line for line in text.splitlines() if line.startswith(label + " &"))
        cells = original.split(" & ")
        cells[7] = replace_once(f"{100 * row.espec:.2f}", ".", "{,}")
        text = replace_once(text, original, " & ".join(cells))
    text = replace_once(text, "O gabarito compartilha o corte e os dados de treinamento",
                        "A referência operacional compartilha o corte do escore OpenRecLink e os dados de treinamento")
    text = replace_once(text, "corte SCORE do OpenRecLink", "corte do escore OpenRecLink")
    text = replace_once(text, r"\end{minipage}", step_ppv_note(
        frames["maxdiff"], frames["group_metrics"], frames["pairwise_tests"]) + r"\end{minipage}")
    target.write_text(transform(target.name, text), encoding="utf-8", newline="\r\n")
    for mode in ("vigilancia", "confirmacao"):
        target = tables_dir / f"tab_equidade_arranjo_grupos_{mode}.tex"
        equity.table_groups(frames["group_metrics"], mode, target)
        text = replace_once(target.read_text(encoding="utf-8"), "Grupos MISSING", "Grupos sem informação")
        target.write_text(transform(target.name, text), encoding="utf-8", newline="\r\n")
    for decision in equity.DECISIONS:
        name = f"tab_equidade_arranjo_rejeicoes_{decision}.tex"
        text = (ROOT / "text/latex/tables" / name).read_text(encoding="utf-8")
        (tables_dir / name).write_text(transform(name, text), encoding="utf-8", newline="\r\n")
    sources = [pd.read_csv(data_dir / name) for name in (
        "referencia_comparador_pos_orl.csv", "arranjo_corte_metricas.csv",
        "funil_gzcmd_final_metricas.csv", "carga_abaixo_corte.csv")]
    (tables_dir / "tab_cap8_referencia.tex").write_text(render_reference(*sources), encoding="utf-8", newline="\r\n")
    name = "tab_cap8_arranjo_corte.tex"
    historical = (ROOT / "text/latex/tables" / name).read_text(encoding="utf-8")
    (tables_dir / name).write_text(render_arrangement(historical), encoding="utf-8", newline="\r\n")
    name = "tab_cap8_ancoras.tex"
    historical = (ROOT / "text/latex/tables" / name).read_text(encoding="utf-8")
    (tables_dir / name).write_text(render_anchors(historical), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    generate(ROOT / "data", ROOT / "text/latex/tables/editorial")
    print("Tabelas editoriais regeneradas; fontes históricas preservadas.")
