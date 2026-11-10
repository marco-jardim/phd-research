"""Recomputação agregada do E5: n1 e n2 >=30 antes de BH por decisão.

Preserva os geradores, manifesto e testes históricos com SHA fixado. Publica
saídas irmãs den30; a troca da fonte editorial exige autorização para revisar
os testes congelados que exigem as 58 comparações históricas por decisão.
"""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import equidade_arranjo as historical
from scripts.tabelas_persona_editorial import transform


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Substituição de layout exige count == 1: {old!r}, count={count}")
    return text.replace(old, new, 1)


def legible_layout(text):
    """Só apresentação: sem \\resizebox, colunas p{} que quebram linha e cabeçalhos curtos em duas linhas.

    O preâmbulo não carrega tabularx; as frações de \\linewidth somam 0,82 (nove colunas) e 0,88 (seis colunas)
    e o preenchimento de \\tabcolsep (3 pt) completa a largura útil (cerca de 0,94 e 0,96). Células, ordem e
    legenda ficam intactas.
    """
    right = r">{\raggedleft\arraybackslash}"
    left = r">{\raggedright\arraybackslash}"
    first = (left + r"p{0.30\linewidth}" + right + r"p{0.07\linewidth}" + right + r"p{0.055\linewidth}"
             + right + r"p{0.055\linewidth}" + right + r"p{0.055\linewidth}" + right + r"p{0.07\linewidth}"
             + right + r"p{0.07\linewidth}" + right + r"p{0.08\linewidth}" + right + r"p{0.065\linewidth}")
    second = (left + r"p{0.32\linewidth}" + left + r"p{0.10\linewidth}" + right + r"p{0.11\linewidth}"
              + right + r"p{0.11\linewidth}" + right + r"p{0.12\linewidth}" + right + r"p{0.12\linewidth}")
    text = replace_once(text, "\\footnotesize\n\\caption{", "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\caption{")
    text = replace_once(text, "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrrrrrr}",
                        "\\begin{tabular}{" + first + "}")
    text = replace_once(text, "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llrrrr}",
                        "\\begin{tabular}{" + second + "}")
    if text.count("\\end{tabular}}\n") != 2:
        raise ValueError("Esperadas duas tabelas com \\resizebox")
    text = text.replace("\\end{tabular}}\n", "\\end{tabular}\n")
    text = replace_once(
        text, r"Decisão & Pred.$+$ & VP & FP & FN & Sens. & VPP & Espec. & Rejeições \\",
        r"Decisão & \shortstack[r]{Pred.\\$+$} & VP & FP & FN & Sens. & VPP & Espec. & Rej. \\")
    text = replace_once(
        text,
        r"Dimensão & Métrica & ASR vigil. & ASR confirm. & Base inteira vigil. (comparação)"
        r" & Base inteira confirm. (comparação) \\",
        r"Dimensão & Métrica & \shortstack[r]{ASR\\vigil.} & \shortstack[r]{ASR\\confirm.}"
        r" & \shortstack[r]{Base int.\\vigil.} & \shortstack[r]{Base int.\\confirm.} \\")
    return replace_once(
        text, "Cascata v3.1: política de quatro ações aplicada à base inteira, incluindo revisão humana.",
        "Cascata v3.1: política de quatro ações aplicada à base inteira, incluindo revisão humana. "
        "Rej.: rejeições após Benjamini-Hochberg sobre o total de comparações da decisão. "
        "Base int.: base inteira (comparação).")


def recompute(source, minimum=30):
    eligible = source.loc[source.n1.ge(minimum) & source.n2.ge(minimum)].copy()
    for _, group in eligible.groupby("decisao", sort=False):
        corrected = historical.legacy.qw3._benjamini_hochberg(
            group.p_value_raw.tolist(), alpha=historical.legacy.ALPHA)
        eligible.loc[group.index, "p_value_fdr"] = corrected
    eligible["reject_fdr"] = eligible.p_value_fdr.le(historical.legacy.ALPHA)
    return eligible.reset_index(drop=True)


def generate(data, editorial):
    source = pd.read_csv(data / "equidade_arranjo_pairwise_tests.csv")
    corrected = recompute(source)
    corrected.to_csv(data / "equidade_arranjo_den30_pairwise_tests.csv", index=False, lineterminator="\n")
    audit = []
    for decision in historical.DECISIONS:
        before = source.loc[source.decisao.eq(decision)]
        after = corrected.loc[corrected.decisao.eq(decision)]
        audit.append(dict(decisao=decision, testes_antes=len(before), testes_depois=len(after),
                          rejeicoes_antes=int(before.reject_fdr.sum()),
                          rejeicoes_depois=int(after.reject_fdr.sum()),
                          excluidos_denominador_menor_30=len(before) - len(after)))
    pd.DataFrame(audit).to_csv(data / "equidade_arranjo_den30_rejeicoes.csv", index=False, lineterminator="\n")
    glob = pd.read_csv(data / "equidade_arranjo_global.csv")
    maximum = pd.read_csv(data / "equidade_arranjo_maxdiff.csv")
    editorial.mkdir(parents=True, exist_ok=True)
    path = editorial / "tab_den30_equidade_arranjo.tex"
    historical.table_summary(glob, maximum, corrected, path)
    text = path.read_text(encoding="utf-8").replace(
        "tab:equidade-arranjo}", "tab:equidade-arranjo-den30}").replace(
        "preservando as omissões por denominador zero",
        "após excluir comparações com denominador menor que 30 em qualquer grupo")
    # A linha do corte 6,0 (7.452 preditos positivos) não consta desta tabela, só de tab_equidade_v31.
    text = text.replace(" A referência histórica de 7.452 preditos positivos pertence ao corte 6,0 da nota final, "
                        "não ao corte SCORE do OpenRecLink nem ao braço P.", "")
    text = text.replace("O gabarito compartilha", "A referência operacional compartilha").replace(
        "corte SCORE do OpenRecLink", "corte do escore OpenRecLink")
    # Especificidade com duas casas, como na tabela editorial principal e no texto do Estudo 4.
    for decision, label in historical.LABELS.items():
        row = glob.loc[glob.decisao.eq(decision)].iloc[0]
        original = next(line for line in text.splitlines() if line.startswith(label + " &"))
        cells = original.split(" & ")
        cells[7] = f"{100 * row.espec:.2f}".replace(".", "{,}")
        text = text.replace(original, " & ".join(cells), 1)
    text = text.replace(r"\end{table}", r"\par\footnotesize Fonte: "
                        r"\path{data/equidade_arranjo_global.csv}, "
                        r"\path{data/equidade_arranjo_maxdiff.csv} e "
                        r"\path{data/equidade_arranjo_den30_pairwise_tests.csv}. "
                        r"Universo: 61.696 potenciais pares e 247 vínculos. Os denominadores são "
                        r"os vínculos na sensibilidade e os positivos preditos no VPP. "
                        r"As rejeições exigem pelo menos 30 observações em cada denominador, "
                        r"antes de Benjamini-Hochberg por decisão. FN inclui vínculos encaminhados "
                        r"ao revisor humano, não somente decisões de não vínculo." + "\n" + r"\end{table}")
    path.write_text(legible_layout(transform(path.name, text)), encoding="utf-8", newline="\n")
    for decision in historical.DECISIONS:
        path = editorial / f"tab_den30_equidade_arranjo_rejeicoes_{decision}.tex"
        historical.legacy.tabela_rejeicoes(
            corrected, path, decisao=decision,
            label=f"tab:equidade-den30-{decision.replace('_', '-')}",
            descr=historical.LABELS[decision] + ", denominadores de pelo menos 30")
        text = path.read_text(encoding="utf-8").replace(
            r"\end{table}", r"\par\footnotesize Fonte: "
            r"\path{data/equidade_arranjo_den30_pairwise_tests.csv}. "
            r"OR é razão de chances. "
            r"Família definida separadamente por decisão, após o filtro de denominador." + "\n" + r"\end{table}")
        if decision in ("arranjo_P_vigilancia", "cascata_v31_vigilancia"):
            text = text.replace(r"\begin{tabular}", r"\resizebox{\linewidth}{!}{%" + "\n" + r"\begin{tabular}")
            text = text.replace(r"\end{tabular}", r"\end{tabular}}")
        path.write_text(transform(path.name, text), encoding="utf-8", newline="\n")
    manifest = dict(denominador_minimo=30, regra="Excluir n1<30 ou n2<30 antes de BH por decisão",
                    fonte="equidade_arranjo_pairwise_tests.csv", treino_executado=False,
                    geradores_historicos_preservados=True, comparacoes=audit)
    (data / "equidade_arranjo_den30_recomputacao.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return corrected


if __name__ == "__main__":
    generate(ROOT / "data", ROOT / "text/latex/tables/editorial")
