"""Versões editoriais W3, somente sobre agregados existentes, sem treinar modelos.

Não altera tabelas históricas, manifestos ou geradores de análises congeladas.
Semente dos experimentos de origem: 42; ablação COMPREC: 42 a 46.
Com --export-qw1, lê somente cv_summary.csv e holdout_metrics.csv do diretório
histórico ignorado data/qw1/ e publica suas métricas agregadas, com semente 42,
em data/review2_qw1_resumo.csv. Não lê predições nem valores por potencial par.
A regeneração editorial sem essa opção não depende de data/qw1/.
"""
from pathlib import Path
import argparse
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import equidade_arranjo, generate_ablation_tables

OUT = ROOT / "text/latex/tables/editorial"


def export_qw1(source, destination):
    """Publica somente métricas já existentes; nenhuma predição individual."""
    cv = pd.read_csv(source / "cv_summary.csv")
    test = pd.read_csv(source / "holdout_metrics.csv")
    frame = cv.merge(test, on="model", validate="one_to_one", suffixes=("_cv", "_teste"))
    frame["semente"] = 42
    frame.to_csv(destination, index=False, lineterminator="\n")
    print(f"QW1: {len(frame)} modelos, somente métricas agregadas exportadas para {destination}")


def number(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def note(text, source):
    return text.replace(r"\end{table}", r"\par\footnotesize " + source + "\n" + r"\end{table}")


def write(name, text, out):
    newline = "\r\n" if name in {
        "tab_equidade_arranjo_grupos_confirmacao.tex",
        "tab_equidade_arranjo_grupos_vigilancia.tex",
    } else "\n"
    (out / name).write_text(text, encoding="utf-8", newline=newline)


def shap_families(out):
    frame = pd.read_csv(ROOT / "data/shap_por_familia.csv")
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Soma descritiva da importância absoluta atribuída às famílias de atributos no Random Forest.}",
             r"\label{tab:shap-importance}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Família & Atributos & Soma & Proporção (\%) \\", r"\midrule"]
    for row in frame.itertuples(index=False):
        lines.append(f"{row.familia} & {row.n_atributos} & {number(row.shap_mean_abs_soma, 4)} & "
                     f"{number(100 * row.fracao_total, 1)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\footnotesize Fonte: \path{data/shap_por_familia.csv}, derivado de "
              r"\path{data/shap_importance.csv} por \path{scripts/shap_por_familia.py}. "
              r"Importância histórica do RF com SMOTE, calculada em 2.000 potenciais pares da própria base de "
              r"treinamento, não em teste independente. Escala de probabilidade quando o método é SHAP. "
              r"A soma conta cada um dos 58 atributos uma vez e não recalcula SHAP conjunto. "
              r"O gerador histórico admite substituir SHAP por Gini sem mudar o nome do arquivo; "
              r"a proveniência do método não é certificada somente pelo CSV. A tabela não ordena atributos "
              r"redundantes nem identifica efeitos causais.", r"\end{table}", ""]
    write("tab_shap_importance.tex", "\n".join(lines), out)


def ablation_tables(out):
    frame = pd.read_csv(ROOT / "data/ablation_results.csv")
    generate_ablation_tables.generate(frame, out)
    for name in ("tab_ablation_best_category.tex", "tab_ablation_top10.tex"):
        text = (out / name).read_text(encoding="utf-8")
        if name == "tab_ablation_best_category.tex":
            rows = text.splitlines()
            for i, row in enumerate(rows):
                if row.startswith("Híbrido OR &"):
                    # Os limiares 8 e 9 empatam no RF; adota-se o 8 usado na CV.
                    rows[i] = row.replace(r"Regras$\geq$9", r"Regras$\geq$8", 1)
            text = "\n".join(rows) + "\n"
        text = text.replace("é agora a mesma", "é a mesma")
        text = note(text, r"Teste com 18.509 potenciais pares e 74 vínculos. OR é união; AND é interseção. "
                    r"Os limiares de regras 8 e 9 empatam no Híbrido OR do RF+SMOTE; apresenta-se 8, "
                    r"como na validação cruzada. A ordenação pelo $F_1$ não demonstra diferença entre "
                    r"as dez primeiras configurações nem equivalência estatística.")
        write(name, text, out)


def historical_tables(out):
    sources = {
        "tab_cv_5fold.tex": r"Fonte: \path{data/cv_5fold_results.csv} e resumo histórico QW1 em "
            r"\path{data/review2_qw1_resumo.csv}. Cinco partições da base com 61.696 potenciais "
            r"pares e 247 vínculos na análise original; no QW1, 25 avaliações nos 70\% de treinamento. "
            r"Os desvios-padrão não são erros-padrão de contrastes pareados. Não se destaca uma configuração.",
        "tab_imbalance_sensitivity.tex": r"Fonte: \path{data/imbalance_sensitivity.csv}, produzida por "
            r"\path{scripts/robustness_analysis.py}. Cinco partições dos 61.696 potenciais pares, "
            r"com 247 vínculos; reamostragem restrita ao treinamento. Média e desvio-padrão entre partições, "
            r"não intervalo de confiança da diferença pareada.",
        "tab_score_bands.tex": r"Total: 61.696 potenciais pares, 247 vínculos. "
            r"Fonte agregada: \path{data/tab_score_bands_proveniencia.csv}. "
            r"Vínculo significa PAR igual a 1 ou 2.",
    }
    for name, source in sources.items():
        text = (ROOT / "text/latex/tables" / name).read_text(encoding="utf-8")
        if name == "tab_cv_5fold.tex":
            text = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", text)
        if name == "tab_score_bands.tex":
            text = text.replace(r"Total & Pares & \% Pares", r"Total & Vínculos & \% Vínculos")
            text = re.sub(r"(?<![0-9])([0-9]+),([0-9]{2})(?![0-9])",
                          lambda m: number(float(m[0].replace(",", ".")), 1), text)
        write(name, note(text, source), out)


def equity_tables(out):
    # Reapresenta os grupos a partir de agregados; não reexecuta os testes.
    frame = pd.read_csv(ROOT / "data/equidade_arranjo_group_metrics.csv")
    for mode in ("vigilancia", "confirmacao"):
        name = f"tab_equidade_arranjo_grupos_{mode}.tex"
        equidade_arranjo.table_groups(frame, mode, out / name)
        text = (out / name).read_text(encoding="utf-8")
        text = text.replace("Grupos MISSING e com menos de 30 potenciais pares não integram os testes, como na análise anterior.",
                            "Grupos sem classificação ou com menos de 30 potenciais pares ficam fora da seleção inicial. "
                            "Na família principal, a comparação exige adicionalmente ao menos 30 observações no "
                            "denominador de cada métrica em ambos os grupos, antes de Benjamini-Hochberg.")
        text = note(text, r"Fonte: \path{data/equidade_arranjo_group_metrics.csv} e "
                    r"\path{data/equidade_arranjo_cobertura.csv}. Universo: 61.696 potenciais pares, 247 vínculos; "
                    r"a cobertura varia por dimensão. Os grupos do nome têm empates e não são quartis de mesmo tamanho. "
                    r"Passo e encerramento são estratos de processo. FN inclui vínculos encaminhados ao revisor humano.")
        write(name, text, out)


def appendix_ablation():
    lines = [r"As tabelas deste apêndice descrevem a ablação saneada com partição por COMPREC, "
             r"nos modos confirmação e vigilância, com MACD ativado ou desativado. Cada tabela apresenta "
             r"cinco sementes e média $\pm$ desvio-padrão amostral. As métricas de precisão, sensibilidade, "
             r"$F_1$ e $F_\beta$ são esperadas, calculadas com as proporções de erro por faixa da configuração "
             r"(\path{gzcmd/eval.py}), sem chamadas a modelo de linguagem. Encaminhados é a contagem "
             r"selecionada para revisão simulada, não o número de respostas de um LLM. "
             r"O $F_1$ esperado não é desempenho observado de um LLM. "
             r"O Random Forest calibrado usa somente os 29 subescores e a nota final, acrescidos das MACD "
             r"no braço ativado, com política v3. O histórico, que incluía escore do OpenRecLink, "
             r"situação de encerramento, identificadores e atributos duplicados, permanece em "
             r"\path{data/ablation_comprec_macd.csv} e \path{data/ablation_comprec_nomacd.csv}; "
             r"a comparação está em \path{data/exp1_vazamento_comparacao.csv}. "
             r"As partições de teste se sobrepõem e não constituem cinco amostras independentes.", ""]
    columns = ["exp_precision", "exp_recall", "exp_f1", "exp_fbeta", "auto_coverage", "llm_used"]
    for suffix, state in (("macd", "ativado"), ("nomacd", "desativado")):
        source = f"data/ablation_comprec_{suffix}_allowlist.csv"
        frame = pd.read_csv(ROOT / source, sep=";")
        summary = pd.read_csv(ROOT / f"data/ablation_comprec_{suffix}_allowlist_summary.csv", sep=";")
        for mode, label in (("confirmacao", "confirmação"), ("vigilancia", "vigilância")):
            group = frame.loc[frame["mode"].eq(mode)].sort_values("seed")
            beta = group.beta.unique()
            if len(beta) != 1 or len(group) != 5:
                raise ValueError("Ablação saneada deve ter cinco sementes e beta constante")
            row_summary = summary.loc[summary.Config.str.startswith(mode + " |")].iloc[0]
            tag = "on" if suffix == "macd" else "off"
            lines += [r"\begin{table}[htbp]", r"\centering",
                      rf"\caption{{Ablação saneada com MACD {state}, modo {label}: métricas esperadas.}}",
                      rf"\label{{tab:ablacao-comprec-{mode}-macd-{tag}}}", r"\small",
                      r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                      r"Semente & $n$ teste & Precisão & Sensibilidade & $F_1$ & $F_\beta$ & Cobertura autom. & Encaminhados \\",
                      r"\midrule"]
            for _, row in group.iterrows():
                values = [number(row[c]) for c in columns[:-1]] + [str(int(row.llm_used))]
                lines.append(f"{int(row.seed)} & {int(row.n_test):,}".replace(",", ".") + " & " + " & ".join(values) + r" \\")
            means = [number(row_summary[c + "_mean"], 1 if c == "llm_used" else 3) + r" $\pm$ "
                     + number(row_summary[c + "_std"], 1 if c == "llm_used" else 3) for c in columns]
            lines += [r"Média $\pm$ DP & n.a. & " + " & ".join(means) + r" \\", r"\bottomrule", r"\end{tabular}}",
                      r"\par\footnotesize Fonte: \path{" + source + "} e resumo no arquivo "
                      r"\path{data/ablation_comprec_" + suffix + r"_allowlist_summary.csv}. "
                      r"Cada teste contém 74 vínculos. $\beta=" + number(beta[0], 1).replace(",", "{,}") + r"$. "
                      r"Contagens por semente são inteiras; a média de encaminhados pode ser fracionária.",
                      r"\end{table}", ""]
    return "\n".join(lines)


def generate(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    shap_families(out)
    ablation_tables(out)
    historical_tables(out)
    equity_tables(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-qw1", action="store_true", help="Copiar métricas agregadas do arquivo histórico privado")
    args = parser.parse_args()
    if args.export_qw1:
        export_qw1(ROOT / "data/qw1", ROOT / "data/review2_qw1_resumo.csv")
    generate()
    (ROOT / "text/latex/apendice_ablacao.tex").write_text(appendix_ablation(), encoding="utf-8", newline="\n")
    print("Tabelas editoriais W3 e apêndice de ablação gerados somente de agregados.")
