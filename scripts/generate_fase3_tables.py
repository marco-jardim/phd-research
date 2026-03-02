"""Generate LaTeX tables for Fase 3 robustness analysis."""

from pathlib import Path
import pandas as pd

TABLES_DIR = Path("text/latex/tables")


def pm(mean, std):
    """Format mean +/- std."""
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def escape(s):
    """Escape LaTeX special chars."""
    return s.replace("_", "\\_").replace(">=", "$\\geq$")


def gen_cv_5fold():
    cv = pd.read_csv("data/cv_5fold_results.csv")
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Valida\\c{c}\\~ao cruzada estratificada (5-\\textit{fold}) das configura\\c{c}\\~oes selecionadas.}",
        "\\label{tab:cv-5fold}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Configura\\c{c}\\~ao & F1 & Precis\\~ao & \\textit{Recall} \\\\",
        "\\midrule",
    ]
    for _, r in cv.iterrows():
        name = escape(r["config"])
        f1 = pm(r["f1_mean"], r["f1_std"])
        p = pm(r["prec_mean"], r["prec_std"])
        rc = pm(r["rec_mean"], r["rec_std"])
        lines.append(f"{name} & {f1} & {p} & {rc} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES_DIR / "tab_cv_5fold.tex").write_text("\n".join(lines), encoding="utf-8")
    print("tab_cv_5fold.tex OK")


def gen_imbalance():
    imb = pd.read_csv("data/imbalance_sensitivity.csv")
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Sensibilidade ao desbalanceamento: Random Forest com diferentes estrat\\'egias de reamostragem (5-\\textit{fold} CV).}",
        "\\label{tab:imbalance-sensitivity}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Estrat\\'egia & F1 & Precis\\~ao & \\textit{Recall} \\\\",
        "\\midrule",
    ]
    for _, r in imb.iterrows():
        name = escape(r["strategy"])
        f1 = pm(r["f1_mean"], r["f1_std"])
        p = pm(r["prec_mean"], r["prec_std"])
        rc = pm(r["rec_mean"], r["rec_std"])
        lines.append(f"{name} & {f1} & {p} & {rc} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES_DIR / "tab_imbalance_sensitivity.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("tab_imbalance_sensitivity.tex OK")


def gen_shap():
    shap_df = pd.read_csv("data/shap_importance.csv")
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Import\\^ancia dos atributos por valores SHAP (m\\'edia do valor absoluto, classe positiva).}",
        "\\label{tab:shap-importance}",
        "\\begin{tabular}{rlr}",
        "\\toprule",
        "\\# & Atributo & $|\\mathrm{SHAP}|$ m\\'edio \\\\",
        "\\midrule",
    ]
    for i, (_, r) in enumerate(shap_df.head(15).iterrows(), 1):
        name = escape(r["feature"])
        lines.append(f"{i} & {name} & {r['shap_mean_abs']:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES_DIR / "tab_shap_importance.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("tab_shap_importance.tex OK")


if __name__ == "__main__":
    gen_cv_5fold()
    gen_imbalance()
    gen_shap()
    print("All Fase 3 tables generated.")
