"""Localiza para português (vírgula decimal, rótulos) as tabelas dos estudos 1 a 3.

Idempotente: aplica-se sobre as tabelas já versionadas em text/latex/tables/.
Uso: python scripts/localizar_tabelas_estudos123.py
"""

from __future__ import annotations

import re
from pathlib import Path

TABLES = Path(__file__).resolve().parent.parent / "text" / "latex" / "tables"

RENAMES = [
    ("Consensus ML-majority (th=0.7, 3 models)", "Consenso ML (limiar 0,7; 3 modelos)"),
    ("Consensus ML-majority (th=0.5, 3 models)", "Consenso ML (limiar 0,5; 3 modelos)"),
    ("Majority (th=0.7, 3 models incl. XGBoost)", "Consenso (limiar 0,7; 3 modelos, incl. XGBoost)"),
    ("Majority (th=0.7, 3 models)", "Consenso (limiar 0,7; 3 modelos)"),
    ("Majority (th=0.5) AND Rules", "Consenso (limiar 0,5) AND Regras"),
    ("Majority AND Rules", "Consenso AND Regras"),
    ("ML-only", "Somente ML"),
    ("Rules-only", "Somente regras"),
    ("Hybrid-OR", "Híbrido OR"),
    ("Hybrid-AND", "Híbrido AND"),
    ("No balancing", "Sem balanceamento"),
    ("Class weight=balanced", "Pesos de classe balanceados"),
    ("Class weight + SMOTE", "Pesos de classe + SMOTE"),
    ("Somente regras ($", "Regras ($"),
    ("Somente ML XGBoost", "Somente ML: XGBoost"),
    ("Somente ML RF", "Somente ML: RF"),
    ("Rules", "Regras"),
    ("Escore$\\geq$8", "Nota final$\\geq$8"),
    ("taxa corrigida", "sensibilidade"),
]

DECIMAL = re.compile(r"(?<![\w.,])(\d+)\.(\d+)(?![\w.])")

FILES_DECIMAL = [
    "tab_cv_5fold.tex",
    "tab_imbalance_sensitivity.tex",
    "tab_ablation_best_category.tex",
    "tab_ablation_top10.tex",
    "tab_shap_importance.tex",
    "tab_pareto_frontier.tex",
]
FILES_LABELS = FILES_DECIMAL + ["tab_impacto_epidemiologico.tex"]


def main() -> None:
    for name in FILES_LABELS:
        path = TABLES / name
        text = path.read_text(encoding="utf-8")
        for old, new in RENAMES:
            text = text.replace(old, new)
        if name in FILES_DECIMAL:
            text = DECIMAL.sub(r"\1,\2", text)
        path.write_text(text, encoding="utf-8")
        print(f"[OK] {name}")

    # Tab. QW1: empate em precisão no holdout (1,000) recebe negrito nas duas linhas.
    qw1 = TABLES / "tab_qw1_deep_learning.tex"
    text = qw1.read_text(encoding="utf-8")
    text = text.replace("&& 0,844 & 1,000 & 0,730", "&& 0,844 & \\textbf{1,000} & 0,730")
    qw1.write_text(text, encoding="utf-8")
    print("[OK] tab_qw1_deep_learning.tex")

    # Tab. de recuperados: a coluna "Já encontrados" soma 40 com a faixa >= 8.
    perfil = TABLES / "tab_perfil_recuperados.tex"
    text = perfil.read_text(encoding="utf-8")
    linha = "\\quad Faixa $\\geq$ 8 & 0 (0,0\\%) & 40 (100,0\\%) \\\\"
    if linha not in text:
        text = text.replace(
            "\\quad Faixa 7--8 & 14 (51,9\\%) & 0 (0,0\\%) \\\\",
            "\\quad Faixa 7--8 & 14 (51,9\\%) & 0 (0,0\\%) \\\\\n" + linha,
        )
    perfil.write_text(text, encoding="utf-8")
    print("[OK] tab_perfil_recuperados.tex")


if __name__ == "__main__":
    main()

