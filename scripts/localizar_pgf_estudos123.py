"""Localiza para português as figuras PGF dos estudos 1 a 3.

Substitui rótulos em inglês ou desatualizados e converte o ponto decimal em
vírgula nos textos (rótulos de eixo, valores sobre barras), sem tocar nas
coordenadas do desenho. Idempotente. Uso: python scripts/localizar_pgf_estudos123.py
"""

from __future__ import annotations

import re
from pathlib import Path

FIG_DOC = Path(__file__).resolve().parent.parent / "text" / "latex" / "fig_doc"

FIGURAS = [
    "score_band_pct.pgf",
    "score_band_volume.pgf",
    "ablation_best_category.pgf",
    "cv_boxplots_v2.pgf",
    "nb01_model_comparison.pgf",
    "imbalance_sensitivity.pgf",
    "pareto_frontier_v2.pgf",
    "epi_deteccao.pgf",
    "epi_revisoes.pgf",
    "epi_custo.pgf",
    "shap_summary.pgf",
    "feature_importance_heatmap.pgf",
]

ROTULOS = [
    ("Zona cinzenta", "Faixa intermediária"),
    (r"\textit{Área cinza}", "Faixa intermediária"),
    ("Área cinza", "Faixa intermediária"),
    ("Vínculos", "Pares verdadeiros"),
    ("Precisão", "VPP"),
    ("Faixa do escore agregado", "Nota final do comparador"),
    ("naive-threshold", "decisão de limiar"),
    ("Naive threshold", "Decisão de limiar"),
    ("Naive ", "Limiar "),
    ("Class weight + SMOTE", "Pesos de classe + SMOTE"),
    ("Class weight=balanced", "Pesos de classe balanceados"),
    ("No balancing", "Sem balanceamento"),
    ("Hybrid-OR", "Híbrido OR"),
    ("Hybrid-AND", "Híbrido AND"),
    ("Majority (th=0.7, 3 models)", "Consenso (limiar 0,7; 3 modelos)"),
    ("Rules", "Regras"),
    ("Importância (gain)", "Importância (ganho)"),
    ("Escore F1", "F1"),
    (r"$\pm$ desvio)", r"$\pm$ desvio-padrão)"),
    ("Logistica", "Logística"),
    ("Regressao", "Regressão"),
]

LINHA = re.compile(
    r"^(\\pgftext\[[^\]]*\]\{\\color\{textcolor\}\{\\rmfamily\\fontsize\{[\d.]+\}\{[\d.]+\}\\selectfont"
    r"(?:\\catcode`\\\^=\\active\\def\^\{\\ifmmode\\sp\\else\\\^\{\}\\fi\}\\catcode`\\%=\\active\\def%\{\\%\})?)"
    r"(.*?)(\}\}%?\s*)$"
)
DECIMAL = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?![\w.])")


def virgula(texto: str) -> str:
    partes = texto.split("$")
    for i, parte in enumerate(partes):
        sub = "{,}" if i % 2 == 1 else ","
        partes[i] = DECIMAL.sub(lambda m: f"{m.group(1)}{sub}{m.group(2)}", parte)
    return "$".join(partes)


def processar(path: Path) -> int:
    alteradas = 0
    saida = []
    for linha in path.read_text(encoding="utf-8").split("\n"):
        m = LINHA.match(linha)
        if m:
            texto = m.group(2)
            novo = texto
            for a, b in ROTULOS:
                novo = novo.replace(a, b)
            novo = virgula(novo)
            if novo != texto:
                alteradas += 1
                linha = m.group(1) + novo + m.group(3)
        saida.append(linha)
    path.write_text("\n".join(saida), encoding="utf-8")
    return alteradas


def main() -> None:
    for nome in FIGURAS:
        path = FIG_DOC / nome
        if not path.exists():
            print(f"[SKIP] {nome}")
            continue
        print(f"[OK] {nome}: {processar(path)} textos alterados")


if __name__ == "__main__":
    main()


