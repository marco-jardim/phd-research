"""Camada editorial: só apresentação; geradores e tabelas históricos intactos."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import equidade_arranjo as equity
from scripts import tabela_referencia_arranjo as reference


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
        parts.append(f"{label}: {int(row.n_grupos_den_ge_30)} de {int(row.n_grupos_elegiveis)} grupos; "
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
        ("Corte + revisão manual do serviço & ambos & n.d.",
         "Corte + revisão manual do serviço & ambos & $\\geq " + lower + "$"),
        (f"Corte + revisão humana integral (cenário) & ambos & até {ceiling}",
         f"Corte + revisão humana integral (cenário) & ambos & {ceiling} (teto)"),
        ("O volume examinado e o limite inferior da revisão do serviço não estão documentados (n.d.).",
         "O volume examinado total não foi registrado. Sob a hipótese de revisão em ordem decrescente de escore, "
         f"o limite inferior é {lower} potenciais pares, com limiar por passo, ou {global_lower} com limiar global. "
         r"São limites derivados do \textit{label}, não registros de carga observada."),
        (f"A revisão integral abaixo do corte é cenário de até {ceiling} potenciais pares, não carga histórica medida.",
         f"A revisão integral abaixo do corte é cenário de revisão de todos os {ceiling} candidatos, "
         "não carga histórica medida."),
        ("CdR, revisão humana da área cinza &", "CdR, revisão humana da área cinza da nota &"),
        ("A revisão da área cinza do CdR é cenário", "A revisão da área cinza da nota do CdR é cenário"),
        ("Zero FP acima do corte também decorre do rótulo.",
         r"Zero FP acima do corte também decorre do \textit{label}."),
        ("revisão do serviço incorporada ao rótulo", r"revisão do serviço incorporada ao \textit{label}"),
        ("n.a.: não se aplica.",
         "O limiar 8 da nota é referência histórica do Estudo~3, não limiar escolhido para P. n.a.: não se aplica."),
        (r"Fontes: \path{data/referencia_comparador_pos_orl.csv}",
         r"Fontes: \path{data/carga_abaixo_corte.csv}, \path{data/referencia_comparador_pos_orl.csv}"),
    ]
    for old, new in changes:
        text = replace_once(text, old, new)
    return text


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
        if "A posteriori" in line:
            line = replace_once(line, "A posteriori", r"\textit{A posteriori}")
        if "O a posteriori reutiliza" in line:
            line = replace_once(line, "O a posteriori reutiliza", r"O \textit{a posteriori} reutiliza")
        if "área cinza" in line:
            line = replace_once(line, "área cinza", "área cinza da nota")
        lines.append(line)
    return "\n".join(lines) + "\n"


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
                        "O padrão-ouro compartilha o corte do escore OpenRecLink e os dados de treinamento")
    text = replace_once(text, "corte SCORE do OpenRecLink", "corte do escore OpenRecLink")
    text = replace_once(text, r"\end{minipage}", step_ppv_note(
        frames["maxdiff"], frames["group_metrics"], frames["pairwise_tests"]) + r"\end{minipage}")
    target.write_text(text, encoding="utf-8")
    for mode in ("vigilancia", "confirmacao"):
        target = tables_dir / f"tab_equidade_arranjo_grupos_{mode}.tex"
        equity.table_groups(frames["group_metrics"], mode, target)
        text = replace_once(target.read_text(encoding="utf-8"), "Grupos MISSING", "Grupos sem informação")
        target.write_text(text, encoding="utf-8")
    for decision in equity.DECISIONS:
        name = f"tab_equidade_arranjo_rejeicoes_{decision}.tex"
        (tables_dir / name).write_bytes((ROOT / "text/latex/tables" / name).read_bytes())
    sources = [pd.read_csv(data_dir / name) for name in (
        "referencia_comparador_pos_orl.csv", "arranjo_corte_metricas.csv",
        "funil_gzcmd_final_metricas.csv", "carga_abaixo_corte.csv")]
    (tables_dir / "tab_cap8_referencia.tex").write_text(render_reference(*sources), encoding="utf-8")
    name = "tab_cap8_arranjo_corte.tex"
    historical = (ROOT / "text/latex/tables" / name).read_text(encoding="utf-8")
    (tables_dir / name).write_text(render_arrangement(historical), encoding="utf-8")


if __name__ == "__main__":
    generate(ROOT / "data", ROOT / "text/latex/tables/editorial")
    print("Tabelas editoriais regeneradas; fontes históricas preservadas.")
