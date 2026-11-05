"""Apresentação S5 dos agregados, sem alterar os geradores registrados.

Os CSVs, cálculos e fingerprints históricos permanecem intactos. A camada
editorial explicita a precisão da especificidade, as exclusões e a carga.
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import equidade_arranjo as equity
from scripts import tabela_referencia_arranjo as reference


def render_reference(referencia, arranjo, cascata, carga):
    """Pós-processa o render histórico; limites vêm do CSV de carga."""
    text = reference.render(referencia, arranjo, cascata)
    totals = carga.loc[carga.passo.astype(str).eq("TOTAL")].set_index("limiar")
    lower = reference.inteiro(totals.loc["por_passo", "pares_acima_limiar"])
    global_lower = reference.inteiro(totals.loc["global", "pares_acima_limiar"])
    text = text.replace("scripts/tabela_referencia_arranjo.py", "scripts/tabelas_arranjo_editorial.py")
    text = text.replace("Corte + revisão manual do serviço & ambos & n.d.",
                        "Corte + revisão manual do serviço & ambos & $\\geq " + lower + "$")
    text = text.replace("até 61.485", "61.485 (teto)")
    text = text.replace(
        "O volume examinado e o limite inferior da revisão do serviço não estão documentados (n.d.).",
        f"O volume examinado total não foi registrado. Sob a hipótese de revisão em ordem decrescente de escore, "
        f"o limite inferior é {lower} potenciais pares, com limiar por passo, ou {global_lower} com limiar global. "
        "São limites derivados do rótulo, não registros de carga observada.")
    text = text.replace(
        "A revisão integral abaixo do corte é cenário de até 61.485 potenciais pares, não carga histórica medida.",
        "A revisão integral abaixo do corte usa o teto de candidatos, não a carga histórica medida.")
    # A substituição do teto na coluna também afeta a nota histórica.
    text = text.replace("cenário de 61.485 (teto) potenciais pares", "cenário de revisão de todos os 61.485 candidatos")
    text = text.replace(
        r"n.a.: não se aplica.",
        r"O limiar 8 da nota é referência histórica do Estudo~3, não limiar escolhido para P. n.a.: não se aplica.")
    text = text.replace(r"Fontes: \path{data/referencia_comparador_pos_orl.csv}",
                        r"Fontes: \path{data/carga_abaixo_corte.csv}, \path{data/referencia_comparador_pos_orl.csv}")
    return text


def generate(data_dir: Path, tables_dir: Path) -> None:
    """Grava só versões editoriais; nunca sobrescreve tabelas históricas."""
    if tables_dir.resolve() == (ROOT / "text/latex/tables").resolve():
        raise ValueError("Use tables/editorial; as tabelas históricas são imutáveis")
    tables_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(data_dir / f"equidade_arranjo_{name}.csv")
              for name in ("global", "maxdiff", "pairwise_tests", "group_metrics")}
    target = tables_dir / "tab_equidade_arranjo.tex"
    equity.table_summary(frames["global"], frames["maxdiff"], frames["pairwise_tests"], target)
    lines = target.read_text(encoding="utf-8").splitlines()
    for decision, label in equity.LABELS.items():
        row = frames["global"].loc[frames["global"].decisao.eq(decision)].iloc[0]
        for i, line in enumerate(lines):
            if line.startswith(label + " &"):
                cells = line.split(" & ")
                cells[7] = f"{100 * row.espec:.2f}".replace(".", "{,}")
                lines[i] = " & ".join(cells)
    text = "\n".join(lines) + "\n"
    text = text.replace("O gabarito compartilha o corte e os dados de treinamento",
                        "O padrão-ouro compartilha o corte do escore OpenRecLink e os dados de treinamento")
    text = text.replace("corte SCORE do OpenRecLink", "corte do escore OpenRecLink")
    text = text.replace(r"\end{minipage}",
                        r" O passo 11 ficou fora dos testes de VPP por ter menos de 30 preditos positivos; "
                        r"a comparação de sensibilidade usa outro denominador.\end{minipage}")
    target.write_text(text, encoding="utf-8")
    for mode in ("vigilancia", "confirmacao"):
        target = tables_dir / f"tab_equidade_arranjo_grupos_{mode}.tex"
        equity.table_groups(frames["group_metrics"], mode, target)
        text = target.read_text(encoding="utf-8").replace("Grupos MISSING", "Grupos sem informação")
        target.write_text(text, encoding="utf-8")
    # As tabelas de rejeições não precisam de transformação. As cópias mantêm
    # o capítulo inteiro sob um único contrato de apresentação.
    for decision in equity.DECISIONS:
        name = f"tab_equidade_arranjo_rejeicoes_{decision}.tex"
        historical = ROOT / "text/latex/tables" / name
        (tables_dir / name).write_bytes(historical.read_bytes())
    sources = [pd.read_csv(data_dir / name) for name in (
        "referencia_comparador_pos_orl.csv", "arranjo_corte_metricas.csv",
        "funil_gzcmd_final_metricas.csv", "carga_abaixo_corte.csv")]
    (tables_dir / "tab_cap8_referencia.tex").write_text(render_reference(*sources), encoding="utf-8")


if __name__ == "__main__":
    generate(ROOT / "data", ROOT / "text/latex/tables/editorial")
    print("Apresentação S5 regenerada dos CSVs; geradores e fingerprints históricos preservados.")
