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
    text = text.replace("O gabarito compartilha", "A referência operacional compartilha").replace(
        "corte SCORE do OpenRecLink", "corte do escore OpenRecLink")
    # Especificidade com duas casas, como na tabela editorial principal e no texto do Estudo 4.
    for decision, label in historical.LABELS.items():
        row = glob.loc[glob.decisao.eq(decision)].iloc[0]
        original = next(line for line in text.splitlines() if line.startswith(label + " &"))
        cells = original.split(" & ")
        cells[7] = f"{100 * row.espec:.2f}".replace(".", "{,}")
        text = text.replace(original, " & ".join(cells), 1)
    path.write_text(text, encoding="utf-8", newline="\n")
    for decision in historical.DECISIONS:
        path = editorial / f"tab_den30_equidade_arranjo_rejeicoes_{decision}.tex"
        historical.legacy.tabela_rejeicoes(
            corrected, path, decisao=decision,
            label=f"tab:equidade-den30-{decision.replace('_', '-')}",
            descr=historical.LABELS[decision] + ", denominadores de pelo menos 30")
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    manifest = dict(denominador_minimo=30, regra="Excluir n1<30 ou n2<30 antes de BH por decisão",
                    fonte="equidade_arranjo_pairwise_tests.csv", treino_executado=False,
                    geradores_historicos_preservados=True, comparacoes=audit)
    (data / "equidade_arranjo_den30_recomputacao.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return corrected


if __name__ == "__main__":
    generate(ROOT / "data", ROOT / "text/latex/tables/editorial")
