"""Apresentação da comparação histórica sem alterar a análise equidade_v31.

tab_equidade_v31_rejeicoes_cascata.tex é conservada como artefato de regressão
do gerador histórico. Não é incluída no capítulo: suas rejeições de vigilância
já constam de tab_equidade_arranjo_rejeicoes_cascata_v31_vigilancia.tex.
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import equidade_v31 as legacy
from scripts.tabelas_arranjo_editorial import replace_once

TABLES = ("tab_equidade_v31.tex", "tab_equidade_v31_rejeicoes_cascata.tex",
          "tab_qw3_group_metrics.tex")


def generate(data_dir: Path, tables_dir: Path) -> None:
    """Reproduz o contrato histórico congelado em diretório separado.

    As versões existentes em tables/ não são saídas literais de equidade_v31;
    sua restauração quebraria test_equidade_arranjo, fixado no manifesto.
    """
    if tables_dir.resolve() == (ROOT / "text/latex/tables").resolve():
        raise ValueError("Use tables/editorial; não sobrescrever a regressão histórica")
    tables_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(data_dir / f"equidade_v31_{name}.csv")
              for name in ("global", "group_metrics", "pairwise_tests", "maxdiff")}
    legacy.tabela_resumo(frames["global"], frames["maxdiff"], frames["pairwise_tests"],
                         tables_dir / TABLES[0])
    legacy.tabela_rejeicoes(frames["pairwise_tests"], tables_dir / TABLES[1],
                            decisao="cascata_v31_vigilancia",
                            label="tab:equidade-v31-rejeicoes-cascata",
                            descr="cascata v3.1 (vigil\\^ancia), aceitação automática")
    legacy.tabela_grupos(frames["group_metrics"], tables_dir / TABLES[2])
    for name in TABLES:
        path = tables_dir / name
        text = path.read_text(encoding="utf-8")
        expected = {TABLES[0]: (0, 2, 0, 1, 0), TABLES[1]: (1, 0, 0, 0, 1),
                    TABLES[2]: (1, 0, 1, 0, 1)}[name]
        substitutions = [("cascata v3.1 (", "cascata v3.1 sobre a base inteira, comparação ("),
                         ("Cascata v3.1 (", "Base inteira v3.1, comparação ("),
                         ("{Cascata v3.1}", "{Base inteira (comparação)}"),
                         (r"Equidade por decis\~ao:", "Equidade das decisões de comparação (corte 6,0, XGBoost e cascata sobre a base inteira):"),
                         (r"\small", r"\footnotesize")]
        for (old, new), count in zip(substitutions, expected):
            if text.count(old) != count:
                raise ValueError(f"Contrato editorial divergente: {name}, {old!r}")
            text = "\n".join(replace_once(line, old, new) if old in line else line
                             for line in text.split("\n"))
        if name != TABLES[0]:
            text = replace_once(text, r"\begin{tabular}", r"\resizebox{\linewidth}{!}{%" + "\n" + r"\begin{tabular}")
            text = replace_once(text, r"\end{tabular}", r"\end{tabular}}")
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    generate(ROOT / "data", ROOT / "text/latex/tables/editorial")
    print("Regeneradas:", ", ".join(TABLES))
