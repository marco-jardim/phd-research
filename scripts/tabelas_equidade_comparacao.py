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
from scripts.tabelas_persona_editorial import transform

TABLES = ("tab_equidade_v31.tex", "tab_equidade_v31_rejeicoes_cascata.tex",
          "tab_qw3_group_metrics.tex")


def generate(data_dir: Path, tables_dir: Path, *, editorial_notes: bool = False) -> None:
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
        # Rótulo histórico (tables/) preservado; o nome editorial "Cascata sobre a base inteira" só na camada editorial.
        rotulo_linha = "Cascata sobre a base inteira (" if editorial_notes else "Base inteira v3.1, comparação ("
        substitutions = [("cascata v3.1 (", "cascata v3.1 sobre a base inteira, comparação ("),
                         ("Cascata v3.1 (", rotulo_linha),
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
        if name == TABLES[0] and editorial_notes:
            # Camada editorial; a tabela histórica mantém uma casa e o cabeçalho "Cascata vig./conf.".
            # Mesma precisão da tabela den30: especificidade com duas casas em todas as decisões.
            text = _especificidade_duas_casas(text, frames["global"])
            old_header = "Cascata vig. & Cascata conf."
            if text.count(old_header) != 1:
                raise ValueError(f"Contrato editorial divergente: {name}, {old_header!r}")
            text = text.replace(old_header, r"\shortstack{Cascata sobre\\a base inteira\\(vigil.)} & "
                                r"\shortstack{Cascata sobre\\a base inteira\\(confirm.)}")
        rejections = (r"A contagem de rejeições usa a família histórica sem filtro por denominador; "
                      r"as diferenças máximas exigem denominadores de pelo menos 30. São critérios "
                      r"distintos, não uma única família filtrada. ")
        universe = (r"Universo: 61.696 potenciais pares, 247 vínculos; "
                    r"a cobertura por dimensão pode ser menor. ")
        tie_note = r"Os grupos históricos do nome contêm empates, não quatro grupos de mesmo tamanho."
        if name == TABLES[2]:
            # Esta tabela traz só sensibilidade e VPP por grupo: sem rejeições, diferenças máximas nem OR.
            source = (r"\par\footnotesize Fonte: \path{data/equidade_v31_group_metrics.csv}. " + universe +
                      r"Grupos sem denominador na métrica aparecem como n.a. " + tie_note)
        else:
            source = (r"\par\footnotesize Fonte: \path{data/equidade_v31_global.csv}, "
                      r"\path{data/equidade_v31_group_metrics.csv}, "
                      r"\path{data/equidade_v31_pairwise_tests.csv} e "
                      r"\path{data/equidade_v31_maxdiff.csv}. " + universe + rejections +
                      ("" if name == TABLES[0] else r"OR é razão de chances. ") + tie_note)
        if editorial_notes:
            text = replace_once(text, r"\end{table}", source + "\n" + r"\end{table}")
        # Nenhuma destas tabelas traz ASR nem REFREC: transform (SEM_ASR) omite essas glosas.
        path.write_text(transform(name, text) if editorial_notes else text, encoding="utf-8", newline="\n")


def _especificidade_duas_casas(text, glob):
    """Reescreve a coluna Espec. da parte superior com duas casas, conferindo a célula de uma casa."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(r"\midrule")) + 1
    for offset, espec in enumerate(glob.espec):
        cells = lines[start + offset].split(" & ")
        if cells[7] != f"{100 * espec:.1f}".replace(".", "{,}"):
            raise ValueError(f"Especificidade divergente na linha {start + offset}: {cells[7]!r}")
        cells[7] = f"{100 * espec:.2f}".replace(".", "{,}")
        lines[start + offset] = " & ".join(cells)
    return "\n".join(lines)


if __name__ == "__main__":
    generate(ROOT / "data", ROOT / "text/latex/tables/editorial", editorial_notes=True)
    print("Regeneradas:", ", ".join(TABLES))
