"""Apresentação editorial separada da regressão histórica, sem novos ajustes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fechar_refrec_final import regenerate_cascade_table
from scripts.tabelas_arranjo_editorial import replace_once
from scripts.tabelas_equidade_comparacao import generate as equity


def generate(tables_dir=ROOT / "text/latex/tables/editorial"):
    if tables_dir.resolve() == (ROOT / "text/latex/tables").resolve():
        raise ValueError("Use tables/editorial; fontes históricas preservadas")
    equity(ROOT / "data", tables_dir, editorial_notes=True)
    target = tables_dir / "tab_cap8_cascata_v31.tex"
    regenerate_cascade_table(output_path=target)
    text = replace_once(target.read_text(encoding="utf-8"),
                        "Cascata final observada por política e modo,",
                        "Cascata sobre a base inteira (comparação), por política e modo,")
    target.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    generate()
