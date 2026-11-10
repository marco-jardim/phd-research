"""Seleção das fontes .tex versionadas, sem os artefatos locais do latexdiff."""
from pathlib import Path


def artefato_latexdiff(path: Path) -> bool:
    """True para `*_diff.tex` e para qualquer arquivo dentro de `*_diff_work/`."""
    return path.name.endswith('_diff.tex') or any(part.endswith('_diff_work') for part in path.parts)


def fontes_tex(base: Path, recursivo: bool = False):
    """Arquivos .tex de `base`, excluindo os artefatos gerados pelo latexdiff."""
    encontrados = base.rglob('*.tex') if recursivo else base.glob('*.tex')
    return [path for path in encontrados if not artefato_latexdiff(path)]
