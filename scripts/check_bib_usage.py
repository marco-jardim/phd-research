"""Confere as chaves citadas na tese contra o .bib usado por ``tese.tex``.

* Considera só a árvore de ``\\input``/``\\include`` de ``text/latex/tese.tex`` (arquivos como
  ``tese_rev_diff.tex`` ficam de fora) e o(s) .bib declarado(s) em ``\\bibliography{...}``.
* Os caminhos partem da raiz do repositório (``--raiz``), não do diretório atual.
* Código de saída: 1 se alguma chave citada não estiver no .bib; 0 caso contrário. Entradas do
  .bib nunca citadas saem como aviso, e só viram falha com ``--estrito``. 2 indica erro de uso
  (``tese.tex`` ou ``\\bibliography`` ausentes).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import citacoes_contexto as cc  # noqa: E402

_COMMAND_RE = re.compile(r"\\([A-Za-z]+)(\*?)")


def cited_keys(doc: cc.Document) -> set[str]:
    """Chaves citadas no corpo do documento (entre \\begin{document} e \\end{document})."""
    keys: set[str] = set()
    body = doc.text[: doc.body_end]
    for match in _COMMAND_RE.finditer(body, doc.body_start):
        call = cc.parse_cite(body, match)
        if call is not None:
            keys.update(call.keys)
    return keys


def analyse(raiz: Path, tex: str = "text/latex/tese.tex") -> dict:
    """Chaves do .bib, citadas, órfãs e ausentes, mais avisos de leitura."""
    tex_path = raiz / tex
    if not tex_path.is_file():
        raise FileNotFoundError(f"arquivo não encontrado: {tex_path}")
    doc = cc.build_document(tex_path, raiz)
    if not cc.find_bib_files(doc):
        raise ValueError(f"{tex_path} não declara \\bibliography{{...}}")
    entries, bib_files = cc.load_bib(doc)
    bib_keys = set(entries)
    cited = cited_keys(doc)
    return {
        "bib_files": bib_files,
        "bib_keys": bib_keys,
        "cited": cited,
        "orphan": sorted(bib_keys - cited),
        "phantom": sorted(cited - bib_keys),
        "warnings": list(doc.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--raiz", default=str(ROOT), help="raiz do repositório (padrão: a deste script)")
    parser.add_argument("--tex", default="text/latex/tese.tex", help="arquivo raiz da tese, relativo à raiz")
    parser.add_argument("--estrito", action="store_true", help="entradas do .bib nunca citadas também falham")
    args = parser.parse_args(argv)
    try:
        result = analyse(Path(args.raiz).resolve(), args.tex)
    except (FileNotFoundError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2

    for warning in result["warnings"]:
        print(f"aviso: {warning}", file=sys.stderr)
    orphan, phantom = result["orphan"], result["phantom"]
    print(f"Total .bib entries: {len(result['bib_keys'])}")
    print(f"Total cited keys:   {len(result['cited'])}")
    print(f"\nOrphan (in .bib but NOT cited in any .tex): {len(orphan)}")
    for key in orphan:
        print(f"  - {key}")
    print(f"\nPhantom (cited in .tex but NOT in .bib): {len(phantom)}")
    for key in phantom:
        print(f"  - {key}")

    if phantom:
        print(f"ERRO: {len(phantom)} chave(s) citada(s) ausente(s) do .bib", file=sys.stderr)
        return 1
    if orphan:
        label = "ERRO" if args.estrito else "aviso"
        print(f"{label}: {len(orphan)} entrada(s) do .bib nunca citada(s)", file=sys.stderr)
        return 1 if args.estrito else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
