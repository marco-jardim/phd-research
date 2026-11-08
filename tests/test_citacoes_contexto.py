"""Testes do extrator de contextos de citação (scripts/citacoes_contexto.py)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.citacoes_contexto import (
    author_label,
    main,
    normalise_doi,
    parse_bibtex,
    run,
)

BIB = """@article{a,
  author = {Silva, Ana and Souza, Bia},
  title = {Um {Título}},
  year = {2020},
  journal = {Revista},
  doi = {https://doi.org/10.1000/ABC}
}
@book{b, author="Lima, Carlos", title="Livro", year=2019, publisher="Ed"}
@comment{@article{fake, title={x}}}
@string{foo = "bar"}
"""


def tese(body: str) -> str:
    return (
        "\\documentclass{abntex2}\n\\begin{document}\n"
        + body
        + "\n\\bibliography{refs}\n\\end{document}\n"
    )


def extract(tmp_path: Path, body: str, files: dict[str, str] | None = None, bib: str = BIB):
    tmp_path.mkdir(parents=True, exist_ok=True)
    allfiles = {"tese.tex": tese(body), "refs.bib": bib}
    allfiles.update(files or {})
    for rel, content in allfiles.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    result = run(tmp_path / "tese.tex", tmp_path / "out", base=tmp_path)
    return result


def by_key(records: list[dict], key: str) -> list[dict]:
    return [r for r in records if r["key"] == key]


def test_nested_input_and_chapter_environment(tmp_path):
    files = {
        "cap1.tex": "\\begin{chapter}{Introdução}\nTexto inicial.\n\\input{secoes/a}\n\\end{chapter}\n",
        "secoes/a.tex": "\n\\section{Parte}\\subsection{Detalhe}\nA frase citada \\cite{a}.\n",
    }
    result = extract(tmp_path, "\\input{cap1}", files)
    (rec,) = result["records"]
    assert rec["key"] == "a"
    assert rec["file"] == "secoes/a.tex"
    assert rec["line"] == 3
    assert rec["chapter"] == "Introdução"
    assert rec["section_path"] == ["Parte", "Detalhe"]
    assert rec["environment"] == "prosa"
    assert rec["apendice"] is False


def test_chapter_command_and_appendix(tmp_path):
    body = (
        "\\begin{chapter}{Um}\nTexto \\cite{a}.\n\\end{chapter}\n"
        "\\appendix\n\\chapter{Apêndice X}\nOutro \\cite{b}.\n"
    )
    result = extract(tmp_path, body)
    rec_a = by_key(result["records"], "a")[0]
    rec_b = by_key(result["records"], "b")[0]
    assert (rec_a["chapter"], rec_a["apendice"]) == ("Um", False)
    assert (rec_b["chapter"], rec_b["apendice"]) == ("Apêndice X", True)


def test_footnote_citation(tmp_path):
    body = "Frase principal.\\footnote{Nota com \\cite{a}.} Outra frase \\cite{b}.\n"
    result = extract(tmp_path, body)
    rec_a = by_key(result["records"], "a")[0]
    rec_b = by_key(result["records"], "b")[0]
    assert rec_a["environment"] == "nota"
    assert rec_a["sentence_text"] == "Nota com [cit]."
    assert rec_b["environment"] == "prosa"
    assert rec_b["sentence_text"] == "Outra frase [cit]."
    assert rec_b["paragraph_text"] == "Frase principal. Outra frase [cit]."


def test_table_and_caption_citations(tmp_path):
    body = "\\begin{table}\n\\caption{Legenda \\cite{b}}\n\\input{tables/t1}\n\\end{table}\n"
    files = {
        "tables/t1.tex": "\\begin{tabular}{ll}\nItem & Fonte \\cite{a} \\\\\n\\end{tabular}\n",
    }
    result = extract(tmp_path, body, files)
    rec_a = by_key(result["records"], "a")[0]
    rec_b = by_key(result["records"], "b")[0]
    assert rec_a["environment"] == "tabela"
    assert rec_a["file"] == "tables/t1.tex"
    assert "Fonte [cit]" in rec_a["sentence_text"]
    assert rec_b["environment"] == "legenda"


def test_list_item_environment(tmp_path):
    body = "\\begin{itemize}\n\\item Primeiro \\cite{a}.\n\\item Segundo \\cite{b}.\n\\end{itemize}\n"
    result = extract(tmp_path, body)
    rec_a = by_key(result["records"], "a")[0]
    assert rec_a["environment"] == "item"
    assert rec_a["paragraph_text"] == "Primeiro [cit]."


def test_optional_page_argument(tmp_path):
    result = extract(tmp_path, "Conforme \\citeonline[p.~10]{a} a regra vale.\n")
    (rec,) = result["records"]
    assert rec["cite_command"] == "citeonline"
    assert rec["args"] == ["p.~10"]
    assert rec["sentence_text"] == "Conforme [cit] a regra vale."


def test_multiple_keys_in_one_command(tmp_path):
    result = extract(tmp_path, "Resultado antigo \\cite{a, b}.\n")
    assert [r["key"] for r in result["records"]] == ["a", "b"]
    assert result["records"][0]["co_cited"] == ["b"]
    assert result["records"][1]["co_cited"] == ["a"]


def test_apud(tmp_path):
    result = extract(tmp_path, "Segundo \\apud{a}{b}, vale isso.\n")
    assert sorted(r["key"] for r in result["records"]) == ["a", "b"]
    assert all(r["cite_command"] == "apud" for r in result["records"])
    assert by_key(result["records"], "a")[0]["co_cited"] == ["b"]


def test_commented_citation_is_ignored(tmp_path):
    body = "Visível \\cite{a}.\n% \\cite{b}\nFim %\\cite{b}\n"
    result = extract(tmp_path, body)
    assert [r["key"] for r in result["records"]] == ["a"]
    row_b = next(r for r in result["rows"] if r["key"] == "b")
    assert row_b["status"] == "nao_citada"


def test_comment_line_does_not_split_paragraph(tmp_path):
    body = "Primeira parte\n% comentário solto\nsegunda parte \\cite{a}.\n"
    result = extract(tmp_path, body)
    assert result["records"][0]["sentence_text"] == "Primeira parte segunda parte [cit]."


def test_missing_key_is_reported(tmp_path):
    result = extract(tmp_path, "Algo \\cite{zzz}.\n")
    row = next(r for r in result["rows"] if r["key"] == "zzz")
    assert row["status"] == "ausente_no_bib"
    resumo = (tmp_path / "out" / "resumo.md").read_text(encoding="utf-8")
    assert "`zzz`" in resumo


def test_uncited_bib_entry_is_reported(tmp_path):
    result = extract(tmp_path, "Algo \\cite{a}.\n")
    row = next(r for r in result["rows"] if r["key"] == "b")
    assert row["status"] == "nao_citada"
    assert row["n_contexts"] == 0
    resumo = (tmp_path / "out" / "resumo.md").read_text(encoding="utf-8")
    assert "## Entradas do .bib nunca citadas" in resumo
    assert "- `b`" in resumo


def test_doi_normalisation(tmp_path):
    assert normalise_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalise_doi("http://dx.doi.org/10.1000/X") == "10.1000/x"
    assert normalise_doi("doi:10.1000/Y") == "10.1000/y"
    result = extract(tmp_path, "Algo \\cite{a}.\n")
    row = next(r for r in result["rows"] if r["key"] == "a")
    assert row["doi"] == "10.1000/abc"
    assert row["author"] == "Silva et al."
    assert row["venue"] == "Revista"
    assert row["status"] == "citada"


def test_stable_id_for_same_key_and_sentence(tmp_path):
    first = extract(tmp_path / "x1", "Frase um \\cite{a}.\n")
    second = extract(
        tmp_path / "x2",
        "\n\n% comentário\nOutra coisa antes.\n\nFrase   um\n\\cite{a}.\n",
    )
    id_first = by_key(first["records"], "a")[0]["id"]
    id_second = next(
        r["id"] for r in second["records"] if r["key"] == "a" and r["sentence_text"].startswith("Frase")
    )
    assert id_first == id_second
    assert len(id_first) == 16
    other = extract(tmp_path / "x3", "Frase um \\cite{b}.\n")
    assert other["records"][0]["id"] != id_first


def test_repeated_key_in_same_sentence_is_merged(tmp_path):
    result = extract(tmp_path, "Texto \\cite{a} e depois \\citeyearpar{a}.\n")
    (rec,) = result["records"]
    assert len(rec["outras_ocorrencias"]) == 1
    resumo = (tmp_path / "out" / "resumo.md").read_text(encoding="utf-8")
    assert "Ocorrências de citação (chave por comando): 2" in resumo


def test_sentence_mapping_with_two_citations(tmp_path):
    body = "A primeira frase afirma X \\cite{a}. A segunda frase afirma Y \\cite{b}.\n"
    result = extract(tmp_path, body)
    rec_a = by_key(result["records"], "a")[0]
    rec_b = by_key(result["records"], "b")[0]
    assert rec_a["sentence_text"] == "A primeira frase afirma X [cit]."
    assert rec_b["sentence_text"] == "A segunda frase afirma Y [cit]."
    assert rec_a["paragraph_text"] == "A primeira frase afirma X [cit]. A segunda frase afirma Y [ref]."
    assert rec_b["paragraph_text"] == "A primeira frase afirma X [ref]. A segunda frase afirma Y [cit]."
    assert rec_a["sentence_latex"] == "A primeira frase afirma X \\cite{a}."
    assert rec_a["id"] != rec_b["id"]


def test_abbreviations_do_not_split_sentences(tmp_path):
    body = (
        "Segundo Silva et al. Mostraram o efeito \\cite{a}. Nova frase.\n\n"
        "Veja Fig. 3 e p. 10 sobre o tema \\cite{b}. Fim.\n"
    )
    result = extract(tmp_path, body)
    assert by_key(result["records"], "a")[0]["sentence_text"] == (
        "Segundo Silva et al. Mostraram o efeito [cit]."
    )
    assert by_key(result["records"], "b")[0]["sentence_text"] == (
        "Veja Fig. 3 e p. 10 sobre o tema [cit]."
    )


def test_pgf_and_figures_are_skipped(tmp_path):
    body = (
        "\\resizebox{\\textwidth}{!}{\\input{fig_doc/x.pgf}}\n\n"
        "\\input{figures/foo}\n\nTexto \\cite{a}.\n"
    )
    files = {"figures/foo.tex": "Oculto \\cite{zzz}.\n", "fig_doc/x.pgf": "\\cite{yyy}\n"}
    result = extract(tmp_path, body, files)
    assert [r["key"] for r in result["records"]] == ["a"]
    assert result["warnings"] == []


def test_missing_input_file_gives_warning(tmp_path):
    result = extract(tmp_path, "\\input{nao_existe}\n\nTexto \\cite{a}.\n")
    assert any("nao_existe" in w for w in result["warnings"])
    assert [r["key"] for r in result["records"]] == ["a"]


def test_bibtex_parser_handles_quotes_macros_and_skips(tmp_path):
    text = (
        "@Article{k1,\n  Author = {Souza, A. and Lima, B. and Costa, C},\n"
        "  TITLE = {{Double} Braced   Title},\n  year = 2021,\n  month = jan,\n"
        '  journal = "Rev " # "Sau{\\\'u}de"\n}\n'
        "@preamble{\"x\"}\n@string{s = \"v\"}\n@comment{@article{fake, title={x}}}\n"
        "@misc{k2, author = {{Ministério da Saúde}}, howpublished = {Web}}\n"
    )
    entries, warnings = parse_bibtex(text)
    assert warnings == []
    assert sorted(entries) == ["k1", "k2"]
    k1 = entries["k1"]
    assert k1.entry_type == "article"
    assert k1.fields["title"] == "{Double} Braced Title"
    assert k1.fields["year"] == "2021"
    assert k1.fields["month"] == "jan"
    assert k1.fields["journal"] == "Rev Sau{\\'u}de"
    assert author_label(k1.fields) == "Souza et al."
    assert author_label(entries["k2"].fields) == "Ministério da Saúde"


def test_bibtex_duplicate_key_warns():
    entries, warnings = parse_bibtex("@misc{x, title={1}}\n@misc{x, title={2}}\n")
    assert entries["x"].fields["title"] == "1"
    assert len(warnings) == 1


def test_cli_writes_outputs(tmp_path, capsys):
    extract(tmp_path / "pre", "Texto \\cite{a}.\n")
    tex = tmp_path / "pre" / "tese.tex"
    out = tmp_path / "saida"
    assert main(["--tex", str(tex), "--out", str(out)]) == 0
    assert "contextos: 1" in capsys.readouterr().out

    lines = (out / "citacoes.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    expected = {
        "id", "key", "cite_command", "args", "file", "line", "chapter", "section_path",
        "environment", "sentence_latex", "sentence_text", "paragraph_text", "co_cited",
    }
    assert expected <= set(record)
    assert not any(k.startswith("_") for k in record)

    with (out / "referencias.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert [r["key"] for r in rows] == ["a", "b"]
    assert rows[0]["status"] == "citada"
    assert rows[0]["files"] == "tese.tex" or rows[0]["files"].endswith("tese.tex")
    assert (out / "resumo.md").read_text(encoding="utf-8").startswith("# Resumo")


def test_cli_missing_tex_returns_error(tmp_path, capsys):
    assert main(["--tex", str(tmp_path / "nada.tex"), "--out", str(tmp_path / "o")]) == 2
    assert "não encontrado" in capsys.readouterr().err
