from pathlib import Path

from scripts.finalizar_pdf_refrec import audit_log, is_build_artifact


def test_cleanup_keeps_sources_and_canonical_pdfs(tmp_path):
    for name in ("tese.pdf", "tese_final.tex", "tese_b1_check.pdf", "tese_refrec.log", "tese_rev_x.aux", "capitulo8.tex"):
        path = tmp_path / name
        path.write_text("example", encoding="utf-8")
        expected = name in {"tese_b1_check.pdf", "tese_refrec.log", "tese_rev_x.aux"}
        assert is_build_artifact(path) == expected


def test_build_validation_rejects_errors_and_undefined_citations():
    assert audit_log("Output written on tese_refrec2.pdf") == ([], [])
    errors, undefined = audit_log("! LaTeX Error: bad input\nLaTeX Warning: Citation `missing' on page 1 undefined\nLaTeX Warning: There were undefined references.")
    assert len(errors) == 1 and len(undefined) == 2


def test_row_archive_preserves_historical_table_bytes_outside_active_tex(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import preservar_m5_m7_row as preserve
    tables = tmp_path / "text/latex/tables"
    tables.mkdir(parents=True)
    (tmp_path / "data").mkdir()
    original = b"historical table [a confirmar]\r\n"
    for name in ("tab_cap8_ancoras_ml.tex", "tab_cap8_ancoras_beta.tex"):
        (tables / name).write_bytes(original)
    monkeypatch.setattr(preserve, "ROOT", tmp_path)
    preserve.main()
    preserve.main()
    entries = json.loads((tmp_path / "data/m5_m7_preservados_row.json").read_text())
    assert len(entries) == 2
    for entry in entries:
        path = tmp_path / entry["archive"]
        assert path.suffix == ".txt" and path.read_bytes() == original
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert not list(tables.glob("*_row.tex"))
