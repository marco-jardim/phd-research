"""Validate the planned queries without executing an institutional search."""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scopus_template_preserves_schema_and_pending_status():
    with (ROOT / "data/protocolo_busca_T6.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        source = csv.DictReader(stream)
        columns = source.fieldnames
        pubmed = {
            row["consulta"]: row["string_exata"]
            for row in source
            if row["base"] == "PubMed" and row["etapa"] == "consulta"
        }
    with (ROOT / "data/protocolo_busca_T6_scopus_template.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        template = csv.DictReader(stream)
        assert template.fieldnames == columns
        rows = list(template)

    instructions = (ROOT / "docs/support_text/scopus_T6_instrucoes.md").read_text(
        encoding="utf-8"
    )
    expressions = re.findall(r"```text\n([^`]+)\n```", instructions)
    assert len(rows) == len(expressions) == 6
    for index, (row, expression, original) in enumerate(
        zip(rows, expressions, ("P1", "P5", "P2", "P3", "P4", "P6")), 1
    ):
        assert None not in row
        assert all(value is not None for value in row.values())
        assert row["base"] == "Scopus"
        assert row["consulta"] == f"Sc{index}"
        assert row["etapa"] == "consulta"
        assert row["data_busca"] == row["n_recuperado"] == ""
        assert row["string_exata"] == expression
        assert "PUBYEAR > 2009 AND PUBYEAR < 2027" in expression
        assert "LANGUAGE(" not in expression
        assert original in row["motivo_ou_nota"]
        assert "não executada" in row["motivo_ou_nota"]
        assert set(re.findall(r'"([^"]+)"', pubmed[original])) == set(
            re.findall(r'"([^"]+)"', expression)
        )

    assert "AFFILCOUNTRY(Brazil)" in rows[3]["string_exata"]
    assert "Brazil" not in rows[4]["string_exata"]
    assert all("TITLE-ABS-KEY(" in row["string_exata"] for row in rows[:2])
    assert all("TITLE-ABS-KEY(" not in row["string_exata"] for row in rows[2:])
