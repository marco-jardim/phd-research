"""Verifica a importação sem depender do checkout externo do CdR."""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "importar_equivalencia_java.py"
SPEC = importlib.util.spec_from_file_location("importar_equivalencia_java", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def test_import_preserves_source_values_and_generates_table(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = [
        f"{who}_{kind}"
        for who in ("paciente", "mae")
        for kind in ("primeiro", "ultimo", "iguais", "raros", "comuns", "soundex", "abreviacoes")
    ] + ["data_igual", "data_um_digito", "data_inverte_dia", "data_inverte_mes", "data_inverte_ano", "nota_final"]
    rows = [
        {"indice": str(i), "subescore": name, "n": "500", "concordancia": "0.112", "diferenca_maxima": "0.25", "causa": "Precisao e contador"}
        for i, name in enumerate(names)
    ]
    with (source / "resultado_agregado.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*IMPORTER.FIELDS, "causa"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    (source / "relatorio_equivalencia.md").write_text(
        'Seed 20150901\n{"data_utc": "2026-11-04T20:23:05+00:00"}', encoding="utf-8"
    )
    output = tmp_path / "output"
    command = [sys.executable, str(SCRIPT), "--source-dir", str(source), "--output-root", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    artifacts = sorted(p for p in output.rglob("*") if p.is_file())
    first_run = {p: p.read_bytes() for p in artifacts}
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert {p: p.read_bytes() for p in artifacts} == first_run
    with (output / "data/equivalencia_java_python.csv").open(encoding="utf-8", newline="") as stream:
        actual = list(csv.DictReader(stream))
    assert len(actual) == 20
    assert [{key: row[key] for key in IMPORTER.FIELDS} for row in actual] == [
        {key: row[key] for key in IMPORTER.FIELDS} for row in rows
    ]
    meta = json.loads((output / "data/equivalencia_java_python_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 20150901
    assert meta["data_utc"] == "2026-11-04T20:23:05+00:00"
    assert meta["commit"] == "0d5b411"
    assert meta["n"] == 500
    table = (output / "text/latex/tables/tab_equivalencia_java.tex").read_text(encoding="utf-8")
    assert table.count(" & 11,2 & 0,25") == 20
    assert r"\label{tab:equivalencia-java}" in table
    assert r"\footnotesize" in table
    assert "não estima a concordância em dados reais" in table


@pytest.mark.parametrize(
    ("name", "cause", "expected"),
    [
        ("data_igual", "Sem divergencia observada", "sem divergência"),
        ("paciente_raros", "Frequencias e nome unico", "tabelas de frequência; nome único; contador"),
        ("mae_soundex", "Soundex adaptado", "Soundex; contador; precisão"),
        ("data_inverte_dia", "Inversoes", "peso da inversão; reconhecimento"),
        ("nota_final", "Soma dos comportamentos", "soma dos componentes"),
    ],
)
def test_mechanism_summary(name, cause, expected):
    assert IMPORTER.mecanismo({"subescore": name, "causa": cause}) == expected
