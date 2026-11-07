"""Integridade dos agregados completos publicados em acesso aberto."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "data/curvas_zenodo.json").read_text(encoding="utf-8"))


def test_public_deposit_metadata():
    assert MANIFEST["zenodo_doi"] == "10.5281/zenodo.24034389"
    assert MANIFEST["zenodo_concept_doi"] == "10.5281/zenodo.24034388"
    assert MANIFEST["zenodo_url"] == "https://zenodo.org/records/24034389"
    assert MANIFEST["licenca"] == "CC-BY-4.0"
    assert MANIFEST["acesso"] == "open"
    expected = {
        f"data/{prefix}{suffix}"
        for prefix in ("fusao_filas", "contexto_obito", "sensibilidade_o3_macd")
        for suffix in ("_curva.csv", "_metadata.json")
    }
    assert len(MANIFEST["arquivos"]) == len(expected)
    assert {entry["nome"] for entry in MANIFEST["arquivos"]} == expected


@pytest.mark.parametrize("entry", MANIFEST["arquivos"], ids=lambda entry: entry["nome"])
def test_published_files_match_git_and_checkout(entry):
    blob = subprocess.run(
        ["git", "show", "HEAD:" + entry["nome"]],
        cwd=ROOT, capture_output=True, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    ).stdout
    assert len(blob) == entry["bytes"]
    assert hashlib.sha256(blob).hexdigest() == entry["sha256"]
    # As regras explícitas de EOL preservam os mesmos bytes no checkout.
    assert (ROOT / entry["nome"]).read_bytes() == blob
