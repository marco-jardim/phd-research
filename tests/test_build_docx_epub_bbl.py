"""Checks for scripts/build_docx.py e build_epub.py: localizacao do .bbl e pin do container do CI."""

import importlib.util
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _carregar(nome, tmp_path):
    spec = importlib.util.spec_from_file_location(nome, RAIZ / "scripts" / f"{nome}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SRC_DIR = tmp_path / "latex"
    module.SRC_DIR.mkdir(parents=True)
    return module


@pytest.fixture(params=["build_docx", "build_epub"])
def modulo(request, tmp_path):
    return _carregar(request.param, tmp_path)


def test_bbl_em_build_final_e_preferido(modulo):
    antigo = modulo.SRC_DIR / "tese.bbl"
    novo = modulo.SRC_DIR / "build" / "final" / "tese.bbl"
    novo.parent.mkdir(parents=True)
    antigo.write_text("antigo", encoding="utf-8")
    novo.write_text("novo", encoding="utf-8")
    assert modulo.find_bbl() == novo


def test_bbl_so_em_build_final(modulo):
    novo = modulo.SRC_DIR / "build" / "final" / "tese.bbl"
    novo.parent.mkdir(parents=True)
    novo.write_text("novo", encoding="utf-8")
    assert modulo.find_bbl() == novo


def test_bbl_antigo_serve_de_fallback(modulo):
    antigo = modulo.SRC_DIR / "tese.bbl"
    antigo.write_text("antigo", encoding="utf-8")
    assert modulo.find_bbl() == antigo


def test_bbl_ausente_mantem_caminho_antigo_no_aviso(modulo, capsys):
    assert modulo.find_bbl() == modulo.SRC_DIR / "tese.bbl"
    (modulo.SRC_DIR / "tese.tex").write_text("test", encoding="utf-8")
    modulo.find_pandoc = lambda: "pandoc-falso"
    modulo.OUT_DIR = modulo.SRC_DIR.parent / "saida"
    modulo.run = lambda *a, **k: 0
    modulo.build()
    assert f"[AVISO] {modulo.SRC_DIR / 'tese.bbl'} nao encontrado" in capsys.readouterr().err


def test_build_subdir_igual_ao_de_build_pdf(tmp_path):
    pdf = _carregar("build_pdf", tmp_path / "a")
    for nome in ("build_docx", "build_epub"):
        assert _carregar(nome, tmp_path / nome).BUILD_SUBDIR == pdf.BUILD_SUBDIR


def test_container_do_ci_fixado_por_digest():
    texto = (RAIZ / ".github" / "workflows" / "latex.yml").read_text(encoding="utf-8")
    linha = next(
        item for item in texto.splitlines() if item.strip().startswith("container: texlive/")
    )
    assert re.search(r"texlive/texlive:latest@sha256:[0-9a-f]{64}\s*$", linha), linha
