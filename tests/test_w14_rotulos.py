"""Rótulos residuais do fechamento, sem modificar resultados históricos."""

from pathlib import Path
import re

from scripts.tabelas_persona_editorial import transform
from scripts.tabelas_review2_w3 import ablation_tables

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def test_residual_fusion_labels_have_local_definitions():
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "com a fusão F4 (posição recíproca de O1 e O3)" in conclusion
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    bias = study.split("O diagnóstico de VPP inferior", 1)[1].split("\n\n", 1)[0]
    assert "F4 (fusão por posição recíproca de O1 e O3)" in bias
    prospective = study.split("Como critério prospectivo de falsificação", 1)[1].split("\n\n", 1)[0]
    assert "F4 é a fusão por posição recíproca de O1 e O3" in prospective
    assert "CM é a fila com atributos de contexto por óbito e MACD" in prospective


def test_ablation_editorial_labels_regenerate_and_are_idempotent(tmp_path):
    ablation_tables(tmp_path)
    for name in ("tab_ablation_best_category.tex", "tab_ablation_top10.tex"):
        generated = (tmp_path / name).read_bytes()
        assert generated == (LATEX / "tables/editorial" / name).read_bytes()
        text = generated.decode("utf-8")
        assert "teste interno QW1 (comparação de classificadores tabulares)" in text
        assert not re.search(r"\bF1\b", text)
        assert transform(name, text) == text


def test_reruns_caption_expands_experiment_label():
    name = "tab_cap8_reruns.tex"
    original = (LATEX / "tables" / name).read_text(encoding="utf-8")
    editorial = (LATEX / "tables/editorial" / name).read_text(encoding="utf-8")
    assert "no Experimento 2 (revisão assistida por LLM)" in editorial
    assert transform(name, original) == editorial
