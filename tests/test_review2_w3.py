"""Regressões da prosa e das tabelas editoriais W3, sem análise por par."""
from pathlib import Path
import re

import pandas as pd

from scripts import tabelas_review2_w3 as tables

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def test_editorial_w3_regenerates_without_changing_historical(tmp_path):
    historical = {p.name: p.read_bytes() for p in (LATEX / "tables").glob("*.tex")}
    tables.generate(tmp_path)
    for path in tmp_path.glob("*.tex"):
        assert path.read_bytes() == (LATEX / "tables/editorial" / path.name).read_bytes()
        for source in re.findall(r"\\path\{(data/[^}]+)\}", path.read_text(encoding="utf-8")):
            assert (ROOT / source).is_file(), source
    assert historical == {p.name: p.read_bytes() for p in (LATEX / "tables").glob("*.tex")}


def test_ablation_appendix_is_generated_expected_not_llm_observation():
    from test_w9a import expected_ablation_appendix

    text = (LATEX / "apendice_ablacao.tex").read_text(encoding="utf-8")
    assert text == expected_ablation_appendix()
    assert "sem chamadas a modelo de linguagem" in text
    assert "não é desempenho observado de um LLM" in text
    assert r"$\beta=0{,}5$" in text and r"$\beta=2{,}0$" in text
    assert "107,0" not in text
    for arm in ("macd", "nomacd"):
        source = f"data/ablation_comprec_{arm}_allowlist.csv"
        rows = pd.read_csv(ROOT / source, sep=";")
        assert source in text
        for count in rows["llm_used"]:
            assert count == int(count)
            assert f" & {int(count)} " + r"\\" in text
            assert f" & {int(count)},0 " + r"\\" not in text
        assert f"data/ablation_comprec_{arm}.csv" in text
    assert text.count("Cada teste contém 74 vínculos") == 4


def test_origin_and_family_importance_have_no_personal_attribution():
    text = (LATEX / "estudo1_comparador.tex").read_text(encoding="utf-8")
    assert "Daniela" not in text and "pela epidemiologista" not in text
    for term in ("OpenRecLink original em C", "revisora única", "extrato pós-CdR",
                 "54 colunas distintas", "38,5", "24,6", "não sustenta ordenar atributos redundantes",
                 "sem registro que permita distinguir"):
        if term == "sem registro que permita distinguir":
            assert term not in text
        else:
            assert term in text
    assert "importância de Gini no arquivo denominado" in text
    assert "com pesos proporcionais ao poder discriminatório" not in text


def test_reference_and_uncertainty_are_explicit_without_mechanism_claims():
    for name in ("estudo2_classificadores.tex", "estudo3_estrategias.tex", "apendice_sensibilidade.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "referência" in text and "D" in text
        assert "0,846" in text or "0{,}846" in text
    e3 = (LATEX / "estudo3_estrategias.tex").read_text(encoding="utf-8")
    assert "não estima isoladamente esse otimismo" in e3
    assert "não permitem distinguir as dez primeiras" in e3
    assert "sugere possível contribuição da diversidade" not in e3
    assert "70/67 não representa o esforço adicional" in e3
    assert "dominando, nesse plano e nessa partição" in e3


def test_equity_uses_feasible_criteria_and_declares_review_limit():
    text = (LATEX / "estudo5_equidade.tex").read_text(encoding="utf-8")
    assert "Esses dois critérios podem ser satisfeitos simultaneamente" in text
    assert "de 42,5\\% para 77,5\\%" in text
    assert "limite com revisor perfeito, não de resultado humano medido" in text
    assert "17/25" in text and "76/78" in text
    assert "estratos de processo" in text
    assert "incompatibilidade geral entre paridade de sensibilidade e de VPP" not in text


def test_qw1_export_is_aggregate_and_uses_existing_metrics(tmp_path):
    frame = pd.read_csv(ROOT / "data/review2_qw1_resumo.csv")
    assert len(frame) == 5
    assert not {"pair_id", "REFREC", "COMPREC", "PAR"} & set(frame.columns)
    assert frame.n_samples.eq(18509).all() and frame.n_pos.eq(74).all()
    source = tmp_path / "source"
    source.mkdir()
    cv_columns = [column for column in frame if column.endswith(("_mean", "_std"))] + ["model", "n_folds"]
    holdout_columns = [column for column in frame if column not in cv_columns and column != "semente"] + ["model"]
    frame[cv_columns].to_csv(source / "cv_summary.csv", index=False)
    frame[holdout_columns].to_csv(source / "holdout_metrics.csv", index=False)
    destination = tmp_path / "summary.csv"
    tables.export_qw1(source, destination)
    regenerated = pd.read_csv(destination)
    pd.testing.assert_frame_equal(frame.sort_index(axis=1), regenerated.sort_index(axis=1))
