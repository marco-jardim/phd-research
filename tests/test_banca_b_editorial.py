"""Regressoes da fatia B: fontes, ressalvas e ordem editorial, sem dados por par."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from scripts import tabelas_banca_b_editorial as tables
from scripts.banca_b_encerramento_agregado import summarize_situence

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def study(number: int) -> str:
    return next(LATEX.glob(f"estudo{number}_*.tex")).read_text(encoding="utf-8")


def test_editorial_tables_regenerate_from_canonical_artifacts(tmp_path, monkeypatch):
    historical = {p.name: p.read_bytes() for p in (LATEX / "tables").glob("*.tex")}
    monkeypatch.setattr(tables, "OUT", tmp_path)
    monkeypatch.setattr(tables.nb03, "OUT", tmp_path)
    tables.main()
    generated = list(tmp_path.glob("*.tex"))
    assert len(generated) == 6
    for path in generated:
        assert path.read_text(encoding="utf-8") == (LATEX / "tables/editorial" / path.name).read_text(encoding="utf-8")
    assert historical == {p.name: p.read_bytes() for p in (LATEX / "tables").glob("*.tex")}


def test_pareto_counts_are_per_model_and_dominated_points_identified():
    text = (LATEX / "tables/editorial/tab_pareto_frontier.tex").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith(("RF+SMOTE &", "XGBoost &"))]
    assert len(rows) == 12
    assert sum("& Não dominado" in row for row in rows) == 10
    assert sum(row.startswith("RF+SMOTE") and "& Não dominado" in row for row in rows) == 7
    assert sum(row.startswith("XGBoost") and "& Não dominado" in row for row in rows) == 3
    for th in ("0,50", "0,90"):
        assert any(row.startswith(f"XGBoost & {th}") and "& Dominado" in row for row in rows)
    grid = pd.read_csv(ROOT / "data/pareto_grid.csv")
    xgb = grid[grid.ml_model.eq("XGBoost")]
    assert sum(not tables.dominated(xgb, row) for _, row in xgb.iterrows()) == 3
    assert "não se trata de dez pontos não dominados na união" in study(3).lower()
    assert "doze pontos" not in study(3)


def test_uniform_nb02_nb03_are_single_sources_with_documented_discrepancy():
    text = study(3)
    for number in (2, 3):
        assert f"data/nb0{number}_results.json" in text
        assert f"scripts/run_nb0{number}_uniform.py" in text
    assert text.count("não está documentada") == 2
    nb02 = json.loads((ROOT / "data/nb02_results.json").read_text(encoding="utf-8"))
    gb = next(row for row in nb02["all_results"] if row["Modelo"].startswith("GB +"))
    assert tables.d(gb["Recall"]) == "0,932"
    assert tables.d(gb["Precision"]) == "0,885"
    assert "sensibilidade de 0,932 e precisão de 0,885" in text
    assert "não são adotados como resultado desta seção" in text
    assert r"XGBoost \cite{Chen2016xgboost} com 200 estimadores" in text


def test_qw1_nonreproduction_remains_adjacent_to_historical_results():
    text = study(2)
    for paragraph in text.split("\n\n"):
        if "0{,}855" in paragraph or "0{,}524" in paragraph:
            assert "não foi reproduzido" in paragraph
            assert "0,01" in paragraph or "mesma tolerância" in paragraph
    assert "RF (0,855 para 0,891) nem TabNet (0,610 para 0,712)" in text
    assert "causa das divergências remanescentes não foi isolada" in text
    note = (LATEX / "tables/editorial/tab_qw1_deep_learning.tex").read_text(encoding="utf-8")
    assert "fora da tolerância absoluta de 0,01" in note


def test_nb01_auc_and_rf_only_balance_without_specificity():
    text = study(2)
    auc = pd.read_csv(ROOT / "data/nb01_comparacao_modelos.csv", sep=";")
    paragraph = next(p for p in text.split("\n\n") if p.startswith("As AUC-ROC do NB01"))
    for value in auc["AUC-ROC"]:
        assert tables.d(value, 4) in paragraph
    assert "partição única, não médias de validação cruzada" in paragraph
    method, rest = text.split(r"\section{Resultados}")
    discussion = rest.split(r"\section{Discussão}")[1]
    for part in (method, discussion):
        assert "A especificidade não foi reportada" in part
        assert "não no XGBoost" in part
    assert "sem estimativa de variabilidade por validação cruzada para esse comparativo" in method


def test_results_and_limitations_are_in_their_requested_sections():
    e2, e3 = study(2), study(3)
    assert e2.index("A construção do vetor de atributos") < e2.index(r"\section{Resultados}")
    assert e2.index(r"\section{Resultados}") < e2.index(r"\input{tables/tab_desempenho_faixas.tex}") < e2.index(r"\section{Discussão}")
    assert e3.index(r"\label{sec:e3-impacto-metodo}") < e3.index("Denomina-se vínculo detectado") < e3.index(r"\section{Resultados}")
    assert e3.index(r"\section{Resultados}") < e3.index("A diferença resulta de 27 vínculos recuperados pelo RF+SMOTE") < e3.index(r"\section{Discussão}")
    assert e3.count(r"\subsection{Limitações}") == 1
    assert e3.rindex(r"\subsection{") == e3.index(r"\subsection{Limitações}")
    paragraph = next(p for p in e3.split("\n\n") if p.startswith("Contra a referência operacional, a leitura muda."))
    for term in ("vazamento de alvo", "partição foi por linha", "limiares foram explorados no teste",
                 "10 dos 27", "sete por tuberculose e três por outras causas", "seis apresentavam campo ausente"):
        assert term in paragraph
    assert "por vínculo recuperado" not in e3
    assert "70/67" in e3 and "67/43" in e3


def test_encerramento_aggregate_is_complete_and_contains_no_pair_rows():
    summary = summarize_situence(pd.Series(["3", "4", "1", None, "x"]))
    assert summary == {"n": 5, "encerrados_obito_tb": 1, "encerrados_obito_outras_causas": 1,
                       "encerrados_obito_total": 2, "outros_encerramentos": 1, "ausentes_ou_nao_numericos": 2}
    path = ROOT / "data/banca_b_encerramento_agregado.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"definicao", "entrada_sha256", "controle", "C_SITUENCE"}
    assert data["controle"] == {"test_n": 18509, "test_positivos": 74, "tp": 67, "fp": 3, "recuperados": 27}
    codes = data["C_SITUENCE"]
    assert codes["n"] == 27
    assert codes["encerrados_obito_total"] == 10
    assert codes["encerrados_obito_tb"] == 7 and codes["encerrados_obito_outras_causas"] == 3
    assert codes["outros_encerramentos"] == 11 and codes["ausentes_ou_nao_numericos"] == 6
    assert not re.search(r"pair_id|row_index|REFREC|COMPREC", path.read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "data/sensibilidade_sem_obito_refrec_meta.json").read_text(encoding="utf-8"))
    assert data["entrada_sha256"] == frozen["entrada_sha256"]
    for name, expected in frozen["fontes_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_limiar_rules_and_comparator_neutrality_are_explicit():
    e1, e3 = study(1), study(3)
    assert "não elege uma implementação como a correta" in e1
    assert "afastamento do Python em relação ao peso 0,8" in e1
    assert "núcleo metodológico" not in e1
    assert "não é o XGBoost de 300 estimadores" in e1
    assert r"A Figura~\ref{fig:shap-summary} provém de \path{scripts/qw1_modal_unified_pipeline.py}: um RF+SMOTE com 150 árvores" in e1
    assert r"\ref{sec:e3-contribuicao-framework}" not in e1
    assert r"\ref{sec:proposta-estrategia}" not in study(2)
    assert "itinerário terapêutico" not in e3
    assert "103 configurações experimentais, 70" in e3
    section = e3.split(r"\label{sec:e3-limiar-regras}")[1].split(r"\subsection{")[0]
    for term in ("0{,}842", "0{,}610", "85,4", r"processo de \textit{linkage} de origem e define PAR=1", r"\ref{cap:estudo4}"):
        assert term in section
