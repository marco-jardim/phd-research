from pathlib import Path
import re

import pandas as pd
from local_data import DATA_ROOT, requires_files
from scripts.referencia_pos_orl import carregar, resumo_extrato

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "text/latex"


def abstracts():
    text = (LATEX / "tese.tex").read_text(encoding="utf-8")
    for start in ("O \\textit{linkage} integra sistemas sem", "Linkage integrates systems without"):
        begin = text.index(start)
        yield text[begin:text.index(r"\vspace{15pt}", begin)].strip()


def test_abstracts_space_limit_and_conservative_operational_comparison():
    resumo, abstract = abstracts()
    for text in (resumo, abstract):
        assert len(text.split()) <= 500
        assert len(text.split("\n\n")) == 4
        assert not re.search("—|–", text)
        assert "P-Platt" in text and "XGBoost" in text
        # Inclui o denominador 247 sem elevar o limite original de 12 números.
        assert '247' in text
        assert len(re.findall(r'\d+(?:[.,]\d+)*', text)) <= 18
    assert r"PAR é o \textit{label} da referência" in resumo
    assert "registered internally before evaluation" in abstract
    assert "preregistered" not in abstract
    assert "Sob a mesma hipótese de revisor perfeito" in resumo
    assert "nove com 29 revisões" in resumo
    assert "the same perfect-reviewer assumption" in abstract
    assert "nine with 29 reviews" in abstract
    assert "verificação seletiva" in resumo and "incorporação" in resumo
    assert "Selective verification" in abstract and "incorporation" in abstract
    assert "intervalos sobrepostos" not in resumo and "intervals overlapping" not in abstract
    assert "16,0" in resumo and "16.0" in abstract
    assert "curva de esforço" in resumo and "effort-versus-recovered-links curve" in abstract
    assert "orçamento humano" in resumo and "human budget" in abstract
    assert "validação registrada em dados novos" in resumo
    assert "registered validation on new data" in abstract


def test_every_study5_input_table_is_referenced_in_chapter():
    chapter = (LATEX / "estudo5_equidade.tex").read_text(encoding="utf-8")
    active = re.sub(r"(?m)^%.*$", "", chapter)
    for name in re.findall(r"\\input\{(tables/[^}]+)\}", active):
        table = (LATEX / name).read_text(encoding="utf-8")
        labels = re.findall(r"\\label\{([^}]+)\}", table)
        assert labels, name
        assert any(r"\ref{" + label + "}" in chapter for label in labels), name


def test_reuse_caveat_present_in_method_limitations_and_conclusion():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    method = study.split(r"\label{sec:e4-arranjo-metodo}")[1].split(r"\section{Resultados}")[0]
    limits = study.split(r"\label{sec:e4-limitacoes}")[1]
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "reutilizam majoritariamente" in method
    assert r"$P\cup B1$" in method and "14" in method
    for text in (method, limits, conclusion):
        assert "não é prospectiva" in text
    for text in (limits, conclusion):
        assert r"\ref{sec:e4-arranjo-metodo}" in text
        assert "reutilização" in text


def test_final_editorial_contract_and_table_placement():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    discussion = study.split(r"\section{Discussão}")[1].split(r"\subsection{Limitações}")[0]
    limits = study.split(r"\subsection{Limitações}")[1]
    for text in (discussion, limits):
        assert "vem inteiramente" not in text
        assert "quatro contra dez" not in text
        assert r"\ref{sec:e4-mesma-carga}" in text or r"\ref{tab:cap8-arranjo-corte-par2}" in text
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "P usaria 121 revisões para 23" in conclusion
    assert "O3 recuperaria 22 na mesma carga" in conclusion
    assert "21 e 4 foram rejeitados ou não examinados" in conclusion
    assert "344 potenciais pares avaliados pelo LLM" in conclusion
    assert "revisor perfeito" in conclusion
    assert "Sob a mesma hipótese de revisor e na mesma carga, entre O1 a O4 e P, nenhuma domina" in conclusion
    chapter = (LATEX / "estudo5_equidade.tex").read_text(encoding="utf-8")
    active = re.sub(r"(?m)^%.*$", "", chapter)
    cutoff = active.index(r"\subsection{Comparação com o corte 6,0}")
    for mode in ("vigilancia", "confirmacao"):
        name = f"tables/editorial/tab_den30_equidade_arranjo_rejeicoes_cascata_v31_{mode}.tex"
        assert active.index(r"\input{" + name + "}") < cutoff
    assert "Não é incluída" in chapter
    assert "tab_equidade_v31_rejeicoes_cascata.tex" in chapter


def test_canonical_names_and_operational_area_cinza_definition():
    for name in ("estudo1_comparador.tex", "estudo2_classificadores.tex", "estudo3_estrategias.tex",
                 "tese.tex", "conclusao.tex", "glossario.tex", "capitulo2.tex",
                 "apendice_busca.tex", "siglas.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "Floresta Aleatória" not in text
        assert r"\textit{Random Forest}" not in text
    formal = (LATEX / "apendice_formalizacao.tex").read_text(encoding="utf-8")
    assert r"\textit{labels}" in formal
    assert "Essas regiões não redefinem a área cinza da nota" in formal
    assert r"$[5,9)$" in formal and r"\ref{sec:area-cinza}" in formal
    for name in ("capitulo1.tex", "glossario.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "área cinza da nota" in text
        assert "área cinza" in text
    intro = (LATEX / "capitulo1.tex").read_text(encoding="utf-8")
    assert "conversão em horas utiliza um parâmetro temporal" in intro
    assert "Esses valores constituem cenário exploratório, não carga histórica medida." in intro
    assert "Sua conversão em horas utiliza um parâmetro temporal e fornece ordem de grandeza, sem medir esforço observado." in intro


def test_notification_year_aggregates_are_portable_and_match_text():
    frame = pd.read_csv(ROOT / "data/extrato_resumo.csv").set_index("metrica")["valor"]
    assert frame["ano_notificacao_min"] == 2013
    assert frame["ano_notificacao_max"] == 2022
    assert frame["n_sem_ano_notificacao"] == 0
    assert frame["n_vinculos_notificacao_2013_2018"] == 63
    assert frame["n_vinculos_notificacao_2019_2022"] == 184
    assert frame["n_vinculos_notificacao_2013_2018"] + frame["n_vinculos_notificacao_2019_2022"] == frame["n_par1"] + frame["n_par2"]
    chapter = (LATEX / "capitulo_dados.tex").read_text(encoding="utf-8")
    assert "entre 2013 e 2022" in chapter
    assert "184 correspondem a notificações de 2019 a 2022 e 63" in chapter
    assert "2 de janeiro" not in chapter and "30 de dezembro" not in chapter


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv")
def test_notification_year_aggregates_reproduce_from_private_input():
    generated = resumo_extrato(carregar(DATA_ROOT / "data/COMPARADORSEMIDENT.csv"))
    published = pd.read_csv(ROOT / "data/extrato_resumo.csv")
    pd.testing.assert_frame_equal(generated, published)


def test_review3_scope_and_experiment1_caveats():
    for name in ("estudo1_comparador.tex", "estudo2_classificadores.tex", "conclusao.tex", "apendice_sensibilidade.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "GZ-CMD v3.1" in text and "arranjo P" in text
        assert r"\path{data/exp1_vazamento_comparacao.csv}" in text
        assert "situação de encerramento" in text
    text = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    for phrase in ("0,941", "0,916", r"55,0\%", r"34,7\%", "conjuntos de teste sobrepostos", "atributos restritos", "amostra revisada de forma independente", "não examinados", "API externa", "causa associada", "direção dominante não é identificável"):
        assert phrase in text, phrase
    objectives = (LATEX / "capitulo3.tex").read_text(encoding="utf-8")
    assert "investigação desnecessária" in objectives
    assert "sem investigação" not in objectives
    for name in ("estudo5_equidade.tex", "conclusao.tex"):
        source = (LATEX / name).read_text(encoding="utf-8")
        assert "não detecta" in source
        assert "nem compara óbitos vinculados e não vinculados" in source
        assert "Soundex do primeiro e do último fragmentos do nome" in source
    assert "e detecção de vínculos}" in (LATEX / "estudo3_estrategias.tex").read_text(encoding="utf-8")
