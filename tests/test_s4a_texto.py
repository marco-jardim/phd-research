from pathlib import Path
import re

import pandas as pd
from local_data import DATA_ROOT, requires_files
from scripts.referencia_pos_orl import carregar, resumo_extrato
from resumo_numeros import TETO_NUMEROS_RESULTADO, numeros_de_resultado
from terminologia import contem

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
        # Revisão 2: P-Platt e XGBoost saíram do Resumo e do Abstract; constam do Estudo 4 e do Estudo 2 (abaixo).
        assert "P-Platt" in (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
        assert "XGBoost" in (LATEX / "estudo2_classificadores.tex").read_text(encoding="utf-8")
        # Inclui o denominador 247.
        assert '247' in text
        # Decisão da autora de 2026-11-08: teto de 24 números de RESULTADO (regra em resumo_numeros.py).
        assert len(numeros_de_resultado(text)) <= TETO_NUMEROS_RESULTADO
    # Revisão 2: PAR~=~2 na referência operacional (a definição de PAR como campo da referência está no Estudo 4).
    assert "têm PAR~=~2 na referência operacional" in resumo
    assert "have PAR~=~2 in the operational reference" in abstract
    assert "PAR é o campo da referência operacional" in (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    assert "after its internal registration, which had no success criterion" in abstract
    assert "preregistered" not in abstract
    assert "Sob revisor perfeito simulado" in resumo
    assert "nove com 29 revisões" in resumo
    assert "Under a simulated perfect reviewer" in abstract
    assert "nine with 29 reviews" in abstract
    assert "Sem suposições sobre a verificação e a incorporação das evidências do comparador" in resumo
    assert "Without assumptions about the verification and the incorporation of comparator evidence" in abstract
    assert "intervalos sobrepostos" not in resumo and "intervals overlapping" not in abstract
    assert "16,0" in resumo and "16.0" in abstract
    assert "curva de esforço" in resumo and "effort curve" in abstract
    # O orçamento humano como critério do ponto de parada ficou na conclusão; o resumo traz o ponto de parada.
    assert "ponto de parada" in resumo and "stopping point" in abstract
    assert "orçamento humano" in (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "a validar em coorte nova" in resumo
    assert "to be validated in a new cohort" in abstract


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
    # Conclusão reescrita (2026-11-08): a ressalva de reutilização das chamadas ao LLM permanece no método e nas
    # limitações do Estudo 4; a conclusão a resume como análise \textit{post hoc} sob revisor simulado.
    for text in (method, limits):
        assert "não é prospectiva" in text
    for text in (limits,):
        assert r"\ref{sec:e4-arranjo-metodo}" in text
        assert "reutilização" in text
    assert r"O achado é \textit{post hoc}" in conclusion
    assert "revisor humano simulado sem erros (revisor perfeito)" in conclusion


def test_final_editorial_contract_and_table_placement():
    study = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    discussion = study.split(r"\section{Discussão}")[1].split(r"\subsection{Limitações}")[0]
    limits = study.split(r"\subsection{Limitações}")[1]
    for text in (discussion, limits):
        assert "vem inteiramente" not in text
        assert "quatro contra dez" not in text
        assert r"\ref{sec:e4-mesma-carga}" in text or r"\ref{tab:cap8-arranjo-corte-par2}" in text
    conclusion = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    assert "P recuperaria 23 pares verdadeiros com 121 revisões" in conclusion
    assert "contra 22 de O3, no máximo um a mais na mesma carga humana" in conclusion
    assert "introduziu 21 falsos positivos" in conclusion
    assert "Na confirmação, foram 27 pares verdadeiros perdidos e 4 falsos positivos" in conclusion
    # Os lotes de 344 potenciais pares avaliados pelo LLM saíram da conclusão por concisão; constam do apêndice.
    assert "344 potenciais pares" in (LATEX / "apendice_gzcmd.tex").read_text(encoding="utf-8")
    assert "revisor perfeito" in conclusion
    assert "Nenhuma ordenação da fila dominou as demais em todas as cargas" in conclusion
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
    # Revisão 2: [5,9) é a "faixa intermediária da nota final"; "área cinza" designa a região abaixo do corte.
    assert contem(formal, "Essas regiões não redefinem a faixa intermediária da nota final", arquivo="text/latex/apendice_formalizacao.tex")
    assert "área cinza da nota" not in formal
    assert r"$[5,9)$" in formal and r"\ref{sec:area-cinza}" in formal
    for name in ("capitulo1.tex", "glossario.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        # Decisão da autora de 2026-11-08: no capítulo 1 (revisado), a faixa [5,9) é "faixa intermediária da
        # nota final" e a grey area (área cinza) designa a região abaixo do corte do OpenRecLink.
        nome_faixa = "faixa intermediária da nota final"  # glossário revisado na revisão 2 (mesma decisão)
        assert contem(text, nome_faixa, arquivo=f"text/latex/{name}")  # migrado: terminologia 3.6
        assert contem(text, "área cinza", arquivo=f"text/latex/{name}")  # migrado: terminologia 3.6
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
    # W5: a janela de notificação dos pares verdadeiros foi movida para o apêndice de origem.
    origin = (LATEX / "apendice_linkage_origem.tex").read_text(encoding="utf-8")
    assert "entre 2013 e 2022" in chapter
    assert "184 correspondem a notificações de 2019 a 2022 e 63" in origin
    assert "2 de janeiro" not in chapter + origin and "30 de dezembro" not in chapter + origin


@requires_files(DATA_ROOT / "data/COMPARADORSEMIDENT.csv")
def test_notification_year_aggregates_reproduce_from_private_input():
    generated = resumo_extrato(carregar(DATA_ROOT / "data/COMPARADORSEMIDENT.csv"))
    published = pd.read_csv(ROOT / "data/extrato_resumo.csv")
    pd.testing.assert_frame_equal(generated, published)


def test_review3_scope_and_experiment1_caveats():
    # estudo2 revisado (revisão 2): o Experimento 1 e data/exp1_vazamento_comparacao.csv ficam no
    # estudo1, na conclusão e no apêndice de sensibilidade; o estudo2 remete ao vazamento sem repeti-los
    # A conclusão reescrita não retoma o Experimento 1; as ressalvas ficam nos capítulos e apêndices de origem.
    # Revisão 2: o apêndice de sensibilidade chama P de "braço P do registro interno" (o Estudo 1 mantém "arranjo P").
    designacao_p = {"estudo1_comparador.tex": "arranjo P", "apendice_sensibilidade.tex": "braço P do registro interno"}
    for name in ("estudo1_comparador.tex", "apendice_sensibilidade.tex"):
        text = (LATEX / name).read_text(encoding="utf-8")
        assert "GZ-CMD v3.1" in text and designacao_p[name] in text
        assert r"\path{data/exp1_vazamento_comparacao.csv}" in text
        assert "situação de encerramento" in text
    estudo4 = (LATEX / "estudo4_gzcmd.tex").read_text(encoding="utf-8")
    for phrase in ("0,941", "0,916", r"55,0\%", r"34,7\%", "as cinco partições de teste se sobrepõem"):
        assert phrase in estudo4, phrase
    assert "atributos restritos" in (LATEX / "apendice_sensibilidade.tex").read_text(encoding="utf-8")
    assert "amostra revisada de forma independente" in (LATEX / "glossario.tex").read_text(encoding="utf-8")
    text = (LATEX / "conclusao.tex").read_text(encoding="utf-8")
    for phrase in ("não examinados", "provedores externos", "a direção dominante e a magnitude do viés não são identificáveis"):
        assert phrase in text, phrase
    objectives = (LATEX / "capitulo3.tex").read_text(encoding="utf-8")
    assert "investigação desnecessária" in objectives
    assert "sem investigação" not in objectives
    # A ressalva sobre a geração de candidatos permanece no Estudo 5; a conclusão reescrita não a repete.
    for name in ("estudo5_equidade.tex",):
        source = (LATEX / name).read_text(encoding="utf-8")
        assert "não detecta" in source
        assert "nem compara óbitos vinculados e não vinculados" in source
        assert "Soundex do primeiro e do último fragmentos do nome" in source
    assert contem((LATEX / "estudo3_estrategias.tex").read_text(encoding="utf-8"), "e detecção de vínculos}", arquivo="text/latex/estudo3_estrategias.tex")  # migrado: terminologia 3.6
