"""S9a: decisão condicionada ao orçamento, teoria e fontes editoriais."""
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd
import pytest

from scripts.tabelas_s8_editorial import same_load, effort

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / 'text/latex'


@pytest.mark.skipif(shutil.which('bibtex') is None, reason='BibTeX não instalado')
def test_dal_pozzolo_case_with_project_bibliography_style(tmp_path):
    """O grupo especial permite caixa alta sem perder o sobrenome na chamada."""
    bib = (L / 'referencias.bib').read_text(encoding='utf-8')
    entry = next(block for block in re.findall(r'@\w+\{[^@]+', bib)
                 if block.startswith('@inproceedings{DalPozzolo2015calibrating,'))
    (tmp_path / 'nome.bib').write_text(entry, encoding='utf-8')
    (tmp_path / 'nome.aux').write_text(
        '\\relax\n\\citation{DalPozzolo2015calibrating}\n'
        '\\bibstyle{abntex2-alf}\n\\bibdata{nome}\n', encoding='utf-8')
    result = subprocess.run(['bibtex', 'nome'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    bbl = (tmp_path / 'nome.bbl').read_text(encoding='utf-8')
    assert r'\bibitem[{\relax Dal Pozzolo} et al. 2015]' in bbl
    assert r'{{\relax DAL POZZOLO}, A. et al.' in bbl


def test_recommendation_does_not_fix_a_learned_probability_order():
    for name in ('capitulo1.tex', 'conclusao.tex'):
        text = (L / name).read_text(encoding='utf-8')
        assert 'ordem decrescente de uma probabilidade ou escore' in text
        assert 'Se a ordenação usar probabilidade aprendida' in text
        assert 'a curva de esforço aplicada a este fluxo' in text
        assert r'\citeonline{Sariyar2010recordlinkage}' in text
        assert 'dentro da variação de perdas' not in text
    text = (L / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert 'uma diferença de um vínculo em 36 não sustenta superioridade' in text
    discussion = text.split(r'\label{sec:e4-discussao}')[1]
    assert 'as perdas são de cobertura, não de erro humano' in discussion
    assert 'o ótimo é dar ao revisor humano o topo da fila' in discussion
    assert r'\ref{prop:duas-acoes-orcamento}' in discussion


def test_theoretical_two_action_proposition_has_scope_threshold_and_exchange_proof():
    text = (L / 'apendice_formalizacao.tex').read_text(encoding='utf-8')
    assert r'\label{prop:duas-acoes-orcamento}' in text
    section = text.split(r'\subsection{Revisão humana e não vínculo sob orçamento}')[1]
    assert 'exclusivamente teórica' in section
    assert r'p_i\geq C_H/C_{FN}' in section
    assert r'C_{FN}(p_j-p_i)<0' in section
    assert 'até o limite $B$' in section and 'ganhos não negativos' in section
    assert 'revisão falível' in section and 'custos variáveis' in section
    assert not re.search(r'CSV|247|36|0,95', section)


def test_policy_projected_count_reads_versioned_workload_not_a_literal():
    data = pd.read_csv(ROOT / 'data/comparador_mesma_carga.csv')
    workload = pd.read_csv(ROOT / 'data/esforco_arranjo.csv')
    text = same_load(data, workload=workload)
    row = workload.loc[workload.modo.eq('vigilancia') & workload.braco.eq('P') & workload.carga.eq('K1_humana')].iloc[0]
    assert row.par2_recuperados_erro05 == 22.05
    assert r'23 & 22{,}05' in text
    changed = workload.copy()
    changed.loc[changed.braco.eq('P'), 'par2_recuperados_erro05'] = 13.37
    assert r'23 & 13{,}37' in same_load(data, workload=changed)
    assert 'se todos os recuperáveis forem revistos por humano' in text
    assert 'h/FP: horas de revisão divididas pelos FP automáticos' in effort(workload)


def test_english_convention_expansions_denominators_and_surveillance():
    text = (L / 'tese.tex').read_text(encoding='utf-8')
    pt = text.split(r'O \textit{linkage} integra', 1)[1].split(r'\vspace{15pt}')[0]
    en = text.split('Linkage integrates', 1)[1].split(r'\vspace{15pt}')[0]
    assert 'modelo de linguagem de grande porte (LLM)' in pt
    assert 'falsos positivos (FP)' in pt
    assert 'large language model (LLM)' in en and 'false positives (FP)' in en
    assert '247' in pt and '247' in en
    assert r'42,5\% na vigilância' in pt and 'surveillance blocking step 11' in en
    for summary in (pt, en):
        assert len(summary.split()) <= 500
        assert r'\textit{Platt scaling}' not in en and r'\textit{Linkage}' not in en
    assert 'nos dois lados' not in pt and 'pontos completos' not in pt
    assert 'Se a ordenação usar probabilidade aprendida' in pt
    assert 'If ordering uses learned probabilities' in en


def test_ic_scope_and_filter_cost_are_explicit():
    text = (L / 'tables/editorial/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8')
    assert 'IC de Wilson de 95\\% descritivo' not in text
    assert 'Cada célula mostra a contagem e seu IC de Wilson' in text
    for name in ('tab_cap8_arranjo_corte.tex', 'tab_cap8_referencia.tex'):
        note = (L / 'tables/editorial' / name).read_text(encoding='utf-8')
        assert '0,95, 0,99 e 1,00' in note
        assert 'data/cenarios_sensibilidade_humana.csv' in note
    study = (L / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert r'$H=22$ revisões humanas e 74 decisões de \texttt{NONMATCH}' in study
    assert 'arredondamento editorial' not in study
    assert r'\texttt{LLM\_REVIEW}' not in study
    assert 'Uma diferença pareada' not in study
    data = (L / 'capitulo_dados.tex').read_text(encoding='utf-8')
    assert 'termo de consentimento' not in data
    assert 'o parecer do CEP abrange' in data
    objectives = (L / 'capitulo3.tex').read_text(encoding='utf-8')
    assert r'\ref{ap:protocolo-busca}' in objectives
    assert 'Pacheco2008hierarchical,Rafael2024accuracy' in objectives
    assert r'foi acrescentada \textit{post hoc}' in objectives
    bib = (L / 'referencias.bib').read_text(encoding='utf-8')
    assert r'author    = {{\relax Dal Pozzolo}, Andrea and Caelen' in bib
