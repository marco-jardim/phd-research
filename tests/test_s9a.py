"""S9a: decisão condicionada ao orçamento, teoria e fontes editoriais."""
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd
import pytest

from scripts.tabelas_s8_editorial import same_load, effort
from terminologia import contem

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
        assert 'fusão por posição recíproca' in text
        assert 'referência cega' in text
        assert 'Se a ordenação usar probabilidade aprendida' in text
        expected = {
            'capitulo1.tex': 'a curva de esforço aplicada a este fluxo',
            'conclusao.tex': 'a curva de esforço aplicada a este fluxo como instrumento de escolha da ordenação e do ponto de parada',
        }
        assert expected[name] in text
        assert r'\citeonline{Sariyar2010recordlinkage}' in text
        assert 'dentro da variação de perdas' not in text
    text = (L / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert contem(text, 'uma diferença de um vínculo em 36 não sustenta superioridade', arquivo='text/latex/estudo4_gzcmd.tex')  # migrado: terminologia 3.6
    discussion = text.split(r'\label{sec:e4-discussao}')[1]
    assert 'as perdas são de cobertura, não de erro humano' in discussion
    assert 'o ótimo é dar ao revisor humano o topo da fila' in discussion
    assert r'\ref{prop:duas-acoes-orcamento}' in discussion


def test_theoretical_two_action_proposition_has_scope_threshold_and_exchange_proof():
    text = (L / 'apendice_formalizacao.tex').read_text(encoding='utf-8')
    assert r'\label{prop:duas-acoes-orcamento}' in text
    section = text.split(r'\subsection{Revisão Humana e Não Vínculo sob Orçamento}')[1]
    assert 'exclusivamente teórica' in section
    assert r'p_i\geq C_H/C_{FN}' in section
    assert r'C_{FN}(p_j-p_i)<0' in section
    assert 'até o limite $B$' in section and 'ganhos não negativos' in section
    assert 'revisão falível' in section and 'custos variáveis' in section
    assert not re.search(r'CSV|247|36|0,95', section)


@pytest.mark.parametrize('mode', ['vigilancia', 'confirmacao'])
@pytest.mark.parametrize('strategy,title', [('P', 'P'), ('P_MATCH_ao_humano', r'P, \texttt{MATCH} ao humano')])
def test_policy_projected_count_reads_strategy_scenario_not_a_literal(mode, strategy, title):
    data = pd.read_csv(ROOT / 'data/comparador_mesma_carga.csv')
    scenarios = pd.read_csv(ROOT / 'data/cenarios_sensibilidade_humana.csv')
    selected = scenarios.modo.eq(mode) & scenarios.estrategia.eq(strategy) & scenarios.s.eq(.95)
    scenario = scenarios.loc[selected].iloc[0]
    label = 'Vigilância' if mode == 'vigilancia' else 'Confirmação'

    def cells(text):
        return next(line.split(' & ') for line in text.splitlines() if line.startswith(f'{label} & {title} & '))

    def value(cell):
        return float(cell.replace('{,}', '.'))

    row = cells(same_load(data, scenarios=scenarios))
    baseline = scenario.vinculos_total - scenario.vinculos_abaixo_corte
    expected = scenario.vinculos_aceitos_sem_humano + scenario.s * scenario.vinculos_ao_humano - baseline
    assert value(row[4]) == pytest.approx(expected)
    assert value(row[5]) == pytest.approx(scenario.sensibilidade_total * 100, abs=.005)
    assert (baseline + value(row[4])) / scenario.vinculos_total == pytest.approx(scenario.sensibilidade_total)
    if strategy == 'P_MATCH_ao_humano':
        assert scenario.vinculos_aceitos_sem_humano == baseline
        assert value(row[4]) == pytest.approx(scenario.s * scenario.vinculos_ao_humano)
    else:
        assert scenario.vinculos_aceitos_sem_humano > baseline
    changed = scenarios.copy()
    changed.loc[selected, 'vinculos_ao_humano'] += 2
    changed.loc[selected, 'sensibilidade_total'] += 2 * scenario.s / scenario.vinculos_total
    mutated = cells(same_load(data, scenarios=changed))
    assert value(mutated[4]) == pytest.approx(expected + 2 * scenario.s)
    assert value(mutated[5]) != value(row[5])
    text = same_load(data, scenarios=scenarios)
    assert contem(text, 'em P, os vínculos aceitos pelo LLM não recebem desconto', arquivo='text/latex/tables/editorial/tab_cap8_mesma_carga.tex')  # migrado: terminologia 3.6
    assert 'em P com MATCH ao humano, todos os recuperáveis passam pelo humano' in text
    assert 'h/FP: horas de revisão divididas pelos FP automáticos' in effort(pd.read_csv(ROOT / 'data/esforco_arranjo.csv'))


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
    # Decisão D2: o parecer cobre o depósito restrito, não um benchmark público.
    assert 'sob os pareceres vinculados aos CAAE 43856720.1.0000.5286 e 43856720.1.3001.5279' in data
    objectives = (L / 'capitulo3.tex').read_text(encoding='utf-8')
    assert r'\ref{ap:protocolo-busca}' in objectives
    assert 'Pacheco2008hierarchical,Rafael2024accuracy' in objectives
    assert r'foi acrescentada \textit{post hoc}' in objectives
    bib = (L / 'referencias.bib').read_text(encoding='utf-8')
    assert r'author    = {{\relax Dal Pozzolo}, Andrea and Caelen' in bib
