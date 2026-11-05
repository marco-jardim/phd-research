"""S10a: coerência dos cenários, notas completas e redação localizada."""
from pathlib import Path
import re

import pytest

from editorial_expectations import arrangement_expected_notes

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / 'text/latex'


@pytest.mark.parametrize('old,new', [('0,99','0,98'),('1,00','0,97'),('0{,}95','0{,}94')])
def test_note_numeric_contract_rejects_changed_scenarios(old, new):
    historical = (L / 'tables/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8')
    expected = arrangement_expected_notes(historical).split(r'\noindent')[1]
    current = (L / 'tables/editorial/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8').split(r'\noindent')[1]
    numeric = lambda t: re.findall(r'\d+(?:[.,]\d+)*',t)
    assert numeric(expected) == numeric(current)
    assert old in current
    assert numeric(expected) != numeric(current.replace(old,new))


def test_local_editorial_redundancies_and_number_forms():
    conclusion = (L / 'conclusao.tex').read_text(encoding='utf-8')
    recommendation = conclusion.split(r'\label{sec:concl-recomendacoes}')[1].split(r'\section{Trabalhos futuros}')[0]
    assert recommendation.lower().count('nenhuma ordenação domina em todas as cargas') == 1
    for name in ('capitulo1.tex','conclusao.tex'):
        text = (L / name).read_text(encoding='utf-8')
        assert 'na vigilância, em $k=96$' in text
        assert 'no topo de 96 pares da vigilância, em $k=96$' not in text
        assert 'descartou 10 dos 23 vínculos' in text
    study = (L / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert 'O2 recupera 9 PAR~=~2 com 11 revisões' in study
    assert 'no feminino, a queda é de 10 pontos' in study
    discussion = study.split(r'\label{sec:e4-discussao}')[1]
    assert r'proposição de duas ações (Proposição~\ref{prop:duas-acoes-orcamento})' in discussion
    assert 'Isso não fixa O3' not in discussion
    formal = (L / 'apendice_formalizacao.tex').read_text(encoding='utf-8')
    assert r'\subsection{Revisão Humana e Não Vínculo sob Orçamento}' in formal
