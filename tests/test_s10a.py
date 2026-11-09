"""S10a: coerência dos cenários, notas completas e redação localizada."""
from pathlib import Path
import re

import pytest

from editorial_expectations import arrangement_expected_notes
from terminologia import contem

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
    recommendation = conclusion.split(r'\label{sec:concl-recomendacoes}')[1].split(r'\section{Validação e próximos passos}')[0]
    assert recommendation.lower().count('nenhuma ordenação dominou em todas as cargas') == 1
    for name in ('capitulo1.tex','conclusao.tex'):
        text = (L / name).read_text(encoding='utf-8')
        if name == 'capitulo1.tex':
            assert 'na vigilância, em $k=96$' in text
        else:
            # Conclusão reescrita: a carga de 96 revisões de P permanece na síntese; o filtro de NONMATCH
            # (96 pares enviados ao LLM, nove dos 22 descartados) fica no Estudo 4, sem repetição na conclusão.
            assert 'P submeteu 344 potenciais pares ao LLM, com 755 chamadas, e encaminhou 96 ao revisor humano' in text
            assert 'descartou nove dos 22 PAR~=~2 do topo' in (L / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
        assert 'no topo de 96 pares da vigilância, em $k=96$' not in text
        # capitulo1.tex (revisado, modo estrito): "pares verdadeiros" é a classe segundo a referência (terminologia 3.6).
        if name == 'capitulo1.tex':
            # o "22" saiu desta frase por ambiguidade; N27 do registro exige 22, k=96 e "nove" no parágrafo.
            assert contem(text, 'descartou nove dos pares verdadeiros do topo', arquivo=f'text/latex/{name}')
        else:
            # Conclusão reescrita: o detalhe do filtro saiu da conclusão e permanece no Estudo 4 (verificado acima).
            assert 'descartou nove dos 22' not in text
    study = (L / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert 'O2 recupera 9 PAR~=~2 com 11 revisões' in study
    assert 'no feminino, a queda é de 10 pontos' in study
    discussion = study.split(r'\label{sec:e4-discussao}')[1]
    assert r'proposição de duas ações (Proposição~\ref{prop:duas-acoes-orcamento})' in discussion
    assert 'Isso não fixa O3' not in discussion
    formal = (L / 'apendice_formalizacao.tex').read_text(encoding='utf-8')
    # Revisão 2: "não vínculo" saiu do título (guia 3.6); a decisão de não aceitar é "rejeição".
    assert contem(formal, r'\subsection{Revisão Humana e Rejeição sob Orçamento}', arquivo='text/latex/apendice_formalizacao.tex')
    assert 'Não Vínculo' not in formal
