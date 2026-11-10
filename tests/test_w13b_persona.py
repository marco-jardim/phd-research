"""Glosas da revisão por persona, sem mudar a recomendação aprovada."""

from pathlib import Path
import re

from latex_fontes import fontes_tex
from resumo_numeros import TETO_NUMEROS_RESULTADO, numeros_de_resultado
from test_s4a_texto import abstracts
from test_w10b import APPROVED
from terminologia import contem


ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / 'text/latex'


def text(name):
    return (LATEX / name).read_text(encoding='utf-8')


def test_persona_blockers_in_bilingual_summaries_and_ceiling():
    pt, en = abstracts()
    for body in (pt, en):
        assert len(body.split()) <= 500
        # Decisão da autora de 2026-11-08: teto de 24 números de RESULTADO (regra em resumo_numeros.py).
        assert len(numeros_de_resultado(body)) <= TETO_NUMEROS_RESULTADO
        assert not re.search('—|–', body)
        assert len(body.split('\n\n')) == 4
    # Redação nova (revisão 2): a decomposição por PAR (um, dois, zero), P-Platt e a ancoragem saíram do resumo;
    # P-Platt consta do Estudo 4 (legenda da dominância) e a ancoragem da seção do revisor de referência.
    # Resumo reescrito (2026-11-10): "sem alterar os vínculos que o corte já aceita" e "todas as aceitações abaixo
    # do corte vieram do LLM" substituem as formas antigas, com o mesmo conteúdo.
    for fragment in ('sem alterar os vínculos aceitos pelo corte',
                     'recuperados abaixo do corte pela revisão manual de origem',
                     'todas as aceitações abaixo do corte vieram do LLM',
                     'sensibilidade automática de ASR na vigilância'):
        # migrado: terminologia 3.6
        assert contem(pt, fragment, arquivo='text/latex/tese.tex') if 'vínculo' in fragment else fragment in pt
    for fragment in ('without altering the links accepted by the cut-off',
                     'recovered below the cut-off by the original manual review',
                     'all acceptances below the cut-off came from the LLM',
                     "the automatic sensitivity of ASR in surveillance"):
        assert fragment in en
    estudo4 = text('estudo4_gzcmd.tex')
    assert 'ASR-Platt (análise de sensibilidade de ASR com \\textit{Platt scaling}' in estudo4
    assert 'Revisor de referência, ancoragem e cascata na base inteira' in estudo4
    assert 'decomposição por PAR' not in pt
    assert 'PAR decomposition' not in en


def test_conclusion_reference_resolves_to_actual_study5_limitations():
    conclusion = text('conclusao.tex')
    study = text('estudo5_equidade.tex')
    assert r'\ref{sec:e5-limitacoes}' not in conclusion
    # Conclusão reescrita: remete ao capítulo do Estudo 5 (cap:estudo5), não a uma subseção; o capítulo e a
    # subseção de discussão (onde estão as limitações) existem e carregam as ressalvas abaixo.
    assert r'\ref{cap:estudo5}' in conclusion
    assert r'\label{cap:estudo5}' in study
    target = 'sec:e5-discussao-equidade'
    section = study.split(rf'\label{{{target}}}', 1)[1]
    assert 'limita a inferência' in section
    assert 'não detecta' in section


def test_conclusion_disambiguates_collisions_without_changing_recommendation():
    source = text('conclusao.tex')
    assert source.count(APPROVED) == 1
    # Conclusão reescrita: as glosas de CH1/CH4, do braço a posteriori e das partições A/A0 saíram da conclusão
    # (concisão); as de P-Platt e das partições permanecem no Estudo 4, no apêndice das estratégias e no glossário.
    assert not re.search(r'(?<![\w.])K[1-4](?![\w.])', source)
    assert 'comparador sem recalibração' not in source
    assert 'comparador de mesma carga' not in source
    assert r'ASR-Platt (análise de sensibilidade de ASR com \textit{Platt scaling}' in text('estudo4_gzcmd.tex')
    assert 'de 22, na execução única sem subamostragem, para medianas de 13 e 18,5' in source
    assert 'partição por linha, com indicadores de óbito' in text('apendice_estrategias.tex')
    assert 'partição original do QW1' in text('glossario.tex')
    assert not re.search(r'\) \(recuperados pela revisão', source)


def test_new_glossary_entries_and_abbreviations_are_present_and_refs_resolve():
    glossary = text('glossario.tex')
    for entry in ('NB01', 'NB02', 'NB03', 'Cenários A a D e controle A0',
                  'F1 a F4 (fusões de filas)', 'C e CM', 'V1 a V4 (estratos de vigilância)',
                  'C1 a C4 (pontos de virada)', r'Braço \textit{a posteriori}',
                  'Ablação restrita', 'Fluxo Modal', 'Meta de recuperação',
                  'Revisor perfeito', 'Revisor de referência', 'Regra de duas ações'):
        assert rf'\item[{entry}]' in glossary
    for code, definition in (('REFREC', 'registro do SIM'), ('COMPREC', 'registro do Sinan'),
                             ('NP', 'fração esperada de falsos vínculos automáticos')):
        row = next(line for line in text('siglas.tex').splitlines() if line.startswith(code + ' '))
        assert definition in row
        assert row.endswith(r'\\[6pt]')
    all_tex = '\n'.join(path.read_text(encoding='utf-8') for path in fontes_tex(LATEX, recursivo=True))
    labels = set(re.findall(r'\\label\{([^}]+)\}', all_tex))
    assert set(re.findall(r'\\ref\{([^}]+)\}', glossary)) <= labels


def test_slides_and_script_do_not_call_aposteriori_a_comparator():
    for name, markup in (('defesa.tex', r'\textit{a posteriori}'),
                         ('roteiro_defesa.md', '*a posteriori*')):
        source = (ROOT / 'text/slides' / name).read_text(encoding='utf-8')
        assert 'comparador ' + markup not in source.lower()
        assert 'braço ' + markup in source
