"""Glosas da revisão por persona, sem mudar a recomendação aprovada."""

from pathlib import Path
import re

from latex_fontes import fontes_tex
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
        assert len(re.findall(r'\d+(?:[.,]\d+)*', body)) <= 18
        assert not re.search('—|–', body)
        assert len(body.split('\n\n')) == 4
    for fragment in ('um, aceito pelo corte', 'dois, recuperado na revisão clerical abaixo dele',
                     'zero, demais', 'separação entre vínculos aceitos pelo corte e recuperados na revisão',
                     'influência das saídas do motor na resposta do LLM (ancoragem)',
                     'P-Platt é análise de sensibilidade', 'recuperação obtida por P'):
        # migrado: terminologia 3.6
        assert contem(pt, fragment, arquivo='text/latex/tese.tex') if ('vínculo' in fragment or 'clerical' in fragment) else fragment in pt
    for fragment in ('one, accepted by the cut-off', 'two, recovered by clerical review below it',
                     'zero, others', 'recovery obtained by P', "P's automatic sensitivity",
                     'separating links accepted by the cut-off from those recovered by review',
                     'influence of engine outputs on LLM responses (anchoring)'):
        assert fragment in en
    assert 'decomposição por PAR' not in pt
    assert 'PAR decomposition' not in en
    assert 'arrangement' not in en


def test_conclusion_reference_resolves_to_actual_study5_limitations():
    conclusion = text('conclusao.tex')
    study = text('estudo5_equidade.tex')
    assert r'\ref{sec:e5-limitacoes}' not in conclusion
    target = 'sec:e5-discussao-equidade'
    assert rf'\ref{{{target}}}' in conclusion
    section = study.split(rf'\label{{{target}}}', 1)[1]
    assert 'limita a inferência' in section
    assert 'não detecta' in section


def test_conclusion_disambiguates_collisions_without_changing_recommendation():
    source = text('conclusao.tex')
    assert source.count(APPROVED) == 1
    assert 'chaves alternativas CH1 (data de nascimento exata' in source
    assert 'CH4 (unidade federativa no lugar do município)' in source
    assert not re.search(r'(?<![\w.])K[1-4](?![\w.])', source)
    assert 'comparador sem recalibração' not in source
    assert 'comparador de mesma carga' not in source
    assert r'braço \textit{a posteriori} (cascata publicada aplicada abaixo do corte, sem recalibração)' in source
    assert r'P-Platt (análise de sensibilidade de P com \textit{Platt scaling})' in source
    assert 'contra 22 do treino sem subamostragem' in source
    assert 'regra de duas ações (revisar ou não vincular)' in source
    assert 'partição por linha, com indicadores de óbito' in source
    assert 'partição original do QW1' in source
    assert 'partição agrupada por REFREC, registro do SIM, sem indicadores de óbito' in source
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
