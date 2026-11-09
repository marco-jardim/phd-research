"""Glosas locais, sem depender da leitura de outro capítulo.

Vocabulário: O1 exige escore/OpenRecLink; O2, nota/comparador; O3, probabilidade
calibrada/classificador; O4, probabilidade/recalibração; P, preservação do corte;
K1, carga humana (ou chave homônima com data exata); K2, carga mista; K3, custo;
NONMATCH, não vínculo; passo de blocagem, geração/chaves. Grupos são cumulativos,
alternativas são disjuntivas, sempre no mesmo parágrafo da primeira menção.
URLs, referências, código e comentários não são prosa. Kimi K3 é nome de modelo.
"""
from pathlib import Path
import csv
import re

import pytest

from resumo_numeros import TETO_NUMEROS_RESULTADO, numeros_de_resultado
from terminologia import normalizar_terminologia

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / 'text/latex'
OWNED = (
    'capitulo1.tex', 'capitulo2.tex', 'capitulo3.tex', 'capitulo_dados.tex',
    'estudo1_comparador.tex', 'estudo2_classificadores.tex',
    'estudo3_estrategias.tex', 'estudo5_equidade.tex', 'conclusao.tex',
    'glossario.tex', 'siglas.tex', 'apendice_busca.tex',
    'apendice_sensibilidade.tex', 'apendice_linkage_origem.tex', 'apendice_pacote.tex',
)
GLOSS_WORDS = {
    'O1': (('escore', 'score'), ('openreclink',)),
    # Alternativas em inglês para o Abstract (revisão 2): o texto em inglês define as filas com
    # "out-of-sample calibrated probability of the classifier", "OpenRecLink score" etc.
    'O2': (('nota', 'final score'), ('comparador', 'cdr', 'comparator')),
    'O3': (('probabilidade', 'probability'), ('calibrada', 'isotônica', 'classificador', 'calibrated', 'classifier')),
    'O4': (('probabilidade', 'probability'), ('recalibrada', 'recalibração', 'recalibrated', 'recalibration')),
    'P': (('corte', 'cut-off'), ('preserva', 'preservação', 'preservado', 'preserved',
                                'recalibração isotônica', 'isotonic recalibration')),
    'K1': (('humana', 'humano', 'data de nascimento exata', 'human', 'exact birth date'),),
    'K2': (('mista', 'misto', 'llm', 'mixed'), ('carga', 'orçamento', 'avaliações', 'load', 'budget', 'evaluations')),
    'K3': (('custo', 'ponderad', 'cost', 'weighted'),),
    'NONMATCH': (('não vínculo', 'não-vínculo', 'não par', 'não-par', 'non-link'),),  # migrado: terminologia 3.6
    'passo de blocagem': (('geração', 'chaves'),),
    'blocking step': (('candidate-generation', 'keys'),),
}


def prose(text):
    text = re.sub(r'(?m)(?<!\\)%.*$', '', text)
    text = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '', text, flags=re.S)
    return re.sub(r'\\(?:path|url|cite\w*|ref|label|input)\s*(?:\[[^]]*\])?\{[^}]*\}', '', text)


def occurrences(paragraph, token):
    return [m for m in re.finditer(r'(?<![\w.-])' + re.escape(token) + r'(?![\w.-])', paragraph)
            if paragraph[max(0, m.start()-5):m.start()] != 'Kimi ']


def assert_unit_first_mentions(text, arquivo=None):
    seen = set()
    for paragraph in re.split(r'\n\s*\n', prose(text)):
        for token, groups in GLOSS_WORDS.items():
            if token not in seen and occurrences(paragraph, token):
                normalized = normalizar_terminologia(paragraph, arquivo=arquivo)  # migrado: terminologia 3.6
                assert all(any(normalizar_terminologia(word, arquivo=arquivo) in normalized for word in alternatives)
                           for alternatives in groups), (token, paragraph)
                seen.add(token)


def summary_bodies():
    text = (LATEX / 'tese.tex').read_text(encoding='utf-8')
    starts = ('O \\textit{linkage} integra sistemas sem identificador comum.',
              'Linkage integrates systems without a common identifier.')
    return [start + text.split(start, 1)[1].split(r'\vspace{15pt}', 1)[0] for start in starts]


def captions(text):
    for match in re.finditer(r'\\caption(?:\[[^]]*\])?\{', text):
        start = end = match.end()
        depth = 1
        while depth:
            if text[end] == '{' and text[end-1] != '\\':
                depth += 1
            elif text[end] == '}' and text[end-1] != '\\':
                depth -= 1
            end += 1
        yield text[start:end-1]


@pytest.mark.parametrize('body', [0, 1], ids=['resumo', 'abstract'])
def test_summary_first_mentions_are_glossed_and_limits_preserved(body):
    text = summary_bodies()[body]
    assert_unit_first_mentions(text, arquivo='text/latex/tese.tex')
    assert len(text.split()) <= 500
    # Decisão da autora de 2026-11-08: teto de 24 números de RESULTADO (regra em resumo_numeros.py).
    assert len(numeros_de_resultado(text)) <= TETO_NUMEROS_RESULTADO
    assert 'NONMATCH' in text
    assert ('blocking step' if body else 'passo de blocagem') in text


@pytest.mark.parametrize('name', OWNED)
def test_captions_and_chapter_openings_have_local_glosses(name):
    text = (LATEX / name).read_text(encoding='utf-8')
    for caption in captions(text):
        assert_unit_first_mentions(caption, arquivo=f'text/latex/{name}')
    opening = re.split(r'\\(?:sub)?section\*?\{', text, maxsplit=1)[0]
    opening = re.sub(r'\\(?:chapter\*?|begin\{chapter\})\{[^\n]*\}', '', opening)
    assert_unit_first_mentions(opening, arquivo=f'text/latex/{name}')


@pytest.mark.parametrize('name', [n for n in OWNED if n not in ('glossario.tex', 'siglas.tex')])
def test_section_and_subsection_first_mentions_have_local_glosses(name):
    text = (LATEX / name).read_text(encoding='utf-8')
    # O cabeçalho e o primeiro parágrafo formam a unidade de apresentação do termo.
    for section in re.split(r'(?=\\(?:sub)?section\*?\{)', text):
        assert_unit_first_mentions(section, arquivo=f'text/latex/{name}')


@pytest.mark.parametrize('token', GLOSS_WORDS)
def test_checker_rejects_unglossed_tokens(token):
    with pytest.raises(AssertionError):
        assert_unit_first_mentions(f'Foi avaliado {token} nesta análise.')


def test_checker_requires_same_paragraph_and_parses_nested_caption():
    with pytest.raises(AssertionError):
        assert_unit_first_mentions('NONMATCH foi avaliado.\n\nDecisão de não vínculo.')
    text = r'\caption{Filtro de \texttt{NONMATCH} (decisão de não vínculo).}'
    assert list(captions(text)) == [r'Filtro de \texttt{NONMATCH} (decisão de não vínculo).']
    assert_unit_first_mentions(next(captions(text)))


def test_search_labels_are_editorial_and_historical_keys_are_preserved():
    text = (LATEX / 'apendice_busca.tex').read_text(encoding='utf-8')
    for prefix, maximum in (('OA', 5), ('SC', 6), ('BV', 3)):
        assert all(re.search(rf'\b{prefix}{number}\b', text) for number in range(1, maximum+1))
    assert not re.search(r'\b(?:O[1-5]|Q[1-5]|S[1-6]|B[1-3])\b', text)
    assert 'chaves dos arquivos de coleta são as históricas' in text
    assert 'Scopus permanece distinto de SC' in text


def test_future_comparisons_and_script_values_track_aggregate_sources():
    with (ROOT / 'data/fusao_filas.csv').open(encoding='utf-8') as handle:
        rows = {(r['fila'], r['metrica']): r for r in csv.DictReader(handle)}
    conclusion = (LATEX / 'conclusao.tex').read_text(encoding='utf-8')
    # Conclusão reescrita: Fisch2024stratppi, "sem encerramento automático" e "não foi avaliada" saíram (concisão).
    for key in ('Callaghan2020stopping', 'Lewis2021certifying', 'Lam2026clerical'):
        assert key in conclusion
    for phrase in ('F4, 1.387 e O3, 23.443', 'O1 exige 2.521 revisões', 'exigiu em média 96 revisões'):
        assert phrase in conclusion
    script = (ROOT / 'text/slides/roteiro_defesa.md').read_text(encoding='utf-8')
    questions = script.split('## Perguntas prováveis')[1].split('## Figuras e critérios')[0]
    assert [int(n) for n in re.findall(r'^### (\d+)\.', questions, re.M)] == list(range(1, 90))
    for queue in ('F3', 'F4', 'O1'):
        value = f'{int(float(rows[queue, "revisoes_36"]["media"])):,}'.replace(',', '.')
        assert value in questions
    adaptive = rows['A1', 'revisoes_23']
    assert f'{float(adaptive["media"]):.2f}'.replace('.', ',') in questions
    assert f'{int(float(adaptive["p10"]))} a {int(float(adaptive["p90"]))}' in questions
    with (ROOT / 'data/sensibilidade_o3_macd.csv').open(encoding='utf-8') as handle:
        macd = {(r['variante'], r['metrica']): float(r['valor']) for r in csv.DictReader(handle)}
    for variant in ('O3_MACD', 'O3_controle'):
        expected = '/'.join(f'{int(macd[variant, "revisoes_"+str(n)]):,}'.replace(',', '.') for n in (9, 23, 36))
        assert expected in questions
        assert f'{macd[variant, "ap_abaixo"]:.3f}'.replace('.', ',') in questions
