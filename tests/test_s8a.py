"""Contrato S8: fontes agregadas, instrumentos literais e apresentação editorial."""
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pytest

from scripts.tabelas_s8_editorial import generate, same_load, effort

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / 'text/latex'


@pytest.mark.parametrize('version,index', [('v3', 0), ('v4', 1)])
def test_prompt_appendix_is_literal_source_bytes(version, index):
    appendix = (L / 'apendice_prompts.tex').read_bytes()
    blocks = re.findall(rb'\\begin\{verbatim\}\r?\n(.*?)\\end\{verbatim\}', appendix, re.S)
    assert len(blocks) == 7
    source = (ROOT / f'gzcmd/gzcmd_{version}_llm_prompt.md').read_bytes()
    assert blocks[index] == source
    assert 'transcrição literal de instrumento fica fora da regra de travessão' in appendix.decode('utf-8')


def test_editorial_tables_regenerate_and_historical_numeric_cells_stay(tmp_path):
    generate(output=tmp_path)
    for name in ('tab_cap8_ancoras_beta', 'tab_cap8_ancoras_decisao', 'tab_cap8_robustez_arranjo',
                 'tab_cap8_mesma_carga', 'tab_cap8_esforco'):
        actual = (tmp_path / (name + '.tex')).read_text(encoding='utf-8')
        assert actual == (L / 'tables/editorial' / (name + '.tex')).read_text(encoding='utf-8')
        if name in ('tab_cap8_ancoras_beta', 'tab_cap8_ancoras_decisao', 'tab_cap8_robustez_arranjo'):
            historical = (L / 'tables' / (name + '.tex')).read_text(encoding='utf-8')
            rows = lambda text: [re.findall(r'\d+(?:[.,]\d+)*', line) for line in text.splitlines()
                                 if ' & ' in line and not line.startswith('Modo &')]
            assert rows(actual) == rows(historical)
    robust = (tmp_path / 'tab_cap8_robustez_arranjo.tex').read_text(encoding='utf-8')
    assert 'Numérico' in robust and 'Partição' in robust
    assert 'P ($k/20$)' not in robust


def test_same_load_rows_and_effort_are_generated_from_versioned_artifacts():
    data = pd.read_csv(ROOT / 'data/comparador_mesma_carga.csv')
    selected = data.loc[data.carga.eq('K1_humana')]
    for mode, expected in [('vigilancia', [19, 12, 22, 22]), ('confirmacao', [3, 9, 6, 5])]:
        rows = selected.loc[selected.modo.eq(mode)]
        assert rows.par2_topk.tolist() == expected
    text = same_load(data)
    altered = data.copy()
    altered.loc[altered.carga.eq('K1_humana'), 'par2_topk'] += 1
    assert same_load(altered) != text
    hours = pd.read_csv(ROOT / 'data/esforco_arranjo.csv')
    assert hours.cenario.all() and not hours.horas_medidas.any()
    assert hours.exploratorio_post_hoc.all()
    changed = hours.copy()
    changed['horas'] += 1
    assert effort(hours) != effort(changed)
    assert 'n.a.' in effort(hours)


def test_fp_position_and_loss_qualification_follow_sources():
    frame = pd.read_csv(ROOT / 'data/fp_arranjo_por_regiao.csv')
    above = frame.loc[frame.passo.eq('TOTAL') & frame.regiao_SCORE.eq('maior_ou_igual')]
    for mode, expected in [('vigilancia', [17, 11]), ('confirmacao', [4, 4])]:
        assert above.loc[above.modo.eq(mode), 'FP'].tolist() == expected
    conclusion = (L / 'conclusao.tex').read_text(encoding='utf-8')
    assert 'não reproduz 13 e 27 dos 36 vínculos recuperados pelo serviço' in conclusion
    assert 'aceita 25 e 12 potenciais pares abaixo do corte, dos quais 21 e 4' in conclusion
    assert '4 e 8 são PAR~=~2' in conclusion
    assert 'pior em todo desfecho' not in conclusion
    assert '17 dos 21 FP' in conclusion and '11 acima do limiar por passo' in conclusion


def test_new_framing_contributions_and_adoption_caveats_in_each_summary():
    for name in ('capitulo1.tex', 'conclusao.tex'):
        text = (L / name).read_text(encoding='utf-8')
        for term in ('curva de esforço', 'decomposição por PAR', 'ancoragem', 'resultado negativo',
                     'registro interno', 'ordem decrescente', 'orçamento humano'):
            assert term in text, (name, term)
    thesis = (L / 'tese.tex').read_text(encoding='utf-8')
    for prefix in (r'O \textit{linkage} integra', r'Linkage integrates'):
        i = thesis.index(prefix)
        text = thesis[i:thesis.index(r'\vspace{15pt}', i)]
        assert len(text.split()) <= 500
        assert len(re.findall(r'\d+(?:[.,]\d+)*', text)) <= 13
        assert '36 PAR~=~2' in text
        assert 'confirmação' in text or 'confirmation' in text
        assert '42,5' in text or '42.5' in text
        assert 'MATCH' in text and 'P-Platt' in text
        assert 'preregistered' not in text and 'intervals overlapping' not in text


def test_order_is_unverified_and_cdr_exposure_not_transferred_to_three_steps():
    data = (L / 'capitulo_dados.tex').read_text(encoding='utf-8')
    assert 'Não foram documentados ordem, limite ou regra de parada' in data
    assert 'Se ela tivesse seguido ordem decrescente de escore do OpenRecLink' in data
    assert 'não nos passos 6, 9 e 11' in data
    assert 'revisora única, sem classificação independente' in data
    assert 'p.~36 a~38' in data
    assert 'parecer do CEP abrange' in data


def test_cutoff_sensitivity_has_no_sampling_interval_in_editorial_table():
    text = (L / 'tables/editorial/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8')
    rows = [line for line in text.splitlines() if line.startswith('(i) Corte & 211 &')]
    assert len(rows) == 2
    assert all(row.split(' & ')[2] == '85,4' for row in rows)
    historical = (L / 'tables/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8')
    assert historical.count('(i) Corte & 211 & 85,4 [80,5; 89,3]') == 2


def test_frozen_code_sha256s_are_not_relaxed():
    for name in ('arranjo_corte_metadata.json', 'equidade_arranjo_manifest.json'):
        metadata = json.loads((ROOT / 'data' / name).read_text(encoding='utf-8'))
        for path, expected in metadata['code_sha256'].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path


def test_effort_figure_uses_the_same_generator_as_slides(tmp_path):
    from scripts.figura_curva_esforco import generate
    target = tmp_path / 'curva.pgf'
    generate(ROOT / 'data/comparador_mesma_carga_curva.csv',
             ROOT / 'data/comparador_mesma_carga.csv', target)
    assert target.read_bytes() == (L / 'figures/curva_esforco.pgf').read_bytes()
    assert target.read_bytes() == (ROOT / 'text/slides/figures/curva_esforco.pgf').read_bytes()
