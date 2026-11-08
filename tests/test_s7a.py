"""Regressões editoriais da conferência S6, sem alterar fontes históricas."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from terminologia import GREY_AREA, contem, normalizar_terminologia

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / 'text/latex'
DATA = ROOT / 'data'


def abstracts():
    thesis = (LATEX / 'tese.tex').read_text(encoding='utf-8')
    for start in (r'O \textit{linkage} integra', r'Linkage integrates'):
        i = thesis.index(start)
        yield thesis[i:thesis.index(r'\vspace{15pt}', i)]


def test_reviewer_acceptances_and_arrangement_outcomes_have_separate_subjects():
    pt, en = abstracts()
    for text in (pt, en):
        assert len(text.split()) <= 500
    assert contem(pt, 'no máximo um vínculo adicional', arquivo='text/latex/tese.tex')  # migrado: terminologia 3.6
    assert '344 pares avaliados pelo modelo de linguagem de grande porte (LLM)' in pt
    assert 'at most one additional link' in en
    assert '344 LLM-reviewed pairs' in en
    assert 'VPP' in pt and 'PPV' in en
    assert '22 revisões humanas' in pt and '22 human reviews' in en
    losses = pd.read_csv(DATA / 'arranjo_corte_perdas_mecanismo.csv')
    totals = pd.read_csv(DATA / 'arranjo_corte_metricas.csv')
    llm = pd.read_csv(DATA / 'arranjo_corte_vpp_llm.csv')
    for mode, mechanisms in [('vigilancia', [1, 6, 6]), ('confirmacao', [1, 22, 4])]:
        rows = losses.loc[losses.braco.eq('P') & losses.modo.eq(mode)].set_index('mecanismo')
        values = [int(rows.loc[key, 'vinculos_perdidos']) for key in (
            'guarda_nota_lt_5', 'AUTO_NONMATCH_custo', 'NONMATCH_LLM')]
        assert values == mechanisms
        observed = totals.loc[totals.braco.eq('P') & totals.modo.eq(mode) & totals.faixa.eq('TODAS')].iloc[0]
        assert sum(values) == observed.perdidos
        reviewer = llm.loc[llm.braco.eq('P') & llm.modo.eq(mode)].iloc[0]
        assert rows.loc['MATCH_LLM', 'nao_vinculos'] == reviewer.FP == observed.fp
    study = (LATEX / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert contem(study, 'a cadeia aceitou 196 vínculos', arquivo='text/latex/estudo4_gzcmd.tex')  # migrado: terminologia 3.6
    assert 'o Sonnet como revisor aceitou' not in study


def test_cutoffs_remain_three_points_not_a_confidence_interval():
    # Pedido C10 da banca: cortes movidos para o apêndice do linkage de origem.
    text = ''.join((LATEX / name).read_text(encoding='utf-8')
                   for name in ('capitulo_dados.tex', 'apendice_linkage_origem.tex'))
    assert '7,0240; 7,8444; 7,9019' in text
    assert '7,8444 a 7,9019' not in text


def test_platt_name_adjusted_p_and_keywords_follow_the_style_guide():
    pt, en = abstracts()
    assert r'análise de sensibilidade com \textit{Platt scaling}' in pt
    assert r'the sensitivity analysis using Platt scaling' in en
    assert '42,5' in pt and '42.5' in en
    assert '0,427' in (LATEX / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    for path in LATEX.glob('*.tex'):
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'calibração sigmoidal|sigmoid calibration', text, re.I), path
        for line in text.splitlines():
            if line.lstrip().startswith('%'):
                continue
            if re.search(r'\brótulos?\b', line):
                # Rótulo de campo de uma busca bibliográfica não é alvo de ML.
                assert path.name == 'apendice_busca.tex' and 'sem rótulos de campo' in line
    thesis = (LATEX / 'tese.tex').read_text(encoding='utf-8')
    assert r'Keywords:} Probabilistic record linkage.' in thesis


def test_conclusion_separates_registration_from_negative_finding_and_has_comparators():
    text = (LATEX / 'conclusao.tex').read_text(encoding='utf-8')
    assert 'não lhe confere força confirmatória' in text
    assert 'registrado antes da avaliação:' not in text
    assert text.count('pior em todo desfecho medido contra') == 0
    reference = pd.read_csv(DATA / 'referencia_comparador_pos_orl.csv').set_index('estrategia')
    cut = reference.loc['pos_orl_corte_nota_9']
    review = reference.loc['pos_orl_revisao_humana_area_cinza']
    fmt = lambda value: f'{value:.1f}'.replace('.', ',')
    assert fmt(cut.sens_automatica * 100) + r'\%' in text
    assert fmt(review.sens_com_humano * 100) + r'\%' in text
    assert f'{int(review.pares_ao_humano):,}'.replace(',', '.') in text
    assert contem(text, 'área cinza da nota projetaria', arquivo='text/latex/conclusao.tex')  # migrado: terminologia 3.6


def test_editorial_area_review_is_a_ceiling_without_numeric_change():
    old = (LATEX / 'tables/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8')
    new = (LATEX / 'tables/editorial/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8')
    assert old.count('21.497 (28)') == 2
    assert new.count('21.497 (teto; 28)') == 2
    numeric = lambda text: re.findall(r'\d+(?:[.,]\d+)*', re.sub(r'(?m)^([^&\n]+ & \d+ & \d+(?:,\d+)?) \[[^]]+\]( &.*)$', r'\1\2', re.sub(r'(?m)^%.*$', '', text)).replace(r'$S_A$ (IC 95\%)', r'$S_A$ (\%)'))
    from editorial_expectations import arrangement_expected_notes
    assert numeric(arrangement_expected_notes(old)) == numeric(new)


def test_fraction_denominators_are_explicitly_minima_not_observed_counts():
    metadata = json.loads((DATA / 'arranjo_corte_s1_metadata.json').read_text(encoding='utf-8'))
    values = [row['grey_low']['e_fn'] for row in metadata['sensitivity_PAR2_grey_rates']['confirmacao'].values()]
    fractions = {Fraction(value).limit_denominator(1000) for value in values}
    assert fractions == {Fraction(1, 2), Fraction(1, 3), Fraction(2, 3)}
    study = (LATEX / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert 'frações irredutíveis $1/2$, $1/3$ e $2/3$' in study
    assert 'denominadores mínimos de dois ou três' in study
    assert 'não denominadores efetivos documentados' in study


def test_study_grey_zone_model_version_chronology_and_range_are_unambiguous():
    study = (LATEX / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    for prefix in ('Nos 21.620', r'A Tabela~\ref{tab:e4-desempenho-faixas-oof}',
                   'O escore de Brier', r'A Tabela~\ref{tab:cap8-ancoras-decisao}'):
        paragraph = next(p for p in study.split('\n\n') if p.startswith(prefix))
        # migrado: terminologia 3.6 (a glosa "(área cinza)" vira o mesmo marcador de \textit{grey area})
        assert not re.search(f'{GREY_AREA}(?! da nota)', normalizar_terminologia(paragraph, arquivo='text/latex/estudo4_gzcmd.tex'))
    assert 'Claude Haiku 5.5 com instruções v4 e sem as saídas do motor' in study
    assert r'S3 às 02:11:52 UTC, conforme \path{data/arranjo_corte_metadata.json} na versão \texttt{16d5876}' in study
    assert 'não no JSON atual' in study
    assert 'e as perdas de nove a 21' not in study
    assert 'mediana 14, mínimo 9 e máximo 21' in study


def test_historical_fingerprints_still_match_exactly():
    for name in ('arranjo_corte_metadata.json', 'equidade_arranjo_manifest.json'):
        metadata = json.loads((DATA / name).read_text(encoding='utf-8'))
        for path, expected in metadata['code_sha256'].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
