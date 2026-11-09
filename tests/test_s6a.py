from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from terminologia import contem
from gzcmd import arranjo_robustez as robust
from gzcmd.arranjo_corte import crossfit_rates
from scripts.carga_abaixo_corte_horas import generate as hours_generate
from scripts.robustez_arranjo_agregados import regenerate
from scripts.tabelas_arranjo_editorial import replace_once, step_ppv_note, render_arrangement

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
LATEX = ROOT / 'text/latex'


def test_step11_note_distinguishes_maximum_difference_from_pairwise_tests():
    maximum, groups, tests = [pd.read_csv(DATA / f'equidade_arranjo_{name}.csv')
                              for name in ('maxdiff', 'group_metrics', 'pairwise_tests')]
    note = step_ppv_note(maximum, groups, tests)
    assert 'não para os testes pareados de VPP' in note
    assert 'que incluem o passo 11' in note
    labels = {'arranjo_P_vigilancia': 'Arranjo P (vigil.)', 'arranjo_P_confirmacao': 'Arranjo P (confirm.)',
              'cascata_v31_vigilancia': 'Base inteira v3.1 (vigil., comparação)',
              'cascata_v31_confirmacao': 'Base inteira v3.1 (confirm., comparação)'}
    for decision, label in labels.items():
        row = maximum.loc[maximum.decisao.eq(decision) & maximum.dimension.eq('passo') & maximum.metric.eq('ppv')].iloc[0]
        step = groups.loc[groups.decisao.eq(decision) & groups.dimension.eq('passo') & groups.group.eq('passo_11')].iloc[0]
        status = 'incluído' if step.n_pred_pos >= 30 else 'excluído'
        assert f'{label}: {int(row.n_grupos_den_ge_30)} de {int(row.n_grupos_elegiveis)} grupos; passo 11 {status}, com {int(step.n_pred_pos)}' in note
        paired = tests.loc[tests.decisao.eq(decision) & tests.dimension.eq('passo') & tests.metric.eq('ppv')]
        assert (paired.group_a.eq('passo_11') | paired.group_b.eq('passo_11')).sum() == 2
    altered = maximum.copy()
    altered.loc[altered.dimension.eq('passo') & altered.metric.eq('ppv'), 'n_grupos_den_ge_30'] = 99
    with pytest.raises(ValueError, match='Denominadores'):
        step_ppv_note(altered, groups, tests)


@pytest.mark.parametrize('text', ['no match', 'old old'])
def test_editorial_replacements_fail_on_absent_or_ambiguous_target(text):
    with pytest.raises(ValueError, match='count == 1'):
        replace_once(text, 'old', 'new')


def test_arrangement_editorial_preserves_every_number():
    original = (LATEX / 'tables/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8')
    current = (LATEX / 'tables/editorial/tab_cap8_arranjo_corte.tex').read_text(encoding='utf-8')
    assert current == render_arrangement(original)
    # Remove comentários de proveniência, que têm o nome de script com S3.
    numeric = lambda text: re.findall(r'\d+(?:[.,]\d+)*', re.sub(r'(?m)^([^&\n]+ & \d+ & \d+(?:,\d+)?) \[[^]]+\]( &.*)$', r'\1\2', re.sub(r'(?m)^%.*$', '', text)).replace(r'$S_A$ (IC 95\%)', r'$S_A$ (\%)'))
    from editorial_expectations import arrangement_expected_notes
    assert numeric(arrangement_expected_notes(original)) == numeric(current)
    assert current.count('61.485 (teto; 36)') == 2
    assert r'\textit{A posteriori}' in current
    # Revisão 2: [5,9) é a "faixa intermediária da nota final"; "área cinza" fica só para a região abaixo do corte.
    arquivo = 'text/latex/tables/editorial/tab_cap8_arranjo_corte.tex'
    assert contem(current, r'faixa intermediária da nota final $[5,9)$', arquivo=arquivo)
    assert 'área cinza' not in current and 'grey area' not in current


def test_workload_hours_reproduce_source_and_historical_time_assumption(tmp_path):
    for name in ('data/carga_abaixo_corte.csv', 'gzcmd/gzcmd_v3_config.yaml'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    generated = hours_generate(tmp_path)
    assert generated.read_bytes() == (DATA / generated.name).read_bytes()
    table = pd.read_csv(generated, dtype=str).set_index('limiar')
    counts = pd.read_csv(DATA / 'carga_abaixo_corte.csv')
    totals = counts.loc[counts.passo.eq('TOTAL')].set_index('limiar')
    for key, row in table.iterrows():
        assert int(row.pares) == totals.loc[key, 'pares_acima_limiar']
        expected = Decimal(row.pares) * Decimal(row.minutos_por_par) / 60
        assert abs(Decimal(row.horas) - expected) < Decimal('1e-12')
        assert Decimal(row.horas_inteiras) * 60 + Decimal(row.minutos_restantes) == Decimal(row.pares) * Decimal(row.minutos_por_par)


def test_frozen_rates_are_stable_across_actual_N_replica_partitions(monkeypatch):
    class Model:
        def __init__(self, seed):
            self.seed = seed
        def fit(self, x, y):
            return self
        def predict_proba(self, x):
            p = .01 + x.signal.to_numpy() * .8 + (self.seed % 100) * .0001
            return np.column_stack([1 - p, p])
    monkeypatch.setattr(robust, 'replica_factory', lambda seed, jobs: lambda *_: Model(seed))
    x = pd.DataFrame({'signal': np.arange(400) / 400})
    y = np.tile([0, 1, 0, 1], 100)
    groups = np.repeat(np.arange(100), 4)
    below = np.ones(400, dtype=bool)
    baseline = None
    probabilities = []
    for replica in robust.grid()[:20]:
        scores, folds, _ = robust.run_replica(x, y, below, groups, par2=y.astype(bool), replica=replica)
        frame = pd.DataFrame({'fold': folds, 'band': 'low', 'y': y,
                              'decision': np.where(x.signal > .4, 'MATCH', 'NONMATCH')})
        rates = crossfit_rates(frame, {'low': {'e_fp': .1, 'e_fn': .2, 'e_unsure': .3}}, range(5))
        if baseline is None:
            baseline = (folds.copy(), rates)
        np.testing.assert_array_equal(folds, baseline[0])
        assert rates == baseline[1]
        probabilities.append(scores['P-Platt'])
    assert any(not np.array_equal(value, probabilities[0]) for value in probabilities[1:])


def test_phase_C_aggregates_and_table_regenerate_from_versioned_csvs(tmp_path):
    table, primary, stability = [pd.read_csv(DATA / f'robustez_arranjo_{name}.csv')
                                 for name in ('replicas', 'par2', 'estabilidade')]
    differences, summary, conclusions = regenerate(table, primary, stability)
    for name, actual in [('diferencas', differences), ('resumo', summary)]:
        expected = pd.read_csv(DATA / f'robustez_arranjo_{name}.csv')
        pd.testing.assert_frame_equal(actual.reindex(columns=expected.columns), expected,
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    from scripts.arranjo_robustez import render_table
    target = tmp_path / 'robustez.tex'
    render_table(conclusions, target)
    # O artefato oficial foi gerado no Windows (CRLF), fixado no .gitattributes.
    official = (LATEX / 'tables/tab_cap8_robustez_arranjo.tex').read_bytes()
    assert b'\r\n' in official
    assert b'\n' not in official.replace(b'\r\n', b'')
    assert b'\r' not in official.replace(b'\r\n', b'')
    assert target.read_bytes().decode('utf-8').replace('\r\n', '\n') == official.decode('utf-8').replace('\r\n', '\n')


def test_chronology_uses_original_git_and_metadata_not_old_editorial_time():
    study = (LATEX / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    # SHAs da história reescrita em 2026-11-06, com datas de autor preservadas.
    for ref in ('10524d921a5faf2b473a6ff912df539d8fe90396', '49a6366b741d0dfdc34087a39bd838b08c49ca97'):
        stamp = subprocess.check_output(['git', 'show', '-s', '--format=%aI', ref], cwd=ROOT, text=True).strip()
        utc = datetime.fromisoformat(stamp).astimezone(timezone.utc)
        assert utc.strftime('%H:%M:%S UTC') in study
    metadata = json.loads(subprocess.check_output(['git', 'show', '49a6366b741d0dfdc34087a39bd838b08c49ca97:data/arranjo_corte_metadata.json'], cwd=ROOT, text=True))
    assert datetime.fromisoformat(metadata['timestamp_utc']).strftime('%H:%M:%S UTC') in study
    assert '04:13 UTC' not in study


def test_new_recommendation_is_not_presented_as_an_evaluated_result():
    text = (LATEX / 'conclusao.tex').read_text(encoding='utf-8')
    recommendation = text.split(r'\label{sec:concl-recomendacoes}')[1]
    assert r'Se for usado, o \texttt{MATCH} segue ao revisor humano' in recommendation
    # Conclusão reescrita: os usos subordinados ao revisor são hipóteses exploratórias (objetivo 4); a validação é a da última seção.
    assert 'seus usos subordinados ao revisor humano, como filtro de \\texttt{NONMATCH} ou fonte de evidência para a probabilidade, permanecem hipóteses exploratórias' in text
    assert contem(recommendation, 'O LLM não deve aceitar vínculos sozinho', arquivo='text/latex/conclusao.tex')
    assert contem(text, 'nas condições avaliadas, o LLM não deve aceitar vínculos sozinho', arquivo='text/latex/conclusao.tex')
    assert 'valor que estaria subestimado se algum dos dez falsos positivos' in recommendation
    assert 'A hipótese exige outra coorte ou período, não usados para fixar o corte nem as regras' in recommendation
    assert 'Com ordenação e orçamento fixados antes dos desfechos e referência cega independente' in recommendation
    # "posição de precaução, não estimativa transferível" permanece no Estudo 4 (restrição à aceitação autônoma pelo LLM).
    assert 'posição de precaução' in (LATEX / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')


def test_abstract_definitions_and_nonrejection_follow_source():
    thesis = (LATEX / 'tese.tex').read_text(encoding='utf-8')
    start_pt = thesis.index(r'O \textit{linkage} integra')
    start_en = thesis.index(r'Linkage integrates')
    pt = thesis[start_pt:thesis.index(r'\vspace{15pt}', start_pt)]
    en = thesis[start_en:thesis.index(r'\vspace{15pt}', start_en)]
    assert len(pt.split()) <= 500 and len(en.split()) <= 500
    # Redação nova (revisão 2): a recalibração isotônica e o P-Platt constam do Estudo 4; a sensibilidade de pelo
    # menos 95% nos passos 6 e 9 consta do Estudo 5; o resumo traz o passo 11 e a ausência de dominância.
    study4 = (LATEX / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert 'que preserva o corte e decide os demais potenciais pares pela política v3.1, com recalibração isotônica abaixo do corte' in study4
    assert r'P-Platt (análise de sensibilidade de P com \textit{Platt scaling}' in study4
    assert 'o arranjo P, que preserva o corte e decide abaixo dele' in pt
    assert 'arrangement P, with the cut-off preserved and decisions made below it' in en
    assert 'pelo menos 95,0' in (LATEX / 'estudo5_equidade.tex').read_text(encoding='utf-8')
    assert '42,5' in pt and '42.5' in en
    assert 'e nos demais foi superior: disparidade herdada do corte, em estrato do processo, não demográfico' in pt
    assert 'and higher in the others: a disparity inherited from the cut-off, tied to a process stratum rather than a demographic one' in en
    assert 'pois nenhuma dominou em todas as cargas' in pt
    assert 'since no ordering was dominant in every workload' in en
    tests = pd.read_csv(DATA / 'equidade_arranjo_pairwise_tests.csv')
    result = tests.loc[tests.decisao.eq('arranjo_P_vigilancia') & tests.dimension.eq('sexo')
                       & tests.metric.eq('ppv')].iloc[0]
    assert not result.reject_fdr
    # A ausência de rejeição demográfica foi substituída pelo resultado operacional.
    assert 'BH-adjusted' not in en and 'não demográfico' in pt
    assert r'\textit{Linkage}' not in en and r'\textit{Platt scaling}' not in en
