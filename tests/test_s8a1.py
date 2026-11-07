"""Regressões da redação final, sem novos resultados experimentais."""
import ast
from pathlib import Path
import re

import pandas as pd
import pytest

from scripts.tabelas_s8a1_editorial import generate, subsampling, llm_filter, human_sensitivity
from scripts.tabelas_s8_editorial import effort, same_load
from scripts.transcrever_prompts_editorial import generate as prompts_generate, role_payloads

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / 'text/latex'


def forbidden_controls(text):
    return [ord(char) for char in text if (ord(char) < 32 and char not in '\r\n') or 127 <= ord(char) <= 159]


@pytest.mark.parametrize('value', ['\t', '\x00', '\x08', '\x1b', '\x7f', '\x85'])
def test_control_detector_rejects_nonprinting_characters(value):
    assert forbidden_controls('texto'+value)
    assert not forbidden_controls('texto\r\nseguinte\n')


def test_all_tex_files_have_no_control_characters():
    violations = {str(path.relative_to(ROOT)): forbidden_controls(path.read_text(encoding='utf-8'))
                  for path in (ROOT / 'text').rglob('*.tex')}
    assert not {path: values for path, values in violations.items() if values}


def test_prompt_roles_and_batch_instruction_match_frozen_source(tmp_path):
    target = tmp_path / 'prompts.tex'
    prompts_generate(output=target)
    assert target.read_bytes() == (L / 'apendice_prompts.tex').read_bytes()
    blocks = re.findall(rb'\\begin\{verbatim\}\r?\n(.*?)\\end\{verbatim\}', target.read_bytes(), re.S)
    assert len(blocks) == 7
    for block, (_, payload) in zip(blocks[2:], role_payloads()):
        assert block == payload.encode('utf-8') + b'\n'
    module = ast.parse((ROOT / 'gzcmd/llm_review.py').read_text(encoding='utf-8'))
    function = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == '_build_messages')
    class_roles = next(ast.literal_eval(node.value) for node in ast.walk(function) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'role_instruction' for t in node.targets))
    assert dict(role_payloads()[:3]) == class_roles
    assert 'Output ONLY the JSON object' in blocks[-1].decode('utf-8')


def test_new_tables_regenerate_only_into_editorial(tmp_path):
    generate(output=tmp_path)
    for path in tmp_path.glob('*.tex'):
        assert path.read_bytes() == (L / 'tables/editorial' / path.name).read_bytes()
    assert len(list(tmp_path.glob('*.tex'))) == 3
    with pytest.raises(ValueError, match='históricas'):
        generate(output=L / 'tables')


def test_subsampling_table_follows_aggregates_and_changes_with_source():
    data = pd.read_csv(ROOT / 'data/subamostragem_treino_resumo.csv')
    below = data.loc[data.population.eq('below') & data.variant.eq('isotonic') & data.ratio.ne('original')]
    top = below.loc[below.metric.eq('par2_top_96')].set_index('ratio')
    assert top.loc['1:1', 'median'] == 13
    assert top.loc['1:5', 'median'] == 18.5
    for metric in ('brier', 'ece'):
        assert below.loc[below.metric.eq(metric), 'delta_median'].gt(0).all()
    text = subsampling(data)
    changed = data.copy()
    changed.loc[changed.metric.eq('par2_top_96'), 'median'] += 1
    assert subsampling(changed) != text
    assert 'não IC' in text and 'prevalência original' in text


def test_filter_distinguishes_llm_queue_from_human_work_and_coverage():
    data = pd.read_csv(ROOT / 'data/filtro_llm.csv')
    row = data.loc[data.modo.eq('vigilancia') & data.k.eq(96)].iloc[0]
    assert (row.H, row.par2_recuperados, row.o3_par2_mesma_carga) == (22, 13, 6)
    assert row.vpn_nonmatch == pytest.approx(65 / 74)
    top = data.loc[data.k.eq(125)]
    assert top.par2_descartados.eq(10).all() and top.par2_topk.eq(23).all()
    text = llm_filter(data)
    assert 'não a carga humana' in text and 'topo 440 incompleto' in text
    changed = data.copy()
    changed['H'] += 1
    assert llm_filter(changed) != text


def test_human_scenarios_are_coverage_not_measured_error():
    data = pd.read_csv(ROOT / 'data/cenarios_sensibilidade_humana.csv')
    assert set(data.s) == {.95, .99, 1.}
    assert data.especificidade_humana_assumida.eq(1).all()
    assert data.amplitude_pp.max() == pytest.approx(.728744939271)
    assert data.loc[~data.estrategia.eq('humano_integral'), 'amplitude_pp'].max() == pytest.approx(.526315789474)
    order = pd.read_csv(ROOT / 'data/cenarios_sensibilidade_humana_ordem.csv')
    assert not order.loc[order.mesma_carga_humana, 'inverte_ordem_estrita'].any()
    text = human_sensitivity(data)
    assert 'Filtro' in text and 'Não se modelam FP humanos' in text
    discussion = (L / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    assert 'As perdas são de cobertura, não de erro humano' in discussion
    assert '0,73 ponto percentual' in discussion and '0,53 nas estratégias' in discussion


def test_acceptances_split_true_and_false_and_match_to_human_comparator():
    data = pd.read_csv(ROOT / 'data/comparador_p_match_humano.csv').set_index('modo')
    assert tuple(data.loc['vigilancia', ['P_revisoes_humanas_total', 'P_par2_recuperados', 'O3_par2_mesma_carga', 'O3_k_min_igualar_P']]) == (121, 23, 22, 125)
    assert tuple(data.loc['confirmacao', ['P_revisoes_humanas_total', 'P_par2_recuperados', 'O3_par2_mesma_carga', 'O3_k_min_igualar_P']]) == (29, 9, 9, 29)
    for name in ('estudo4_gzcmd.tex', 'conclusao.tex'):
        text = (L / name).read_text(encoding='utf-8')
        assert 'dos quais 21 e 4 foram rejeitados ou não examinados' in text
        assert '4 e 8 são PAR~=~2' in text
        assert 'aceita 25 e 12 potenciais pares abaixo do corte que o serviço rejeitou' not in text
    text = same_load(pd.read_csv(ROOT / 'data/comparador_mesma_carga.csv'))
    assert 'P, \\texttt{MATCH} ao humano' in text
    assert '344 (755)' in text
    assert 'revisor perfeito dos dois lados' in text


def test_effort_uses_saved_hours_per_lost_link_and_keeps_llm_separate():
    data = pd.read_csv(ROOT / 'data/esforco_arranjo.csv')
    text = effort(data)
    assert 'Horas poupadas' in text and 'por vínculo perdido' in text
    assert 'Pares LLM' in text and 'sem equivalência temporal' in text
    changed = data.copy()
    changed['horas_poupadas_por_vinculo_perdido'] += 100
    assert effort(changed) != text


def test_hypothesis_bias_and_recommendation_are_not_claimed_as_validation():
    intro = (L / 'capitulo1.tex').read_text(encoding='utf-8')
    conclusion = (L / 'conclusao.tex').read_text(encoding='utf-8')
    assert 'gerou a hipótese' in intro and 'A avaliação gerou a hipótese' in conclusion
    assert 'A tese sustenta' not in conclusion
    for text, expected in (
        (intro, 'investigação do LLM como aceitador'),
        (conclusion, 'a investigação do LLM como aceitador e como filtro de descarte antes do revisor humano, associada à investigação de sua ancoragem'),
    ):
        assert 'Se a revisão' in text and 'O3 e O4 também' in text
        assert expected in text
    assert 'O LLM pode atuar somente como filtro' in conclusion
    assert r'Se for utilizado, seu \texttt{MATCH} vai ao revisor humano' in conclusion
    assert 'Os potenciais pares fora do orçamento não são examinados e devem ser registrados como não examinados' in conclusion
    assert 'a regra de duas ações atribui-lhes não vínculo por custo' in conclusion
    assert 'não equivale a óbito não notificado' in conclusion
    assert 'validação registrada em dados novos' in conclusion


def test_tables_figure_and_derivation_have_explicit_references():
    text = (L / 'estudo4_gzcmd.tex').read_text(encoding='utf-8')
    for label in ('fig:cap8-curva-esforco', 'tab:cap8-mesma-carga', 'tab:cap8-esforco',
                  'tab:cap8-subamostragem', 'tab:cap8-filtro-llm', 'tab:cap8-sensibilidade-humana'):
        assert r'\ref{'+label+'}' in text
    assert r'p\geq C_H/C_{FN}' in text
    assert 'não no critério' in text
    assert r'\citeonline{DalPozzolo2015calibrating}' in text
    assert 'Sariyar2010recordlinkage' in text
