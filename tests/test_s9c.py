import ast
import json
from pathlib import Path
import runpy
import subprocess

import pytest

from local_data import DATA_ROOT, requires_files
from scripts.auditar_eol import audit
from scripts.filtro_llm import resolve

ROOT = Path(__file__).resolve().parents[1]
FILTER_AUDIT = json.loads((ROOT / 'data/s9c_filtro_duplicatas.json').read_text(encoding='utf-8'))


def test_byte_sensitive_references_have_explicit_coherent_eol():
    rows = audit()
    assert len(rows) > 100
    assert not [row for row in rows if not row['ok']]
    names = {row['path'] for row in rows}
    assert {'data/cenarios_sensibilidade_humana.csv',
            'data/comparador_p_match_humano_metadata.json',
            'text/latex/apendice_prompts.tex',
            'text/latex/tables/editorial/tab_cap8_filtro_llm.tex'} <= names


def test_eol_audit_rejects_missing_and_wrong_rules(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'data').mkdir()
    (tmp_path / 'tests/test_generated.py').write_text(
        'def test_output():\n    assert Path("data/generated.csv").read_bytes() == b"x\\n"\n',
        encoding='utf-8', newline='\n')
    (tmp_path / 'data/generated.csv').write_bytes(b'x\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    assert not next(row for row in audit(tmp_path) if row['path'] == 'data/generated.csv')['ok']
    (tmp_path / '.gitattributes').write_text('data/generated.csv text eol=crlf\n', encoding='utf-8')
    assert not next(row for row in audit(tmp_path) if row['path'] == 'data/generated.csv')['ok']
    (tmp_path / '.gitattributes').write_text('data/generated.csv text eol=lf\n', encoding='utf-8')
    assert next(row for row in audit(tmp_path) if row['path'] == 'data/generated.csv')['ok']


@pytest.mark.parametrize('directory', ['anthropic_sem_mo_v4_arranjo_corte',
    'anthropic_sem_mo_v4_arranjo_corte_supl', 'anthropic_sem_mo_v4_arranjo_robustez'])
def test_private_mode_audit_skips_if_a_run_dir_source_is_absent(tmp_path, monkeypatch, directory):
    from scripts import filtro_llm
    missing = f'data/sprint3b/{directory}/exp2_llm_sync_confirmacao.json'
    existing = tmp_path / 'present.json'
    existing.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(filtro_llm, 'resolve', lambda root, name: tmp_path / missing if name == missing else existing)
    module = runpy.run_path(str(ROOT / 'tests/test_cenarios_sensibilidade_humana.py'))
    marks = module['test_private_mode_audit_regenerates_count_only_outputs'].pytestmark
    assert any(mark.name == 'skipif' and mark.args == (True,)
               and missing in mark.kwargs['reason'].replace('\\', '/')
               for mark in marks)


def test_frozen_cost_monitor_limitation_is_documented_not_silently_patched():
    # Characterization only: the historical hash is fixed. No process/API starts.
    tree = ast.parse((ROOT / 'scripts/filtro_llm_executar.py').read_text(encoding='utf-8'))
    run_mode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_mode')
    commands = [n.value for n in ast.walk(run_mode) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert '  $cost = 0.0' in commands
    assert any("-Filter '*.transport.json'" in c for c in commands)
    assert not any('$journals.Count' in c for c in commands)
    doc = (ROOT / 'gzcmd/S9c_desvios_filtro.md').read_text(encoding='utf-8')
    assert 'sem erro' in doc and 'não foi alterado' in doc
    meta = json.loads((ROOT / 'data/filtro_llm_metadata.json').read_text(encoding='utf-8'))
    assert meta['new_execution']['total_pairs'] == 83
    assert meta['new_execution']['total_api_calls'] == 201


@requires_files(*(resolve(DATA_ROOT, entry['source']) for entry in FILTER_AUDIT['sources']))
def test_private_duplicate_audit_matches_all_historical_response_sources():
    from collections import Counter
    meta = json.loads((ROOT / 'data/filtro_llm_metadata.json').read_text(encoding='utf-8'))
    sources = {entry['source'] for mode in meta['response_audit'].values() for entry in mode['sources']}
    assert sources == {entry['source'] for entry in FILTER_AUDIT['sources']}
    total = 0
    for entry in FILTER_AUDIT['sources']:
        rows = json.loads(resolve(DATA_ROOT, entry['source']).read_text(encoding='utf-8'))['results']
        duplicate_rows = sum(count - 1 for count in Counter(row['pair_id'] for row in rows).values())
        assert len(rows) == entry['rows']
        assert duplicate_rows == entry['duplicates'] == 0
        total += len(rows)
    assert total == FILTER_AUDIT['total_source_rows'] == 2651
