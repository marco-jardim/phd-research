from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gzcmd import arranjo_robustez as robust
from gzcmd.arranjo_corte import MODEL, REMOVED_FIELDS, crossfit_rates, nested_cutoff_probabilities
from gzcmd.ml_oof import _classifier


class Model:
    def fit(self, x, y):
        self.offset = float(np.mean(y)) * .01
        return self

    def predict_proba(self, x):
        p = .02 + .9 * x.signal.to_numpy() + self.offset
        return np.column_stack([1 - p, p])


def synthetic():
    x = pd.DataFrame({'signal': np.arange(400) / 400})
    y = np.tile([0, 1, 0, 1], 100)
    return x, y, np.ones(400, dtype=bool), np.repeat(np.arange(100), 4)


def test_factory_changes_only_registered_parameters():
    y = np.array([0, 0, 1])
    baseline = _classifier(y, 42).get_params()
    actual = robust.replica_factory(101, 2)(y, 1190).get_params()
    assert actual['random_state'] == 101
    assert actual['n_jobs'] == 2
    for key in baseline.keys() - {'random_state', 'n_jobs'}:
        if isinstance(baseline[key], float) and np.isnan(baseline[key]):
            assert np.isnan(actual[key])
        else:
            assert actual[key] == baseline[key]


def test_fixed_grid_order():
    grade = robust.grid()
    assert len(grade) == 40
    assert asdict(grade[0]) == dict(replica='N00', bloco='N', fold_seed=42, xgb_seed=42, n_jobs=8)
    assert (grade[1].xgb_seed, grade[1].n_jobs) == (42, 2)
    assert [(r.xgb_seed, r.n_jobs) for r in grade[:20]] == [(s, j) for s in (42, *range(101, 110)) for j in (8, 2)]
    assert [r.fold_seed for r in grade[20:]] == list(range(1000, 1200, 10))


def test_fold_seed_not_xgb_seed_changes_splits_and_r0_exact(monkeypatch):
    x, y, below, groups = synthetic()
    factories = []

    def factory(seed, jobs):
        factories.append((seed, jobs))
        return lambda *_: Model()

    monkeypatch.setattr(robust, 'replica_factory', factory)
    reference, fold, _ = nested_cutoff_probabilities(
        x, y, below, groups, par2=y.astype(bool), include_a3=False, model_factory=lambda *_: Model())
    observed, same, _ = robust.run_replica(x, y, below, groups, par2=y.astype(bool), replica=robust.grid()[0])
    for arm in reference:
        np.testing.assert_array_equal(observed[arm], reference[arm])
    np.testing.assert_array_equal(fold, same)
    _, same, _ = robust.run_replica(x, y, below, groups, par2=y.astype(bool), replica=robust.grid()[2])
    np.testing.assert_array_equal(fold, same)
    _, changed, _ = robust.run_replica(x, y, below, groups, par2=y.astype(bool), replica=robust.grid()[20])
    assert not np.array_equal(fold, changed)
    assert factories == [(42, 8), (101, 8), (42, 8)]


def test_rates_exclude_own_new_fold_and_ignore_new_responses():
    frame = pd.DataFrame({'fold': np.repeat([0, 1], 20), 'band': ['low'] * 40,
                          'y': np.tile([0, 1], 20), 'decision': ['NONMATCH'] * 20 + ['MATCH'] * 20})
    fixed = {'low': {'e_fp': .1, 'e_fn': .2, 'e_unsure': .3}}
    rate = crossfit_rates(frame, fixed, [0, 1])
    assert rate[0]['low'].e_fp == 1
    assert rate[1]['low'].e_fp == 0
    changed = frame.copy()
    changed.loc[changed.fold.eq(0), 'decision'] = 'UNSURE'
    assert crossfit_rates(changed, fixed, [0, 1])[0] == rate[0]
    # N changes no fold membership and therefore no rate value.
    for _ in robust.grid()[:20]:
        assert crossfit_rates(frame, fixed, [0, 1]) == rate
    # Production rate estimator has no S2, supplement or robustness argument.
    from scripts import arranjo_robustez as cli
    import inspect
    source = inspect.getsource(cli.child_replica)
    assert 'frozen_responses(spec, data_root, mode)' in source
    assert 'known_responses(' not in source and 'all_responses(' not in source


def test_rate_source_allowlist_ignores_new_run(tmp_path, monkeypatch):
    from scripts import arranjo_robustez as cli
    frozen = tmp_path / 'frozen.json'
    robust.atomic_json(frozen, {'results': [{'pair_id': 'a', 'decision': 'MATCH'}]})
    robust.atomic_json(tmp_path / 'robustness.json', {'results': [{'pair_id': 'a', 'decision': 'NONMATCH'}]})
    monkeypatch.setattr(cli, 'input_path', lambda data_root, name: tmp_path / name)
    spec = {'reviewer_rates': {'sources': [{'mode': 'vigilancia', 'path': 'frozen.json'}]}}
    assert cli.frozen_responses(spec, tmp_path, 'vigilancia') == {'a': 'MATCH'}
    robust.atomic_json(tmp_path / 'robustness.json', {'results': [{'pair_id': 'b', 'decision': 'UNSURE'}]})
    assert cli.frozen_responses(spec, tmp_path, 'vigilancia') == {'a': 'MATCH'}


@pytest.mark.parametrize('metric,direction', [('aceitos', 1), ('fp_abaixo', -1), ('perdidos', -1),
                                             ('pares_humano', -1), ('pares_llm', -1)])
def test_dominance_truth_table(metric, direction):
    p = dict(aceitos=10, fp_abaixo=5, perdidos=10, pares_humano=10, pares_llm=10)
    assert not robust.dominates(p, p)
    q = {**p, metric: p[metric] + direction}
    assert robust.dominates(q, p)
    assert not robust.dominates(p, q)
    tradeoff = {**q, 'fp_abaixo': 6} if metric != 'fp_abaixo' else {**q, 'aceitos': 9}
    assert not robust.dominates(tradeoff, p)


@pytest.mark.parametrize('counts,inverse,expected', [([18, 18], [1, 1], 'estável'),
    ([20, 17], [0, 0], 'mista'), ([9, 9], [0, 0], 'não sustentada'),
    ([20, 20], [0, 2], 'não sustentada'), ([9, 10], [0, 0], 'mista')])
def test_classification(counts, inverse, expected):
    assert robust.classify(counts, inverse) == expected


def test_stop_rule_boundaries():
    robust.safety_stop(2000, cost=6.5, projected_seconds=28800)
    for kwargs in ({'n_new': 2001}, {'n_new': 0, 'cost': 6.5000001},
                   {'n_new': 0, 'projected_seconds': 28800.1}):
        with pytest.raises(RuntimeError, match='STOP'):
            robust.safety_stop(**kwargs)


def test_clopper_pearson_endpoints():
    assert robust.clopper_pearson(0)[0] == 0
    assert robust.clopper_pearson(20)[1] == 1
    assert robust.clopper_pearson(10)[0] == pytest.approx(1 - robust.clopper_pearson(10)[1])


def test_checkpoint_identity_rejects_divergence(tmp_path):
    path = tmp_path / 'oof'
    np.savez_compressed(path.with_suffix('.npz'), p=[.1])
    robust.atomic_json(path.with_suffix('.json'), {'identity': {'seed': 42, 'hash': 'a'}})
    assert robust.checkpoint_matches(path, {'seed': 42, 'hash': 'a'})
    assert not robust.checkpoint_matches(path, {'seed': 101, 'hash': 'a'})
    assert not robust.checkpoint_matches(path, {'seed': 42, 'hash': 'b'})


def test_replace_backoff_windows_permission(tmp_path, monkeypatch):
    original = robust.os.replace
    calls = []

    def locked(source, destination):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError('Windows lock')
        original(source, destination)

    monkeypatch.setattr(robust.os, 'replace', locked)
    monkeypatch.setattr(robust.time, 'sleep', lambda _: None)
    robust.atomic_json(tmp_path / 'out.json', {'ok': True})
    assert len(calls) == 3


def test_supplement_checks_actual_dossier_hash(tmp_path):
    from scripts import arranjo_corte_s2_suplementar as supplement

    class View:
        def to_json(self, **kwargs):
            return '{"value":1}'

    digest = hashlib.sha256(View().to_json().encode()).hexdigest()
    run = {'mode': 'vigilancia', 'model': MODEL, 'prompt_version': 'v4',
           'prompt_sha256': supplement.sha256(supplement.PROMPT), 'complete': True,
           'removed_fields': list(REMOVED_FIELDS), 'dossier_sha256': {'a': digest},
           'results': [{'pair_id': 'a', 'decision': 'MATCH'}]}
    path = tmp_path / 'run.json'
    robust.validate_dossiers(path, run, 'vigilancia', {'a': View()}, prompt_sha256=run['prompt_sha256'])
    altered = deepcopy(run)
    altered['dossier_sha256']['a'] = 'different'
    with pytest.raises(ValueError, match='Dossier cache identity'):
        robust.validate_dossiers(path, altered, 'vigilancia', {'a': View()}, prompt_sha256=run['prompt_sha256'])
    identity = path.with_name('run_journal')
    identity.mkdir()
    robust.atomic_json(identity / 'identity.json', altered)
    with pytest.raises(ValueError, match='Dossier cache identity'):
        robust.validate_dossiers(path, run, 'vigilancia', {'a': View()}, prompt_sha256=run['prompt_sha256'])


def test_technical_amendment_preserves_execution_identity():
    from scripts import arranjo_robustez as cli
    spec = {'sha256': {'code': 'new'}, 'versions': {'sklearn': '1.8.0', 'xgboost': '3.2.0'},
            'amendments': [{'execution_identity': {'prereg_sha256': 'original', 'sha256': {'code': 'old'}}}]}
    original = cli.identity_for(spec, 'original', robust.grid()[0], False)
    assert original['sha256'] == {'code': 'old'}
    assert original['prereg_sha256'] == 'original'
    assert cli.identity_for(spec, 'amended', robust.grid()[0], False)['sha256'] == {'code': 'new'}


def test_historical_fingerprints_are_exact_and_robustness_outputs_unchanged():
    root = Path(__file__).resolve().parents[1]
    for name in ('data/arranjo_corte_metadata.json', 'data/equidade_arranjo_manifest.json'):
        metadata = json.loads((root / name).read_text(encoding='utf-8'))
        for path, digest in metadata['code_sha256'].items():
            assert robust.sha256(root / path) == digest
    metadata = json.loads((root / 'data/robustez_arranjo_metadata.json').read_text(encoding='utf-8'))
    for name, digest in metadata['outputs_sha256'].items():
        assert robust.sha256(root / f'data/robustez_arranjo_{name}.csv') == digest
    assert robust.sha256(root / 'data/arranjo_corte_metricas.csv') == 'd5f9f3c790802c8dcf7bb2d50c333390c5523f002aafdd7315e136e1c9945d16'
    assert robust.sha256(root / 'data/arranjo_corte_par2.csv') == '2f762158f44c9d5ceaa04c5788878e51e6d0c22ea65700e064a0a82d652b9f0e'


def test_phase_a_no_api_no_outcome_metrics_and_allowlist():
    from scripts import arranjo_robustez as cli
    import inspect
    source = inspect.getsource(cli.phase_a) + inspect.getsource(cli.child_replica)
    assert 'run_mode(' not in source and 's3.metrics(' not in source
    assert 'par2_metrics(' not in source and 'phase_c(' not in source
    out = pd.DataFrame({'band': ['low'], 'action': ['NONMATCH'], 'y': [1], 'PAR': [2]})
    summary = cli.summarize_actions(out, 'vigilancia', 'P')
    assert list(summary) == ['modo', 'braco', 'banda', 'acao', 'n']


def test_verification_rejects_mismatched_hash_before_training(tmp_path):
    path = tmp_path / 'spec.yaml'
    path.write_text('confirmatory: false\nprincipal_arm_unchanged: P\nsha256: {}\n')
    with pytest.raises(ValueError, match='SHA256'):
        robust.verify_prereg(path, 'wrong', root=tmp_path, data_root=tmp_path)


def test_action_stability_without_labels():
    a = pd.DataFrame({'pair_id': ['a', 'b'], 'p_cal': [.1, .2], 'action': ['NONMATCH', 'LLM_REVIEW']})
    b = pd.DataFrame({'pair_id': ['b', 'a'], 'p_cal': [.2, .3], 'action': ['LLM_REVIEW', 'LLM_REVIEW']})
    result = robust.action_stability(b, a, ['a', 'b'])
    assert result['p_alterados'] == 1
    assert result['proporcao_acao_alterada'] == .5
    assert result['jaccard_llm'] == .5
