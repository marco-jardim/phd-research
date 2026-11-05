"""Registro e execução sequencial da robustez não confirmatória P/P-Platt."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from gzcmd.arranjo_corte import (
    below_cutoff, crossfit_rates, numeric_features, pending_cost, sha256, summarize_actions,
)
from gzcmd.arranjo_robustez import (
    action_stability, atomic_json, checkpoint_matches, classify, clopper_pearson,
    dominates, grid, replace_backoff, resolve_input, run_replica, safety_stop, validate_dossiers, verify_prereg,
)
from gzcmd.bands import BandAssigner
from gzcmd.config import load_config
from gzcmd.guardrails import apply_guardrails
from gzcmd.ml_oof import pair_ids
from gzcmd.runner import build_engine_from_config
from scripts import arranjo_corte_s1_simular as s1
from scripts import arranjo_corte_s2_suplementar as supplement
from scripts import arranjo_corte_s2_executar as executor
from scripts import arranjo_corte_s3_avaliar as s3

LOCAL = ROOT / '.local/arranjo_robustez'
PREREG = ROOT / 'gzcmd/prereg_robustez_arranjo.yaml'
OUT = ROOT / 'data/sprint3b/anthropic_sem_mo_v4_arranjo_robustez'
NOTE = ROOT / 'gzcmd/execucao_robustez_arranjo.json'
ARMS = ('P', 'P-Platt')
NEW_CODE = ('gzcmd/arranjo_robustez.py', 'scripts/arranjo_robustez.py',
            'scripts/arranjo_corte_s2_suplementar.py', 'tests/test_arranjo_robustez.py',
            'tests/test_arranjo_corte_s3.py', 'gzcmd/M_robustez_arranjo.md')
REFERENCES = ('gzcmd/prereg_arranjo_corte.yaml', 'data/arranjo_corte_s1_metadata.json',
              'data/arranjo_corte_taxas_crossfit.csv', 'data/arranjo_corte_metricas.csv',
              'data/arranjo_corte_par2.csv', 'data/arranjo_corte_metadata.json',
              'data/arranjo_corte_s1_resumo.csv', 'data/pcal_ml_oof.csv',
              '.local/arranjo_corte/oof.npz')


def input_path(data_root, name):
    return resolve_input(ROOT, data_root, name)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def pushed_file(path):
    name = path.relative_to(ROOT).as_posix()
    commit = git('log', '-1', '--format=%H', '--', name)
    if not commit:
        raise ValueError(f'Not committed: {name}')
    recorded = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
    if recorded.replace(b'\r\n', b'\n') != path.read_bytes().replace(b'\r\n', b'\n'):
        raise ValueError(f'Working file differs from commit: {name}')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit,
                    'origin/pq/arranjo-corte'], cwd=ROOT, check=True)
    return commit


def verified(data_root, expected):
    spec = verify_prereg(PREREG, expected, root=ROOT, data_root=data_root)
    return spec, pushed_file(PREREG)


def preregister(data_root, design):
    if PREREG.exists():
        raise ValueError('Existing preregistration requires an authorized amendment')
    if git('branch', '--show-current') != 'pq/arranjo-corte':
        raise ValueError('Wrong execution branch')
    subprocess.run(['git', 'merge-base', '--is-ancestor', '893a2bb', 'HEAD'], cwd=ROOT, check=True)
    paths = set(NEW_CODE + REFERENCES + tuple(s1.CODE))
    paths.update(('gzcmd/gzcmd_v3_config.yaml', 'gzcmd/error_rates_by_band_v4.yaml',
                  'gzcmd/gzcmd_v4_llm_prompt.md', 'data/COMPARADORSEMIDENT.csv'))
    for source in s1.cohort_sources(data_root):
        paths.add(source['path'])
    for mode in s1.MODES:
        for folder in ('anthropic_sem_mo_v4', 'anthropic_sem_mo_v4_lote31',
                       'anthropic_sem_mo_v4_lote31_refrec',
                       'anthropic_sem_mo_v4_arranjo_corte', 'anthropic_sem_mo_v4_arranjo_corte_supl'):
            name = f'data/sprint3b/{folder}/exp2_llm_sync_{mode}.json'
            paths.add(name)
            identity = str(Path(name).with_name(Path(name).stem + '_journal') / 'identity.json').replace('\\', '/')
            if input_path(data_root, identity).exists():
                paths.add(identity)
        for arm in ('P', 'P-Platt', 'B1'):
            paths.add(f'.local/arranjo_corte/policy_{mode}_{arm}.csv')
    spec = {
        'schema_version': 1, 'confirmatory': False, 'principal_arm_unchanged': 'P',
        'parent_commit': 'd526c3f', 'base_commit': git('rev-parse', 'HEAD'),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'design_sha256': sha256(design), 'approved_design': design.read_text(encoding='utf-8'),
        'sha256': {name: sha256(input_path(data_root, name)) for name in sorted(paths)},
        'grid': [asdict(r) for r in grid()], 'versions': {'sklearn': '1.8.0', 'xgboost': '3.2.0'},
        'arms': list(ARMS), 'include_a3': False,
        'include_a3_fallback': 'Only if R0 P differs without A3; rerun R0 and all replicas with A3',
        'criteria': {'dominance': {'aceitos': 'maximize', 'fp_abaixo': 'minimize',
                                  'perdidos': 'minimize', 'pares_humano': 'minimize', 'pares_llm': 'minimize'},
                     'tolerance': 0, 'one_strict': True, 'denominator_each_block': 20,
                     'stable': 'k>=18 in both blocks and inverse<=1 in both',
                     'not_sustained': 'k<10 in both or inverse>=2 in either', 'otherwise': 'mista',
                     'invalid': 'non-dominant, retained in denominator',
                     'secondary_vigilancia': 'P minus P-Platt FP >0 in >=18/20 of each block',
                     'interval': 'Clopper-Pearson descriptive 95%', 'par2_posterior_losses': [9, 17]},
        'stop_rule': {'complete_grid': True, 'new_pairs_more_than': 2000,
                      'usd_more_than': 6.50, 'projection_hours_more_than': 8,
                      'estimated_usd_per_pair': pending_cost(1)},
        'reviewer_rates': {'sources': s1.cohort_sources(data_root), 'new_responses_reestimate': False,
                           'crossfit': 'exclude evaluated REFREC fold; M3, same fixed fallback'},
        'phase_order': ['committed_pushed_prereg', 'A', 'committed_pushed_queue', 'B', 'C'],
        'llm': {'provider': 'anthropic', 'model': 'claude-haiku-5-5', 'prompt': 'v4',
                'workers': 4, 'modes_sequential': True, 'out': OUT.relative_to(ROOT).as_posix()},
    }
    PREREG.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding='utf-8')
    verify_prereg(PREREG, sha256(PREREG), root=ROOT, data_root=data_root)
    print('PREREG_SHA256', sha256(PREREG), flush=True)


def known_responses(data_root, mode):
    # The historical source loader has worktree-local output constants. Resolve
    # ignored runs from the read-only data clone, never copying dossier fields.
    old_base, old_out = supplement.BASE_OUT, supplement.OUT
    supplement.BASE_OUT = input_path(data_root, 'data/sprint3b/anthropic_sem_mo_v4_arranjo_corte/exp2_llm_sync_' + mode + '.json').parent
    supplement.OUT = input_path(data_root, 'data/sprint3b/anthropic_sem_mo_v4_arranjo_corte_supl/exp2_llm_sync_' + mode + '.json').parent
    try:
        decisions, paths = supplement.responses(data_root, mode, include_supplement=True)
    finally:
        supplement.BASE_OUT, supplement.OUT = old_base, old_out
    validate_response_paths(data_root, mode, decisions, paths)
    return decisions, paths


def validate_response_paths(data_root, mode, decisions, paths):
    """Wrap the unchanged historical response loader with dossier validation."""
    import run_exp2_sem_model_outputs as runner
    from gzcmd.runner import run_v3
    historical, _ = run_v3(input_csv=data_root / 'data/COMPARADORSEMIDENT.csv',
                           config_path=s1.CONFIG, mode=mode, p_cal='fit_platt', policy_version='v3')
    selected = historical.loc[pair_ids(historical).isin(decisions)]
    views = {d.pair_id: runner.abl.DossierView(d, True)
             for d in runner.ant.exp2.build_dossiers(selected, only_llm_review=False)}
    for path in paths:
        validate_dossiers(path, supplement.read_run(path, mode), mode, views,
                          prompt_sha256=sha256(s1.PROMPT))


def inputs(data_root):
    raw = s1.load_raw(data_root / 'data/COMPARADORSEMIDENT.csv')
    below = below_cutoff(raw)
    x = numeric_features(raw)
    if len(x.columns) != 30:
        raise ValueError('Expected 30 features')
    par = pd.to_numeric(raw.PAR, errors='raise').to_numpy(dtype=int)
    ids = pair_ids(raw)
    groups = raw.REFREC.str.strip()
    if groups.isna().any() or groups.eq('').any():
        raise ValueError('Missing REFREC')
    return raw, below, x, par, ids, groups.to_numpy(dtype=str)


def policy_path(replica, mode, arm):
    return LOCAL / replica.replica / f'policy_{mode}_{arm}.csv'


def identity_for(spec, expected, replica, include_a3):
    execution = next((a['execution_identity'] for a in spec.get('amendments', [])
                      if a.get('execution_identity', {}).get('prereg_sha256') == expected), None)
    return {'replica': asdict(replica), 'prereg_sha256': expected,
            'sha256': execution['sha256'] if execution else spec['sha256'],
            'versions': spec['versions'], 'include_a3': include_a3}


def frozen_responses(spec, data_root, mode):
    frozen = {}
    for source in spec['reviewer_rates']['sources']:
        if source['mode'] == mode:
            run = json.loads(input_path(data_root, source['path']).read_text(encoding='utf-8'))
            frozen.update({r['pair_id']: r['decision'].strip().upper() for r in run['results']})
    return frozen


def child_replica(data_root, expected, index, include_a3):
    spec, _ = verified(data_root, expected)
    replica = grid()[index]
    directory = LOCAL / replica.replica
    directory.mkdir(parents=True, exist_ok=True)
    identity = identity_for(spec, expected, replica, include_a3)
    checkpoint = directory / 'oof'
    if checkpoint_matches(checkpoint, identity) and all(
            policy_path(replica, m, a).exists() for m in s1.MODES for a in ARMS):
        print('CHECKPOINT_VALID', replica.replica, flush=True)
        return
    raw, below, x, par, ids, groups = inputs(data_root)
    start = perf_counter()
    probabilities, folds, calibration = run_replica(
        x, np.isin(par, [1, 2]).astype(int), below, groups,
        par2=par == 2, replica=replica, include_a3=include_a3)
    training_seconds = perf_counter() - start
    temporary = checkpoint.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, pair_id=ids.to_numpy(dtype=str), fold=folds, **probabilities)
    replace_backoff(temporary, checkpoint.with_suffix('.npz'))
    cfg = load_config(s1.CONFIG)
    bands = BandAssigner.from_config(cfg).assign(x['nota final'])
    summaries, rate_rows = [], []
    for mode in s1.MODES:
        # Only the frozen three cohorts appear in this file. S2/supplement/
        # robustness responses are solely cache hits, never rate estimators.
        frozen = frozen_responses(spec, data_root, mode)
        records = pd.DataFrame({'pair_id': ids, 'fold': folds, 'band': bands,
                                'y': np.isin(par, [1, 2]).astype(int), 'par': par})
        records['decision'] = records.pair_id.map(frozen)
        rates = crossfit_rates(records.loc[records.decision.notna()],
                               cfg.llm_review.error_rates_by_band, list(range(5)))
        for fold, values in rates.items():
            for band, rate in values.items():
                rate_rows.append({'modo': mode, 'fold': fold, 'banda': band, **asdict(rate)})
        for arm in (*ARMS, *(('B1',) if index == 0 else ())):
            parts = []
            for fold in range(5):
                mask = (folds == fold) & (np.ones(len(raw), dtype=bool) if arm == 'B1' else below)
                frame = pd.DataFrame({'pair_id': ids[mask], 'nota final': x.loc[mask, 'nota final'],
                                      'band': bands[mask], 'p_cal': probabilities[arm][mask]})
                guard = apply_guardrails(frame, policy_version='v3.1')
                frame['guardrail'], frame['guardrail_reason'] = guard.guardrail, guard.reason
                engine = build_engine_from_config(cfg, mode=mode, policy_version='v3.1', rates_yaml=s1.RATES)
                engine.llm_error_by_band = rates[fold]
                parts.append(engine.triage(frame))
            out = pd.concat(parts).sort_index()
            if int(out.action.eq('LLM_REVIEW').sum()) > cfg.decision_policy.modes[mode].llm_max_calls_per_window:
                raise RuntimeError('STOP: global LLM budget exceeded')
            out[['pair_id', 'band', 'p_cal', 'action']].to_csv(policy_path(replica, mode, arm), index=False)
            if arm in ARMS:
                summary = summarize_actions(out, mode, arm)
                for key, value in asdict(replica).items():
                    summary[key] = value
                summaries.append(summary)
    rate_frame = pd.DataFrame(rate_rows)[s1.RATE_COLUMNS]
    rate_frame.to_csv(directory / 'taxas.csv', index=False)
    if replica.bloco == 'N':
        expected_rates = pd.read_csv(input_path(data_root, 'data/arranjo_corte_taxas_crossfit.csv'))
        pd.testing.assert_frame_equal(rate_frame, expected_rates, check_exact=False, atol=1e-14, rtol=0)
    pd.concat(summaries, ignore_index=True).to_csv(directory / 'acoes.csv', index=False)
    atomic_json(checkpoint.with_suffix('.json'), {'identity': identity, 'training_seconds': training_seconds,
                'wall_seconds': perf_counter() - start, 'calibration': calibration,
                'fallback_folds': [c['fold'] for c in calibration if c['principal'] == 'platt'],
                'policy_sha256': {f'{m}/{a}': sha256(policy_path(replica, m, a))
                                  for m in s1.MODES for a in ARMS}})
    print('REPLICA_COMPLETE', replica.replica, 'TRAINING_SECONDS', training_seconds, flush=True)


def memory_gate():
    import psutil
    free = psutil.virtual_memory().available / 2**30
    killed = 0
    if free < 6:
        processes = {p.pid: p.info for p in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline'])}
        for pid, info in processes.items():
            command = ' '.join(info.get('cmdline') or [])
            if ('joblib.externals.loky' in command and 'miniconda' in command.lower()
                    and info['ppid'] not in processes):
                try:
                    psutil.Process(pid).terminate()
                    killed += 1
                except psutil.NoSuchProcess:
                    continue
        free = psutil.virtual_memory().available / 2**30
    atomic_json(LOCAL / 'memory_last.json', {'free_gib': free, 'orphan_loky_terminated': killed})
    if free < 3:
        raise RuntimeError('STOP: free memory below 3 GiB')


def start_child(data_root, expected, index, include_a3):
    memory_gate()
    args = [str(Path(__file__).resolve()), '--data-root', str(data_root), '--prereg-sha256', expected,
            '_replica', '--index', str(index)] + (['--include-a3'] if include_a3 else [])
    argument_line = ' '.join('"' + a.replace('"', '\\"') + '"' for a in args)
    q = executor.ps_quote
    lines = ["$ErrorActionPreference = 'Stop'",
             '$p = Start-Process -FilePath ' + q(sys.executable) + ' -ArgumentList ' + q(argument_line)
             + ' -WorkingDirectory ' + q(ROOT) + ' -RedirectStandardOutput ' + q(LOCAL / f'{index:02d}.stdout.log')
             + ' -RedirectStandardError ' + q(LOCAL / f'{index:02d}.stderr.log') + ' -PassThru',
             'if (-not $p.HasExited) { Wait-Process -InputObject $p }', '$p.WaitForExit()', 'exit $p.ExitCode']
    path = LOCAL / 'child.ps1'
    path.write_text('\n'.join(lines), encoding='utf-8')
    result = subprocess.run(['pwsh', '-NoProfile', '-NonInteractive', '-File', str(path)], cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f'STOP: replica {index} failed, see local logs; exit={result.returncode}')


def r0_gate(data_root):
    replica = grid()[0]
    original = np.load(input_path(data_root, '.local/arranjo_corte/oof.npz'), allow_pickle=False)
    current = np.load(LOCAL / replica.replica / 'oof.npz', allow_pickle=False)
    np.testing.assert_array_equal(current['pair_id'], original['pair_id'])
    np.testing.assert_array_equal(current['fold'], original['fold'])
    for arm in ARMS:
        np.testing.assert_allclose(current[arm], original[arm], atol=1e-14, rtol=0)
    published = pd.read_csv(input_path(data_root, 'data/pcal_ml_oof.csv')).set_index('pair_id').loc[current['pair_id']]
    np.testing.assert_array_equal(current['fold'], published.fold.to_numpy())
    np.testing.assert_allclose(current['B1'], published.p_cal_ml.to_numpy(), atol=1e-14, rtol=0)
    metadata = json.loads(input_path(data_root, 'data/arranjo_corte_metadata.json').read_text(encoding='utf-8'))
    for mode in s1.MODES:
        for arm in (*ARMS, 'B1'):
            path = policy_path(replica, mode, arm)
            if sha256(path) != metadata['policy_sha256'][path.name]:
                raise ValueError(f'R0 policy SHA256 mismatch: {mode}/{arm}')
    # S3 gate is deliberately isolated: only equality with already-published
    # R0 outcomes is checked. No non-R0 outcome is evaluated during phase A.
    raw, below, x, par, ids, _ = inputs(data_root)
    base = pd.DataFrame({'pair_id': ids, 'par': par, 'y': np.isin(par, [1, 2]).astype(int),
                         'nota': x['nota final'], 'abaixo': below}).set_index('pair_id')
    reference = pd.read_csv(input_path(data_root, 'data/arranjo_corte_metricas.csv'))
    reference_par2 = pd.read_csv(input_path(data_root, 'data/arranjo_corte_par2.csv'))
    for mode in s1.MODES:
        known, paths = known_responses(data_root, mode)
        for arm in (*ARMS, 'B1'):
            frame = base.copy()
            frame['action'] = np.where(frame.abaixo, 'NONMATCH', 'MATCH')
            policy = pd.read_csv(policy_path(replica, mode, arm)).set_index('pair_id')
            frame.loc[policy.index, 'action'] = policy.action
            frame = s3.replay(frame, known)
            needed = set(frame.index[frame.action.eq('LLM_REVIEW')])
            costs, usage, _ = s3.selected_usage(paths, needed)
            for scope in ('TODAS', '[5,9)'):
                selected = frame if scope == 'TODAS' else frame.loc[frame.nota.ge(5) & frame.nota.lt(9)]
                observed = s3.metrics(selected, mode, arm, scope, usage, costs)
                expected = reference.loc[reference.modo.eq(mode) & reference.braco.eq(arm) & reference.faixa.eq(scope)].iloc[0]
                for key, value in observed.items():
                    if isinstance(value, (float, int)):
                        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=0)
            primary = s3.par2_metrics(frame, mode, arm)
            expected = reference_par2.loc[reference_par2.modo.eq(mode) & reference_par2.braco.eq(arm)].iloc[0]
            for key in ('aceitos', 'humano', 'perdidos'):
                if primary[key] != expected[key]:
                    raise ValueError('R0 PAR=2 gate mismatch')
    atomic_json(LOCAL / 'r0_gate.json', {'passed': True, 'S1_S3_exact': True})
    print('R0_GATES_PASS probabilities folds rates policy_sha256 S3 PAR2', flush=True)


def phase_a(data_root, expected):
    spec, commit = verified(data_root, expected)
    LOCAL.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    # Audit complete identities once, before treating any pair as a cache hit.
    for mode in s1.MODES:
        known, _ = known_responses(data_root, mode)
        if len(known) != {'vigilancia': 1850, 'confirmacao': 476}[mode]:
            raise ValueError('Known pool differs from approved design')
        atomic_json(LOCAL / f'known_{mode}.json', known)
        frozen = frozen_responses(spec, data_root, mode)
        atomic_json(LOCAL / f'frozen_{mode}.json', frozen)
    include_a3 = False
    start_child(data_root, expected, 0, include_a3)
    # The only authorized A3 fallback concerns P probabilities, not any other gate.
    original = np.load(input_path(data_root, '.local/arranjo_corte/oof.npz'), allow_pickle=False)
    observed = np.load(LOCAL / 'N00/oof.npz', allow_pickle=False)
    if not np.allclose(original['P'], observed['P'], atol=1e-14, rtol=0):
        include_a3 = True
        original.close()
        observed.close()
        start_child(data_root, expected, 0, include_a3)
    else:
        original.close()
        observed.close()
    r0_gate(data_root)
    accumulated = {m: set() for m in s1.MODES}
    for i, replica in enumerate(grid()):
        if i:
            start_child(data_root, expected, i, include_a3)
        for mode in s1.MODES:
            for arm in ARMS:
                policy = pd.read_csv(policy_path(replica, mode, arm))
                accumulated[mode].update(policy.loc[policy.action.eq('LLM_REVIEW'), 'pair_id'])
        n = sum(len(accumulated[m] - json.loads((LOCAL / f'known_{m}.json').read_text(encoding='utf-8')).keys()) for m in s1.MODES)
        safety_stop(n)
        if i >= 1:
            timing = [json.loads((LOCAL / r.replica / 'oof.json').read_text(encoding='utf-8'))['training_seconds'] for r in grid()[:i + 1]]
            # R0/n_jobs=2 measured separately; use each thread-count mean for
            # the fixed full grid, including elapsed setup and triage overhead.
            remaining = grid()[i + 1:]
            by_jobs = {j: np.mean([t for r, t in zip(grid(), timing) if r.n_jobs == j]) for j in (8, 2)}
            projection = perf_counter() - started + sum(by_jobs[r.n_jobs] for r in remaining)
            safety_stop(n, projected_seconds=projection)
            print('PROJECTION_SECONDS', projection, flush=True)
    summaries = [pd.read_csv(LOCAL / r.replica / 'acoes.csv') for r in grid()]
    pd.concat(summaries, ignore_index=True).to_csv(ROOT / 'data/robustez_arranjo_acoes.csv', index=False)
    counts = {}
    for mode in s1.MODES:
        known = json.loads((LOCAL / f'known_{mode}.json').read_text(encoding='utf-8'))
        pending = sorted(accumulated[mode] - known.keys())
        pd.DataFrame({'pair_id': pending}).to_csv(LOCAL / f'pendentes_{mode}.csv', index=False)
        counts[mode] = len(pending)
    report = {'stage': 'A_complete_no_API_no_new_outcomes', 'prereg_commit': commit,
              'prereg_sha256': expected, 'replicas_complete': 40, 'include_a3': include_a3,
              'counts': counts, 'pending_total': sum(counts.values()), 'api_calls': 0,
              'estimated_usd': pending_cost(sum(counts.values())), 'wall_seconds': perf_counter() - started,
              'replica_seconds': {r.replica: json.loads((LOCAL / r.replica / 'oof.json').read_text(encoding='utf-8'))['training_seconds'] for r in grid()}}
    atomic_json(LOCAL / 'fase_a.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def prepare_llm(data_root, expected):
    _, commit = verified(data_root, expected)
    phase = json.loads((LOCAL / 'fase_a.json').read_text(encoding='utf-8'))
    if phase['replicas_complete'] != 40 or phase['prereg_sha256'] != expected:
        raise ValueError('Complete phase A required')
    counts, hashes = {}, {}
    for mode in s1.MODES:
        source = LOCAL / f'pendentes_{mode}.csv'
        queue = pd.read_csv(source)
        if list(queue) != ['pair_id'] or not queue.pair_id.is_unique:
            raise ValueError('Invalid queue schema')
        target = ROOT / f'data/robustez_arranjo_fila_{mode}.csv'
        target.write_bytes(source.read_bytes())
        counts[mode], hashes[mode] = len(queue), sha256(target)
    n = sum(counts.values())
    safety_stop(n)
    note = {'prereg_commit': commit, 'prereg_sha256': expected, 'counts': counts,
            'pending_total': n, 'estimated_usd': pending_cost(n), 'queue_sha256': hashes,
            'phase_a_sha256': sha256(LOCAL / 'fase_a.json'),
            'frozen_rates_sha256': sha256(ROOT / 'data/arranjo_corte_taxas_crossfit.csv'),
            'new_responses_reestimate_rates': False}
    atomic_json(NOTE, note)
    print(json.dumps(note, indent=2), flush=True)


def queue_gate(data_root, expected):
    spec, _ = verified(data_root, expected)
    commit = pushed_file(NOTE)
    note = json.loads(NOTE.read_text(encoding='utf-8'))
    execution_hashes = {expected} | {a['execution_identity']['prereg_sha256']
                                   for a in spec.get('amendments', []) if 'execution_identity' in a}
    if note['prereg_sha256'] not in execution_hashes or sha256(LOCAL / 'fase_a.json') != note['phase_a_sha256']:
        raise ValueError('Phase A identity changed')
    for mode in s1.MODES:
        queue = ROOT / f'data/robustez_arranjo_fila_{mode}.csv'
        pushed_file(queue)
        if sha256(queue) != note['queue_sha256'][mode] or sha256(LOCAL / f'pendentes_{mode}.csv') != sha256(queue):
            raise ValueError('Committed queue identity mismatch')
    if sha256(ROOT / 'data/arranjo_corte_taxas_crossfit.csv') != note['frozen_rates_sha256']:
        raise ValueError('Frozen reviewer rates changed')
    safety_stop(note['pending_total'])
    return note, commit


def phase_b(data_root, expected, env_file):
    note, commit = queue_gate(data_root, expected)
    from dotenv import load_dotenv
    load_dotenv(env_file)
    executor.OUT, executor.LOCAL = OUT, LOCAL
    OUT.mkdir(parents=True, exist_ok=True)
    started, runs = perf_counter(), []
    for mode in s1.MODES:
        if note['counts'][mode]:
            runs.append(executor.run_mode(mode))
            safety_stop(note['pending_total'], cost=sum(r['cost_estimate_usd'] for r in runs))
    audit = executor.audit_secrets()
    queue_gate(data_root, expected)
    report = {'stage': 'B_complete', 'queue_commit': commit, 'runs': runs, 'secret_audit': audit,
              'total_pairs': sum(r['n_reviewed'] for r in runs), 'wall_seconds': perf_counter() - started,
              'total_cost_estimate_usd': sum(r['cost_estimate_usd'] for r in runs)}
    atomic_json(OUT / 'execucao.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def all_responses(data_root, mode):
    known, paths = known_responses(data_root, mode)
    path = OUT / f'exp2_llm_sync_{mode}.json'
    if path.exists():
        run = supplement.read_run(path, mode)
        wanted = {r['pair_id'] for r in run['results']}
        queue = set(pd.read_csv(LOCAL / f'pendentes_{mode}.csv').pair_id)
        if wanted != queue or wanted & known.keys():
            raise ValueError('Robustness run coverage differs from queue')
        import run_exp2_sem_model_outputs as runner
        from gzcmd.runner import run_v3
        historical, _ = run_v3(input_csv=data_root / 'data/COMPARADORSEMIDENT.csv', config_path=s1.CONFIG,
                               mode=mode, p_cal='fit_platt', policy_version='v3')
        selected = historical.loc[pair_ids(historical).isin(wanted)]
        views = {d.pair_id: runner.abl.DossierView(d, True)
                 for d in runner.ant.exp2.build_dossiers(selected, only_llm_review=False)}
        validate_dossiers(path, run, mode, views, prompt_sha256=sha256(s1.PROMPT))
        known.update({r['pair_id']: r['decision'].strip().upper() for r in run['results']})
        paths.append(path)
    return known, paths


def phase_c(data_root, expected):
    spec, prereg_commit = verified(data_root, expected)
    note, queue_commit = queue_gate(data_root, expected)
    execution = json.loads((OUT / 'execucao.json').read_text(encoding='utf-8'))
    if execution['stage'] != 'B_complete' or execution['secret_audit']['secret_matches'] != 0:
        raise ValueError('Clean complete B required')
    started = perf_counter()
    raw, below, x, par, ids, _ = inputs(data_root)
    base = pd.DataFrame({'pair_id': ids, 'par': par, 'y': np.isin(par, [1, 2]).astype(int),
                         'nota': x['nota final'], 'abaixo': below}).set_index('pair_id')
    rows, par2, stability = [], [], []
    for mode in s1.MODES:
        known, paths = all_responses(data_root, mode)
        needed = set()
        for replica in grid():
            for arm in ARMS:
                policy = pd.read_csv(policy_path(replica, mode, arm))
                needed.update(policy.loc[policy.action.eq('LLM_REVIEW'), 'pair_id'])
        costs, usage, _ = s3.selected_usage(paths, needed)
        for replica in grid():
            metadata = json.loads((LOCAL / replica.replica / 'oof.json').read_text(encoding='utf-8'))
            if metadata['identity'] != identity_for(spec, note['prereg_sha256'], replica, metadata['identity']['include_a3']):
                raise ValueError('Replica checkpoint identity mismatch')
            for arm in ARMS:
                path = policy_path(replica, mode, arm)
                if sha256(path) != metadata['policy_sha256'][f'{mode}/{arm}']:
                    raise ValueError('Replica policy identity mismatch')
                policy = pd.read_csv(path)
                reference = pd.read_csv(policy_path(grid()[0], mode, arm))
                unlabeled = action_stability(policy, reference, base.index[base.abaixo])
                with np.load(LOCAL / replica.replica / 'oof.npz', allow_pickle=False) as scores:
                    with np.load(LOCAL / 'N00/oof.npz', allow_pickle=False) as baseline:
                        delta = np.abs(scores[arm] - baseline[arm])
                        unlabeled.update(p_alterados=int((delta > 1e-14).sum()), delta_p_max=float(delta.max()))
                stability.append({**asdict(replica), 'modo': mode, 'braco': arm, **unlabeled,
                                  'fallback_folds': len(metadata['fallback_folds'])})
                frame = base.copy()
                frame['action'] = np.where(frame.abaixo, 'NONMATCH', 'MATCH')
                policy = policy.set_index('pair_id')
                frame.loc[policy.index, 'action'] = policy.action
                frame = s3.replay(frame, known)
                par2.append({**asdict(replica), **s3.par2_metrics(frame, mode, arm)})
                for scope in ('TODAS', '[5,9)'):
                    selected = frame if scope == 'TODAS' else frame.loc[frame.nota.ge(5) & frame.nota.lt(9)]
                    result = s3.metrics(selected, mode, arm, scope, usage, costs)
                    llm_match = selected.action.eq('LLM_REVIEW') & selected.final.eq('MATCH') & selected.abaixo
                    denominator = int(llm_match.sum())
                    result.update({'vpp_llm_abaixo': float(selected.loc[llm_match, 'y'].mean()) if denominator else np.nan,
                                   'llm_match_abaixo': denominator,
                                   'llm_match_verdadeiros_abaixo': int(selected.loc[llm_match, 'y'].sum()),
                                   **asdict(replica), 'valida': True})
                    rows.append(result)
    table, primary, unlabeled = pd.DataFrame(rows), pd.DataFrame(par2), pd.DataFrame(stability)
    differences = []
    numeric = list(table.select_dtypes(include='number').columns.difference(['fold_seed', 'xgb_seed', 'n_jobs']))
    for (rid, block, mode, scope), part in table.groupby(['replica', 'bloco', 'modo', 'faixa'], sort=False):
        indexed = part.set_index('braco')
        p, q = indexed.loc['P'], indexed.loc['P-Platt']
        differences.append({'replica': rid, 'bloco': block, 'modo': mode, 'faixa': scope,
                            **{k: p[k] - q[k] for k in numeric},
                            'platt_domina': dominates(q, p), 'inversa': dominates(p, q), 'valida': True})
    diff = pd.DataFrame(differences)
    summary, conclusions = [], {}
    for mode in s1.MODES:
        ks, inverse = [], []
        for block in ('N', 'P'):
            d = diff.loc[diff.modo.eq(mode) & diff.bloco.eq(block) & diff.faixa.eq('TODAS')]
            k, inv = int(d.platt_domina.sum()), int(d.inversa.sum())
            ks.append(k)
            inverse.append(inv)
            low, high = clopper_pearson(k)
            summary.append({'tipo': 'dominancia', 'modo': mode, 'bloco': block, 'k': k, 'n': 20,
                            'proporcao': k / 20, 'ic95_inf': low, 'ic95_sup': high, 'inversa': inv,
                            'fp_positivo': int(d.fp.gt(0).sum())})
        conclusions[mode] = {'classificacao': classify(ks, inverse), 'dominancia_k20': ks,
                             'inversa_k20': inverse}
    for (block, mode, arm), part in primary.groupby(['bloco', 'modo', 'braco']):
        other = primary.loc[primary.bloco.eq(block) & primary.modo.eq(mode)].pivot(index='replica', columns='braco', values='aceitos')
        summary.append({'tipo': 'PAR2', 'bloco': block, 'modo': mode, 'braco': arm,
                        'platt_aceita_pelo_menos_P': float(other['P-Platt'].ge(other.P).sum() / 20),
                        'perde_mais_aposteriori': float(part.perdidos.gt(9 if mode == 'vigilancia' else 17).sum() / 20)})
    # Complete descriptive distributions, separately by block, mode, arm/scope.
    for kind, frame, groups in (
        ('desfecho', table, ['bloco', 'modo', 'braco', 'faixa']),
        ('diferenca_P_menos_Platt', diff, ['bloco', 'modo', 'faixa']),
        ('PAR2_distribuicao', primary, ['bloco', 'modo', 'braco']),
        ('estabilidade', unlabeled, ['bloco', 'modo', 'braco'])):
        columns = frame.select_dtypes(include='number').columns.difference(['fold_seed', 'xgb_seed', 'n_jobs'])
        for keys, part in frame.groupby(groups):
            for column in columns:
                values = part[column].dropna()
                summary.append({'tipo': kind, **dict(zip(groups, keys)), 'metrica': column,
                                'mediana': values.median(), 'minimo': values.min(), 'maximo': values.max(),
                                'iqr': values.quantile(.75) - values.quantile(.25),
                                'positivo': values.gt(0).sum() / 20, 'zero': values.eq(0).sum() / 20,
                                'negativo': values.lt(0).sum() / 20})
    outputs = {'replicas': table, 'diferencas': diff, 'resumo': pd.DataFrame(summary),
               'par2': primary, 'estabilidade': unlabeled}
    for name, frame in outputs.items():
        frame.to_csv(ROOT / f'data/robustez_arranjo_{name}.csv', index=False)
    render_table(conclusions, ROOT / 'text/latex/tables/tab_cap8_robustez_arranjo.tex')
    report = {'confirmatory': False, 'principal_arm_unchanged': 'P', 'prereg_commit': prereg_commit,
              'prereg_sha256': expected, 'queue_commit': queue_commit, 'conclusoes': conclusions,
              'phase_a': json.loads((LOCAL / 'fase_a.json').read_text(encoding='utf-8')),
              'phase_b': execution, 'phase_c_seconds': perf_counter() - started,
              'new_responses_reestimate_rates': False, 'registered_sha256': spec['sha256'],
              'outputs_sha256': {name: sha256(ROOT / f'data/robustez_arranjo_{name}.csv') for name in outputs}}
    atomic_json(ROOT / 'data/robustez_arranjo_metadata.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def render_table(conclusions, path):
    lines = [r'% Gerada por scripts/arranjo_robustez.py, fase C.',
             r'\begin{tabular}{lrrrl}', r'\toprule',
             r'Modo & N ($k/20$) & P ($k/20$) & Inversa (N; P) & Classificação \\', r'\midrule']
    for mode, row in conclusions.items():
        k, inv = row['dominancia_k20'], row['inversa_k20']
        lines.append(f"{'Vigilância' if mode == 'vigilancia' else 'Confirmação'} & {k[0]}/20 & {k[1]}/20 & {inv[0]}/20; {inv[1]}/20 & {row['classificacao']}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\par\noindent\footnotesize Análise não confirmatória. P permanece o braço principal. '
              r'Dominância fraca de P-Platt sobre P, com ao menos uma desigualdade estrita, '
              r'nos vínculos aceitos, falsos positivos abaixo do corte, vínculos perdidos e '
              r'potenciais pares ao revisor humano e ao modelo de linguagem. '
              r'Os modos são avaliados separadamente. Fonte: '
              r'\texttt{data/robustez\_arranjo\_resumo.csv}.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--prereg-sha256')
    parser.add_argument('--preregister', action='store_true')
    parser.add_argument('--design', type=Path)
    parser.add_argument('--env-file', type=Path)
    sub = parser.add_subparsers(dest='command')
    for command in ('fase-a', 'preparar-llm', 'fase-b', 'fase-c'):
        sub.add_parser(command)
    replica = sub.add_parser('_replica')
    replica.add_argument('--index', type=int, required=True, choices=range(40))
    replica.add_argument('--include-a3', action='store_true')
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    if args.preregister:
        if not args.design:
            parser.error('--design required')
        preregister(data_root, args.design)
    elif args.command == '_replica':
        child_replica(data_root, args.prereg_sha256, args.index, args.include_a3)
    elif args.command == 'fase-a':
        phase_a(data_root, args.prereg_sha256)
    elif args.command == 'preparar-llm':
        prepare_llm(data_root, args.prereg_sha256)
    elif args.command == 'fase-b':
        if not args.env_file:
            parser.error('--env-file required')
        phase_b(data_root, args.prereg_sha256, args.env_file)
    elif args.command == 'fase-c':
        phase_c(data_root, args.prereg_sha256)
    else:
        parser.error('Choose --preregister or phase')


if __name__ == '__main__':
    main()
