"""Análise não confirmatória registrada, sem alterar o braço principal P."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta
import yaml

from .arranjo_corte import nested_cutoff_probabilities, pending_cost, sha256
from .ml_oof import _classifier


@dataclass(frozen=True)
class Replica:
    replica: str
    bloco: str
    fold_seed: int
    xgb_seed: int
    n_jobs: int


def grid():
    return ([Replica(f'N{i:02d}', 'N', 42, seed, jobs)
             for i, (seed, jobs) in enumerate(
                 (s, j) for s in (42, *range(101, 110)) for j in (8, 2))]
            + [Replica(f'P{i:02d}', 'P', seed, 42, 8)
               for i, seed in enumerate(range(1000, 1200, 10))])


def replica_factory(xgb_seed, n_jobs):
    return lambda y, _seed: _classifier(y, xgb_seed).set_params(n_jobs=n_jobs)


def run_replica(x, y, below, groups, *, par2, replica, include_a3=False, audit=None):
    return nested_cutoff_probabilities(
        x, y, below, groups, par2=par2, seed=replica.fold_seed,
        model_factory=replica_factory(replica.xgb_seed, replica.n_jobs),
        include_a3=include_a3, audit=audit)


def dominates(a, b):
    """a domina b, com tolerância zero e ao menos uma desigualdade estrita."""
    keys = ('aceitos', 'fp_abaixo', 'perdidos', 'pares_humano', 'pares_llm')
    delta = np.array([(a[k] - b[k]) * (1 if k == 'aceitos' else -1) for k in keys])
    return bool(np.all(delta >= 0) and np.any(delta > 0))


def clopper_pearson(k, n=20, alpha=.05):
    if not 0 <= k <= n or n <= 0:
        raise ValueError('Invalid binomial counts')
    return (0. if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
            1. if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)))


def classify(counts, inverse):
    if len(counts) != 2 or len(inverse) != 2:
        raise ValueError('Both fixed blocks required')
    if all(k >= 18 for k in counts) and all(k <= 1 for k in inverse):
        return 'estável'
    if all(k < 10 for k in counts) or any(k >= 2 for k in inverse):
        return 'não sustentada'
    return 'mista'


def safety_stop(n_new, *, cost=None, projected_seconds=0):
    cost = pending_cost(n_new) if cost is None else cost
    if n_new > 2000 or cost > 6.50 or projected_seconds > 8 * 3600:
        raise RuntimeError(f'STOP: pares={n_new}, custo_USD={cost:.8f}, projeção_s={projected_seconds:.3f}')


def resolve_input(root, data_root, name):
    path = Path(root) / name
    return path if path.exists() else Path(data_root) / name


def verify_prereg(prereg, expected_hash, *, root, data_root):
    """Check every pinned byte, including new code/tests, before doing work."""
    if not expected_hash or sha256(prereg) != expected_hash:
        raise ValueError('Preregistration SHA256 mismatch')
    spec = yaml.safe_load(Path(prereg).read_text(encoding='utf-8'))
    if spec.get('confirmatory') is not False or spec.get('principal_arm_unchanged') != 'P':
        raise ValueError('Invalid registered scientific scope')
    import sklearn
    import xgboost
    if sklearn.__version__ != '1.8.0' or xgboost.__version__ != '3.2.0':
        raise ValueError('Registered sklearn/xgboost versions differ')
    for name, digest in spec['sha256'].items():
        if sha256(resolve_input(root, data_root, name)) != digest:
            raise ValueError(f'Preregistered SHA256 mismatch: {name}')
    return spec


def replace_backoff(source, destination):
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.1 * 2 ** attempt)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    replace_backoff(temporary, path)


def checkpoint_matches(path, identity):
    path = Path(path)
    return (path.with_suffix('.npz').exists() and path.with_suffix('.json').exists()
            and json.loads(path.with_suffix('.json').read_text(encoding='utf-8')).get('identity') == identity)


def action_stability(current, reference, below_ids):
    current = current.set_index('pair_id').loc[below_ids]
    reference = reference.set_index('pair_id').loc[below_ids]
    a = set(current.index[current.action.eq('LLM_REVIEW')])
    b = set(reference.index[reference.action.eq('LLM_REVIEW')])
    delta = np.abs(current.p_cal.to_numpy() - reference.p_cal.to_numpy())
    return {'p_alterados': int((delta > 1e-14).sum()), 'delta_p_max': float(delta.max()),
            'proporcao_acao_alterada': float(current.action.ne(reference.action).mean()),
            'jaccard_llm': len(a & b) / len(a | b) if a | b else 1.}
