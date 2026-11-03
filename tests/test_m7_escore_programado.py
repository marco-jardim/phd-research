"""Minimal contract tests for the new score, plus restricted rule execution."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pure_deterministic_bounded():
    score = load_module(ROOT / "gzcmd/escore_programado.py").escore_programado
    for features in [{}, {"nota final": 0.0}, {"nota final": 10.0},
                     {"NOME qtd frag raros": 2, "DTNASC dt iguais": 1},
                     {"NOME prim frag igual": 1, "CODMUNRES local igual": 0}]:
        before = copy.deepcopy(features)
        values = [score(features) for _ in range(5)]
        assert features == before
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in values)
        assert len(set(values)) == 1


def test_random_feature_domain():
    import json
    score = load_module(ROOT / "gzcmd/escore_programado.py").escore_programado
    dictionary = json.loads((ROOT / "data/m7/e2_generation.json").read_text(encoding="utf-8"))["prompt"]["feature_dictionary"]
    rng = np.random.default_rng(42)
    for _ in range(200):
        features = {f["name"]: float(rng.uniform(*f["observed_range"])) for f in dictionary}
        assert 0 <= score(features) <= 1


def test_rule_interpreter_rejects_code():
    common = load_module(ROOT / "scripts/m7_common.py")
    assert common.rule_value("f['x'] >= 1 and not f['x'] < 0", {"x": 1})
    assert common.rule_value("f['x'] >= -1", {"x": 0})
    function = common.validated_score("def escore_programado(features):\n    v = features.get('x', 0)\n    return 0.0 if v is None else min(1.0, max(0.0, float(v)))", {"x"})
    assert function({"x": None}) == 0
    assert function({"x": .5}) == .5
    with pytest.raises(ValueError):
        common.rule_value("__import__('os').getcwd()", {})
    with pytest.raises(ValueError):
        common.validated_score("def escore_programado(features):\n    return open('secret')", {"x"})


@pytest.mark.parametrize("code", [
    "def escore_programado(features):\n    features['x'] = 1\n    return 0.5",
    "def escore_programado(features):\n    features.clear()\n    return 0.5",
    "def escore_programado(features):\n    import os\n    return 0.5",
    "def escore_programado(features):\n    return features.get('PAR', 0)",
    "def escore_programado(features):\n    return features.get('pair_id', 0)",
])
def test_generated_program_boundary(code):
    common = load_module(ROOT / "scripts/m7_common.py")
    with pytest.raises(ValueError):
        common.validated_score(code, {"x"})
