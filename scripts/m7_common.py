"""Shared, privacy-preserving infrastructure for the M7 experiments.

Only feature dictionaries, model parameters and aggregate statistics may cross
the API boundary. Row-level inputs remain in memory, never in prompts.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/m7"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call_json(name, payload, instruction):
    """One paid request, no retries; checkpoint before sending and after receiving."""
    from run_exp2_anthropic import AnthropicReviewer, DEFAULT_MODEL
    path = OUT / f"{name}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("response") is not None:
            if existing["prompt"] != payload or existing["system"] != instruction:
                raise ValueError("Checkpoint differs from requested prompt")
            return existing["response"]
        raise RuntimeError("Unfinished API call: inspect ledger before explicitly retrying")
    ledger = OUT / "calls.json"
    calls = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else []
    if len(calls) >= 60:
        raise RuntimeError("M7 API budget exhausted")
    calls.append({"name": name, "model": DEFAULT_MODEL, "status": "started"})
    save("calls.json", calls)
    save(f"{name}.json", {"system": instruction, "prompt": payload, "response": None})
    reviewer = AnthropicReviewer(api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url="https://api.anthropic.com", model=DEFAULT_MODEL,
        fallback_model=DEFAULT_MODEL, temperature=0.0, max_tokens=12000,
        timeout_s=240.0, max_retries=1, retry_delay_s=0.0)
    response, latency, input_tokens, output_tokens = reviewer._call_api([
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    calls[-1].update(status="completed", input_tokens=input_tokens,
                     output_tokens=output_tokens, latency_s=latency)
    save("calls.json", calls)
    save(f"{name}.json", {"system": instruction, "prompt": payload, "response": response,
                          "input_tokens": input_tokens, "output_tokens": output_tokens})
    return response


def rule_value(expression, features):
    """Interpret scalar boolean expressions without eval or arbitrary code."""
    tree = ast.parse(expression, mode="eval")
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
            return node.value
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "f":
            if not isinstance(node.slice, ast.Constant) or node.slice.value not in features:
                raise ValueError("Unknown feature")
            return features[node.slice.value]
        if isinstance(node, ast.BoolOp):
            values = [bool(visit(v)) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not visit(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            if isinstance(node.op, ast.USub):
                return -node.operand.value
            if isinstance(node.op, ast.UAdd):
                return node.operand.value
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = visit(right_node)
                if isinstance(op, ast.GtE): result = left >= right
                elif isinstance(op, ast.Gt): result = left > right
                elif isinstance(op, ast.LtE): result = left <= right
                elif isinstance(op, ast.Lt): result = left < right
                elif isinstance(op, ast.Eq): result = left == right
                elif isinstance(op, ast.NotEq): result = left != right
                else: raise ValueError("Forbidden comparison")
                if not result: return False
                left = right
            return True
        raise ValueError(f"Forbidden rule node: {type(node).__name__}")
    return bool(visit(tree))


def load_data():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from gzcmd.config import load_config
    from gzcmd.ml_oof import numeric_features, pair_ids, SEED, N_SPLITS
    source = ROOT / "data/COMPARADORSEMIDENT.csv"
    raw = pd.read_csv(source, sep=";", dtype=str, low_memory=False)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    split_by = load_config(ROOT / "gzcmd/gzcmd_v3_config.yaml").evaluation.split_by
    x = numeric_features(raw)
    y = pd.to_numeric(raw.PAR, errors="raise").isin([1, 2]).astype(int).to_numpy()
    ids = pair_ids(raw)
    oof_path = ROOT / "data/pcal_ml_oof.csv"
    meta = json.loads(oof_path.with_suffix(".json").read_text(encoding="utf-8"))
    import sklearn
    import xgboost
    from gzcmd.ml_oof import METHOD
    expected = {"method": METHOD, "seed": SEED, "folds": N_SPLITS, "split_by": split_by,
                "source_sha256": sha(source), "code_sha256": sha(ROOT / "gzcmd/ml_oof.py"),
                "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}
    if meta["fingerprint"] != expected or meta["csv_sha256"] != sha(oof_path):
        raise ValueError("M1 cache fingerprint mismatch; refusing to change existing cache")
    oof = pd.read_csv(oof_path)
    assert oof.pair_id.is_unique and set(ids) == set(oof.pair_id)
    oof = oof.set_index("pair_id").loc[ids].reset_index()
    nota = x["nota final" if "nota final" in x else "nota_final"].to_numpy()
    np.testing.assert_allclose(oof.nota, nota)
    groups = raw[split_by.upper()].str.strip().to_numpy() if split_by != "row" else None
    splitter = StratifiedGroupKFold if groups is not None else StratifiedKFold
    outer = splitter(N_SPLITS, shuffle=True, random_state=SEED)
    splits = outer.split(x, y, groups) if groups is not None else outer.split(x, y)
    for fold, (train, test) in enumerate(splits):
        if groups is not None:
            assert set(groups[train]).isdisjoint(groups[test])
        assert (oof.fold.iloc[test] == fold).all()
    assert meta["features"] == list(x.columns)
    return raw, x, y, nota, ids, oof


def feature_dictionary(x):
    """Header-derived semantics, not a claim to reverse engineer comparator code."""
    meanings = {"prim frag igual": "igualdade do primeiro fragmento",
        "ult frag igual": "igualdade do último fragmento", "qtd frag iguais": "contagem de fragmentos iguais",
        "qtd frag raros": "contagem de fragmentos raros coincidentes",
        "qtd frag comuns": "contagem de fragmentos comuns coincidentes",
        "qtd frag muito parec": "contagem de fragmentos muito parecidos",
        "qtd frag abrev": "contagem de abreviações compatíveis",
        "dt iguais": "igualdade de nascimento", "dt ap 1digi": "nascimento difere em aproximadamente um dígito",
        "dt inv dia": "indicador de inversão envolvendo dia", "dt inv mes": "indicador de inversão envolvendo mês",
        "dt inv ano": "indicador de inversão envolvendo ano", "uf igual": "UF igual",
        "uf prox": "proximidade de UF", "local igual": "município igual", "local prox": "proximidade de município",
        "via igual": "via de endereço igual", "via prox": "proximidade da via",
        "numero igual": "número do endereço igual", "compl prox": "proximidade do complemento",
        "texto prox": "proximidade textual do endereço", "tokens jacc": "Jaccard de tokens de endereço"}
    result = []
    for name in x:
        prefix, _, suffix = name.partition(" ")
        result.append({"name": name, "meaning": ("nota composta do comparador" if name.startswith("nota")
                      else prefix + ": " + meanings.get(suffix, suffix)),
                      "observed_range": [float(x[name].min()), float(x[name].max())],
                      "missing": "ausente convertido a zero pela allowlist M1; zero não distingue ausência de discordância"})
    return result


def metric(y, score, probability=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    return {"n": len(y), "positive": int(y.sum()),
            "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            "ap": float(average_precision_score(y, score)) if y.sum() else None,
            "brier": float(brier_score_loss(y, probability)) if probability is not None else None,
            "median_positive_raw": float(np.median(score[y == 1])) if y.sum() else None,
            "median_positive_calibrated": float(np.median(probability[y == 1])) if probability is not None and y.sum() else None}


def fit_anchor(score, nota, y):
    from validar_ancoras_ml import fit_masked_anchor_platt, rates
    masks = [nota >= 9, nota < 5]
    targets, audit = rates(y, masks)
    model, diagnostic = fit_masked_anchor_platt(score, masks, targets)
    return model, {"rates": audit, "fit": diagnostic}


def anchor_predict(model, score):
    from validar_ancoras_ml import predict
    return predict(model, score)


def validated_score(code, allowed_features):
    """Narrow, non-I/O Python subset; reject rather than execute arbitrary LLM code."""
    tree = ast.parse(code)
    allowed_nodes = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign,
        ast.Return, ast.If, ast.IfExp, ast.Expr, ast.Constant, ast.Name, ast.Load, ast.Store,
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Attribute, ast.Subscript,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.And, ast.Or,
        ast.Not, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.Dict, ast.keyword)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Exactly one function required")
    fn = tree.body[0]
    if fn.name != "escore_programado" or [a.arg for a in fn.args.args] != ["features"] or fn.decorator_list:
        raise ValueError("Unexpected function signature")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Forbidden score syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder name")
        if isinstance(node, ast.Assign):
            if any(not isinstance(target, ast.Name) or target.id == "features" for target in node.targets):
                raise ValueError("Assignments must be local scalars, not input mutation")
        if isinstance(node, ast.Attribute) and not (isinstance(node.value, ast.Name)
                and node.value.id == "features" and node.attr == "get"):
            raise ValueError("Only features.get attributes allowed")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in {"min", "max", "abs", "float", "int"}:
                raise ValueError("Forbidden call")
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise ValueError("Forbidden callee")
            if isinstance(node.func, ast.Attribute):
                if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value not in allowed_features:
                    raise ValueError("Unknown feature access")
        if isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "features"
                    and isinstance(node.slice, ast.Constant) and node.slice.value in allowed_features):
                raise ValueError("Forbidden subscript")
    scope = {"__builtins__": {}, "min": min, "max": max, "abs": abs, "float": float, "int": int}
    exec(compile(tree, "<m7-reviewed-score>", "exec"), scope)
    return scope["escore_programado"]


def score_rows(function, x):
    import numpy as np
    scores = np.array([function(row) for row in x.to_dict(orient="records")], dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Generated score outside [0,1]")
    return scores
