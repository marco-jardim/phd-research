"""Recomputa os agregados Modal com probabilidades persistidas, sem treinamento.

Carrega somente funções puras do fonte via AST, nunca importa Modal. A validação
histórica é bloqueante e ocorre em diretório temporário antes de qualquer escrita.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.feature_engineering import engineer_features, split_data

PIPELINE = ROOT / "scripts/qw1_modal_unified_pipeline.py"
META = ROOT / "data/qw1_modal_recompute_meta.json"
FILES = ("ablation_results.csv", "pareto_grid.csv", "epidemiological_impact.csv")
LOG = logging.getLogger(__name__)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_functions(source, names, data_root=ROOT / "data"):
    """Extrai funções nomeadas, redirecionando apenas as três entradas agregadas."""
    class LocalPaths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith("/root/data/"):
                return ast.copy_location(
                    ast.Constant(str(data_root / node.value.removeprefix("/root/data/"))), node
                )
            return node

    tree = ast.parse(source)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError("Funções solicitadas ausentes do fonte")
    module = ast.fix_missing_locations(LocalPaths().visit(ast.Module(body=functions, type_ignores=[])))
    namespace = {"np": np, "pd": pd}
    exec(compile(module, str(PIPELINE), "exec"), namespace)
    return namespace


def load_holdout():
    """Recria o split canônico e verifica a identidade e a ordem de cada linha."""
    pd.options.future.infer_string = False
    df = pd.read_csv(ROOT / "data/COMPARADORSEMIDENT.csv", sep=";", dtype=str)
    df.columns = [c.split(",")[0] for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="coerce")
    X, y = engineer_features(df)
    _, X_test, _, y_test = split_data(X, y, test_size=0.3, random_state=42)
    scored = pd.read_csv(ROOT / "data/qw1/xgboost_holdout_scored.csv")
    # O exportador Modal gravou a posição no split, não o índice do extrato.
    np.testing.assert_array_equal(scored.row_index, np.arange(len(X_test)))
    np.testing.assert_array_equal(scored.y_true, y_test)
    np.testing.assert_array_equal(scored.nota_final, X_test["nota final"])
    return X_test, scored


def legacy_rules(row):
    """Regra defeituosa congelada de 1042959, somente para validação prévia."""
    def v(key):
        value = row.get(key, 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    score = 0.0
    if v("NOME qtd frag iguais") >= 2:
        score += 3
    if v("NOME compativel bool") > 0:
        score += 1
    if v("DTNASC dt iguais") == 1:
        score += 3
    elif v("DTNASC dt ap 1digi") == 1:
        score += 2
    if v("NOMEMAE qtd frag iguais") >= 2:
        score += 2
    elif v("NOMEMAE compativel bool") == 1:
        score += 1
    if v("CODMUNRES local igual") == 1:
        score += 1
    if v("ENDERECO via igual") == 1:
        score += 0.5
    nota_final = v("nota final")
    if nota_final >= 9:
        score += 2
    elif nota_final >= 8:
        score += 1
    return score


def xgb_mask(frame, filename):
    col = {FILES[0]: "config", FILES[1]: "ml_model", FILES[2]: "method"}[filename]
    return frame[col].astype(str).str.contains("XGBoost", case=False, regex=False)


def recompute(scored, source, originals):
    """Executa os mesmos agregadores históricos em arquivos temporários."""
    approved_temp = Path(tempfile.gettempdir()) / "opencode"
    with tempfile.TemporaryDirectory(prefix="qw1_recompute_",
                                     dir=approved_temp if approved_temp.exists() else None) as directory:
        base = Path(directory)
        for sub in ("csvs", "probas", "inputs"):
            (base / sub).mkdir()
        scored.to_csv(base / "probas/xgboost_holdout_scored.csv", index=False)
        pd.DataFrame({"y_true": scored.y_true, "proba": scored.proba_xgboost}).to_csv(
            base / "probas/xgboost_holdout_probas.csv", index=False
        )
        for filename, frame in originals.items():
            # Os dois agregadores auxiliares só acrescentam linhas ausentes.
            frame = frame.loc[~xgb_mask(frame, filename)]
            frame.to_csv(base / "inputs" / filename, index=False)
        names = ["evaluate_binary_predictions", "update_ablation_results",
                 "update_pareto_grid", "update_epidemiological_impact"]
        functions = load_functions(source, names, base / "inputs")
        functions["update_ablation_results"](base, None, None, LOG)
        functions["update_pareto_grid"](base, LOG)
        functions["update_epidemiological_impact"](base, LOG)
        return {filename: pd.read_csv(base / "csvs" / filename) for filename in FILES}


def assert_exact(expected, actual):
    """Comparação exata após a mesma leitura CSV, sem tolerância numérica."""
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True),
                                  check_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--baseline-ref", default="1042959")
    args = parser.parse_args()
    source = PIPELINE.read_text(encoding="utf-8")
    X_test, scored = load_holdout()
    rules_old = X_test.apply(legacy_rules, axis=1).to_numpy(dtype=float)
    np.testing.assert_array_equal(rules_old, scored.rules_score)
    baseline_bytes = {
        filename: subprocess.run(
            ["git", "show", f"{args.baseline_ref}:data/{filename}"], cwd=ROOT,
            check=True, capture_output=True,
        ).stdout for filename in FILES
    }
    originals = {filename: pd.read_csv(io.BytesIO(content))
                 for filename, content in baseline_bytes.items()}
    input_paths = [ROOT / "data/COMPARADORSEMIDENT.csv",
                   ROOT / "data/qw1/xgboost_holdout_scored.csv", PIPELINE,
                   ROOT / "scripts/ablation_study.py", ROOT / "scripts/feature_engineering.py",
                   Path(__file__)]
    input_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                    for path in input_paths}
    historical = recompute(scored.assign(rules_score=rules_old), source, originals)
    validation = {}
    for filename in FILES:
        mask = xgb_mask(originals[filename], filename)
        actual = historical[filename]
        assert_exact(originals[filename].loc[mask], actual.loc[xgb_mask(actual, filename)])
        validation[filename] = int(mask.sum())
        print(f"VALIDAÇÃO EXATA {filename}: {mask.sum()} linhas XGBoost, todas as colunas iguais")
    if validation[FILES[0]] != 33:
        raise ValueError("Esperadas exatamente 33 configurações XGBoost")
    print(f"Escore histórico reproduzido em {len(X_test)} linhas; máximo {rules_old.max()}")
    if args.validate_only:
        return
    from scripts.ablation_study import regras_alta_confianca

    modal_rule = load_functions(source, ["regras_alta_confianca"])["regras_alta_confianca"]
    rules_new = X_test.apply(modal_rule, axis=1).to_numpy(dtype=float)
    np.testing.assert_array_equal(rules_new, X_test.apply(regras_alta_confianca, axis=1))
    if np.array_equal(rules_new, rules_old):
        raise ValueError("A função Modal ainda usa o escore defeituoso")
    corrected = recompute(scored.assign(rules_score=rules_new), source, originals)
    # Antes de escrever, verificar TODOS os agregados e os controles intocados.
    for filename in FILES:
        before, after = originals[filename], corrected[filename]
        assert_exact(before.loc[~xgb_mask(before, filename)],
                     after.loc[~xgb_mask(after, filename)])
        current = pd.read_csv(ROOT / "data" / filename)
        assert_exact(before.loc[~xgb_mask(before, filename)],
                     current.loc[~xgb_mask(current, filename)])
        if filename != FILES[0]:
            assert_exact(before, after)  # Nenhum desses dois agregadores usa regras.
    before = originals[FILES[0]].loc[xgb_mask(originals[FILES[0]], FILES[0])]
    after = corrected[FILES[0]].loc[xgb_mask(corrected[FILES[0]], FILES[0])]
    diff = []
    for (_, old), (_, new) in zip(before.iterrows(), after.iterrows(), strict=True):
        assert old["config"] == new["config"]
        diff.append({"config": old["config"], "category": old["category"],
                     "before": {col: float(old[col]) if col == "f1" else int(old[col])
                                for col in ("f1", "tp", "fp", "fn")},
                     "after": {col: float(new[col]) if col == "f1" else int(new[col])
                               for col in ("f1", "tp", "fp", "fn")}})
    # Preservar byte a byte cabeçalho e as 70 linhas originais do checkout.
    path = ROOT / "data" / FILES[0]
    prefix = b"".join(path.read_bytes().splitlines(keepends=True)[:71])
    assert not xgb_mask(originals[FILES[0]].iloc[:70], FILES[0]).any()
    assert len(originals[FILES[0]]) == 103
    tail = after.to_csv(index=False, header=False, lineterminator="\n").encode("utf-8")
    path.write_bytes(prefix + tail)
    assert path.read_bytes().startswith(prefix)
    metadata = {
        "data_utc": datetime.now(timezone.utc).isoformat(), "seed": 42, "test_size": 0.3,
        "baseline_ref": args.baseline_ref, "input_sha256": input_hashes,
        "baseline_sha256_lf": {name: hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
                               for name, content in baseline_bytes.items()},
        "validation_exact_rows": validation, "validation_columns": "all; check_exact=True",
        "legacy_rules_equal_rows": len(X_test), "canonical_rules_equal_rows": len(X_test),
        "original_70_preserved_bytewise": True,
        "rules_max_before": float(rules_old.max()), "rules_max_after": float(rules_new.max()),
        "positive_rules_counts_after": {str(th): int(((scored.y_true == 1) & (rules_new >= th)).sum())
                                        for th in (5, 6, 7, 8, 9)},
        "unchanged_artifacts": list(FILES[1:]),
        "output_sha256": {name: sha256(ROOT / "data" / name) for name in FILES},
        "diff_33": diff,
    }
    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"CORREÇÃO: {len(diff)} configurações; {sum(d['before'] != d['after'] for d in diff)} alteradas")
    print(f"Regras corrigidas: máximo {rules_new.max()}; positivos por corte: "
          f"{metadata['positive_rules_counts_after']}")
    print(f"70 linhas originais preservadas byte a byte; metadados: {META}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
