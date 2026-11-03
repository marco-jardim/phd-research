"""M7 E1: tree-space audit. Run directly with the Miniconda Python."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m7_common import (ROOT, OUT, load_data, save, feature_dictionary, call_json, rule_value)
from gzcmd.ml_oof import _classifier, SEED


def rule_stats(mask, scope, y):
    selected = mask & scope
    tp, fp = int(y[selected].sum()), int((1 - y[selected]).sum())
    return {"scope_n": int(scope.sum()), "selected": int(selected.sum()), "tp": tp, "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "coverage": float(selected.sum() / scope.sum()),
            "recall": tp / int(y[scope].sum()) if y[scope].sum() else None}


def main():
    _, x, y, nota, ids, oof = load_data()
    folds = oof.fold.to_numpy()
    train, test = folds != 1, folds == 1
    model = _classifier(y[train], SEED)
    model.fit(x.loc[train], y[train])
    booster = model.get_booster()
    dump = booster.get_dump(with_stats=True)
    gains = [sum(float(v) for v in re.findall(r"gain=([0-9.eE+-]+)", tree)) for tree in dump]
    top = sorted(range(len(dump)), key=lambda i: gains[i], reverse=True)[:40]
    trees = [{"tree": i, "total_split_gain": gains[i], "dump": dump[i]} for i in top]
    save("e1_tree_dump.json", {"fold": 1, "train_n": int(train.sum()), "trees": dump,
                               "top40_tree_indices": top, "ranking": "sum of split gain, not causal importance"})
    importance = {kind: booster.get_score(importance_type=kind) for kind in ["gain", "total_gain", "weight", "cover"]}
    # Exact TreeSHAP through native XGBoost: no optional shap dependency/version coupling.
    contrib = booster.predict(xgb.DMatrix(x), pred_contribs=True)
    gray = (nota >= 5) & (nota < 9)
    errors = {}
    for mode in ["vigilancia", "confirmacao"]:
        path = ROOT / f"data/funil_gzcmd_{mode}.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False).set_index("pair_id").loc[ids]
        else:
            from gzcmd.runner import run_v3
            from funil_gzcmd_v2 import safe_funnel
            output, _ = run_v3(input_csv=ROOT / "data/COMPARADORSEMIDENT.csv",
                config_path=ROOT / "gzcmd/config.yaml", mode=mode, p_cal="ml_xgb", reuse_ml_oof=True)
            df = safe_funnel(output).loc[ids]
        np.testing.assert_array_equal(df.y.to_numpy(), y)
        errors[f"{mode}_fn"] = df.decisao_motor.eq("AUTO_NONMATCH").to_numpy() & (y == 1)
        errors[f"{mode}_fp"] = df.decisao_motor.eq("AUTO_MATCH").to_numpy() & (y == 0)
    groups = {"gray_positive": gray & (y == 1), **errors}
    def aggregate(mask):
        return {"n": int(mask.sum()), "features": {
            name: {"mean_shap": float(contrib[mask, j].mean()),
                   "mean_abs_shap": float(abs(contrib[mask, j]).mean())}
            for j, name in enumerate(x)} if mask.any() else {}}
    shap_full = {name: aggregate(mask) for name, mask in groups.items()}
    shap_train = {name: aggregate(mask & train) for name, mask in groups.items()}
    save("e1_shap.json", {"method": "native pred_contribs exact TreeSHAP; log-odds margin",
         "all_rows_descriptive": shap_full, "training_only_sent_to_llm": shap_train,
         "importance": importance, "bias_mean": float(contrib[:, -1].mean())})
    # Keep top-ranked trees first, cap prompt by characters, never cut an individual tree.
    chosen, chars = [], 0
    for tree in trees:
        if chars + len(tree["dump"]) > 155000:
            break
        chosen.append(tree)
        chars += len(tree["dump"])
    payload = {"feature_dictionary": feature_dictionary(x.loc[train]), "fold": 1,
        "training": "fold != 1; no pair rows; aggregate SHAP only from training",
        "trees_total": len(dump), "trees_included": chosen, "importance": importance,
        "shap_training_aggregates": shap_train,
        "problem": "SIM versus Sinan-TB record linkage. Spelling errors, missing birth, date inversion, migration.",
        "request": "Specify what this tree ensemble computes, rank probable failure modes with responsible features. Propose one high-band and one gray consistency rule, boolean over f['feature name']. Rules mean accept MATCH, not NONMATCH. Do not use p_cal or labels. Return JSON keys specification (string), failure_modes (list), high_rule (string), gray_rule (string), limitations (list). Expressions only and/or/not, comparisons, numeric literals, f subscripts. No arithmetic/functions. No pair examples."}
    result = call_json("e1_audit", payload, "You audit model parameters, not individual people. Return only valid JSON. Distinguish hypotheses from observed failure rates; no claim of exact specification from truncated trees.")
    masks = {"llm_high": np.array([rule_value(result["high_rule"], row) for row in x.to_dict("records")]),
             "llm_gray": np.array([rule_value(result["gray_rule"], row) for row in x.to_dict("records")]),
             "current_guard": (nota >= 9) & (oof.p_cal_ml.to_numpy() >= .9)}
    records = []
    for name, mask in masks.items():
        for scope_name, scope in [("high", nota >= 9), ("gray", gray)]:
            for split_name, split in [("pooled_exploratory", np.ones(len(y), bool)), ("heldout_fold1", test)]:
                records.append({"rule": name, "scope": scope_name, "split": split_name,
                                **rule_stats(mask, scope & split, y),
                                **{key + "_selected": int((mask & scope & split & emask).sum()) for key, emask in errors.items()}})
    # Two additional, independent audits cover ALL folds. The initial audit is
    # not reused for these rules because historical error masks may depend
    # indirectly on heldout outcomes through M1's earlier OOF classifiers.
    crossfit_masks = {"crossfit_high": np.zeros(len(y), dtype=bool),
                      "crossfit_gray": np.zeros(len(y), dtype=bool)}
    crossfit_audit = [{"heldout_folds": [1], "call": "e1_audit", "train_n": int(train.sum()),
        "test_n": int(test.sum()), "high_rule": result["high_rule"], "gray_rule": result["gray_rule"],
        "role": "descriptive_audit_only_not_reused_in_crossfit_rules",
        "note": "Historical engine error masks in this initial audit came from frozen M1 OOF cache; indirect dependence is disclosed."}]
    for heldout in [[0, 1, 4], [2, 3]]:
        held = np.isin(folds, heldout)
        fit = ~held
        clf = _classifier(y[fit], SEED).fit(x.loc[fit], y[fit])
        b = clf.get_booster()
        td = b.get_dump(with_stats=True)
        tg = [sum(float(v) for v in re.findall(r"gain=([0-9.eE+-]+)", t)) for t in td]
        indices = sorted(range(len(td)), key=lambda i: tg[i], reverse=True)[:40]
        selected_trees, total_chars = [], 0
        for i in indices:
            if total_chars + len(td[i]) > 155000:
                break
            selected_trees.append({"tree": i, "total_split_gain": tg[i], "dump": td[i]})
            total_chars += len(td[i])
        train_x, train_y = x.loc[fit], y[fit]
        cs = b.predict(xgb.DMatrix(train_x), pred_contribs=True)
        rawp = clf.predict_proba(train_x)[:, 1]
        train_gray = gray[fit]
        train_groups = {"gray_positive": train_gray & (train_y == 1),
            "training_model_fn_proxy": (rawp < .1) & (train_y == 1),
            "training_model_fp_proxy": (rawp >= .9) & (train_y == 0)}
        sg = {name: {"n": int(m.sum()), "features": {
                feature: {"mean_shap": float(cs[m, j].mean()), "mean_abs_shap": float(abs(cs[m, j]).mean())}
                for j, feature in enumerate(x)} if m.any() else {}} for name, m in train_groups.items()}
        name = "e1_audit_excluding_" + "_".join(map(str, heldout))
        new_payload = {"feature_dictionary": feature_dictionary(train_x), "heldout_folds": heldout,
            "training": "Only the complementary persisted folds; no heldout outcomes or pair rows.",
            "trees_total": len(td), "trees_included": selected_trees,
            "importance": {kind: b.get_score(importance_type=kind) for kind in ["gain", "total_gain", "weight", "cover"]},
            "shap_training_aggregates": sg,
            "error_proxy_note": "Training-model threshold proxies, NOT historical motor error masks: avoids indirect heldout dependence via cached OOF model selection.",
            "problem": payload["problem"], "request": payload["request"]}
        response = call_json(name, new_payload,
            "You audit model parameters, not individual people. Return only valid JSON. Distinguish hypotheses from observed failure rates; no claim of exact specification from truncated trees.")
        held_rows = x.loc[held].to_dict("records")
        crossfit_masks["crossfit_high"][held] = [rule_value(response["high_rule"], row) for row in held_rows]
        crossfit_masks["crossfit_gray"][held] = [rule_value(response["gray_rule"], row) for row in held_rows]
        crossfit_audit.append({"heldout_folds": heldout, "call": name, "train_n": int(fit.sum()),
            "test_n": int(held.sum()), "train_test_overlap": 0,
            "high_rule": response["high_rule"], "gray_rule": response["gray_rule"],
            "role": "independent_grouped_oof_rule"})
    for name, mask in crossfit_masks.items():
        for scope_name, scope in [("high", nota >= 9), ("gray", gray)]:
            for split_name, split in [("grouped_oof_all", np.ones(len(y), bool))] + [
                    (f"fold_{f}", folds == f) for f in range(5)]:
                records.append({"rule": name, "scope": scope_name, "split": split_name,
                    **rule_stats(mask, scope & split, y),
                    **{key + "_selected": int((mask & scope & split & emask).sum()) for key, emask in errors.items()}})
    save("e1_crossfit_audit.json", crossfit_audit)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "e1_rules.csv", index=False)
    save("e1_summary.json", {"fold": 1, "trees_in_prompt": len(chosen), "gray_positive_n": int((gray & (y == 1)).sum()),
        "error_counts": {k: int(v.sum()) for k, v in errors.items()}, "rules": records, "crossfit": crossfit_audit,
        "warning": "Single fold1-audit rules pooled over all rows are posthoc; initial error masks have potential indirect label dependence. Independent crossfit rules NEVER reuse the initial response: two fresh calls hold out groups [0,1,4] and [2,3] of the original five folds, using only training parameters and training-only SHAP/error proxies. All predictions are held out, but training fraction is 40%/60%, not 80% as in five separate folds."})
    print(frame.to_string(index=False))
    print("E1 errors", json.dumps({k: int(v.sum()) for k, v in errors.items()}))


if __name__ == "__main__":
    main()
