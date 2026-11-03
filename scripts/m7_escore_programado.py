"""M7 E2: label-free score synthesis, training-only selection/calibration and transport."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m7_common import (ROOT, OUT, load_data, save, feature_dictionary, call_json,
    validated_score, score_rows, metric, fit_anchor, anchor_predict, sha)
from gzcmd.ml_oof import SEED, _classifier


def generate(dictionary):
    request = {"feature_dictionary": dictionary,
        "problem": "Link death records SIM with Sinan-TB notifications. True links may have spelling variation, inverted day/month, missing birth date, different municipality due to migration. False matches can share common name fragments. No pair examples, no labels, no learned model parameters are available.",
        "request": "Write a pure Python escore_programado(features: dict) -> float, in [0,1], interpretable weighted rules. Missing keys default zero. Do not mutate features. Only one function (with Portuguese explanatory docstring); no imports, loops, comprehensions, nested functions, globals, annotations or decorators. Only arithmetic, comparisons, if, assignment, return; calls allowed min/max/abs/float/int and features.get('known feature',0). No math exp: use bounded affine score or rational sigmoid. No data/labels embedded. Treat zero as potentially missing; do not mechanically veto migration or uncertain birth. Return JSON: code (string), semantics (list), limitations (list)."}
    first = call_json("e2_generation", request, "Write a semantic linkage score, never infer identities. Return JSON only.")
    refinement = {**request, "previous_code": first["code"],
        "refinement": "Review and refine the function semantically, particularly double-counting correlated fragments, rare/common fragments, missingness and migration. No validation metrics or labels are provided: remain label-free. Return same JSON schema. Improve interpretability rather than promise accuracy."}
    second = call_json("e2_refinement", refinement, "Review the proposed program and return improved pure function as JSON only. Do not invent empirical performance.")
    return [first, second]


def choose_candidate(scores, y, nota, train):
    """Fixed candidates, selected on external-training rows only (no fitted score)."""
    grey = train & (nota >= 5) & (nota < 9)
    from sklearn.metrics import roc_auc_score
    values = [float(roc_auc_score(y[grey], score[grey])) for score in scores]
    return int(np.argmax(values)), values


def linear_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=SEED))


def main():
    raw, x, y, nota, ids, oof = load_data()
    # Dictionary bounds are feature metadata only, no distribution summaries or outcomes.
    versions = generate(feature_dictionary(x))
    scores = [score_rows(validated_score(v["code"], set(x)), x) for v in versions]
    folds, gray = oof.fold.to_numpy(), (nota >= 5) & (nota < 9)
    p_program, s_program = np.full(len(y), np.nan), np.full(len(y), np.nan)
    p_program_direct = np.full(len(y), np.nan)
    p_linear, s_linear, p_nota = [np.full(len(y), np.nan) for _ in range(3)]
    audit = []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        selected, aucs = choose_candidate(scores, y, nota, train)
        score = scores[selected]
        # A bounded score is converted to log-odds before Platt, preserving rank.
        # Also report direct-score Platt as a predeclared scale/bounds sensitivity.
        margin = logit(np.clip(score, 1e-6, 1 - 1e-6))
        program_anchor, pa = fit_anchor(margin[train], nota[train], y[train])
        direct_anchor, direct_audit = fit_anchor(score[train], nota[train], y[train])
        p_program[test] = anchor_predict(program_anchor, margin[test])
        p_program_direct[test] = anchor_predict(direct_anchor, score[test])
        s_program[test] = score[test]
        # Linear anchor fit uses inner-OOF margins, not fitted training margins.
        train_idx = np.flatnonzero(train)
        inner_margin = np.full(train.sum(), np.nan)
        for inner_train, inner_test in StratifiedKFold(5, shuffle=True, random_state=SEED + fold + 1).split(x.loc[train], y[train]):
            model = linear_model().fit(x.iloc[train_idx[inner_train]], y[train_idx[inner_train]])
            inner_margin[inner_test] = model.decision_function(x.iloc[train_idx[inner_test]])
        la_model, la = fit_anchor(inner_margin, nota[train], y[train])
        model = linear_model().fit(x.loc[train], y[train])
        s_linear[test] = model.decision_function(x.loc[test])
        p_linear[test] = anchor_predict(la_model, s_linear[test])
        nota_model, na = fit_anchor(nota[train], nota[train], y[train])
        p_nota[test] = anchor_predict(nota_model, nota[test])
        audit.append({"fold": fold, "selected_version": selected, "training_gray_auc": aucs,
            "train_n": int(train.sum()), "test_n": int(test.sum()), "train_test_overlap": 0,
            "program_anchor": pa, "program_direct_anchor": direct_audit,
            "linear_anchor": la, "nota_anchor": na})
        print(f"E2 fold={fold} selected={selected} train-only AUC={aucs}", flush=True)
    methods = {"programado": (s_program, p_program), "programado_platt_direto": (s_program, p_program_direct), "nota": (nota, p_nota),
        "logistica_l2": (s_linear, p_linear), "xgb_oof_isotonica": (oof.p_cal_ml.to_numpy(), oof.p_cal_ml.to_numpy())}
    rows = []
    for name, (score, prob) in methods.items():
        for scope, mask in [("total", np.ones(len(y), bool)), ("gray", gray)]:
            rows.append({"model": name, "scope": scope, **metric(y[mask], score[mask], prob[mask])})
    validation = pd.DataFrame(rows)
    validation.to_csv(OUT / "e2_validation.csv", index=False)
    # Persist aggregate diagnostics only, never feature rows or labels/pair rows.
    bands = []
    for fold in range(5):
        for version, score in enumerate(scores):
            for low, high in [(-1, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 11)]:
                mask = (folds != fold) & (nota >= low) & (nota < high)
                bands.append({"fold": fold, "version": version, "band_low": low, "band_high": high,
                              **metric(y[mask], score[mask])})
    pd.DataFrame(bands).to_csv(OUT / "e2_training_bands.csv", index=False)
    # PASSO is reserved for stratification, never a model feature or API input.
    passo_col = next(c for c in raw if c == "PASSO" or c.startswith("PASSO,"))
    passo = pd.to_numeric(raw[passo_col].str.replace(",", ".", regex=False), errors="raise").to_numpy()
    unique = np.unique(passo)
    if len(unique) < 2:
        raise ValueError("Transport requires at least two PASSO strata")
    cut = float(unique[len(unique) // 2 - 1])
    strata = passo <= cut
    transport = []
    transport_audit = []
    for direction, source in [("early_to_late", strata), ("late_to_early", ~strata)]:
        target = ~source
        fit_idx, valid_idx = train_test_split(np.flatnonzero(source), test_size=.25, stratify=y[source], random_state=SEED)
        fit_mask = np.zeros(len(y), bool)
        fit_mask[fit_idx] = True
        selected, selection = choose_candidate(scores, y, nota, fit_mask)
        score = scores[selected]
        # Transport endpoint is AUC, so do not fabricate an anchor fit when a
        # blocking stratum has fewer than M5's required 30 high/low anchors.
        if (nota[fit_idx] >= 9).sum() >= 30 and (nota[fit_idx] < 5).sum() >= 30:
            anchor, diag = fit_anchor(logit(np.clip(score[fit_idx], 1e-6, 1 - 1e-6)), nota[fit_idx], y[fit_idx])
        else:
            anchor, diag = None, {"status": "not_fitted_insufficient_anchors",
                "high_n": int((nota[fit_idx] >= 9).sum()), "low_n": int((nota[fit_idx] < 5).sum()),
                "note": "Transport AUC uses raw scores; source-only candidate selection remains fitted, no target rates substituted."}
        model = _classifier(y[fit_idx], SEED).fit(x.iloc[fit_idx], y[fit_idx])
        for name in ["programado", "xgb"]:
            within = score[valid_idx] if name == "programado" else model.predict_proba(x.iloc[valid_idx])[:, 1]
            outside = score[target] if name == "programado" else model.predict_proba(x.loc[target])[:, 1]
            within_prob = anchor_predict(anchor, logit(np.clip(within, 1e-6, 1 - 1e-6))) if name == "programado" and anchor is not None else None
            outside_prob = anchor_predict(anchor, logit(np.clip(outside, 1e-6, 1 - 1e-6))) if name == "programado" and anchor is not None else None
            for scope in ["total", "gray"]:
                wm = np.ones(len(valid_idx), bool) if scope == "total" else gray[valid_idx]
                tm = np.ones(target.sum(), bool) if scope == "total" else gray[target]
                a = metric(y[valid_idx][wm], within[wm], within_prob[wm] if within_prob is not None else None)
                b = metric(y[target][tm], outside[tm], outside_prob[tm] if outside_prob is not None else None)
                transport.append({"direction": direction, "model": name, "scope": scope,
                    "source_validation_n": a["n"], "source_validation_positive": a["positive"],
                    "target_n": b["n"], "target_positive": b["positive"],
                    "within_auc": a["auc"], "transport_auc": b["auc"],
                    "auc_drop": a["auc"] - b["auc"] if a["auc"] is not None and b["auc"] is not None else None})
        transport_audit.append({"direction": direction, "selected_version": selected, "training_selection_auc": selection,
            "fit_n": len(fit_idx), "source_validation_n": len(valid_idx), "target_n": int(target.sum()),
            "program_anchor": diag})
    transport_df = pd.DataFrame(transport)
    transport_df.to_csv(OUT / "e2_transport.csv", index=False)
    # Deployment example selected on fold1 training only, never claimed to be a fitted global probability.
    best, selection_auc = choose_candidate(scores, y, nota, folds != 1)
    code = versions[best]["code"]
    module = '"""M7: escore sem gradiente, sintetizado por Haiku; não é probabilidade calibrada.\n\nVersão escolhida apenas no treino do fold 1; não incorporada ao motor.\n"""\n\n' + code + "\n"
    (ROOT / "gzcmd/escore_programado.py").write_text(module, encoding="utf-8")
    save("e2_summary.json", {"seed": SEED, "folds": audit, "validation": rows,
        "transport": transport, "transport_audit": transport_audit, "transport_passo_cut": cut,
        "saved_version": best, "saved_selection_train_fold1_auc": selection_auc,
        "saved_module_sha256": sha(ROOT / "gzcmd/escore_programado.py"),
        "selection": "two label-free candidates; select using training-gray AUC only per fold. Candidate score is not fitted; its calibration uses training anchor rates only.",
        "limitations": ["pair split, not person split", "PASSO is a blocking/process stratum, not geographical/temporal external validation", "source validation uses 25% holdout; XGB transport discriminative raw probability, no target labels or target calibration", "no validation outcomes were sent to LLM; refinement is semantic only"]})
    print("VALIDATION\n" + validation.to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print("TRANSPORT\n" + transport_df.to_string(index=False, float_format=lambda v: f"{v:.8f}"))


if __name__ == "__main__":
    main()
