"""M5: prevalências das mesmas âncoras, agora sobre a margem OOF do XGBoost.

Executar: python scripts/validar_ancoras_ml.py [--reuse-margins]
Não altera M1, seu cache ou o pacote externo. Não persiste dados identificadores.
O adaptador de máscaras é necessário: a API pública fit_anchor_platt define as
âncoras pelo próprio escore, enquanto aqui elas continuam definidas pela NOTA.
Reutiliza o núcleo numérico do pacote; a equivalência é testada sobre a nota.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent / "gzcmd-record-linkage" / "src" / "gzcmd_record_linkage"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE.parent))
from gzcmd.ml_oof import N_SPLITS, SEED, _classifier, numeric_features, pair_ids  # noqa: E402
from gzcmd.config import load_config
from gzcmd.gzcmd_v3_policy_engine import Budget, Costs, LLMError, PolicyEngineV3  # noqa: E402
from gzcmd_record_linkage.calibration import (  # noqa: E402
    PlattModel, _logit_mean_and_gradient, fit_anchor_platt,
)

NAMES = {"platt_nota": "Platt (nota)", "ancoras_ml": "Âncoras (margem ML)",
         "isotonica_ml": "Isotônica ML (OOF)",
         "exp1_sem_limites_ativos": "E1: limites ampliados, sem L2",
         "exp2_ancoras_internas": "E2: momentos por banda de nota",
         "exp3_ancoras_aninhadas": "E3: âncoras estritamente aninhadas"}
PRIMARY = ("platt_nota", "ancoras_ml", "isotonica_ml")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rates(y: np.ndarray, masks: list[np.ndarray]) -> tuple[list[float], list[dict]]:
    """Único uso dos rótulos no ajuste por âncoras: taxas agregadas por conjunto.

    Sem suavização quando 0<r<1; Jeffreys apenas na fronteira (API exige 0<r<1).
    Ambas as taxas são registradas para não apresentar alvo suavizado como medido.
    """
    targets, audit = [], []
    for m in masks:
        n, pos = int(m.sum()), int(y[m].sum())
        if n < 30:
            raise ValueError(f"Âncora insuficiente: N={n} < 30")
        measured = pos / n
        target = measured if 0 < measured < 1 else (pos + 0.5) / (n + 1)
        targets.append(target)
        audit.append({"n": n, "positivos": pos, "taxa_medida": measured,
                      "alvo": target, "suavizada": target != measured})
    return targets, audit


def fit_masked_anchor_platt(
    score: np.ndarray, masks: list[np.ndarray], targets: list[float], *,
    l2: float = 1e-6, alpha_max: float = 10.0, beta_max: float = 50.0,
) -> tuple[PlattModel, dict]:
    """Extensão mínima de fit_anchor_platt a máscaras externas, sem receber y.

    Mesmo objetivo, kernel/gradiente, inicialização, limites e tolerâncias do
    pacote. Para >2 conjuntos (E2), cada momento tem peso igual, como no pacote.
    Uma busca multistart separada é somente E1; não muda o ajuste principal.
    """
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all() or len(masks) != len(targets) or len(masks) < 2:
        raise ValueError("Escores finitos e pelo menos dois momentos alinhados exigidos")
    if any(len(m) != len(score) or np.asarray(m).dtype != bool or m.sum() < 30 for m in masks):
        raise ValueError("Máscaras booleanas alinhadas com pelo menos 30 pares exigidas")
    if any(not 0 < r < 1 for r in targets):
        raise ValueError("Alvos devem estar no intervalo aberto (0,1)")
    groups = [score[m] for m in masks]
    target_logits = logit(targets)

    def objective(w):
        loss, gradient = l2 * float(w @ w), 2 * l2 * w
        for s, target in zip(groups, target_logits):
            value, grad = _logit_mean_and_gradient(s, w)
            residual = value - target
            loss += residual ** 2
            gradient = gradient + 2 * residual * grad
        return loss, gradient

    spread = float(groups[0].mean() - groups[1].mean())
    if spread <= 0:
        raise ValueError("Média da margem positiva não supera a negativa")
    alpha0 = float(np.clip((target_logits[0] - target_logits[1]) / spread, 0, alpha_max))
    beta0 = float(np.clip(target_logits[0] - alpha0 * groups[0].mean(), -beta_max, beta_max))
    starts = [[alpha0, beta0]]
    if l2 == 0:  # E1: verificar se resultado ruim decorre de limites/ótimo local.
        starts += [[1.0, 0.0], [0.5, -3.0], [2.0, 0.0]]
    fits = [minimize(objective, start, jac=True, method="L-BFGS-B",
                     bounds=[(0, alpha_max), (-beta_max, beta_max)],
                     options={"maxiter": 100 if l2 else 1000, "ftol": 1e-10, "gtol": 1e-10})
            for start in starts]
    result = min(fits, key=lambda r: r.fun)
    if not np.isfinite(result.x).all():
        raise RuntimeError("Ajuste não finito")
    model = PlattModel(slope=float(result.x[0]), intercept=float(result.x[1]))
    residuals = [_logit_mean_and_gradient(s, result.x)[0] - t
                 for s, t in zip(groups, target_logits)]
    return model, {"slope": model.slope, "intercept": model.intercept,
                   "success": bool(result.success), "message": str(result.message),
                   "objective": float(result.fun), "max_logit_residual": float(max(abs(np.array(residuals)))),
                   "means": [float(expit(model.slope * s + model.intercept).mean()) for s in groups],
                   "targets": list(targets), "residuals_logit": list(map(float, residuals)),
                   "l2": l2, "alpha_max": alpha_max, "beta_max": beta_max}


def predict(model: PlattModel, score: np.ndarray) -> np.ndarray:
    return expit(model.slope * score + model.intercept)


def reconstruct_margins(x, y, nota, *, split_by="row", groups=None) -> tuple[pd.DataFrame, list[dict]]:
    """Mesmo 5x5, fábrica e seed de M1; extrai margem antes da isotônica.

    E3 ajusta âncoras apenas sobre margens OOF internas e taxas do treino externo.
    Nenhum fit de E3 (classificador, taxas ou calibrador) vê o teste externo.
    """
    margin = np.full(len(y), np.nan)
    iso = np.full(len(y), np.nan)
    nested_anchor = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    audit = []
    if split_by not in {"row", "refrec", "comprec"}:
        raise ValueError("Invalid split_by")
    grouped = split_by != "row"
    if grouped and (groups is None or len(groups) != len(y) or pd.isna(groups).any()):
        raise ValueError("Aligned nonmissing groups required")
    groups = np.asarray(groups) if grouped else None
    splitter = StratifiedGroupKFold if grouped else StratifiedKFold
    outer = splitter(N_SPLITS, shuffle=True, random_state=SEED)
    outer_splits = outer.split(x, y, groups) if grouped else outer.split(x, y)
    for fold, (train, test) in enumerate(outer_splits):
        if grouped:
            assert set(groups[train]).isdisjoint(groups[test])
        inner_prob, inner_margin = np.full(len(train), np.nan), np.full(len(train), np.nan)
        inner = splitter(N_SPLITS, shuffle=True, random_state=SEED + fold + 1)
        inner_splits = inner.split(x.iloc[train], y[train], groups[train]) if grouped else inner.split(x.iloc[train], y[train])
        for itrain, itest in inner_splits:
            fit_idx, pred_idx = train[itrain], train[itest]
            assert not np.intersect1d(fit_idx, test).size
            assert not np.intersect1d(pred_idx, test).size
            if grouped:
                assert set(groups[fit_idx]).isdisjoint(groups[pred_idx])
            model = _classifier(y[fit_idx], SEED)
            model.fit(x.iloc[fit_idx], y[fit_idx])
            inner_prob[itest] = model.predict_proba(x.iloc[pred_idx])[:, 1]
            inner_margin[itest] = model.predict(x.iloc[pred_idx], output_margin=True)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(inner_prob, y[train])
        masks = [nota[train] >= 9, nota[train] < 5]
        targets, measured = rates(y[train], masks)
        anchor, diagnostic = fit_masked_anchor_platt(inner_margin, masks, targets)
        model = _classifier(y[train], SEED)
        model.fit(x.iloc[train], y[train])
        margin[test] = model.predict(x.iloc[test], output_margin=True)
        iso[test] = calibrator.predict(model.predict_proba(x.iloc[test])[:, 1])
        nested_anchor[test] = predict(anchor, margin[test])
        folds[test] = fold
        audit.append({"fold": fold, "train_n": len(train), "test_n": len(test),
                      "split_by": split_by, "group_overlap": 0 if grouped else None,
                      "taxas_treino": measured, "ajuste": diagnostic})
        print(f"OOF fold={fold}: train={len(train)}, test={len(test)}, 6 fits, interseção=0", flush=True)
    if not all(np.isfinite(a).all() for a in (margin, iso, nested_anchor)):
        raise ValueError("Predições OOF incompletas")
    return pd.DataFrame({"margin_oof": margin, "isotonica_recalculada": iso,
                         "exp3_ancoras_aninhadas": nested_anchor, "fold": folds}), audit


def reliability(y: np.ndarray, p: np.ndarray, kind: str) -> list[dict]:
    """ECE: 10 intervalos uniformes; curva adicional: decis de massa aproximada.

    Decis por qcut sem separar empates (bordas duplicadas removidas); isotônica
    pode produzir menos de dez grupos. Isso fica explícito no CSV/relatório.
    """
    if kind == "uniforme":
        labels = np.minimum((p * 10).astype(int), 9)
        bins = range(10)
    else:
        labels = np.asarray(pd.qcut(p, 10, labels=False, duplicates="drop"))
        if np.isnan(labels).all():
            labels = np.zeros(len(p), dtype=int)
        bins = sorted(set(labels))
    rows = []
    for b in bins:
        m = labels == b
        n = int(m.sum())
        rows.append({"tipo_bin": kind, "bin": int(b) + 1, "n": n,
                     "positivos": int(y[m].sum()), "p_min": float(p[m].min()) if n else np.nan,
                     "p_max": float(p[m].max()) if n else np.nan,
                     "p_media": float(p[m].mean()) if n else np.nan,
                     "taxa_observada": float(y[m].mean()) if n else np.nan,
                     "contribuicao_ece": n / len(y) * abs(float(y[m].mean() - p[m].mean())) if n else 0.0})
    return rows


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if not (np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()):
        raise ValueError("Probabilidades inválidas")
    return {"n": len(y), "positivos": int(y.sum()), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "ece_10": sum(r["contribuicao_ece"] for r in reliability(y, p, "uniforme")),
            "auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "mediana_vinculos": float(np.median(p[y == 1])),
            "mediana_nao_vinculos": float(np.median(p[y == 0])),
            "fracao_vinculos_p_lt_01": float((p[y == 1] < .1).mean()),
            "fracao_vinculos_p_gt_09": float((p[y == 1] > .9).mean())}


def decision(nota, y, probabilities) -> pd.DataFrame:
    """Mesma simulação v3 de validar_ancoras.py; NÃO é a nova política v3.1."""
    g = (nota >= 5) & (nota < 9)
    sg, yg = nota[g], y[g]
    bands = np.where(sg < 6, "grey_low", np.where(sg < 7, "grey_mid", np.where(sg < 8, "grey_high", "near_high")))
    errors = {"grey_low": LLMError(.08, .12), "grey_mid": LLMError(.06, .10),
              "grey_high": LLMError(.04, .08), "near_high": LLMError(.02, .06)}
    efp, efn = np.array([errors[b].e_fp for b in bands]), np.array([errors[b].e_fn for b in bands])
    modes = {"vigilancia": (10., 50., .25, .85, .15, 2000),
             "confirmacao": (100., 20., .25, .95, .10, 1000)}
    rows = []
    for mode, (cfp, cfn, cllm, tmatch, tnon, budget) in modes.items():
        for name in PRIMARY:
            engine = PolicyEngineV3(Costs(cfp, cfn, cllm), errors, Budget(llm_max=budget),
                                    min_auto_match=tmatch, max_auto_nonmatch=tnon, policy_version="v3")
            act = engine.triage(pd.DataFrame({"p_cal": probabilities[name][g], "band": bands})).action.to_numpy()
            rev = act == "LLM_REVIEW"
            fn, fp = int(((act == "NONMATCH") & (yg == 1)).sum()), int(((act == "MATCH") & (yg == 0)).sum())
            loss = fn * cfn + fp * cfp + rev.sum() * cllm
            loss_config = loss + ((rev & (yg == 1)) * efn * cfn).sum() + ((rev & (yg == 0)) * efp * cfp).sum()
            rows.append({"modo": mode, "calibracao": name, "revisao": int(rev.sum()),
                         "auto_match": int((act == "MATCH").sum()), "auto_nonmatch": int((act == "NONMATCH").sum()),
                         "fn_auto": fn, "fp_auto": fp, "perda_revisor_perfeito": float(loss),
                         "perda_revisor_config": float(loss_config)})
    return pd.DataFrame(rows)


def table(valid: pd.DataFrame, dec: pd.DataFrame) -> str:
    def num(v):
        return f"{v:.6f}".replace(".", ",")
    lines = ["% Gerado por scripts/validar_ancoras_ml.py; não altera tabelas históricas.",
             r"\begin{table}[!ht]", r"\centering", r"\footnotesize",
             r"\caption{Calibração sobre a margem OOF do XGBoost, com âncoras definidas pela nota ($A^+:s\geq9$, $A^-:s<5$). Métricas na área cinza ($5\leq s<9$). Platt da nota é histórico, ajustado na própria base; isotônica ML usa validação aninhada. Âncoras usam somente duas prevalências agregadas medidas no extrato, não constituindo validação externa. ECE: dez intervalos uniformes.}",
             r"\label{tab:cap8-ancoras-ml}", r"\begin{tabular}{|l|r|r|r|r|r|}", r"\hline",
             r"Calibração & Brier & Log-loss & ECE & AUC & Mediana ($Y=1$) \\", r"\hline"]
    for k in PRIMARY:
        r = valid[(valid.calibracao == k) & (valid.escopo == "cinza")].iloc[0]
        lines.append(NAMES[k] + " & " + " & ".join(num(r[c]) for c in ["brier", "log_loss", "ece_10", "auc", "mediana_vinculos"]) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\smallskip", r"\begin{tabular}{|l|l|r|r|r|r|r|}", r"\hline",
              r"Modo & Calibração & Revisão & FN auto & FP auto & Perda perfeita & Perda config. \\", r"\hline"]
    for _, r in dec.iterrows():
        mode = "Vigilância" if r.modo == "vigilancia" else "Confirmação"
        vals = [str(int(r[c])) for c in ["revisao", "fn_auto", "fp_auto"]]
        vals += [f"{r[c]:.1f}".replace(".", ",") for c in ["perda_revisor_perfeito", "perda_revisor_config"]]
        lines.append(f"{mode} & {NAMES[r.calibracao]} & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\smallskip",
              r"\parbox{\linewidth}{\footnotesize Simulação histórica v3, sem guardas, com os mesmos custos, tetos, orçamento e erros por banda das tabelas anteriores. Perda configurada é expectativa sobre as classes observadas, não resposta de revisor executado.}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    ap.add_argument("--oof", type=Path, default=ROOT / "data/pcal_ml_oof.csv")
    ap.add_argument("--config", type=Path, default=ROOT / "gzcmd/gzcmd_v3_config.yaml")
    ap.add_argument("--split-by", choices=("row", "refrec", "comprec"))
    ap.add_argument("--reuse-margins", action="store_true", help="Reutiliza somente cache M5 com hashes/versões válidos")
    args = ap.parse_args()
    import sklearn
    import xgboost

    raw = pd.read_csv(args.csv, sep=";", dtype=str, low_memory=False)
    raw.columns = [c.split(",")[0] for c in raw.columns]
    split_by = args.split_by or load_config(args.config).evaluation.split_by
    groups = raw[split_by.upper()].str.strip().to_numpy() if split_by != "row" else None
    x = numeric_features(raw)
    nota = x["nota final" if "nota final" in x else "nota_final"].to_numpy()
    y = pd.to_numeric(raw.PAR, errors="raise").isin([1, 2]).astype(int).to_numpy()
    ids = pair_ids(raw)
    oof = pd.read_csv(args.oof)
    metadata = json.loads(args.oof.with_suffix(".json").read_text(encoding="utf-8"))
    fingerprint = {"method": "nested-xgb-isotonic-5x5-v1", "seed": SEED, "folds": N_SPLITS, "split_by": split_by,
                   "source_sha256": sha(args.csv), "code_sha256": sha(ROOT / "gzcmd/ml_oof.py"),
                   "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}
    if metadata.get("fingerprint") != fingerprint or metadata.get("csv_sha256") != sha(args.oof):
        raise ValueError("Cache M1 não corresponde à fonte/código/versões. Recalcule M1 antes de M5.")
    if metadata.get("features") != list(x.columns):
        raise ValueError("Allowlist/ordem de features difere do cache M1")
    if not oof.pair_id.is_unique or set(ids) != set(oof.pair_id):
        raise ValueError("Cobertura pair_id incompatível")
    oof = oof.set_index("pair_id").loc[ids].reset_index()
    np.testing.assert_allclose(oof.nota, nota, atol=1e-12)
    # Esta chamada pública e a extensão por máscaras devem coincidir na nota.
    base_masks = [nota >= 9, nota < 5]
    targets, anchors = rates(y, base_masks)
    package_model = fit_anchor_platt(pd.Series(nota), r_plus=targets[0], r_minus=targets[1])
    local_model, _ = fit_masked_anchor_platt(nota, base_masks, targets)
    equivalence = float(np.max(abs(predict(package_model, nota) - predict(local_model, nota))))
    assert equivalence < 1e-10, equivalence
    print(f"Equivalência adaptador/API pública sobre nota: max |delta p|={equivalence:.3g}", flush=True)

    cache_path = ROOT / "data/ancoras_ml_margens_oof.csv"
    meta_path = cache_path.with_suffix(".json")
    fp = {**fingerprint, "script_sha256": sha(Path(__file__)), "m1_csv_sha256": sha(args.oof),
          "package_sha256": sha(PACKAGE / "calibration.py")}
    cache_metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if (args.reuse_margins and cache_path.exists() and cache_metadata.get("fingerprint") == fp
            and cache_metadata.get("csv_sha256") == sha(cache_path)):
        margins = pd.read_csv(cache_path).set_index("pair_id").loc[ids].reset_index()
        nested_audit = cache_metadata["nested_audit"]
        print("Cache M5 de margens validado por hashes e versões.", flush=True)
    else:
        margins, nested_audit = reconstruct_margins(x, y, nota, split_by=split_by, groups=groups)
        margins.insert(0, "pair_id", ids.to_numpy())
        margins.to_csv(cache_path, index=False)
        meta_path.write_text(json.dumps({"fingerprint": fp, "csv_sha256": sha(cache_path),
                                         "nested_audit": nested_audit}, indent=2), encoding="utf-8")
    np.testing.assert_array_equal(margins.fold, oof.fold)
    iso_delta = float(np.max(abs(margins.isotonica_recalculada.to_numpy() - oof.p_cal_ml.to_numpy())))
    assert iso_delta < 1e-12, f"Isotônica reconstruída diverge: {iso_delta}"
    print(f"Reprodução M1: max |delta isotônica|={iso_delta:.3g}; folds idênticos", flush=True)
    score = margins.margin_oof.to_numpy()
    anchor_model, fit_audit = fit_masked_anchor_platt(score, base_masks, targets)
    probs = {"platt_nota": oof.p_cal_platt.to_numpy(), "ancoras_ml": predict(anchor_model, score),
             "isotonica_ml": oof.p_cal_ml.to_numpy()}
    loose, loose_audit = fit_masked_anchor_platt(score, base_masks, targets, l2=0, alpha_max=100, beta_max=500)
    probs["exp1_sem_limites_ativos"] = predict(loose, score)
    # E2 é um experimento oracle de informação agregada adicional, NÃO sem rótulos.
    extra_masks = base_masks + [(nota >= a) & (nota < a + 1) for a in range(5, 9)]
    extra_targets, extra_rates = rates(y, extra_masks)
    internal, internal_audit = fit_masked_anchor_platt(score, extra_masks, extra_targets)
    probs["exp2_ancoras_internas"] = predict(internal, score)
    probs["exp3_ancoras_aninhadas"] = margins.exp3_ancoras_aninhadas.to_numpy()
    g = (nota >= 5) & (nota < 9)
    validation, curves = [], []
    for name, p in probs.items():
        for scope, m in [("total", np.ones(len(y), dtype=bool)), ("cinza", g)]:
            validation.append({"calibracao": name, "escopo": scope, **metrics(y[m], p[m])})
            for kind in ["uniforme", "decil"]:
                curves.extend({"calibracao": name, "escopo": scope, **r}
                              for r in reliability(y[m], p[m], kind))
    sensitivity, sensitivity_audit = [], []
    for plus in [9., 9.5]:
        for minus in [5., 4.]:
            masks = [nota >= plus, nota < minus]
            ts, rs = rates(y, masks)
            model, diag = fit_masked_anchor_platt(score, masks, ts)
            p = predict(model, score)
            sensitivity.append({"s_plus": plus, "s_minus": minus,
                                "n_plus": rs[0]["n"], "n_minus": rs[1]["n"],
                                "r_plus_medida": rs[0]["taxa_medida"], "r_minus_medida": rs[1]["taxa_medida"],
                                "r_plus_alvo": ts[0], "r_minus_alvo": ts[1],
                                "max_logit_residual": diag["max_logit_residual"], **metrics(y[g], p[g])})
            sensitivity_audit.append({"s_plus": plus, "s_minus": minus, "taxas": rs, "ajuste": diag})
    valid, conf, sens = pd.DataFrame(validation), pd.DataFrame(curves), pd.DataFrame(sensitivity)
    dec = decision(nota, y, probs)
    diagnostics = []
    for group, m in [("A+", base_masks[0]), ("A-", base_masks[1]), ("cinza", g)]:
        for target in [0, 1]:
            z = score[m & (y == target)]
            diagnostics.append({"grupo": group, "y": target, "n": len(z),
                                **{f"margem_q{q}": float(np.quantile(z, q / 100)) for q in [0, 10, 50, 90, 100]},
                                "fracao_sigmoid_lt_001": float((expit(z) < .01).mean()),
                                "fracao_sigmoid_gt_099": float((expit(z) > .99).mean())})
    diagnostic_df = pd.DataFrame(diagnostics)
    out = ROOT / "data"
    for frame, suffix in [(valid, "validacao"), (conf, "confiabilidade"), (dec, "decisao"),
                          (sens, "sensibilidade"), (diagnostic_df, "diagnostico")]:
        frame.to_csv(out / f"ancoras_ml_{suffix}.csv", index=False)
    audit = {"seed": SEED, "folds": N_SPLITS, "fingerprint": fp,
             "adapter_api_delta": equivalence, "isotonic_reproduction_delta": iso_delta,
             "anchors": anchors, "base_fit": fit_audit, "exp1": loose_audit,
             "exp2": {"rates": extra_rates, "fit": internal_audit},
             "exp3": nested_audit, "sensitivity": sensitivity_audit}
    (out / "ancoras_ml_ajustes.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (ROOT / "text/latex/tables/tab_cap8_ancoras_ml.tex").write_text(table(valid, dec), encoding="utf-8")
    print("\nMETRICAS\n" + valid.to_string(index=False, float_format=lambda f: f"{f:.8f}"))
    print("\nSENSIBILIDADE\n" + sens.to_string(index=False, float_format=lambda f: f"{f:.8f}"))
    print("\nDECISAO V3 SEM GUARDAS\n" + dec.to_string(index=False))
    print("\nDIAGNOSTICO MARGENS\n" + diagnostic_df.to_string(index=False, float_format=lambda f: f"{f:.6f}"))
    print("\nAJUSTE PRINCIPAL\n" + json.dumps(fit_audit, indent=2))
    print("\nCONFIABILIDADE CINZA — BINS EXTREMOS\n" + conf[
        conf.calibracao.isin(PRIMARY) & conf.escopo.eq("cinza")
        & conf.tipo_bin.eq("uniforme") & conf.bin.isin([1, 10])
    ].to_string(index=False, float_format=lambda f: f"{f:.8f}"))
    write_report(valid, sens, dec, diagnostic_df, conf, audit, out)


def write_report(valid, sens, dec, diagnostic_df, conf, audit, out):
    grey = valid[valid.escopo == "cinza"].set_index("calibracao")
    base, iso = grey.loc["ancoras_ml"], grey.loc["isotonica_ml"]
    # Definição operacional de 'próximos', fixada no script: tolerância de 25%
    # mais 1e-4 absoluta para evitar divisão por ECE quase zero. Não é teste formal.
    close_brier = base.brier <= 1.25 * iso.brier + 1e-4
    close_ece = base.ece_10 <= 1.25 * iso.ece_10 + 1e-4
    success = close_brier and close_ece and base.mediana_vinculos > .5 and base.mediana_nao_vinculos < .1
    verdict = "SIM, descritivamente neste extrato" if success else "NÃO: o critério conjunto não é satisfeito"
    extreme_bins = conf[conf.calibracao.isin(PRIMARY) & conf.escopo.eq("cinza")
                        & conf.tipo_bin.eq("uniforme") & conf.bin.isin([1, 10])]
    e1, e2, e3 = [grey.loc[k] for k in ["exp1_sem_limites_ativos", "exp2_ancoras_internas", "exp3_ancoras_aninhadas"]]
    nested_slopes = [r["ajuste"]["slope"] for r in audit["exp3"]]
    distributions = diagnostic_df.set_index(["grupo", "y"])
    pos_anchor_neg = distributions.loc[("A+", 0)]
    neg_anchor_pos = distributions.loc[("A-", 1)]
    findings = [
        f"Âncoras vs isotônica na área cinza: Brier {(base.brier / iso.brier - 1) * 100:+.2f}%, ECE {(base.ece_10 / iso.ece_10 - 1) * 100:+.2f}% e log-loss {(base.log_loss / iso.log_loss - 1) * 100:+.2f}%. A mediana alta dos vínculos ({base.mediana_vinculos:.8f}) resolve o problema de ordenação da nota, mas NÃO demonstra probabilidades bem calibradas.",
        f"E1: Brier={e1.brier:.8f}, ECE={e1.ece_10:.8f}, resíduo máximo={audit['exp1']['max_logit_residual']:.3g}. Igualar quase exatamente os momentos não melhora a calibração cinzenta; não é um efeito dos limites α/β, da penalização ou da inicialização examinados.",
        f"E2: Brier={e2.brier:.8f}, ECE={e2.ece_10:.8f} (ainda {(e2.ece_10 / iso.ece_10 - 1) * 100:.2f}% acima da isotônica), resíduo máximo logit={audit['exp2']['fit']['max_logit_residual']:.6f}. Mesmo fornecendo taxas internas oracle, uma única sigmoide de dois parâmetros não satisfaz todos os momentos nem remove a sobreconfiança. Só adicionar momentos por banda ao mesmo Platt não basta.",
        f"E3: Brier={e3.brier:.8f}, ECE={e3.ece_10:.8f}, log-loss={e3.log_loss:.8f}. O aninhamento rigoroso não resgata o resultado (ECE {(e3.ece_10 / iso.ece_10 - 1) * 100:.2f}% acima da isotônica). Inclinações variam de {min(nested_slopes):.6f} a {max(nested_slopes):.6f} entre folds, coerente com instabilidade das taxas extremas raras e da escala das margens.",
        "Conclusão diagnóstica: os controles não encontraram falha numérica que explique a discrepância. Há evidência de especificação/transporte inadequados dos dois momentos extremos para a área cinza e de sobreconfiança, não de perda da capacidade discriminativa do classificador. Isso não prova uma impossibilidade teórica geral das âncoras: testa esta seleção pela nota e esta família Platt neste extrato.",
        "Melhoria prioritária: manter a isotônica supervisionada aninhada como referência operacional quando há gabarito. Para uma alternativa de poucos rótulos, medir âncoras internas por consistência independente de campos em amostra separada e substituir o Platt global por curva monotônica regularizada com restrições agregadas e pooling entre estratos/folds; avaliar por pessoa e fora da amostra. E2 mostra que a informação interna sozinha, sem flexibilidade de forma, teve benefício insuficiente."
    ]
    text = ["# M5 — Calibração por âncoras da margem supervisionada", "", f"## Veredito: {verdict}", "",
            f"Brier próximo: {close_brier}; ECE próximo: {close_ece}; mediana Y=1 >0,5: {base.mediana_vinculos > .5}; mediana Y=0 <0,1: {base.mediana_nao_vinculos < .1}.",
            "Proximidade operacional: erro ≤1,25×isotônica +0,0001 (não é margem de equivalência inferencial).",
            "", "## Síntese dos resultados e recomendação", "", *findings,
            "", "## Reprodução e protocolo", "", "```powershell",
            "python scripts/validar_ancoras_ml.py", "python scripts/validar_ancoras_ml.py --reuse-margins", "```", "",
            f"Python Miniconda; seed 42; split_by={audit['fingerprint']['split_by']}; mesma fábrica XGBoost/allowlist de M1; 5 folds externos × (5 internos + 1 ajuste externo), 300 árvores por ajuste. Nenhum LLM executado.",
            f"Equivalência do adaptador à API sobre a nota: max |Δp|={audit['adapter_api_delta']:.12g}. Reprodução da isotônica M1: max |Δp|={audit['isotonic_reproduction_delta']:.12g}; folds iguais (asserções no comando).",
            "Margem extraída por predict(output_margin=True), não pelo logit da isotônica (que perde informação e possui 0/1). O cache M1 original é somente lido e validado por hashes, features, seed e versões; margens M5 têm cache separado com hashes de fonte/código/pacote/CSV e versões.",
            "Gabarito PAR∈{1,2}, leitura dtype=str, decimal vírgula convertido. pair_id é o mesmo hash de M1, com unicidade/cobertura verificadas. Somente hashes, margens, probabilidades e agregados são persistidos; nenhum identificador ou campo pessoal.",
            "", "### Limite real da API do pacote", "",
            "fit_anchor_platt seleciona A+/A- por cortes no próprio argumento nota. Passar a margem com cortes 9/5, ou recalcular cortes para a margem, TROCARIA os conjuntos. Para preservar exatamente as âncoras originais, implementou-se extensão local por máscaras, reutilizando _logit_mean_and_gradient do pacote e reproduzindo objetivo, inicialização, limites, L2 e tolerâncias. A chamada pública é executada como controle sobre a nota e comparada ao adaptador a cada execução. O pacote externo não foi alterado. A extensão por máscaras é uma adaptação explícita, não uma alegação de que a API pública aceita máscaras.",
            "", "### Rótulos, OOF e limites de interpretação", "",
            "A margem de cada par nunca é treinada no rótulo desse par. O ajuste principal usa apenas as duas taxas agregadas das âncoras extremas; não usa rótulos cinzentos diretamente. Taxas medidas exatas (sem arredondamento a cinco casas do script histórico). Os rótulos são reutilizados para avaliação offline, não para ajustar individualmente o Platt ancorado.",
            "O Platt da nota é o comparador histórico in-sample de pcal_ml_oof.csv, NÃO calibração por âncoras da nota. A isotônica é estritamente aninhada. Já ajustar um único Platt às margens OOF globais não é uma avaliação totalmente aninhada: escores de outros folds podem depender do rótulo de um par avaliado, e taxas extremas incluem seus próprios rótulos no total. E3 elimina essa dependência: cada calibrador recebe apenas margens OOF internas e prevalências do treino externo. Separação por par, não por pessoa; registros compartilhados continuam sendo uma limitação. Nenhum resultado prova calibração sem gabarito nem validação externa.",
            "", "## Métricas copiadas da saída do comando", "", "```text",
            valid.to_string(index=False, float_format=lambda f: f"{f:.8f}"), "```", "",
            "Brier/log-loss menores são melhores; log-loss trunca p em [10^-6,1−10^-6] igualmente para todos, inclusive isotônica. AUC/AP são discriminação, não calibração. ECE usa dez intervalos uniformes [0,0,1), …, [0,9,1], ponderados por N. Com 151 vínculos em 21.620 cinzentos, Brier/ECE podem esconder erro na classe rara: por isso medianas e frações condicionais também são exigidas.",
            "Curvas completas em ancoras_ml_confiabilidade.csv: intervalos uniformes (inclusive vazios) e decis por quantis. Empates isotônicos não são repartidos artificialmente; qcut remove bordas duplicadas, podendo haver menos de dez grupos. A contribuição ECE dos intervalos uniformes soma o ECE da validação.",
            "", "### Sobreconfiança nos dois extremos cinzentos", "", "```text",
            extreme_bins.to_string(index=False, float_format=lambda f: f"{f:.8f}"), "```", "",
            "Compare p_media com taxa_observada, não apenas a quantidade de pares: o bin de baixa probabilidade pode subestimar vínculos raros mesmo tendo ótimo desempenho majoritário; o bin ≥0,9 pode superestimar o valor preditivo positivo. Esses são intervalos de probabilidade, não bandas da nota.",
            "", "## Sensibilidade aos cortes", "", "```text",
            sens.to_string(index=False, float_format=lambda f: f"{f:.8f}"), "```", "",
            f"Mediana cinzenta Y=1 varia de {sens.mediana_vinculos.min():.8f} a {sens.mediana_vinculos.max():.8f} (amplitude {sens.mediana_vinculos.max() - sens.mediana_vinculos.min():.8f}). O conjunto de avaliação [5,9) permanece fixo.",
            "Se uma âncora tem prevalência 0/1, a restrição logit não tem solução finita: usa-se Jeffreys (pos+0,5)/(N+1) SOMENTE nessa fronteira; taxas medidas e alvos distintos ficam registrados. Cortes mais estritos não garantem estimativas mais estáveis: A+ é pequeno e A- tem pouquíssimos positivos.",
            "", "## Diagnóstico: implementação versus hipótese de transporte", "", "```text",
            json.dumps(audit["base_fit"], indent=2), "```", "", "```text",
            diagnostic_df.to_string(index=False, float_format=lambda f: f"{f:.6f}"), "```", "",
            "O resíduo logit e o status do otimizador distinguem falha de encaixe dos momentos de falha de calibração fora das âncoras. Igualar duas médias não identifica a curva P(Y=1|margem) na área intermediária. As âncoras são selecionadas pela nota, que também é feature do classificador: P(Y|margem,âncora) não precisa ser P(Y|margem,cinza). Distribuições/quantis por classe acima permitem verificar sobreposição e saturação sem presumir que não há suporte comum. scale_pos_weight do XGBoost altera a escala probabilística bruta; a margem não é um log-odds populacional já calibrado. Platt continua monótono na margem e preserva sua ordenação, ao contrário de corrigir a ordenação insuficiente da nota.",
            f"Nas âncoras positivas, os {int(pos_anchor_neg.n)} não vínculos têm margem entre {pos_anchor_neg.margem_q0:.6f} e {pos_anchor_neg.margem_q100:.6f}; entre não vínculos cinzentos a margem máxima é {distributions.loc[('cinza', 0), 'margem_q100']:.6f}. Nas âncoras negativas, os {int(neg_anchor_pos.n)} vínculos têm mediana de margem {neg_anchor_pos.margem_q50:.6f}, contra {distributions.loc[('cinza', 1), 'margem_q50']:.6f} nos vínculos cinzentos. As distribuições têm suporte sobreposto: 'âncoras só cobrem extremos' refere-se à seleção pela nota, não a separação perfeita das margens. Os números são consistentes com seleção não representativa e saturação dos escores fáceis, sem provar isoladamente violação da independência condicional. AUC da isotônica pode mudar por patamares/transformações diferentes por fold; AUC maior do Platt ancorado global não implica calibração melhor.",
            "", "## Três experimentos rápidos executados", "",
            "1. **E1 — limites/L2/ótimo local:** mesmas duas âncoras, L2=0, limites α≤100, |β|≤500 e quatro inicializações. Compara com α≤10, |β|≤50 e L2=10^-6. Se não resolver a discrepância, afrouxar o otimizador não basta.",
            "2. **E2 — âncoras internas estratificadas:** adiciona quatro momentos de prevalência nas bandas [5,6), …, [8,9), com igual peso logit; ainda só dois parâmetros Platt. É experimento oracle/otimista: usa informação agregada dos próprios cinzentos avaliados, NÃO estimativa independente nem âncora disponível sem rótulos. Incompatibilidade dos seis momentos aparece nos resíduos do JSON. Próximo experimento prospectivo: âncoras internas definidas por consistência independente de campos, prevalências medidas numa amostra separada e ajuste monotônico mais flexível com restrições de médias por estrato.",
            "3. **E3 — aninhamento rigoroso:** calibra por âncoras nas margens OOF internas, mede taxas só no treino externo, aplica à margem do teste externo. Controla dependência indireta do ajuste OOF global e flutuação de prevalências entre folds. Não muda os modelos ou o cache de M1. Auditoria por fold em ancoras_ml_ajustes.json registra taxas, N e resíduos.",
            "", "### Ajustes dos experimentos (saída numérica)", "", "```json",
            json.dumps({"exp1": audit["exp1"], "exp2": audit["exp2"]}, indent=2), "```", "",
            "Isotônica com restrições agregadas é uma melhoria concreta para a próxima rodada, mas duas médias sozinhas deixam muitos graus de liberdade: exigir regularização/suavidade e âncoras internas independentes, escolher hiperparâmetros só no treino e avaliar por fold/pessoa. EM com prevalência conhecida só é justificável sob hipóteses de mistura/label shift verificáveis; a seleção pela nota pode violá-las. Não foi executado um quarto experimento nem apresentado EM/isotônica restrita como solução comprovada.",
            "", "## Decisão — mesma política histórica, sem guardas", "", "```text",
            dec.to_string(index=False), "```", "",
            "PolicyEngineV3 policy_version=v3; vigilância: Cfp=10, Cfn=50, Cllm=0,25, tetos 0,85/0,15, orçamento 2.000; confirmação: 100/20/0,25, tetos 0,95/0,10, orçamento 1.000. Erros por banda idênticos ao validar_ancoras.py. Não se usa v3.1, guardas ou custo humano: objetivo é isolar calibração nas tabelas históricas. Perda perfeita soma FP/FN automáticos + custo de revisão; perda configurada acrescenta erros esperados do revisor sobre as classes observadas. Não houve revisão LLM real.",
            "", "## Artefatos", "",
            "Script scripts/validar_ancoras_ml.py; CSVs ancoras_ml_{validacao,confiabilidade,decisao,sensibilidade,diagnostico,margens_oof}.csv; JSONs de cache/ajustes; tabela nova tab_cap8_ancoras_ml.tex. Tabelas/capítulo/configurações existentes e pacote externo não são modificados."]
    (out / "ancoras_ml_relatorio.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"\nVEREDITO: {verdict}; Brier próximo={close_brier}; ECE próximo={close_ece}")
    print(f"Sensibilidade: mediana vínculos cinzentos [{sens.mediana_vinculos.min():.8f}, {sens.mediana_vinculos.max():.8f}]")


if __name__ == "__main__":
    main()
