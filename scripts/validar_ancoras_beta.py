"""M5b: Beta monotônica por momentos, com restrição cinza não supervisionada.

python scripts/validar_ancoras_beta.py
Reutiliza exclusivamente os escores OOF de M5, sem retreinar ou alterar M1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validar_ancoras_ml as m5  # noqa: E402

SEED = 42
NAMES = {"platt_nota": "Platt (nota)", "ancoras_ml": "Platt por âncoras (ML)",
         "beta_ancoras": "Beta por âncoras", "beta_ancoras_cinza": "Beta + restrição cinza",
         "isotonica_ml": "Isotônica supervisionada"}


def beta_design(raw_probability: np.ndarray) -> np.ndarray:
    s = np.asarray(raw_probability, dtype=float)
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("Probabilidades brutas finitas em [0,1] exigidas")
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(s), -np.log1p(-s), np.ones(len(s))])


def fit_beta_moments(s, masks, targets, *, seed=SEED, starts=12):
    """WLS por momentos de médias, pesos inversos à variância binomial plug-in.

    Não recebe rótulos. Aceita âncoras pequenas e taxas 0/1 sem inventar pureza:
    Jeffreys é usado só na variância, nunca para substituir o alvo medido.
    A restrição cinza tem variância plug-in, não inclui erro do estimador FS.
    """
    design = beta_design(s)
    targets = np.asarray(targets, dtype=float)
    masks = [np.asarray(mask) for mask in masks]
    if len(masks) != len(targets) or len(masks) < 3:
        raise ValueError("Pelo menos três momentos alinhados exigidos")
    if any(mask.dtype != bool or mask.shape != (len(design),) or not mask.any() for mask in masks):
        raise ValueError("Máscaras booleanas alinhadas e não vazias exigidas")
    if not (np.isfinite(targets).all() and ((0 <= targets) & (targets <= 1)).all()):
        raise ValueError("Alvos finitos em [0,1] exigidos")
    n = np.array([mask.sum() for mask in masks], dtype=float)
    variance_rate = (targets * n + .5) / (n + 1)
    se = np.sqrt(variance_rate * (1 - variance_rate) / n)
    # Agrupar probabilidades idênticas acelera os multistarts sem aproximação.
    groups = []
    for mask in masks:
        unique, counts = np.unique(design[mask], axis=0, return_counts=True)
        groups.append((unique, counts / counts.sum()))

    def means_and_jac(w):
        means, jac = [], []
        for xx, freq in groups:
            p = expit(xx @ w)
            means.append(p @ freq)
            jac.append((freq * p * (1 - p)) @ xx)
        return np.array(means), np.array(jac)

    def residual(w):
        return (means_and_jac(w)[0] - targets) / se

    def jacobian(w):
        return means_and_jac(w)[1] / se[:, None]

    rng = np.random.default_rng(seed)
    initial = [np.array(v, dtype=float) for v in [(1, 1, 0), (1, 1, -5), (.1, .1, -3), (2, .1, 0)]]
    initial += [np.r_[rng.uniform(.05, 4, 2), rng.uniform(-8, 2)] for _ in range(max(0, starts - 4))]
    # Produtos n×3 são pequenos: BLAS multithread custa mais que o cálculo e
    # tornava a análise de sensibilidade desnecessariamente lenta no Windows.
    with threadpool_limits(limits=1, user_api="blas"):
        fits = [least_squares(residual, w, jac=jacobian, x_scale="jac",
                              bounds=([0, 0, -100], [1000, 1000, 100]),
                              max_nfev=5000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
                for w in initial[:starts]]
    result = min(fits, key=lambda fit: np.sum(fit.fun ** 2))
    means, jac = means_and_jac(result.x)
    singular = np.linalg.svd(jac / se[:, None], compute_uv=False)
    objective = float(np.sum(result.fun ** 2))
    near = [f for f in fits if np.sum(f.fun ** 2) <= objective + 1e-6 * (1 + objective)]
    near_params = np.array([f.x for f in near])
    with threadpool_limits(limits=1, user_api="blas"):
        near_predictions = np.array([expit(design @ f.x) for f in near])
    audit = {"a": float(result.x[0]), "b": float(result.x[1]), "c": float(result.x[2]),
             "success": bool(result.success), "message": str(result.message), "nfev": result.nfev,
             "objective": objective, "n": n.astype(int).tolist(),
             "targets": targets.tolist(), "means": means.tolist(), "residuals": (means - targets).tolist(),
             "standard_errors_plugin": se.tolist(), "weighted_residuals": result.fun.tolist(),
             "jacobian_singular_values": singular.tolist(),
             "jacobian_condition": float(singular[0] / max(singular[-1], 1e-300)),
             "active_bounds": result.active_mask.tolist(), "bounds_upper": [1000, 1000, 100],
             "near_optimal_start_count": len(near),
             "near_optimal_parameter_min": near_params.min(axis=0).tolist(),
             "near_optimal_parameter_max": near_params.max(axis=0).tolist(),
             "near_optimal_max_prediction_spread": float(np.ptp(near_predictions, axis=0).max()),
             "seed": seed, "starts": len(fits)}
    return expit(design @ result.x), audit


def fs_posterior(weights: np.ndarray, *, base: float = 2., starts=8, seed=SEED):
    """Estima prior por EM, com razão de verossimilhança FS fixa, sem y.

    L(pi) proporcional a produto[(1-pi)+pi*LR_i]; q=pi*LR/(1-pi+pi*LR).
    Usa todos os pares bloqueados, antes de somar q somente na área cinza.
    Assume LR válida na população selecionada, hipótese explicitada no relatório.
    """
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or base <= 1:
        raise ValueError("Pesos finitos e base >1 exigidos")
    log_lr = np.log(base) * weights
    rng = np.random.default_rng(seed)
    initial = [1e-4, .001, .01, .1] + rng.uniform(.00001, .5, max(0, starts - 4)).tolist()
    fits = []
    for prior in initial[:starts]:
        for iteration in range(10000):
            q = expit(logit(prior) + log_lr)
            updated = float(np.clip(q.mean(), 1e-12, 1 - 1e-12))
            if abs(updated - prior) < 1e-12:
                prior = updated
                break
            prior = updated
        ll = float(np.logaddexp(np.log1p(-prior), np.log(prior) + log_lr).sum())
        fits.append((ll, prior, iteration + 1))
    best = max(fits)
    q = expit(logit(best[1]) + log_lr)
    return q, {"method": "FS fixed likelihood ratio, unlabeled prior EM", "base": base,
               "prior": best[1], "iterations": best[2], "log_likelihood_relative": best[0],
               "prior_range_multistart": [min(f[1] for f in fits), max(f[1] for f in fits)],
               "total_expected": float(q.sum()), "seed": seed, "starts": len(fits)}


def infer_fs_base(weights):
    """Audita a unidade pelo piso de discordância completa e parâmetros do cap.5.

    m=(.99,.75,.95), u=(.01,.02,.05). O piso teórico base 2 é
    -12.84813778786, igual ao piso do extrato. Não assume sigmoid(SCORE).
    """
    m, u = np.array([.99, .75, .95]), np.array([.01, .02, .05])
    ln_floor = np.log((1 - m) / (1 - u)).sum()
    observed = float(np.min(weights))
    errors = {str(base): abs(ln_floor / np.log(base) - observed) for base in [2., np.e, 10.]}
    base = min([2., np.e, 10.], key=lambda b: errors[str(b)])
    if errors[str(base)] > 1e-6:
        raise ValueError("Não foi possível auditar unidade do peso FS pelo piso documentado")
    return base, {"m": m.tolist(), "u": u.tolist(), "observed_floor": observed,
                  "expected_floor": float(ln_floor / np.log(base)), "errors_by_base": errors}


def binomial_mixture(nota, *, trials=20, seed=SEED):
    """Fallback explícito se SCORE/m/u não existem: discretização fixa da nota.

    k=round(20*clip(nota/10,0,1)); mistura binomial de duas componentes por EM.
    O componente de maior média é o presumido vínculo, não verdade identificada.
    A nota não é contagem binomial; essa hipótese artificial limita o estimador.
    """
    k = np.rint(trials * np.clip(np.asarray(nota) / 10, 0, 1)).astype(int)
    rng = np.random.default_rng(seed)
    best = None
    for init in range(8):
        prior = float(rng.uniform(.001, .2))
        theta = np.sort(rng.uniform(.1, .95, 2))
        for iteration in range(3000):
            l0 = np.log1p(-prior) + k * np.log(theta[0]) + (trials - k) * np.log1p(-theta[0])
            l1 = np.log(prior) + k * np.log(theta[1]) + (trials - k) * np.log1p(-theta[1])
            q = expit(l1 - l0)
            new_prior = float(np.clip(q.mean(), 1e-8, 1 - 1e-8))
            new_theta = np.clip([((1 - q) * k).sum() / ((1 - q).sum() * trials),
                                 (q * k).sum() / (q.sum() * trials)], 1e-8, 1 - 1e-8)
            delta = max(abs(new_prior - prior), max(abs(new_theta - theta)))
            prior, theta = new_prior, new_theta
            if delta < 1e-10:
                break
        l0 = np.log1p(-prior) + k * np.log(theta[0]) + (trials - k) * np.log1p(-theta[0])
        l1 = np.log(prior) + k * np.log(theta[1]) + (trials - k) * np.log1p(-theta[1])
        ll = float(np.logaddexp(l0, l1).sum())
        if best is None or ll > best[0]:
            best = (ll, expit(l1 - l0), prior, theta.tolist(), iteration + 1)
    return best[1], {"method": "two-binomial EM on rounded 20*nota/10", "trials": trials,
                     "prior": best[2], "theta": best[3], "iterations": best[4], "seed": seed}


def success_metrics(y, p, iso_ece):
    high = p >= .9
    predicted = float(p[high].mean()) if high.any() else np.nan
    observed = float(y[high].mean()) if high.any() else np.nan
    ece = m5.metrics(y, p)["ece_10"]
    return {"high_n": int(high.sum()), "high_predicted": predicted, "high_observed": observed,
            "high_gap": predicted - observed, "success_ece": bool(ece <= 1.2 * iso_ece),
            "success_high": bool(high.any() and abs(predicted - observed) <= .02),
            "success_joint": bool(ece <= 1.2 * iso_ece and high.any() and abs(predicted - observed) <= .02)}


def simulate_decisions(nota, y, probabilities):
    # Reusa o motor histórico e seus custos de M5 sem alterar PRIMARY global.
    frames = []
    for name, p in probabilities.items():
        repeated = {key: p for key in m5.PRIMARY}
        frame = m5.decision(nota, y, repeated)
        frame = frame[frame.calibracao.eq(m5.PRIMARY[0])].copy()
        frame["calibracao"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def latex_table(valid):
    lines = ["% Gerado por scripts/validar_ancoras_beta.py", r"\begin{table}[!ht]", r"\centering", r"\footnotesize",
             r"\caption{Calibração na área cinza definida pela nota do comparador ($5\leq v<9$; 21.620 potenciais pares, 151 vínculos). As duas últimas colunas referem-se a $p\geq0{,}9$.}",
             r"\label{tab:cap8-ancoras-beta}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Calibração & Brier & ECE & Prevista & Observada \\", r"\midrule"]
    for name in NAMES:
        row = valid[valid.calibracao.eq(name) & valid.escopo.eq("cinza")].iloc[0]
        values = ["n.a." if pd.isna(row[c]) else f"{row[c]:.6f}".replace(".", ",")
                  for c in ["brier", "ece_10", "high_predicted", "high_observed"]]
        lines.append(NAMES[name] + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\parbox{\linewidth}{\footnotesize Beta+ usa posteriores de Fellegi e Sunter, com prevalência estimada sem rótulos. Platt da nota é histórico; isotônica supervisionada é aninhada. n.a.: ausência de previsões no intervalo superior.}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=ROOT / "data/COMPARADORSEMIDENT.csv")
    args = ap.parse_args()
    raw = pd.read_csv(args.csv, sep=";", dtype=str, low_memory=False)
    ids = m5.pair_ids(raw)
    nota = pd.to_numeric(raw["nota final"].str.replace(",", ".", regex=False), errors="raise").to_numpy()
    # Estimador cinza definido e executado ANTES de ler PAR.
    grey = (nota >= 5) & (nota < 9)
    score_cols = [c for c in raw if c == "SCORE" or c.startswith("SCORE,")]
    if score_cols:
        weights = pd.to_numeric(raw[score_cols[0]].str.replace(",", ".", regex=False), errors="raise").to_numpy()
        base, floor_audit = infer_fs_base(weights)
        fs, estimator = fs_posterior(weights, base=base)
        estimator["score_column"] = score_cols[0]
        estimator["unit_audit"] = floor_audit
    else:
        fs, estimator = binomial_mixture(nota)
    estimated_count = float(fs[grey].sum())
    estimator["grey_expected"] = estimated_count
    estimator["grey_prevalence"] = estimated_count / grey.sum()
    print("ESTIMADOR SEM ROTULOS\n" + json.dumps(estimator, indent=2), flush=True)
    margin_path = ROOT / "data/ancoras_ml_margens_oof.csv"
    margin_meta = json.loads(margin_path.with_suffix(".json").read_text(encoding="utf-8"))
    if margin_meta["fingerprint"]["source_sha256"] != m5.sha(args.csv) or margin_meta["csv_sha256"] != m5.sha(margin_path):
        raise ValueError("Cache de margens M5 incompatível com a fonte/CSV")
    margins = pd.read_csv(margin_path).set_index("pair_id")
    oof_path = ROOT / "data/pcal_ml_oof.csv"
    oof_meta = json.loads(oof_path.with_suffix(".json").read_text(encoding="utf-8"))
    if oof_meta["fingerprint"]["source_sha256"] != m5.sha(args.csv) or oof_meta["csv_sha256"] != m5.sha(oof_path):
        raise ValueError("Cache M1 incompatível")
    oof = pd.read_csv(oof_path).set_index("pair_id")
    if not margins.index.is_unique or not oof.index.is_unique or set(ids) != set(margins.index) or set(ids) != set(oof.index):
        raise ValueError("Cobertura/unicidade pair_id incompatível")
    margins, oof = margins.loc[ids], oof.loc[ids]
    np.testing.assert_array_equal(margins.fold, oof.fold)
    np.testing.assert_allclose(oof.nota, nota, atol=1e-12)
    s = expit(margins.margin_oof.to_numpy())
    y = pd.to_numeric(raw.PAR, errors="raise").isin([1, 2]).astype(int).to_numpy()
    masks = [nota < 5, (nota >= 9) & (nota < 9.5), nota >= 9.5]
    targets = [float(y[mask].mean()) for mask in masks]
    anchor_rows = [{"conjunto": name, "n": int(mask.sum()), "positivos": int(y[mask].sum()), "prevalencia": target}
                   for name, mask, target in zip(["A-", "A+1", "A+2"], masks, targets)]
    anchor_frame = pd.DataFrame(anchor_rows)
    print("ANCORAS\n" + anchor_frame.to_string(index=False), flush=True)
    pt, _ = m5.rates(y, [nota >= 9, nota < 5])
    platt, _ = m5.fit_masked_anchor_platt(margins.margin_oof.to_numpy(), [nota >= 9, nota < 5], pt)
    beta, beta_audit = fit_beta_moments(s, masks, targets)
    print(f"Beta ajustada: objetivo={beta_audit['objective']:.8f}, nfev={beta_audit['nfev']}", flush=True)
    beta_plus, plus_audit = fit_beta_moments(s, masks + [grey], targets + [estimator["grey_prevalence"]])
    print(f"Beta+ ajustada: objetivo={plus_audit['objective']:.8f}, nfev={plus_audit['nfev']}", flush=True)
    probabilities = {"platt_nota": oof.p_cal_platt.to_numpy(),
                     "ancoras_ml": m5.predict(platt, margins.margin_oof.to_numpy()),
                     "beta_ancoras": beta, "beta_ancoras_cinza": beta_plus,
                     "isotonica_ml": oof.p_cal_ml.to_numpy()}
    iso_ece = m5.metrics(y[grey], probabilities["isotonica_ml"][grey])["ece_10"]
    validation, curves = [], []
    for name, p in probabilities.items():
        for scope, mask in [("cinza", grey), ("total", np.ones(len(y), dtype=bool))]:
            reference_ece = m5.metrics(y[mask], probabilities["isotonica_ml"][mask])["ece_10"]
            validation.append({"calibracao": name, "escopo": scope, **m5.metrics(y[mask], p[mask]),
                               **success_metrics(y[mask], p[mask], reference_ece),
                               "optimizer_success": {"beta_ancoras": beta_audit["success"],
                                                     "beta_ancoras_cinza": plus_audit["success"]}.get(name, True)})
            for kind in ["uniforme", "decil"]:
                curves.extend({"calibracao": name, "escopo": scope, **r}
                              for r in m5.reliability(y[mask], p[mask], kind))
    sensitivity = []
    # A+ precisa continuar dividido: subir o início 9->9,5 colapsaria A+1.
    # Cruzar início 9/9,5 com A- 5/4, preservando divisão em 9,5/9,75.
    for lower, split in [(9., 9.5), (9.5, 9.75)]:
        for minus in [5., 4.]:
            mm = [nota < minus, (nota >= lower) & (nota < split), nota >= split]
            if not all(mask.any() for mask in mm):
                raise ValueError("Sensibilidade produziu âncora vazia")
            tt = [float(y[mask].mean()) for mask in mm]
            for constrained in [False, True]:
                pred, fit = fit_beta_moments(s, mm + ([grey] if constrained else []),
                                             tt + ([estimator["grey_prevalence"]] if constrained else []))
                sensitivity.append({"s_plus": lower, "s_split": split, "s_minus": minus,
                                    "restricao_cinza": constrained,
                                    **{f"n_{i}": int(mask.sum()) for i, mask in enumerate(mm)},
                                    **{f"r_{i}": target for i, target in enumerate(tt)},
                                    **m5.metrics(y[grey], pred[grey]),
                                    **success_metrics(y[grey], pred[grey], iso_ece),
                                    "a": fit["a"], "b": fit["b"], "c": fit["c"], "objective": fit["objective"],
                                    "optimizer_success": fit["success"], "nfev": fit["nfev"]})
                print(f"Sensibilidade: A+={lower}/{split}, A-={minus}, cinza={constrained}, objetivo={fit['objective']:.6f}", flush=True)
    valid, conf, sens = pd.DataFrame(validation), pd.DataFrame(curves), pd.DataFrame(sensitivity)
    dec = simulate_decisions(nota, y, probabilities)
    estimator["grey_observed_evaluation_only"] = int(y[grey].sum())
    estimator["difference_expected_minus_observed"] = estimated_count - int(y[grey].sum())
    audit = {"source_sha256": m5.sha(args.csv), "script_sha256": m5.sha(Path(__file__)),
             "margin_csv_sha256": m5.sha(margin_path), "m1_csv_sha256": m5.sha(oof_path),
             "seed": SEED, "estimator": estimator, "beta": beta_audit, "beta_plus": plus_audit}
    out = ROOT / "data"
    for frame, suffix in [(valid, "validacao"), (conf, "confiabilidade"), (dec, "decisao"),
                          (sens, "sensibilidade"), (anchor_frame, "conjuntos")]:
        frame.to_csv(out / f"ancoras_beta_{suffix}.csv", index=False)
    (out / "ancoras_beta_ajustes.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (ROOT / "text/latex/tables/tab_cap8_ancoras_beta.tex").write_text(latex_table(valid), encoding="utf-8")
    print("\nMETRICAS\n" + valid.to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print("\nSENSIBILIDADE\n" + sens.to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print("\nDECISAO SEM GUARDAS\n" + dec.to_string(index=False))
    print("\nAJUSTES\n" + json.dumps(audit, indent=2))
    write_document(valid, sens, dec, anchor_frame, audit)


def write_document(valid, sens, dec, anchors, audit):
    text = ["# M5b: calibração Beta por momentos", "", "## Modelo e derivação", "",
            "Referência verificada: Kull, Silva Filho e Flach (2017), AISTATS, PMLR 54:623-631, https://proceedings.mlr.press/v54/kull17a.html. O artigo descreve calibração supervisionada; seu uso por momentos agregados aqui é uma extensão experimental, não validação do procedimento pelo artigo.",
            "Com densidades condicionais Beta(s;α1,β1) e Beta(s;α0,β0), o logaritmo da razão posterior é c+(α1−α0)ln(s)+(β1−β0)ln(1−s). Definindo a=α1−α0 e b=β0−β1, obtém-se p=expit(a ln(s)−b ln(1−s)+c). Impõe-se a,b≥0: dp/ds=p(1−p)[a/s+b/(1−s)]≥0. A identidade está contida: a=b=1,c=0. A calibração Platt da margem é o caso a=b (antes do clipping).",
            "A motivação gaussiana usual de Platt decorre de distribuições condicionais de mesma variância para o escore. Isso é condição suficiente para log-odds linear, não pressuposto necessário de toda regressão logística. Árvores podem produzir escores saturados e distribuições assimétricas, motivando os dois coeficientes da Beta, sem garantir sua correção.",
            "s=expit(margem OOF do XGBoost), truncada em [10^-6,1−10^-6]. As margens são as mesmas do M5, 5 folds externos com seed 42; caches são validados pela fonte, hash do CSV, cobertura de pair_id e folds. Não se usa logit da isotônica. Não houve retreinamento nem chamada LLM.",
            "", "## Momentos e algoritmo", "", "```text", anchors.to_string(index=False), "```", "",
            "Três médias: A− com nota<5, A+1 com 9≤nota<9,5 e A+2 com nota≥9,5. O alvo é a prevalência medida exata de cada âncora. A+1 tem menos de 30 pares e taxa de fronteira; isso é reportado, não ocultado por agrupamento ou suavização do alvo. Beta+ acrescenta a média cinza estimada sem ler PAR. Nenhum rótulo cinzento individual entra no ajuste.",
            "Objetivo: soma_j [(mean_j(p)−r_j)/SE_j]^2, SE_j=sqrt(rtilde_j(1−rtilde_j)/n_j), rtilde=(n_j*r_j+0,5)/(n_j+1). Jeffreys estabiliza SOMENTE o peso, não altera r_j. Beta usa três momentos; Beta+ usa quatro para três parâmetros, logo não há garantia de satisfazer todos simultaneamente. Otimização least_squares com Jacobiano analítico e escala adaptativa x_scale=jac, 12 inicializações fixas/aleatórias, seed 42, a,b∈[0,1000], c∈[−100,100], até 5.000 avaliações e tolerâncias 10^-11. Uma primeira tentativa sem escala adaptativa e com a≤100 atingiu 2.000 avaliações em Beta+ sem convergir; seus resultados provisórios foram substituídos pelos ajustes auditados abaixo. Reportam-se limites ativos, resíduos, valores singulares do Jacobiano e dispersão das previsões entre inicializações quase ótimas (objetivo≤melhor+10^-6*(1+melhor)). Três momentos não garantem identificabilidade quando as distribuições de s são semelhantes ou saturadas.",
            "", "## Estimador da restrição cinza sem rótulos", "", "```json", json.dumps(audit["estimator"], indent=2), "```", "",
            "O extrato possui SCORE,C,15,0, peso de Fellegi e Sunter do OpenRecLink, distinto da nota final do comparador. Usou-se esse peso, não a mistura binomial. A base 2 é auditada pelo piso de discordância total: soma log2((1−m)/(1−u)), com m=(0,99;0,75;0,95), u=(0,01;0,02;0,05), parâmetros documentados na Tabela tab:parametros-orl do capítulo 5. O piso coincide com o mínimo do extrato dentro de 10^-6. Não se pressupõe que sigmoid(SCORE) seja posterior.",
            "Define-se LR_i=2^SCORE_i e estima-se π sem rótulos por EM sobre todos os pares: q_i=expit(logit(π)+ln(2)*SCORE_i), π_novo=mean(q). Isso maximiza soma ln[(1−π)+π LR_i], omitindo a densidade de não vínculos constante em π. O alvo cinza é E=sum(q_i para nota em [5,9)). EM usa multistart e não consulta PAR. A comparação com 151 é somente avaliação posterior.",
            "A razão FS configurada pode não ser a razão correta para o universo bloqueado/exportado, com campos dependentes e parâmetros fixos. O erro do alvo cinza pode ser elevado. O SE plug-in do quarto momento ignora incerteza/viés do modelo FS e pode conferir peso excessivo a uma estimativa errada; os resultados devem ser lidos como teste dessa hipótese, não prevalência conhecida. O fallback binomial está implementado para fontes sem SCORE: k=round(20*clip(nota/10,0,1)), EM de duas binomiais, componente de maior média presumido vínculo. Não foi usado aqui e não seria justificado como contagem real de critérios.",
            "", "## Avaliação e critério pré-definido", "",
            "Sucesso conjunto: ECE cinza≤1,2×ECE da isotônica e |média prevista−frequência observada|≤0,02 em p≥0,9, com pelo menos uma observação nesse intervalo. Avaliação descritiva no mesmo extrato, não teste de equivalência. ECE usa dez intervalos uniformes; a curva adicional usa decis por quantis sem partir empates. Log-loss trunca todas as probabilidades em [10^-6,1−10^-6]. Isotônica é supervisionada estritamente aninhada; Platt da nota é histórico in-sample. As calibrações globais por momentos sobre margens OOF não são totalmente aninhadas: há dependência indireta entre folds; M5 já examinou a variante aninhada. Os rótulos das âncoras são usados como prevalências, e os cinzentos somente para avaliar. O escore FS e a nota compartilham campos, portanto o quarto momento não é evidência independente.",
            "", "```text", valid.to_string(index=False, float_format=lambda v: f"{v:.8f}"), "```", "",
            "## Sensibilidade", "", "```text", sens.to_string(index=False, float_format=lambda v: f"{v:.8f}"), "```", "",
            "Avaliação cinza permanece [5,9). Cortes A−<5/<4 e início A+≥9/≥9,5. Para manter três âncoras não vazias, A+≥9,5 é dividido em [9,5;9,75) e ≥9,75; simplesmente manter divisão 9,5 produziria A+1 vazio. Taxas zero/um são mantidas no alvo, com pesos estabilizados. Nenhum corte é selecionado pelo desempenho cinzento.",
            "", "## Decisão histórica sem guardas", "", "```text", dec.to_string(index=False), "```", "",
            "Reusa a simulação v3 de M5, com seus custos, tetos, erros por banda e orçamento históricos; não altera a política/configuração de outros experimentos. Perdas configuradas são expectativas do revisor, não revisões executadas.",
            "", "## Auditoria dos ajustes", "", "```json", json.dumps({k: audit[k] for k in ["beta", "beta_plus"]}, indent=2), "```", "",
            "## Reprodução", "", "```powershell", "python scripts/validar_ancoras_beta.py",
            "python -m pytest tests/test_validar_ancoras_beta.py -q -o addopts=", "```"]
    grey = valid[valid.escopo.eq("cinza")].set_index("calibracao")
    for name in ["beta_ancoras", "beta_ancoras_cinza"]:
        row = grey.loc[name]
        text += ["", f"**{NAMES[name]}: sucesso conjunto={row.success_joint}**. Brier={row.brier:.8f}, ECE={row.ece_10:.8f}; intervalo superior previsto={row.high_predicted:.8f}, observado={row.high_observed:.8f}, diferença={row.high_gap:.8f}."]
    text += ["", f"Ajuste Beta: convergiu={audit['beta']['success']}, condição do Jacobiano={audit['beta']['jacobian_condition']:.6g}, amplitude máxima de p entre soluções quase ótimas={audit['beta']['near_optimal_max_prediction_spread']:.6f}. Isso avalia identificabilidade prática, não só sucesso numérico do algoritmo.",
             f"Ajuste Beta+: convergiu={audit['beta_plus']['success']}; massa cinza ajustada={audit['beta_plus']['means'][-1] * 21620:.6f}, alvo FS={audit['estimator']['grey_expected']:.6f}; a restrição adicional é um momento ponderado, não igualdade imposta exatamente quando incompatível com os demais. Limites ativos={audit['beta_plus']['active_bounds']}."]
    text += ["", "Se o critério falha, ampliar a família não basta para atribuir confiabilidade às âncoras: separar a incerteza do alvo FS da flexibilidade da curva, medir prevalências internas em amostra independente e regularizar curvas monotônicas com pooling são prioridades. Não se recomenda substituir a isotônica supervisionada sem validação independente e por pessoa."]
    (ROOT / "gzcmd/M5_calibracao_ancoras.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
