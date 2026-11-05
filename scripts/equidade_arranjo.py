"""Study 5 subgroup analysis of P, reusing equidade_v31 without modifying it.

Automatic acceptance includes MATCH from the engine or the LLM; human referrals
remain nonaccepted. No training, new threshold fitting or API calls occur here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import below_cutoff, sha256
from gzcmd.ml_oof import pair_ids
from scripts import equidade_v31 as legacy
from scripts.arranjo_corte_s1_simular import LOCAL, MODES, load_raw, verify_prereg
from scripts.arranjo_corte_s2_suplementar import responses
from scripts.arranjo_corte_s3_avaliar import build_frames, require

DECISIONS = ("arranjo_P_vigilancia", "arranjo_P_confirmacao",
             "cascata_v31_vigilancia", "cascata_v31_confirmacao")
LABELS = {"arranjo_P_vigilancia": "Arranjo P (vigil.)",
          "arranjo_P_confirmacao": "Arranjo P (confirm.)",
          "cascata_v31_vigilancia": "Base inteira v3.1 (vigil., comparação)",
          "cascata_v31_confirmacao": "Base inteira v3.1 (confirm., comparação)"}
TEST_COLUMNS = ["decisao", "tipo", "dimension", "metric", "group_a", "group_b", "test", "n1", "n2",
                "num_a", "num_b", "p_value_raw", "p_value_fdr", "reject_fdr", "odds_ratio", "or_ci_low",
                "or_ci_high", "rate_a", "rate_b", "rate_diff", "rate_diff_ci_low", "rate_diff_ci_high"]
SORT_KEYS = {"global": ["decisao"], "group_metrics": ["decisao", "dimension", "group"],
             "pairwise_tests": ["decisao", "dimension", "metric", "group_a", "group_b"],
             "maxdiff": ["decisao", "dimension", "metric"]}


def analyze(frame, decision, mode, final):
    """Adapter for the legacy human-column naming contract; no global mutation."""
    temporary = frame.copy()
    alias = f"cascata_v31_{mode}"
    temporary[alias] = final.final.eq("MATCH").astype(int)
    temporary[f"humano_{mode}"] = final.final.isin(["UNSURE", "DEFERRED"]).astype(int)
    glob, groups, tests = legacy.analisar_decisao(temporary, alias)
    for table in (glob, groups, tests):
        table["decisao"] = decision
    return glob, groups, tests


def coverage(frame, decision, final):
    """Account for every pair, including groups excluded by the original analysis."""
    work = frame.copy()
    work["accepted"] = final.final.eq("MATCH").astype(int)
    records = []
    for dim in legacy.DIMENSOES:
        eligible = set(legacy.elegiveis(work, dim))
        for group, subset in work.groupby(dim, dropna=False, observed=True, sort=True):
            records.append({"decisao": decision, "tipo": legacy.TIPO[dim], "dimension": dim,
                            "group": str(group), "incluido_analise": str(group) in eligible,
                            **legacy.contagens(subset, "accepted")})
    return pd.DataFrame(records)


def check_reference(outputs, data_root):
    """Require numerical reproduction of all four existing cascade aggregates."""
    references = {}
    for name, keys in SORT_KEYS.items():
        path = data_root / f"data/equidade_v31_{name}.csv"
        expected = pd.read_csv(path)
        expected = expected.loc[expected.decisao.isin(DECISIONS[2:])]
        actual = outputs[name].loc[outputs[name].decisao.isin(DECISIONS[2:])]
        pd.testing.assert_frame_equal(
            actual[expected.columns].sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
            check_dtype=False, rtol=1e-12, atol=1e-12)
        references[str(path)] = sha256(path)
    return references


def paired_groups(groups):
    fields = ["n_pred_pos", "tp", "fp", "fn", "tn", "sens", "sens_low", "sens_high", "ppv",
              "ppv_low", "ppv_high", "espec", "espec_low", "espec_high", "fpr", "n_humano", "n_humano_true"]
    tables = []
    for mode in MODES:
        a = groups.loc[groups.decisao.eq(f"arranjo_P_{mode}")].set_index(["tipo", "dimension", "group"])
        b = groups.loc[groups.decisao.eq(f"cascata_v31_{mode}")].set_index(["tipo", "dimension", "group"])
        pd.testing.assert_frame_equal(a[["n_total", "n_true"]], b[["n_total", "n_true"]])
        result = a[["n_total", "n_true"]].join(a[fields].add_suffix("_P")).join(b[fields].add_suffix("_publicada"))
        result["modo"] = mode
        tables.append(result.reset_index())
    return pd.concat(tables, ignore_index=True)


def table_groups(groups, mode, path):
    # The renderer's first slot is called xgb_sens6; only presentation aliases
    # change here. The analysis always uses the actual P decisions from S3.
    a = groups.loc[groups.decisao.eq(f"arranjo_P_{mode}")].copy()
    b = groups.loc[groups.decisao.eq(f"cascata_v31_{mode}")].copy()
    a["decisao"], b["decisao"] = "xgb_sens6", "cascata_v31_vigilancia"
    legacy.tabela_grupos(pd.concat([a, b], ignore_index=True), path)
    text = path.read_text(encoding="utf-8")
    text = text.replace("XGBoost com limiar de sensibilidade equivalente ao corte 6{,}0",
                        "arranjo corte OpenRecLink com o braço P abaixo do corte")
    text = text.replace("XGBoost", "Arranjo P")
    text = text.replace("cascata v3.1 (vigil\\^ancia)", "cascata v3.1 (" + (
        "vigil\\^ancia" if mode == "vigilancia" else "confirma\\c{c}\\~ao") + ")")
    text = text.replace("tab:qw3-group-metrics", f"tab:equidade-arranjo-grupos-{mode}")
    text = text.replace("cascata v3.1 (", "cascata v3.1 sobre a base inteira, comparação (")
    text = text.replace("{Cascata v3.1}", "{Base inteira (comparação)}")
    text = text.replace(r"\begin{tabular}", r"\resizebox{\linewidth}{!}{%" + "\n" + r"\begin{tabular}")
    text = text.replace(r"\end{tabular}", r"\end{tabular}}")
    text = text.replace(r"\end{table}", r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize "
                        r"A aceitação automática inclui MATCH do motor ou do modelo de linguagem; UNSURE e DEFERRED seguem ao revisor humano e contam como não aceitos. Grupos MISSING e com menos de 30 potenciais pares não integram os testes, como na análise anterior. A faixa etária exclui 403 potenciais pares, com um vínculo, sem idade classificada; a contagem integral consta no CSV de cobertura. Intervalos e testes não corrigem a dependência por REFREC."
                        r"\end{minipage}\end{table}")
    path.write_text(text, encoding="utf-8")


def table_summary(glob, maximum, tests, path):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\caption{Equidade por decisão: desempenho global e diferença máxima entre subgrupos, em pontos percentuais, considerando grupos com denominador $\geq 30$. A última coluna superior conta rejeições após Benjamini-Hochberg sobre a família de comparações de cada decisão.}",
             r"\label{tab:equidade-arranjo}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Decisão & Pred.$+$ & VP & FP & FN & Sens. & VPP & Espec. & Rejeições \\", r"\midrule"]
    for decision in DECISIONS:
        row = glob.loc[glob.decisao.eq(decision)].iloc[0]
        te = tests.loc[tests.decisao.eq(decision)]
        cells = [LABELS[decision], *[legacy.fmt_int(row[k]) for k in ("n_pred_pos", "tp", "fp", "fn")],
                 *[legacy.fmt_pct(row[k]) for k in ("sens", "ppv", "espec")],
                 f"{int(te.reject_fdr.sum())}/{len(te)}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\par\medskip", r"\resizebox{\textwidth}{!}{%",
              r"\begin{tabular}{llrrrr}", r"\toprule",
              r"Dimensão & Métrica & P vigil. & P confirm. & Base inteira vigil. (comparação) & Base inteira confirm. (comparação) \\", r"\midrule"]
    for dims in (legacy.DEMOGRAFICAS, legacy.OPERACIONAIS):
        for dim in dims:
            for metric, label in (("sens", "Sens."), ("ppv", "VPP")):
                cells = [legacy.NOMES_DIM[dim], label]
                for decision in DECISIONS:
                    row = maximum.loc[maximum.decisao.eq(decision) & maximum.dimension.eq(dim)
                                      & maximum.metric.eq(metric)].iloc[0]
                    cells.append(legacy.fmt_pct(row.max_diff))
                lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}\footnotesize FN inclui todos os vínculos não aceitos automaticamente, inclusive os encaminhados ao revisor humano; não corresponde apenas a NONMATCH. BH é aplicada separadamente por decisão aos valores de $p$ definidos, preservando as omissões por denominador zero. Ausência de rejeição não demonstra paridade; pares com o mesmo REFREC são dependentes e os testes não ajustam essa correlação. A comparação entre decisões não é um teste formal. O gabarito compartilha o corte e os dados de treinamento, com ressalvas de verificação e incorporação. A referência histórica de 7.452 preditos positivos pertence ao corte 6,0 da nota final, não ao corte SCORE do OpenRecLink nem ao braço P.\end{minipage}",
              r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(data_root, prereg_hash):
    specification, prereg_commit = verify_prereg(data_root, prereg_hash)
    s3 = json.loads((ROOT / "data/arranjo_corte_metadata.json").read_text(encoding="utf-8"))
    require(s3["prereg_sha256"] == prereg_hash, "S3 preregistration differs")
    require(sha256(ROOT / "data/arranjo_corte_taxas_crossfit.csv") == s3["frozen_rates_sha256"], "Frozen rates changed")
    for name, value in s3["code_sha256"].items():
        require(sha256(ROOT / name) == value, "S3 code fingerprint changed")
    frame, meta = legacy.build_frame(data_root / "data/COMPARADORSEMIDENT.csv")
    require(not meta["raca_cor_presente"], "Race/color column requires author review")
    frame["bandeira"] = frame.bandeira.astype("string").replace(
        {"controle": "demais_situacoes", "tb_origem": "encerrada_obito"})
    frame = frame.set_index("pair_id", drop=False)
    require(frame.index.is_unique and len(frame) == 61696 and int(frame.y_true.sum()) == 247, "Fairness universe differs")
    require(int(frame.y_pred.sum()) == 7452, "Historical comparator threshold 6.0 was not reproduced")
    raw = load_raw(data_root / "data/COMPARADORSEMIDENT.csv")
    par = pd.to_numeric(raw.PAR).astype(int)
    base = pd.DataFrame({"pair_id": pair_ids(raw), "par": par, "y": par.isin([1, 2]).astype(int),
                         "nota": pd.to_numeric(raw["nota final"].str.replace(",", ".", regex=False)),
                         "abaixo": below_cutoff(raw)}).set_index("pair_id")
    require(set(frame.index) == set(base.index), "Fairness pair IDs differ from S3")
    frame = frame.loc[base.index]
    require(np.array_equal(frame.y_true.to_numpy(), base.y.to_numpy()), "Fairness labels differ from S3")
    summary = pd.read_csv(ROOT / "data/arranjo_corte_s1_resumo.csv")
    oof = np.load(LOCAL / "oof.npz", allow_pickle=False)
    globs, groups, tests, cover = [], [], [], []
    policies, response_sources = {}, {}
    for mode in MODES:
        known, paths = responses(data_root, mode, include_supplement=True)
        finals, hashes, _ = build_frames(data_root, mode, base, known, summary, oof)
        policies.update(hashes)
        response_sources[mode] = {str(p): sha256(p) for p in paths}
        for decision, final in ((f"arranjo_P_{mode}", finals["P"]),
                                (f"cascata_v31_{mode}", finals["publicada"])):
            g, gr, te = analyze(frame, decision, mode, final)
            globs.append(g)
            groups.append(gr)
            tests.append(te)
            cover.append(coverage(frame, decision, final))
    outputs = {"global": pd.concat(globs, ignore_index=True),
               "group_metrics": pd.concat(groups, ignore_index=True),
               "pairwise_tests": pd.concat(tests, ignore_index=True)[TEST_COLUMNS]}
    outputs["maxdiff"] = legacy.diferencas_maximas(outputs["group_metrics"])
    references = check_reference(outputs, data_root)
    coverage_table = pd.concat(cover, ignore_index=True)
    closed = coverage_table.groupby(["decisao", "dimension"])[["n_total", "n_true"]].sum()
    require(closed.n_total.eq(61696).all() and closed.n_true.eq(247).all(), "Subgroup accounting does not close")
    comparison = paired_groups(outputs["group_metrics"])
    for name, table in {**outputs, "cobertura": coverage_table, "comparacao": comparison}.items():
        table.to_csv(ROOT / f"data/equidade_arranjo_{name}.csv", index=False)
    tables = ROOT / "text/latex/tables"
    table_summary(outputs["global"], outputs["maxdiff"], outputs["pairwise_tests"], tables / "tab_equidade_arranjo.tex")
    for mode in MODES:
        table_groups(outputs["group_metrics"], mode, tables / f"tab_equidade_arranjo_grupos_{mode}.tex")
        for decision in (f"arranjo_P_{mode}", f"cascata_v31_{mode}"):
            path = tables / f"tab_equidade_arranjo_rejeicoes_{decision}.tex"
            legacy.tabela_rejeicoes(outputs["pairwise_tests"], path, decisao=decision,
                                    label=f"tab:equidade-arranjo-rejeicoes-{decision.replace('_', '-')}",
                                    descr=LABELS[decision] + ", aceitação automática")
            text = path.read_text(encoding="utf-8").replace(r"\small", r"\footnotesize\setlength{\tabcolsep}{3pt}")
            text = text.replace(r"\end{table}", r"\par\smallskip{\footnotesize BH por decisão, com omissão dos valores indefinidos por denominador zero. Os testes não corrigem a dependência entre potenciais pares que compartilham REFREC.}\end{table}")
            path.write_text(text, encoding="utf-8")
    missing_age = coverage_table.loc[coverage_table.decisao.eq(DECISIONS[0])
                                    & coverage_table.dimension.eq("faixa_idade") & ~coverage_table.incluido_analise]
    manifest = {"analysis_name": "estudo5_equidade_arranjo_P", "created_utc": datetime.now(timezone.utc).isoformat(),
                "seed": "n/a: no random sampling, training, API calls or new thresholds",
                "rows": 61696, "links": 247, "raca_cor_presente": False,
                "alpha": legacy.ALPHA, "n_min": legacy.N_MIN, "decisoes": list(DECISIONS),
                "dimensoes_demograficas": legacy.DEMOGRAFICAS, "dimensoes_operacionais": legacy.OPERACIONAIS,
                "prereg_commit": prereg_commit, "prereg_sha256": prereg_hash,
                "frozen_rates_sha256": s3["frozen_rates_sha256"],
                "registered_input_sha256": specification["sha256"], "policy_sha256": policies,
                "response_sources_sha256": response_sources, "reference_sha256": references,
                "code_sha256": {name: sha256(ROOT / name) for name in (
                    "scripts/equidade_arranjo.py", "scripts/equidade_v31.py", "scripts/qw3_fairness_analysis.py",
                    "tests/test_equidade_arranjo.py")},
                "published_reference_reproduced": True, "historical_corte6_predictions": 7452,
                "missing_age_pairs": int(missing_age.n_total.sum()), "missing_age_links": int(missing_age.n_true.sum()),
                "missing_group_rule": "Original inference excludes MISSING and groups with n<30; coverage CSV includes every group to close 61696/247",
                "BH_rule": "Same legacy routine; correction per decision over finite p-values; no pooling across decisions",
                "REFREC_dependence": "Pairwise tests do not adjust within-REFREC correlation; p-values may be optimistic",
                "human_outcomes_observed": False, "chapter_tex_modified": False}
    (ROOT / "data/equidade_arranjo_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    display = comparison[["modo", "dimension", "group", "n_total", "n_true"]].copy()
    for method in ("P", "publicada"):
        for name in ("sens", "ppv"):
            display[f"{name}_{method}_pct"] = comparison[f"{name}_{method}"].map(
                lambda v: "n.a." if pd.isna(v) else f"{100*v:.1f}".replace(".", ","))
        display[f"fp_{method}"] = comparison[f"fp_{method}"]
        display[f"fpr_{method}_pct"] = comparison[f"fpr_{method}"].map(lambda v: f"{100*v:.3f}".replace(".", ","))
        display[f"humano_{method}"] = comparison[f"n_humano_{method}"]
    print("EQUIDADE_P_PUBLICADA_LADO_A_LADO")
    print(display.to_string(index=False))
    te = outputs["pairwise_tests"]
    print("REJEICOES_BH")
    print(te.loc[te.reject_fdr, ["decisao", "tipo", "dimension", "metric", "group_a", "group_b",
                               "test", "p_value_raw", "p_value_fdr", "rate_diff", "odds_ratio"]].to_string(index=False))
    counts = te.groupby("decisao").agg(testes=("reject_fdr", "size"), rejeicoes=("reject_fdr", "sum"),
                                      efetivos=("p_value_raw", "count"))
    print("RESUMO_BH\n" + counts.to_string())
    print("GLOBAL\n" + outputs["global"].to_string(index=False))
    print("SUBGROUP_ACCOUNTING_CLOSED", len(closed), "HISTORICAL_CORTE6", 7452,
          "MISSING_AGE", manifest["missing_age_pairs"], manifest["missing_age_links"])
    return outputs, coverage_table, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prereg-sha256", required=True)
    args = parser.parse_args()
    run(args.data_root.resolve(), args.prereg_sha256)


if __name__ == "__main__":
    main()
