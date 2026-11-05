"""S3: observed arrangement evaluation, aggregate outputs only, no API calls."""
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

from gzcmd.arranjo_corte import below_cutoff, sha256, summarize_actions
from gzcmd.intervals import wilson95
from gzcmd.llm_usage_cost import PRICES_USD_PER_MILLION, selected_cost
from gzcmd.ml_oof import pair_ids
from scripts.arranjo_corte_s1_simular import LOCAL, MODES, load_raw, verify_prereg
from scripts.arranjo_corte_s2_suplementar import responses
from scripts import arranjo_corte_orl_gzcmd as posterior

ARMS = ("corte_sozinho", "publicada", "B1", "humano_integral", "aposteriori",
        "P", "P-Platt", "A3", "P-PAR2-grey")
NAMES = {"corte_sozinho": "(i) Corte", "publicada": "Cascata sobre a base inteira", "B1": "B1",
         "humano_integral": "(ii) Humano integral", "aposteriori": "A posteriori (sem registro prévio)",
         "P": "P", "P-Platt": "P-Platt", "A3": "A3", "P-PAR2-grey": "P-PAR2-grey"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replay(frame, decisions):
    """Resolve only the selected LLM branch; UNSURE stays a human referral."""
    out = frame.copy()
    mask = out.action.eq("LLM_REVIEW")
    selected = pd.Series(out.index.map(decisions), index=out.index)
    require(not selected.loc[mask].isna().any(), "Selected LLM pair lacks an observed response")
    out["final"] = out.action.astype(str)
    out.loc[mask, "final"] = selected.loc[mask]
    require(out.final.isin(["MATCH", "NONMATCH", "UNSURE", "DEFERRED"]).all(),
            "Unknown or missing final decision")
    return out


def metrics(frame, mode, arm, scope, usage, costs):
    positive = frame.y.eq(1)
    accepted = frame.final.eq("MATCH")
    human = frame.final.isin(["UNSURE", "DEFERRED"])
    lost = frame.final.eq("NONMATCH")
    require((accepted.astype(int) + human.astype(int) + lost.astype(int)).eq(1).all(),
            "Final pair decisions do not partition the universe")
    n = int(positive.sum())
    auto = int((positive & accepted).sum())
    referred = int((positive & human).sum())
    rejected = int((positive & lost).sum())
    require(auto + referred + rejected == n, "Positive endpoint partition does not close")
    low, high = wilson95(auto, n)
    selected = set(frame.index[frame.action.eq("LLM_REVIEW")])
    require(selected <= costs.keys() and selected <= usage.keys(), "Incomplete selected token journal")
    return {"modo": mode, "braco": arm, "faixa": scope, "pares": len(frame), "vinculos": n,
            "aceitos": auto, "sens_auto": auto / n, "ic95_inf": low, "ic95_sup": high,
            "fp": int((~positive & accepted).sum()),
            "fp_abaixo": int((~positive & accepted & frame.abaixo).sum()),
            "perdidos": rejected, "vinc_humano": referred, "pares_humano": int(human.sum()),
            "pares_llm": len(selected), "chamadas_llm": sum(usage[p]["calls"] for p in selected),
            "tokens_entrada": sum(usage[p]["input_tokens"] for p in selected),
            "tokens_saida": sum(usage[p]["output_tokens"] for p in selected),
            "custo_llm_usd": sum(costs[p] for p in sorted(selected)),
            "sens_humano_05": (auto + .95 * referred) / n, "sem_resposta_llm": 0}


def par2_metrics(frame, mode, arm):
    subset = frame.loc[frame.par.eq(2)]
    require(len(subset) == 36, "PAR=2 primary endpoint requires exactly 36 links")
    counts = {"aceitos": int(subset.final.eq("MATCH").sum()),
              "humano": int(subset.final.isin(["UNSURE", "DEFERRED"]).sum()),
              "perdidos": int(subset.final.eq("NONMATCH").sum())}
    require(sum(counts.values()) == 36, "PAR=2 outcomes do not form a partition")
    row = {"modo": mode, "braco": arm, "vinculos": 36}
    for name, count in counts.items():
        low, high = wilson95(count, 36)
        row.update({name: count, name + "_proporcao": count / 36,
                    name + "_ic95_inf": low, name + "_ic95_sup": high})
    return row


def selected_usage(paths, ids):
    costs, provenance = selected_cost(paths, ids)
    usage = {}
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        selected = {r["pair_id"] for r in run["results"]} & ids
        for pid in selected:
            require(pid not in usage, "Duplicate selected usage across cohorts")
            journal = path.with_name(path.stem + "_journal") / f"{pid}.transport.json"
            value = json.loads(journal.read_text(encoding="utf-8"))
            require(isinstance(value["calls"], int) and value["calls"] >= 0, "Invalid call count")
            usage[pid] = value
    require(set(usage) == ids, "Selected usage coverage is incomplete")
    return costs, usage, provenance


def build_frames(data_root, mode, base, known, s1_summary, oof):
    frames = {}
    cut = base.copy()
    cut["action"] = np.where(cut.abaixo, "NONMATCH", "MATCH")
    cut["final"] = cut.action
    frames["corte_sozinho"] = cut
    human = cut.copy()
    human.loc[human.abaixo, ["action", "final"]] = "DEFERRED"
    frames["humano_integral"] = human
    published_path = data_root / f"data/funil_gzcmd_{mode}.csv"
    published = pd.read_csv(published_path, low_memory=False).set_index("pair_id")
    require(published.index.is_unique and set(published.index) == set(base.index),
            "Published cascade universe differs")
    published = published.loc[base.index]
    require(np.array_equal(published.y.to_numpy(), base.y.to_numpy()), "Published labels differ")
    original = base.copy()
    original["action"] = published.action
    original = replay(original, known)
    require(np.array_equal(original.final.to_numpy(), published.final.to_numpy()), "Published response replay differs")
    frames["publicada"] = original
    post = original.copy()
    post.loc[~post.abaixo, ["action", "final"]] = "MATCH"
    frames["aposteriori"] = post
    policy_hashes = {}
    for arm in ("B1", "P", "P-Platt", "A3", "P-PAR2-grey"):
        path = LOCAL / f"policy_{mode}_{arm}.csv"
        policy = pd.read_csv(path).set_index("pair_id")
        expected_ids = base.index if arm == "B1" else base.index[base.abaixo]
        require(policy.index.is_unique and set(policy.index) == set(expected_ids),
                f"S1 policy universe differs: {mode}/{arm}")
        observed = summarize_actions(policy, mode, arm).sort_values(["banda", "acao"]).reset_index(drop=True)
        expected = s1_summary.loc[s1_summary.modo.eq(mode) & s1_summary.braco.eq(arm)]
        expected = expected.sort_values(["banda", "acao"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
        score_arm = "P" if arm == "P-PAR2-grey" else arm
        score = dict(zip(oof["pair_id"], oof[score_arm]))
        np.testing.assert_allclose(policy.p_cal.to_numpy(), policy.index.map(score).to_numpy(),
                                   rtol=0, atol=1e-14)
        frame = cut.copy()
        frame.loc[policy.index, "action"] = policy.action
        frames[arm] = replay(frame, known)
        policy_hashes[path.name] = sha256(path)
    return frames, policy_hashes, published_path


def pct(value):
    return f"{100 * value:.1f}".replace(".", ",")


def interval(value, low, high):
    return f"{pct(value)} [{pct(low)}; {pct(high)}]"


def integer(value):
    return f"{int(value):,}".replace(",", ".")


def render_tex(table, primary, path):
    caption = (
        "Arranjo sequencial do corte do OpenRecLink e GZ-CMD v3.1, com Claude Haiku 5.5, instruções v4 "
        "e dossiê sem saídas do motor: aceitação automática, revisão e custo "
        "na base inteira e na área cinza. O padrão-ouro contém 247 vínculos na base e "
        "151 em $[5,9)$. PAR~=~1 é definido pelo corte, com viés de verificação; "
        "as classes PAR~=~1 e PAR~=~2 também participam do ajuste supervisionado, "
        "com viés de incorporação. Predições fora da amostra por REFREC não "
        "equivalem a validação externa desse padrão-ouro."
    )
    header = (r"Braço & $V_A$ & $S_A$ (IC 95\%) & FP (abaixo) & $V_N$ & $H$ ($V_H$) "
              r"& LLM & Chamadas & US\$ & $S_H$ (5\%) \\")
    lines = [r"% Gerada por scripts/arranjo_corte_s3_avaliar.py; não editar manualmente.",
             r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{longtable}{lrrrrrrrrr}", r"\caption{" + caption + "}",
             r"\label{tab:cap8-arranjo-corte}\\", r"\toprule", header,
             r"\midrule", r"\endfirsthead", r"\toprule", header, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endlastfoot"]
    for mode in MODES:
        for scope in ("TODAS", "[5,9)"):
            label = "Vigilância" if mode == "vigilancia" else "Confirmação"
            label += ": base inteira" if scope == "TODAS" else ": área cinza $[5,9)$"
            lines.append(r"\multicolumn{10}{l}{\textbf{" + label + r"}} \\")
            for arm in ARMS:
                r = table.loc[table.modo.eq(mode) & table.braco.eq(arm) & table.faixa.eq(scope)].iloc[0]
                cells = [NAMES[arm], integer(r.aceitos), interval(r.sens_auto, r.ic95_inf, r.ic95_sup),
                         f"{integer(r.fp)} ({integer(r.fp_abaixo)})", integer(r.perdidos),
                         f"{'até ' if arm == 'humano_integral' and scope == 'TODAS' else ''}{integer(r.pares_humano)} ({integer(r.vinc_humano)})", integer(r.pares_llm),
                         integer(r.chamadas_llm), f"{r.custo_llm_usd:.4f}".replace(".", ","),
                         pct(r.sens_humano_05)]
                lines.append(" & ".join(cells) + r" \\")
            if mode != MODES[-1] or scope != "[5,9)":
                lines.append(r"\midrule")
    lines += [r"\end{longtable}",
              r"\noindent\parbox{\linewidth}{\footnotesize $V_A$: vínculos aceitos automaticamente, inclusive MATCH do modelo de linguagem (LLM); $V_N$: vínculos perdidos por NONMATCH; $H$: potenciais pares ao revisor humano, com vínculos $V_H$ entre parênteses. UNSURE do LLM segue ao humano uma única vez. LLM e humano se sobrepõem nesse encaminhamento, sem somar novamente sua carga humana. $S_A=V_A/n$; IC de Wilson de 95\% descritivo, sem correção pela dependência entre potenciais pares ou pelos vieses do gabarito. $S_H=(V_A+0{,}95V_H)/n$ é cenário de erro humano de 5\%, não resultado humano observado. Custos usam somente os registros de consumo dos potenciais pares selecionados por cada braço e faixa, com tarifas históricas por tokens, não faturas. O a posteriori reutiliza a cascata sobre a base inteira sem recalibração. Fonte: \texttt{data/arranjo\_corte\_metricas.csv} e \texttt{data/arranjo\_corte\_par2.csv}.}",
              r"\begin{longtable}{lrrr}",
              r"\caption{Partição dos 36 vínculos PAR~=~2 abaixo do corte. Cada célula mostra a contagem e seu IC de Wilson de 95\%, em porcentagem. As ressalvas sobre o gabarito e a dependência entre pares permanecem.}",
              r"\label{tab:cap8-arranjo-corte-par2}\\", r"\toprule",
              r"Braço & Aceitos (IC 95\%) & Humano (IC 95\%) & Perdidos (IC 95\%) \\",
              r"\midrule"]
    for mode in MODES:
        lines.append(r"\multicolumn{4}{l}{\textbf{" + ("Vigilância" if mode == "vigilancia" else "Confirmação") + r"}} \\")
        for arm in ARMS:
            row = primary.loc[primary.modo.eq(mode) & primary.braco.eq(arm)].iloc[0]
            cells = [NAMES[arm]]
            for name in ("aceitos", "humano", "perdidos"):
                cells.append(f"{int(row[name])} [{pct(row[name + '_ic95_inf'])}; {pct(row[name + '_ic95_sup'])}]")
            lines.append(" & ".join(cells) + r" \\")
        if mode != MODES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(data_root, prereg_hash):
    specification, commit = verify_prereg(data_root, prereg_hash)
    s1_meta = json.loads((ROOT / "data/arranjo_corte_s1_metadata.json").read_text(encoding="utf-8"))
    note = specification["amendments"][-1]
    require(note["original_S1_prereg_sha256"] == s1_meta["prereg_sha256"], "S1 preregistration lineage differs")
    rates_path = ROOT / "data/arranjo_corte_taxas_crossfit.csv"
    require(sha256(rates_path) == s1_meta["frozen_rates_sha256"] == note["frozen_rates_sha256"],
            "Frozen reviewer rates changed")
    require(sha256(ROOT / "data/arranjo_corte_s1_resumo.csv") == s1_meta["summary_sha256"], "S1 aggregate changed")
    raw = load_raw(data_root / "data/COMPARADORSEMIDENT.csv")
    below = below_cutoff(raw)
    par = pd.to_numeric(raw.PAR, errors="raise").astype(int)
    nota = pd.to_numeric(raw["nota final"].str.replace(",", ".", regex=False), errors="raise")
    base = pd.DataFrame({"pair_id": pair_ids(raw), "par": par, "y": par.isin([1, 2]).astype(int),
                         "nota": nota, "abaixo": below}).set_index("pair_id")
    require(int(base.y.sum()) == 247 and int(base.loc[base.nota.ge(5) & base.nota.lt(9), "y"].sum()) == 151,
            "Expected positive denominators 247/151 not reproduced")
    require(sha256(ROOT / "scripts/arranjo_corte_orl_gzcmd.py") == sha256(data_root / "scripts/arranjo_corte_orl_gzcmd.py"),
            "A posteriori script differs from the supplied original")
    aposteriori = posterior.execute(LOCAL / "aposteriori_s3")
    posterior_metadata = json.loads((LOCAL / "aposteriori_s3/meta.json").read_text(encoding="utf-8"))
    aposteriori.to_csv(ROOT / "data/arranjo_corte_aposteriori.csv", index=False)
    s1_summary = pd.read_csv(ROOT / "data/arranjo_corte_s1_resumo.csv")
    oof = np.load(LOCAL / "oof.npz", allow_pickle=False)
    rows, primary_rows, provenance, policies, input_hashes = [], [], [], {}, {}
    published_expected = {"vigilancia": (194, 34, 111, 347), "confirmacao": (132, 0, 127, 44)}
    post_expected = {"vigilancia": (221, 34, 9, 90, 330), "confirmacao": (219, 0, 17, 51, 25)}
    for mode in MODES:
        known, paths = responses(data_root, mode, include_supplement=True)
        frames, policy_hashes, published_path = build_frames(data_root, mode, base, known, s1_summary, oof)
        policies.update(policy_hashes)
        input_hashes[str(published_path)] = sha256(published_path)
        needed = set().union(*(set(f.index[f.action.eq("LLM_REVIEW")]) for f in frames.values()))
        costs, usage, sources = selected_usage(paths, needed)
        provenance.append({"modo": mode, "sources": sources})
        for arm in ARMS:
            frame = frames[arm]
            primary_rows.append(par2_metrics(frame, mode, arm))
            for scope in ("TODAS", "[5,9)"):
                selected = frame if scope == "TODAS" else frame.loc[frame.nota.ge(5) & frame.nota.lt(9)]
                result = metrics(selected, mode, arm, scope, usage, costs)
                rows.append(result)
                if scope == "TODAS" and arm == "publicada":
                    require(tuple(result[k] for k in ("aceitos", "fp", "pares_humano", "pares_llm")) == published_expected[mode],
                            "Published cascade reference was not reproduced")
                if scope == "TODAS" and arm == "aposteriori":
                    require(tuple(result[k] for k in ("aceitos", "fp", "perdidos", "pares_humano", "pares_llm")) == post_expected[mode],
                            "A posteriori reference was not reproduced")
                    old = aposteriori.loc[aposteriori.modo.eq(mode) & aposteriori.faixa.eq(scope)
                                         & aposteriori.arranjo.eq("corte_mais_cascata")].iloc[0]
                    require(result["aceitos"] == old.vinculos_automaticos
                            and result["perdidos"] == old.vinculos_rejeitados_automaticamente
                            and abs(result["custo_llm_usd"] - old.custo_llm_usd) < 1e-10,
                            "Independent a posteriori regeneration differs")
    table, primary = pd.DataFrame(rows), pd.DataFrame(primary_rows)
    output = ROOT / "data/arranjo_corte_metricas.csv"
    par2_output = ROOT / "data/arranjo_corte_par2.csv"
    tex = ROOT / "text/latex/tables/tab_cap8_arranjo_corte.tex"
    table.to_csv(output, index=False)
    primary.to_csv(par2_output, index=False)
    render_tex(table, primary, tex)
    metadata = {"stage": "S3_observed_evaluation_no_API_calls", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "prereg_commit": commit, "prereg_sha256": prereg_hash,
                "S1_prereg_commit": s1_meta["prereg_commit"], "S1_prereg_sha256": s1_meta["prereg_sha256"],
                "frozen_rates_sha256": sha256(rates_path), "policy_sha256": policies,
                "registered_input_sha256": specification["sha256"], "published_input_sha256": input_hashes,
                "aposteriori_regeneration": {key: posterior_metadata[key] for key in (
                    "conferencias", "cortes_SCORE_por_PASSO", "avisos_proveniencia", "fontes", "custos_journals")},
                "cost_sources": provenance, "prices_usd_per_million": PRICES_USD_PER_MILLION,
                "cost_note": "Historical tariffs applied to recorded tokens of selected pairs only; not an invoice",
                "human_outcomes_observed": False, "human_error_scenario": .05,
                "verification_bias": "PAR=1 is defined by the ORL cutoff; results are conditional on the available reference",
                "incorporation_bias": "Reference labels also participate in supervised calibration and reviewer-rate estimation; cross-fitting is not external gold-standard validation",
                "wilson_note": "Descriptive 95% intervals; no correction for pair dependence or reference biases",
                "new_responses_reestimate_rates": False, "chapter_tex_modified": False,
                "code_sha256": {name: sha256(ROOT / name) for name in (
                    "scripts/arranjo_corte_s3_avaliar.py", "scripts/arranjo_corte_orl_gzcmd.py",
                    "scripts/arranjo_corte_s2_suplementar.py", "tests/test_arranjo_corte_s3.py")},
                "output_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
                    output, par2_output, tex, ROOT / "data/arranjo_corte_aposteriori.csv")}}
    (ROOT / "data/arranjo_corte_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    printable = table.copy()
    printable["sens_auto_IC95_pct"] = [interval(r.sens_auto, r.ic95_inf, r.ic95_sup) for r in table.itertuples()]
    printable["H_pares_vinc"] = [f"{r.pares_humano} ({r.vinc_humano})" for r in table.itertuples()]
    printable["sens_H05_pct"] = table.sens_humano_05.map(pct)
    print("\nS3_TABELA_COMPLETA")
    print(printable[["modo", "faixa", "braco", "aceitos", "sens_auto_IC95_pct", "fp", "fp_abaixo",
                     "perdidos", "H_pares_vinc", "pares_llm", "chamadas_llm", "custo_llm_usd", "sens_H05_pct"]].to_string(index=False))
    print("\nS3_PAR2_36")
    p = primary[["modo", "braco"]].copy()
    for name in ("aceitos", "humano", "perdidos"):
        p[name + "_IC95_pct"] = [f"{int(r[name])} [{pct(r[name + '_ic95_inf'])}; {pct(r[name + '_ic95_sup'])}]"
                                  for _, r in primary.iterrows()]
    print(p.to_string(index=False))
    print("FROZEN_RATES_UNCHANGED", sha256(rates_path))
    return table, primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prereg-sha256", required=True)
    args = parser.parse_args()
    evaluate(args.data_root.resolve(), args.prereg_sha256)


if __name__ == "__main__":
    main()
