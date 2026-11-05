"""Filtro exploratório offline: protocolo, cobertura cega, desfechos agregados."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gzcmd.arranjo_corte import MODEL, below_cutoff, sha256
from gzcmd.arranjo_robustez import validate_dossiers
from gzcmd.intervals import wilson95
from gzcmd.llm_dossier import build_dossiers
from gzcmd.ml_oof import pair_ids
from gzcmd.runner import LoadConfig, load_comparador_csv
from scripts.arranjo_corte_s1_simular import (
    MODES, PROMPT, cohort_sources, validated_responses,
)
from scripts.comparador_mesma_carga import (
    align_original, load_raw, numeric, text_sha256,
)
from scripts.run_exp2_ablacao_dossie import DossierView
from scripts.run_exp2_sem_model_outputs import audit_view

PROTOCOL = ROOT / "gzcmd/protocolo_filtro_llm.yaml"
LOCAL = ROOT / ".local/filtro_llm"
KS = (17, 29, 58, 96, 121, 125, 440)
RUN_DIRS = (
    "anthropic_sem_mo_v4_arranjo_corte",
    "anthropic_sem_mo_v4_arranjo_corte_supl",
    "anthropic_sem_mo_v4_arranjo_robustez",
)
OUTPUTS = ("filtro_llm_cobertura.csv", "filtro_llm.csv", "filtro_llm_metadata.json")
RESULT_COLUMNS = [
    "modo", "k", "nonmatch", "H", "par2_topk", "par2_recuperados",
    "par2_descartados", "nao_vinculos_descartados", "vpn_nonmatch",
    "vpn_ic95_inf", "vpn_ic95_sup", "o3_par2_mesma_carga", "diferenca_par2",
    "par2_esperados_humano_05", "o3_par2_esperados_humano_05",
    "sens_humano_perfeito", "sens_humano_05", "o3_sens_humano_perfeito",
    "o3_sens_humano_05", "diferenca_sens_humano_05",
]


def resolve(data_root, name):
    local = ROOT / name
    return local if local.exists() else Path(data_root) / name


def frozen_sources(data_root):
    spec = yaml.safe_load((ROOT / "gzcmd/error_rates_by_band_v4.yaml").read_text(encoding="utf-8"))
    names = [s["path"] for s in spec["sources"] if s["model"] == MODEL]
    base = ROOT if all((ROOT / name).exists() for name in names) else Path(data_root)
    return base, cohort_sources(base)


def input_paths(data_root):
    base, sources = frozen_sources(data_root)
    names = [
        "data/COMPARADORSEMIDENT.csv", "data/pcal_ml_oof.csv",
        "data/pcal_ml_oof.json", "data/comparador_mesma_carga_curva.csv",
        "gzcmd/error_rates_by_band_v4.yaml", "gzcmd/prereg_arranjo_corte.yaml",
        "gzcmd/gzcmd_v4_llm_prompt.md", "gzcmd/arranjo_robustez.py",
        "gzcmd/arranjo_corte.py", "gzcmd/llm_dossier.py", "gzcmd/loader.py",
        "gzcmd/ml_oof.py", "scripts/arranjo_corte_s1_simular.py",
        "scripts/comparador_mesma_carga.py", "scripts/run_exp2_ablacao_dossie.py",
        "scripts/run_exp2_sem_model_outputs.py", "gzcmd/llm_usage_cost.py",
    ]
    paths = {name: resolve(data_root, name) for name in names}
    runs = {s["path"]: base / s["path"] for s in sources}
    for directory in RUN_DIRS:
        for mode in MODES:
            name = f"data/sprint3b/{directory}/exp2_llm_sync_{mode}.json"
            path = resolve(data_root, name)
            if path.exists():
                runs[name] = path
    paths.update(runs)
    for name, path in runs.items():
        identity = path.with_name(path.stem + "_journal") / "identity.json"
        if identity.exists():
            paths[str(Path(name).with_name(path.stem + "_journal") / "identity.json").replace("\\", "/")] = identity
    return paths


def register_inputs(data_root):
    spec = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if spec["input_sha256"]:
        raise ValueError("Protocol already fingerprinted; do not overwrite")
    spec["input_sha256"] = {name: sha256(path) for name, path in input_paths(data_root).items()}
    PROTOCOL.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    print("PROTOCOL_SHA256", sha256(PROTOCOL))


def verify_protocol(data_root):
    spec = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    actual = {name: sha256(path) for name, path in input_paths(data_root).items()}
    if not actual or spec["input_sha256"] != actual:
        raise ValueError("Protocol input SHA256 mismatch")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "gzcmd/protocolo_filtro_llm.yaml"], cwd=ROOT, text=True,
    ).strip()
    if not commit:
        raise ValueError("Commit and push protocol before coverage/outcomes")
    recorded = subprocess.check_output(["git", "show", commit + ":gzcmd/protocolo_filtro_llm.yaml"], cwd=ROOT)
    if yaml.safe_load(recorded) != spec:
        raise ValueError("Protocol differs from committed version")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/pq/arranjo-corte"], cwd=ROOT, check=True)
    return spec, commit


def select_responses(frozen, extras):
    """P's frozen decisions win; extras ordered S2, supplement, robustness."""
    selected = {}
    observed = {}
    for source in (frozen, *extras):
        for pid, decision in source.items():
            if decision not in {"MATCH", "NONMATCH", "UNSURE"}:
                raise ValueError("Invalid reviewer decision")
            observed.setdefault(pid, set()).add(decision)
            selected.setdefault(pid, decision)
    return selected, {pid for pid, decisions in observed.items() if len(decisions) > 1}


def coverage_row(mode, top_ids, selected, conflicts, k):
    wanted = list(top_ids[:k])
    if len(wanted) != k or len(set(wanted)) != k:
        raise ValueError("Queue shorter than k or not unique")
    n = sum(pid in selected for pid in wanted)
    missing = k - n
    return {
        "modo": mode, "k": k, "respostas_validas": n, "pendentes": missing,
        "cobertura": n / k, "completa": missing == 0,
        "escalate": missing > 0 and k <= 125,
        "conflitos_decisao_topk": len(set(wanted) & conflicts),
        "custo_pendentes_usd_sem_cache": round(missing * .00316, 8),
    }


def require_complete(top_ids, decisions):
    if not set(top_ids) <= decisions.keys():
        raise ValueError("STOP: incomplete coverage; outcomes forbidden")


def outcomes(mode, top_ids, decisions, labels, o3_counts):
    require_complete(top_ids, decisions)
    k = len(top_ids)
    rejected = [pid for pid in top_ids if decisions[pid] == "NONMATCH"]
    if any(labels[pid] not in {0, 2} for pid in top_ids):
        raise ValueError("Below-cutoff labels must be PAR=0 or PAR=2")
    h = k - len(rejected)
    positives = sum(labels[pid] == 2 for pid in top_ids)
    discarded = sum(labels[pid] == 2 for pid in rejected)
    recovered = positives - discarded
    nonlinks = len(rejected) - discarded
    baseline = int(o3_counts[h]) if h else 0
    low, high = wilson95(nonlinks, len(rejected)) if rejected else (None, None)
    return {
        "modo": mode, "k": k, "nonmatch": len(rejected), "H": h,
        "par2_topk": positives, "par2_recuperados": recovered,
        "par2_descartados": discarded, "nao_vinculos_descartados": nonlinks,
        "vpn_nonmatch": nonlinks / len(rejected) if rejected else None,
        "vpn_ic95_inf": low, "vpn_ic95_sup": high,
        "o3_par2_mesma_carga": baseline, "diferenca_par2": recovered - baseline,
        "par2_esperados_humano_05": .95 * recovered,
        "o3_par2_esperados_humano_05": .95 * baseline,
        "sens_humano_perfeito": (211 + recovered) / 247,
        "sens_humano_05": (211 + .95 * recovered) / 247,
        "o3_sens_humano_perfeito": (211 + baseline) / 247,
        "o3_sens_humano_05": (211 + .95 * baseline) / 247,
        "diferenca_sens_humano_05": .95 * (recovered - baseline) / 247,
    }


def interpret(differences):
    if not differences:
        return "indisponivel"
    if min(differences) >= 0 and max(differences) > 0:
        return "ajuda"
    if min(differences) < 0 and max(differences) <= 0:
        return "prejudica"
    return "misto"


def collect(data_root):
    """No PAR evaluation, no fit, no policy run: reconstruct feature-only dossiers."""
    source = resolve(data_root, "data/COMPARADORSEMIDENT.csv")
    # Do not parse PAR in the queue/coverage phase.
    raw = pd.read_csv(source, sep=";", dtype=str, low_memory=False,
                      usecols=lambda c: c.split(",")[0] != "PAR")
    raw.columns = [c.split(",")[0] for c in raw.columns]
    below = below_cutoff(raw, check_partition=False)
    ids = pair_ids(raw)
    original = align_original(raw, resolve(data_root, "data/pcal_ml_oof.csv"), source)
    queue = pd.DataFrame({"pair_id": ids, "p_cal_ml": original.p_cal_ml})
    top = queue.loc[below].sort_values(["p_cal_ml", "pair_id"], ascending=[False, True]).pair_id.tolist()[:max(KS)]
    base, sources = frozen_sources(data_root)
    run_paths = {}
    for mode in MODES:
        frozen = [base / s["path"] for s in sources if s["mode"] == mode]
        extra = [resolve(data_root, f"data/sprint3b/{d}/exp2_llm_sync_{mode}.json") for d in RUN_DIRS]
        run_paths[mode] = (frozen, [p for p in extra if p.exists()])
    wanted = set()
    for frozen, extra in run_paths.values():
        for path in frozen + extra:
            run = json.loads(path.read_text(encoding="utf-8"))
            wanted.update(row["pair_id"] for row in run["results"])
    # The canonical loader supplies the same linkage features as historical run_v3.
    # PAR/y, if loaded internally, are immediately removed and never consulted.
    features = load_comparador_csv(source, cfg=LoadConfig(macd_enabled=True))
    features = features.drop(columns=[c for c in features if c.split(",")[0] in {"PAR", "y"}])
    selected_features = features.loc[pair_ids(features).isin(wanted)]
    views = {d.pair_id: DossierView(d, True) for d in build_dossiers(selected_features, only_llm_review=False)}
    for view in views.values():
        audit_view(view)
    decisions, conflicts, audit = {}, {}, {}
    for mode in MODES:
        frozen_paths, extra_paths = run_paths[mode]
        frozen, embedded = validated_responses(base, mode, views, sources)
        all_sources = []
        source_audit = []
        for path in frozen_paths + extra_paths:
            run = json.loads(path.read_text(encoding="utf-8"))
            validate_dossiers(path, run, mode, views, prompt_sha256=sha256(PROMPT))
            rows = {}
            for row in run["results"]:
                decision = row["decision"].strip().upper()
                if decision not in {"MATCH", "NONMATCH", "UNSURE"}:
                    raise ValueError("Invalid structured decision")
                if row["pair_id"] in rows:
                    raise ValueError("Duplicate pair within source")
                rows[row["pair_id"]] = decision
            all_sources.append(rows)
            source_audit.append({"source": path.relative_to(base).as_posix() if path.is_relative_to(base) else path.relative_to(ROOT).as_posix(),
                                 "responses": len(rows), "sha256": sha256(path)})
        decisions[mode], conflicts[mode] = select_responses(frozen, all_sources)
        audit[mode] = {"valid_cached_pairs": len(decisions[mode]),
                       "conflicting_pairs_all_sources": len(conflicts[mode]),
                       "embedded_identity_cohorts": embedded, "sources": source_audit}
    return top, decisions, conflicts, audit


def generate(data_root, out=ROOT / "data", *, local=LOCAL, coverage_only=False):
    spec, commit = verify_protocol(data_root)
    out, local = Path(out), Path(local)
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    top, decisions, conflicts, audit = collect(data_root)
    coverage = pd.DataFrame([coverage_row(m, top, decisions[m], conflicts[m], k) for m in MODES for k in KS])
    coverage.to_csv(out / OUTPUTS[0], index=False, lineterminator="\n")
    pending_summary = {}
    for mode in MODES:
        pending = [pid for pid in top[:125] if pid not in decisions[mode]]
        pd.DataFrame({"pair_id": pending}).to_csv(local / f"pendentes_{mode}.csv", index=False, lineterminator="\n")
        pending_summary[mode] = {"pending_unique_k_le125": len(pending),
                                 "estimated_usd_no_cache": round(.00316 * len(pending), 8)}
    # Coverage is persisted before labels are loaded or any endpoint is computed.
    if coverage_only:
        print(coverage.to_string(index=False))
        print("PENDING", json.dumps(pending_summary))
        return coverage, None, None
    complete = coverage.loc[coverage.completa]
    results = []
    if not complete.empty:
        raw = load_raw(resolve(data_root, "data/COMPARADORSEMIDENT.csv"))
        below_cutoff(raw)  # Validate frozen universe only after coverage.
        labels = dict(zip(pair_ids(raw), numeric(raw.PAR).astype(int)))
        curve = pd.read_csv(resolve(data_root, "data/comparador_mesma_carga_curva.csv"))
        o3 = curve.loc[curve.ordenacao.eq("O3_p_cal_ml")]
        if not o3.k.is_unique:
            raise ValueError("O3 curve has duplicate k")
        o3_counts = dict(zip(o3.k, o3.par2_topk))
        # Confirm the stored O3 curve is exactly the current ranked queue.
        running = 0
        for k, pid in enumerate(top, 1):
            running += labels[pid] == 2
            if o3_counts.get(k) != running:
                raise ValueError("O3 curve differs from current queue/labels")
        for row in complete.itertuples():
            results.append(outcomes(row.modo, top[:row.k], decisions[row.modo], labels, o3_counts))
    table = pd.DataFrame(results, columns=RESULT_COLUMNS)
    table.to_csv(out / OUTPUTS[1], index=False, lineterminator="\n")
    metadata = {
        "analysis": "exploratory_post_hoc", "post_hoc": True, "registered": "internal",
        "date": spec["date"], "protocol_commit": commit, "protocol_sha256": sha256(PROTOCOL),
        "api_calls": 0, "model_fits": 0, "input_sha256": spec["input_sha256"],
        "response_audit": audit, "pending": pending_summary,
        "stop_rule_triggered": bool(coverage.escalate.any()),
        "interpretation": {m: interpret([r["diferenca_par2"] for r in results if r["modo"] == m]) for m in MODES},
        "evaluated_k": {m: [r["k"] for r in results if r["modo"] == m] for m in MODES},
        "limitations": spec["limitations"],
        "response_reuse_warning": "As respostas do LLM também alimentaram as taxas do revisor e o braço P.",
        "human_review": "Perfeito; erro de 5% é cenário, não desfecho observado; sem FP hipotéticos.",
        "wilson": "IC 95% descritivo; sem correção de dependência entre pares; n=0 indefinido.",
        "coverage_blinding": "Fila sem PAR; dossiês só com features via loader canônico, sem ajuste/model_outputs; cobertura persistida antes dos rótulos.",
        "code_sha256_lf": {name: text_sha256(ROOT / name) for name in (
            "scripts/filtro_llm.py", "tests/test_filtro_llm.py",
        )},
        "output_sha256": {name: sha256(out / name) for name in OUTPUTS[:2]},
    }
    (out / OUTPUTS[2]).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return coverage, table, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/git/phd-research"))
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--register-inputs", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    if args.register_inputs:
        register_inputs(args.data_root)
    else:
        coverage, table, metadata = generate(args.data_root, args.out, coverage_only=args.coverage_only)
        if metadata is not None:
            print(coverage.to_string(index=False))
            print(table.to_string(index=False))
            print("INTERPRETATION", json.dumps(metadata["interpretation"]))
            print("PENDING", json.dumps(metadata["pending"]))
            if metadata["stop_rule_triggered"]:
                print("ESCALATE: cobertura incompleta em k <= 125; pendências apenas em .local/filtro_llm")


if __name__ == "__main__":
    main()
