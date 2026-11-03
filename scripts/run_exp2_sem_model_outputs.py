"""Exp2 Anthropic, exact historical cohorts, without the model_outputs group.

Successful pairs are reused (including the three-pair smoke test). Only structured
allowlisted results and pseudonymous IDs are persisted; never dossiers/API text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_exp2_ablacao_dossie as abl
import run_exp2_anthropic as ant
from comparar_revisores_exp2 import gold, load_run
from gzcmd.llm_review import VALID_REASON_CODES

FORBIDDEN = {"model_outputs", "p_cal", "base_choice", "band", "evr", "guardrail", "guardrail_reason"}
ALLOWED_RESULT = {"pair_id", "decision", "confidence", "reason_codes", "quality_flags", "protocol",
                  "agent_a_decision", "agent_b_decision", "arbiter_decision", "total_latency_s", "total_tokens"}
OUT = ROOT / "data/sprint3b/anthropic_sem_mo"
PROVIDER = "anthropic"
PROMPT_DIR = ROOT / "gzcmd"


def configure_provider(provider):
    """Keep Anthropic defaults unchanged; select Go explicitly for this ablation."""
    global OUT, PROVIDER
    PROVIDER = provider
    OUT = ROOT / f"data/sprint3b/{'opencode_go' if provider == 'opencode-go' else 'anthropic'}_sem_mo"


def configure_prompt(version):
    """Select the shared loader before constructing either provider's reviewers.

    Keep the historical v3 effective-text hash for checkpoint compatibility;
    v4 hashes the entire UTF-8 prompt file, also recorded for both versions.
    """
    if version not in ("v3", "v4"):
        raise ValueError("Unsupported prompt version")
    path = PROMPT_DIR / f"gzcmd_{version}_llm_prompt.md"
    ant.llm_review._PROMPT_PATH = path
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    effective = ant.llm_review._load_system_prompt() + ant.llm_review._load_user_template()
    return {"prompt_version": version, "prompt_file_sha256": file_hash,
            "prompt_sha256": file_hash if version == "v4" else hashlib.sha256(effective.encode()).hexdigest()}


def configure_go_transport():
    import requests
    import run_exp2_opencode_go as go
    wrapped = go._RequestsWithSession(requests, go._session_id("deepseek-sem-model-outputs"),
                                     go.EXTRA_BY_MODEL[go.MODEL])

    class MeteredGoRequests(abl.MeteredRequests):
        def __getattr__(self, name):
            # _call_api catches requests.RequestException; keep that module API.
            return getattr(self.inner, name)

    go.llm_review.requests = MeteredGoRequests(wrapped)


def original_path(mode):
    folder = "opencode_go" if PROVIDER == "opencode-go" else "anthropic"
    return ROOT / f"data/sprint3b/{folder}/exp2_llm_sync_{mode}.json"


def artifact_prefix():
    return "exp2_deepseek_sem_mo" if PROVIDER == "opencode-go" else "exp2_haiku_sem_mo"


def retry_file_operation(operation):
    """Windows readers/virus scanners can briefly deny atomic replacement."""
    for attempt in range(12):
        try:
            return operation()
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.02 * (2 ** attempt), .5))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    retry_file_operation(lambda: tmp.write_text(content, encoding="utf-8"))
    # Five replacement retries after the first attempt (0.2s through 3s).
    for delay in (0, .2, .4, .8, 1.6, 3):
        if delay:
            time.sleep(delay)
        try:
            tmp.replace(path)
            return
        except PermissionError:
            pass
    # A locked canonical checkpoint must never terminate paid inference. An
    # immutable alternate remains recoverable, even if the next save rewrites tmp.
    alternate = path.with_name(path.name + f".fallback-{time.time_ns()}.json")
    retry_file_operation(lambda: alternate.write_text(content, encoding="utf-8"))
    print(f"CHECKPOINT_FALLBACK {path.name} {alternate.name}", flush=True)


def read_json(path):
    return retry_file_operation(lambda: json.loads(path.read_text(encoding="utf-8")))


def recover_candidates(candidates, *, result_checkpoint=False):
    valid = []
    existing = []
    for candidate in candidates:
        try:
            existing.append((candidate.stat().st_mtime_ns, candidate))
        except FileNotFoundError:
            # A concurrent writer may already have renamed a transport tmp.
            continue
    existing.sort(key=lambda item: item[0], reverse=True)
    for modified_at, candidate in existing:
        try:
            value = read_json(candidate)
            if not result_checkpoint:
                return value
            rank = (len(value.get("results", [])), modified_at)
            valid.append((rank, value))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            continue
    return max(valid, key=lambda item: item[0])[1] if valid else None


def recover_json(path, *, result_checkpoint=False):
    candidates = [path, path.with_suffix(".tmp"), *path.parent.glob(path.name + ".fallback-*.json")]
    return recover_candidates(candidates, result_checkpoint=result_checkpoint)


def ensure_journal_identity(journal, identity, *, checkpoint_validated=False):
    """Refuse orphan results of unknown origin; migrate only a validated checkpoint."""
    path = journal / "identity.json"
    stored = recover_json(path)
    if stored is not None:
        if stored != identity:
            raise ValueError("Journal identity changed; refusing mixed execution")
        return
    has_results = any(".result." in item.name for item in journal.iterdir())
    if has_results and not checkpoint_validated:
        raise ValueError("Journal results require identity.json before checkpoint-free recovery")
    save(path, identity)
    if recover_json(path) != identity:
        raise ValueError("Journal identity could not be persisted")


# In-memory replacement only: preserve the original ablation runner on disk.
# Its per-call metering must use the same Windows-safe persistence boundary.
abl.save = save


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def load_ids(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            ids = [row["pair_id"] for row in csv.DictReader(stream)]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("results", data.get("pair_ids", data.get("selected_pair_ids", [])))
        ids = [row["pair_id"] if isinstance(row, dict) else row for row in data]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[0-9a-f]{16}", p) for p in ids):
        raise ValueError("Expected unique SHA256[:16] pair IDs")
    return ids


def audit_view(view):
    def check(value):
        if isinstance(value, dict):
            assert not FORBIDDEN.intersection(value), "Engine output survived ablation"
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    assert set(view.payload) == {"schema_version", "pair_id", "features"}
    check(view.payload)


def safe_result(result):
    row = abl.safe_result(result)
    assert set(row) <= ALLOWED_RESULT
    assert row["decision"] in abl.DECISIONS
    assert set(row["reason_codes"]) <= VALID_REASON_CODES
    assert all(isinstance(v, bool) for v in row["quality_flags"].values())
    # Even a flagged result has no provider free text in this allowlist.
    return row


def prepare(mode, ids):
    df, _ = ant.exp2.run_v3(input_csv=str(ROOT / "data/COMPARADORSEMIDENT.csv"),
                            config_path=str(ROOT / "gzcmd/gzcmd_v3_config.yaml"), mode=mode,
                            p_cal="fit_platt", policy_version="v3")
    # The historical IDs, not today's triage policy, define this experiment.
    dossiers = {d.pair_id: d for d in ant.exp2.build_dossiers(df, only_llm_review=False)}
    missing = set(ids) - dossiers.keys()
    print(f"COHORT {len(ids)} / rebuilt {len(set(ids) & dossiers.keys())}", flush=True)
    if missing:
        print("MISSING_PAIR_IDS " + json.dumps(sorted(missing)), flush=True)
        raise ValueError(f"Historical cohort missing {len(missing)} rebuilt dossiers")
    views = {pid: abl.DossierView(dossiers[pid], True) for pid in ids}
    for view in views.values():
        audit_view(view)
    # Field names only, never dossier values: supports an independent rg audit.
    def keys(value):
        found = set()
        if isinstance(value, dict):
            found.update(value)
            for child in value.values():
                found.update(keys(child))
        elif isinstance(value, list):
            for child in value:
                found.update(keys(child))
        return found
    OUT.mkdir(parents=True, exist_ok=True)
    abl.save(OUT / f"dossier_field_audit_{mode}.json",
             {"mode": mode, "n_dossiers": len(views), "forbidden_present": False,
              "field_names": sorted(set().union(*(keys(v.payload) for v in views.values())))})
    return views


def parse_metered(text):
    try:
        return ant.llm_review._extract_json(text)
    except (ValueError, KeyError):
        stats = abl._LOCAL.stats
        stats["response_parse_errors"] += 1
        abl.persist_transport(stats)
        raise


def review_pair(pid, view, model, journal):
    stat_path = journal / f"{pid}.transport.json"
    result_path = journal / f"{pid}.result.json"
    stats = recover_json(stat_path) or abl.meter()
    abl._LOCAL.stats, abl._LOCAL.stats_path = stats, stat_path
    if PROVIDER == "opencode-go":
        import run_exp2_opencode_go as go
        reviewer = go.llm_review.LLMReviewer(
            api_key=os.environ["OPENCODE_GO_API_KEY"], base_url=go.BASE_URL,
            model=model, fallback_model=model, **go.GEN_PARAMS)
        try:
            row = safe_result(reviewer.review(view))
            abl.save(result_path, row)
            return row, None
        except Exception as exc:
            return None, {"pair_id": pid, "type": type(exc).__name__, "status": getattr(exc, "status", None)}
    reviewer = ant.AnthropicReviewer(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com",
                                      model=model, fallback_model=model, max_tokens=2500,
                                      timeout_s=120, max_retries=5, retry_delay_s=3)
    client = reviewer._get_client()

    class Facade:
        messages = abl.MeteredMessages(client.messages, stats)

    reviewer._client = Facade()
    try:
        row = safe_result(reviewer.review(view))
        abl.save(result_path, row)
        return row, None
    except Exception as exc:
        # Exception messages may contain provider text/credentials: never persist.
        return None, {"pair_id": pid, "type": type(exc).__name__, "status": getattr(exc, "status", None)}
    finally:
        client.close()


def aggregate(journal):
    total = abl.meter()
    groups = {}
    for path in journal.iterdir():
        if ".transport." in path.name:
            groups.setdefault(path.name[:16], []).append(path)
    for candidates in groups.values():
        stats = recover_candidates(candidates)
        if stats is None:
            continue
        for key, value in stats.items():
            if key == "error_statuses":
                for status, count in value.items():
                    total[key][status] = total[key].get(status, 0) + count
            elif key == "max_request_input_tokens":
                total[key] = max(total[key], value)
            else:
                total[key] += value
    return total


def run(args):
    prompt_metadata = configure_prompt(getattr(args, "prompt_version", "v3"))
    key_name = "OPENCODE_GO_API_KEY" if PROVIDER == "opencode-go" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_name):
        raise ValueError(f"Missing {key_name} (value suppressed)")
    ids = load_ids(args.pares)
    original = original_path(args.modo)
    allow_new_cohort = getattr(args, "allow_new_cohort", False)
    if not allow_new_cohort and set(ids) != set(load_ids(original)):
        raise ValueError("--pares must identify the exact historical cohort; use --limit for smoke "
                         "or --allow-new-cohort for a v3.1 lot")
    views = prepare(args.modo, ids)
    fingerprints = {p: hashlib.sha256(v.to_json(sort_keys=True).encode()).hexdigest() for p, v in views.items()}
    source_hash = hashlib.sha256(args.pares.read_bytes()).hexdigest()
    path = args.saida
    journal = path.parent / (path.stem + "_journal")
    journal.mkdir(parents=True, exist_ok=True)
    previous = recover_json(path, result_checkpoint=True) or {}
    identity = {"mode": args.modo, "model": args.modelo, "dossier_sha256": fingerprints,
                "prompt_sha256": prompt_metadata["prompt_sha256"],
                "prompt_version": prompt_metadata["prompt_version"],
                "selected_pair_ids": ids, "source_sha256": source_hash,
                "cohort": "v31_lot" if allow_new_cohort else "historical"}
    if previous and any(previous.get(k, "v3" if k == "prompt_version" else None) != v
                        for k, v in identity.items()):
        raise ValueError("Checkpoint identity changed; refusing mixed execution")
    ensure_journal_identity(journal, identity, checkpoint_validated=bool(previous))
    results = {r["pair_id"]: r for r in previous.get("results", [])}
    recovered_pair_ids = {p.name[:16] for p in journal.iterdir() if ".result." in p.name}
    for pid in recovered_pair_ids:
        row = recover_json(journal / f"{pid}.result.json")
        assert row is not None
        assert row["pair_id"] in ids
        results[row["pair_id"]] = row
    errors = {r["pair_id"]: r for r in previous.get("errors", []) if r["pair_id"] not in results}
    output = {**identity, "experiment": "exp2_llm_sync", "timestamp": previous.get("timestamp", utcnow()),
              "n_review_pairs": len(ids), "workers": args.workers,
              "removed_fields": abl.REMOVED_FIELDS, "elapsed_s": previous.get("elapsed_s", 0),
              "metadata": {"provider": "Anthropic Messages API", "model_exact": args.modelo,
                           "max_tokens": 2500, "temperature_sent": False, "extended_thinking": False,
                           "max_retries": 5, "retry_delay_s": 3, "timeout_s": 120,
                           "protocol": "unchanged Agent-A + Agent-B + arbiter on disagreement",
                            **prompt_metadata, "free_response_fields_saved": False,
                            "checkpoint": "every completed pair + durable per-call usage journal"}}
    if PROVIDER == "opencode-go":
        import run_exp2_opencode_go as go
        if args.modelo != go.MODEL:
            raise ValueError("Go ablation requires the historical DeepSeek model")
        configure_go_transport()
        output["metadata"].update({"provider": "OpenCode Go", **go.GEN_PARAMS,
                                   **go.EXTRA_BY_MODEL[go.MODEL], "temperature_sent": True,
                                   "retry_delay_s": 2.0})
    started, base_elapsed = time.monotonic(), output["elapsed_s"]

    def checkpoint():
        output["results"] = [results[p] for p in ids if p in results]
        output["errors"] = [errors[p] for p in ids if p in errors]
        output["failed_pair_ids"] = [p for p in ids if p not in results]
        output["n_reviewed"], output["n_errors"] = len(results), len(errors)
        output["complete"] = len(results) == len(ids)
        output["elapsed_s"] = round(base_elapsed + time.monotonic() - started, 2)
        output["updated_at"] = utcnow()
        output["transport"] = aggregate(journal)
        output["cost_estimate_usd"] = (None if PROVIDER == "opencode-go" else abl.haiku_cost_estimate(output["transport"]))
        output["cost_note"] = ("Go uses subscription quota; billed USD unavailable." if PROVIDER == "opencode-go" else "Estimate only: standard/global USD/M input .10, output .50, cache read .01, cache write .125; not invoice. Source: platform.claude.com/docs/en/about-claude/pricing, 2026-11-02 (existing ablation).")
        output["metrics"] = {"mode": args.modo, "n_reviewed": len(results), "n_errors": len(errors),
                             "total_elapsed_s": output["elapsed_s"],
                             "decisions": dict(Counter(r["decision"] for r in results.values())),
                             "protocols": dict(Counter(r["protocol"] for r in results.values()))}
        abl.save(path, output)

    checkpoint()
    chosen = ids[:args.limit] if args.limit else ids
    pending_ids = iter(p for p in chosen if p not in results)
    ant._extract_json = parse_metered
    if PROVIDER == "opencode-go":
        # Capture once: parse_metered itself calls the module's original parser.
        import run_exp2_opencode_go as go
        original_extract = go.llm_review._extract_json
        def go_parse_metered(text):
            try:
                return original_extract(text)
            except (ValueError, KeyError):
                abl._LOCAL.stats["response_parse_errors"] += 1
                abl.persist_transport(abl._LOCAL.stats)
                raise
        go.llm_review._extract_json = go_parse_metered
    consecutive_errors = 0
    stop = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}

        def submit():
            pid = next(pending_ids, None)
            if pid is not None:
                active[pool.submit(review_pair, pid, views[pid], args.modelo, journal)] = pid

        for _ in range(args.workers):
            submit()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                pid = active.pop(future)
                row, error = future.result()
                if error:
                    consecutive_errors += 1
                    errors[pid] = error
                    print("ERROR " + json.dumps(error), flush=True)
                    stop = stop or error["type"] == "FatalAPIError" or consecutive_errors >= 2
                else:
                    consecutive_errors = 0
                    results[pid] = row
                    errors.pop(pid, None)
                    print(f"RESULT {args.modo} {len(results)}/{len(ids)} {pid} {row['decision']} {row['protocol']}", flush=True)
                checkpoint()
                if not stop:
                    submit()
    checkpoint()
    print("RUN_END " + json.dumps({k: output[k] for k in ("mode", "n_reviewed", "n_errors", "complete", "elapsed_s", "transport", "cost_estimate_usd")}), flush=True)
    if stop:
        raise RuntimeError("Stopped after fatal/two consecutive errors; sanitized errors checkpointed")
    return output


def ratio(a, b):
    return a / b if b else None


def summarize():
    import pandas as pd
    from scipy.stats import binomtest
    g = gold()
    assert g.index.is_unique
    summaries, paired = [], []
    for mode in ("vigilancia", "confirmacao"):
        run = recover_json(OUT / f"exp2_llm_sync_{mode}.json", result_checkpoint=True)
        assert run is not None
        old, _, _ = load_run(original_path(mode), "old")
        new = pd.DataFrame([{"pair_id": row["pair_id"], "dec_new": row["decision"],
                             "conf_new": row.get("confidence")} for row in run["results"]]).set_index("pair_id")
        assert new.index.is_unique and set(new.index) <= set(old.index)
        table = new.join(old).join(g)
        assert not table.y.isna().any()
        for label in ("old", "new"):
            dec, y = table[f"dec_{label}"], table.y
            vp = int(((dec == "MATCH") & (y == 1)).sum())
            fp = int(((dec == "MATCH") & (y == 0)).sum())
            fn = int(((dec == "NONMATCH") & (y == 1)).sum())
            vn = int(((dec == "NONMATCH") & (y == 0)).sum())
            up = int(((dec == "UNSURE") & (y == 1)).sum())
            un = int(((dec == "UNSURE") & (y == 0)).sum())
            summaries.append({"mode": mode, "arm": label, "n": len(table), "expected": len(old),
                              "VP": vp, "FP": fp, "FN": fn, "VN": vn, "UNSURE": up + un,
                              "UNSURE_pos": up, "UNSURE_neg": un,
                              "sens_excl_unsure": ratio(vp, vp + fn), "spec_excl_unsure": ratio(vn, vn + fp),
                              "sens_unsure_as_match": ratio(vp + up, vp + fn + up),
                              "spec_unsure_as_match": ratio(vn, vn + fp + un),
                              "sens_unsure_as_nonmatch": ratio(vp, vp + fn + up),
                              "spec_unsure_as_nonmatch": ratio(vn + un, vn + fp + un)})
        old_correct = ((table.dec_old == "MATCH") & (table.y == 1)) | ((table.dec_old == "NONMATCH") & (table.y == 0))
        new_correct = ((table.dec_new == "MATCH") & (table.y == 1)) | ((table.dec_new == "NONMATCH") & (table.y == 0))
        b = int((old_correct & ~new_correct).sum())
        c = int((~old_correct & new_correct).sum())
        transitions = Counter(f"{a}->{b}" for a, b in zip(table.dec_old, table.dec_new))
        comparison = {"mode": mode, "n_common": len(table),
                      "fn_old_to_match": int(((table.y == 1) & (table.dec_old == "NONMATCH") & (table.dec_new == "MATCH")).sum()),
                      "fn_old_to_unsure": int(((table.y == 1) & (table.dec_old == "NONMATCH") & (table.dec_new == "UNSURE")).sum()),
                      "fp_new_not_old": int(((table.y == 0) & (table.dec_old != "MATCH") & (table.dec_new == "MATCH")).sum()),
                      "mcnemar_old_correct_new_not": b, "mcnemar_old_not_new_correct": c,
                      "mcnemar_exact_p": float(binomtest(b, b + c, .5).pvalue) if b + c else 1.0,
                      "transitions": dict(transitions), "transport": run["transport"],
                      "review_errors": run["n_errors"], "cost_estimate_usd": run["cost_estimate_usd"],
                      "elapsed_s": run["elapsed_s"], "failed_pair_ids": run["failed_pair_ids"],
                      "provider_pii_flags": sum(r["quality_flags"].get("pii_leak_detected", False) for r in run["results"])}
        paired.append(comparison)
        table[["dec_old", "dec_new", "nota_final"]].to_csv(ROOT / f"data/{artifact_prefix()}_pares_{mode}.csv")
        for row in summaries[-2:]:
            if row["arm"] == "new":
                row.update({k: v for k, v in comparison.items() if not isinstance(v, (dict, list)) and k != "mode"})
                row.update(run["transport"])
                row.pop("error_statuses", None)
    pd.DataFrame(summaries).to_csv(ROOT / f"data/{artifact_prefix()}_resumo.csv", index=False)
    abl.save(OUT / "comparacao.json", {"metrics": summaries, "paired": paired})
    metric_table = ["| Modo | Braço | N | VP | FP | FN | VN | UNSURE (+/−) | Sens. sem U | Esp. sem U |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        rates = [f"{row[k]:.6f}" if row[k] is not None else "NA"
                 for k in ("sens_excl_unsure", "spec_excl_unsure")]
        metric_table.append(f"| {row['mode']} | {row['arm']} | {row['n']} | {row['VP']} | {row['FP']} | "
                            f"{row['FN']} | {row['VN']} | {row['UNSURE']} ({row['UNSURE_pos']}/{row['UNSURE_neg']}) | "
                            f"{rates[0]} | {rates[1]} |")
    execution = OUT / "execucao.json"
    execution_metadata = json.loads(execution.read_text(encoding="utf-8-sig")) if execution.exists() else {}
    recovery_path = OUT / "recuperacao_windows.json"
    recovery = read_json(recovery_path) if recovery_path.exists() else {}
    smoke_path = OUT / "fumaca_inspecao.json"
    smoke = read_json(smoke_path) if smoke_path.exists() else {}
    report = ["# Exp2 — Haiku 5.5 sem model_outputs", "", f"Gerado em {utcnow()}.", "",
              "Modelo exato: claude-haiku-5-5. Mesmos pares históricos por modo, reconstrução v3/fit_platt; "
              "mesmo prompt, parser e protocolo A+B+árbitro. Removido todo o bloco model_outputs "
              "(p_cal, base_choice, band, evr, guardrail, guardrail_reason). Subescores, nota e MACD mantidos. "
              "O prompt não foi alterado e pode mencionar os campos ausentes.", "",
              "max_tokens=2500; sem temperature e sem extended thinking; até quatro pares concorrentes (modos sequenciais). "
              "Retries com backoff exponencial, cinco tentativas; checkpoint por par e journal de tokens por chamada. "
              "Três pares de fumaça por modo reaproveitados; nenhum sucesso é chamado novamente.", "",
              "Gabarito PAR ∈ {1,2}; identificadores dtype=str aparados, SHA256(COMPREC|REFREC|PASSO)[:16]. "
              "Métricas old/new nos pares com resposta em ambos. UNSURE explicitamente separado; são mostradas "
              "taxas excluindo UNSURE e duas políticas inclusivas (UNSURE como MATCH e como NONMATCH). "
              "McNemar exato bilateral compara acerto estrito contra gabarito, contando UNSURE como não acerto. "
              "Não há ajuste de multiplicidade; execução estocástica única, comparação histórica não controla variação temporal. "
              "Amostras condicionadas à triagem não representam todo o extrato.", "",
              "## Matriz e taxas", "", "\n".join(metric_table), "",
              "Taxas inclusivas e todos os contadores estão em `exp2_haiku_sem_mo_resumo.csv`.", "",
              "## Comparação pareada, chamadas, tokens, custo e tempo", "", "```json",
              json.dumps(paired, ensure_ascii=False, indent=2), "```", "",
              "Custo estimado, não fatura: tarifas standard/global já documentadas na ablação, "
              "USD/M entrada 0.10, saída 0.50, cache leitura 0.01 e escrita 0.125 (até 100k). "
              "Fonte: https://platform.claude.com/docs/en/about-claude/pricing (2026-11-02). "
              "Tempo elapsed_s soma sessões de API incluindo fumaça/retries; exclui reconstrução do pipeline. "
              "O arquivo execucao.json registra duração de parede da execução completa final.", "",
              "Execução final: " + json.dumps(execution_metadata, ensure_ascii=False), "",
              "## Fumaça e recuperação de checkpoint", "", "```json",
              json.dumps({"smoke": smoke, "windows_checkpoint_recovery": recovery}, ensure_ascii=False, indent=2), "```", "",
              "A troca atômica bloqueada pelo Windows recebeu cinco retries com backoff de 0,2 até 3 segundos; "
              "bloqueio persistente produz checkpoint alternativo sem abortar inferência. "
              "Na retomada, o journal recuperou sucessos além do JSON/tmp; nenhum sucesso foi reexecutado. "
              "elapsed_s acumula as sessões de API; wall_seconds inclui reconstrução e intervalo de recuperação "
              "desde o primeiro início do lote completo, excluindo a fumaça prévia.", "",
              "Persistência restrita a IDs pseudônimos, decisões estruturadas, códigos enumerados, flags e contadores; "
              "sem dossiês, evidências livres ou respostas brutas. Ver auditoria.json para auditoria de segredo e esquema."]
    if PROVIDER == "opencode-go":
        report = ["# Exp2: DeepSeek V4.1 Flash sem model_outputs", "", f"Gerado em {utcnow()}.", "",
                  "Modelo deepseek-v4.1-flash via OpenCode Go. Mesmos pares históricos por modo, pipeline v3/fit_platt, "
                  "mesmos prompts, parser e protocolo Agent-A + Agent-B + árbitro. Removido todo o grupo model_outputs "
                  "(p_cal, base_choice, band, evr, guardrail, guardrail_reason). Subescores, nota e MACD mantidos. "
                  "O prompt não foi alterado e pode mencionar campos ausentes.", "",
                  "Parâmetros do run original: temperature=0, top_p=1, top_k=1, presence_penalty=0, frequency_penalty=0, "
                  "max_tokens=800, thinking disabled, reasoning_effort none; timeout por chamada=120 s, "
                  "quatro tentativas com backoff inicial de 2 s, sem fallback distinto. Até quatro pares concorrentes; "
                  "modos sequenciais. O processo do lote não tem timeout de 120 s.", "",
                  "Gabarito PAR ∈ {1,2}, identificadores dtype=str aparados, SHA256(COMPREC|REFREC|PASSO)[:16]. "
                  "Métricas nos pares com resposta nos dois braços. UNSURE separado, taxas excluindo UNSURE e "
                  "duas políticas inclusivas no CSV (UNSURE como MATCH ou NONMATCH). McNemar exato bilateral "
                  "compara acerto estrito; UNSURE conta como não acerto. Sem ajuste de multiplicidade. "
                  "Comparação histórica, execução única: não separa o efeito da ablação da variação temporal. "
                  "Coortes condicionadas à triagem não representam todo o extrato.", "",
                  "## Matriz e taxas", "", "\n".join(metric_table), "",
                  f"Taxas inclusivas: `{artifact_prefix()}_resumo.csv`. Comparações individuais nos CSVs `_pares_<modo>`.", "",
                  "## Comparação pareada e execução", "", "```json", json.dumps(paired, ensure_ascii=False, indent=2), "```", "",
                  "Tempo elapsed_s inclui chamadas/retries, exclui reconstrução. Go consome quota de assinatura; "
                  "não há valor faturado disponível. execucao.json registra tempo de parede do lote. "
                  "Sondagem de quota separada em quota_probe.json; suas chamadas não entram no transporte dos pares.", "",
                  "Execução: " + json.dumps(execution_metadata, ensure_ascii=False), "",
                  "Persistência restrita a IDs pseudônimos, escores, decisões estruturadas, códigos enumerados, flags "
                  "e contadores. Nenhum dossiê, resposta bruta ou evidência livre salvo. Checkpoint por par com retry "
                  "na troca atômica e fallback recuperável no Windows; sucessos não são repetidos. Ver auditoria.json."]
        report += ["", "## Interpretação", "",
                   "FN explícitos devem ser interpretados junto com UNSURE e FP. Sensibilidade de 100% após excluir "
                   "UNSURE não significa recuperação de todos os vínculos, pois retira do denominador os positivos "
                   "sem decisão. O teste de McNemar acima usa acerto estrito, não apenas ausência de FN."]
        for mode in ("vigilancia", "confirmacao"):
            old, new = [row for row in summaries if row["mode"] == mode]
            report += ["", f"{mode}: FN {old['FN']} para {new['FN']}, FP {old['FP']} para {new['FP']}, "
                       f"UNSURE {old['UNSURE']} para {new['UNSURE']}; "
                       f"acertos estritos {old['VP'] + old['VN']} para {new['VP'] + new['VN']}. "
                       "A redução de FN não deve ser apresentada isoladamente como melhora global."]
    (ROOT / f"data/{artifact_prefix()}_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit_files()
    print("SUMMARY " + json.dumps({"metrics": summaries, "paired": paired}, ensure_ascii=False), flush=True)


def audit_files():
    key = os.environ.get("OPENCODE_GO_API_KEY" if PROVIDER == "opencode-go" else "ANTHROPIC_API_KEY")
    assert key, "Secret must be loaded for in-memory audit"
    secrets = [value for name in ("OPENCODE_GO_API_KEY", "ANTHROPIC_API_KEY", "SAGE_API_KEY")
               if (value := os.environ.get(name))]
    files = [*OUT.rglob("*.json"), *OUT.glob("*.log"),
             *ROOT.glob(f"data/{artifact_prefix()}_*.csv"), ROOT / f"data/{artifact_prefix()}_relatorio.md", Path(__file__)]
    for path in files:
        content = path.read_text(encoding="utf-8")
        assert not any(secret in content for secret in secrets), "Secret audit failed (value suppressed)"
    for mode in ("vigilancia", "confirmacao"):
        run = json.loads((OUT / f"exp2_llm_sync_{mode}.json").read_text(encoding="utf-8"))
        for row in run["results"]:
            assert set(row) <= ALLOWED_RESULT
            assert set(row["reason_codes"]) <= VALID_REASON_CODES
    abl.save(OUT / "auditoria.json", {"timestamp": utcnow(), "files_scanned": len(files), "secret_matches": 0,
                                     "free_response_fields_saved": False, "dossiers_saved": False,
                                     "result_schema_allowlist_valid": True})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("anthropic", "opencode-go"), default="anthropic")
    parser.add_argument("--modelo")
    parser.add_argument("--modo", choices=("vigilancia", "confirmacao"))
    parser.add_argument("--pares", type=Path)
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--saida-dir", type=Path,
                        help="Output directory; defaults filename to exp2_llm_sync_<modo>.json")
    parser.add_argument("--prompt-version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-new-cohort", action="store_true",
                        help="Accept a --pares list that differs from the historical cohort (v3.1 lot)")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.saida is not None and args.saida_dir is not None:
        parser.error("Use either --saida or --saida-dir, not both")
    if args.saida_dir is not None and args.modo:
        args.saida = args.saida_dir / f"exp2_llm_sync_{args.modo}.json"
    if not args.summarize_only and (not all((args.modo, args.pares, args.saida)) or args.limit < 0):
        parser.error("--modo, --pares and --saida/--saida-dir required; --limit must be nonnegative")
    return args


def main():
    global OUT
    args = parse_args()
    configure_provider(args.provider)
    if args.saida_dir is not None:
        OUT = args.saida_dir
    elif args.saida is not None:
        OUT = args.saida.parent
    if args.modelo is None:
        args.modelo = "deepseek-v4.1-flash" if args.provider == "opencode-go" else "claude-haiku-5-5"
    logging.disable(logging.CRITICAL)
    if args.summarize_only:
        summarize()
    else:
        run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # No traceback/provider exception text: this is also the background log boundary.
        print("FATAL " + type(exc).__name__, flush=True)
        import traceback
        print("FATAL_FRAMES " + json.dumps([{ "function": f.name, "line": f.lineno }
                                             for f in traceback.extract_tb(exc.__traceback__)]), flush=True)
        sys.exit(1)
