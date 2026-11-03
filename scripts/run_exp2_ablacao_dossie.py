"""M4: paired dossier ablation (engine outputs removed), without persisting PII.

The full and ablated arms use the same original Exp2 prompts and parser.
Only pseudonymous pair IDs, linkage scores and structured decisions may be saved.

Run from the repository root (Python environment with the original runners' deps)::

    python scripts/run_exp2_ablacao_dossie.py --limit 2 --workers 4 --try-kimi
    python scripts/run_exp2_ablacao_dossie.py --workers 4
    python scripts/run_exp2_ablacao_dossie.py --summarize-only
    python scripts/run_exp2_ablacao_dossie.py --audit-only

Successful checkpoints are reused automatically. ``--arm sem_pcal`` explicitly
selects the output-free dossier, whereas the default ``--arm ambos`` performs the
paired experiment. No API calls occur under prepare/summarize/audit-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Removing the entire group also removes indirect proxies for calibrated
# probability / engine disposition (band, EVR and guardrail explanations).
REMOVED_FIELDS = [
    "model_outputs.p_cal", "model_outputs.base_choice", "model_outputs.band",
    "model_outputs.evr", "model_outputs.guardrail", "model_outputs.guardrail_reason",
]


def ablate(dossier: dict) -> dict:
    result = copy.deepcopy(dossier)
    result.pop("model_outputs", None)
    return result


OUT = ROOT / "data/sprint3b/ablacao"
ORIGINALS = {
    "Kimi K2.5": ROOT / "data/sprint3b/exp2_llm_sync_vigilancia.json",
    "DeepSeek V4.1 Flash": ROOT / "data/sprint3b/opencode_go/exp2_llm_sync_vigilancia.json",
    "Haiku 5.5": ROOT / "data/sprint3b/anthropic/exp2_llm_sync_vigilancia.json",
}
MODELS = {"deepseek": "deepseek-v4.1-flash", "haiku": "claude-haiku-5-5", "kimi": "kimi-k3"}
LABELS = {"deepseek": "DeepSeek V4.1 Flash", "haiku": "Haiku 5.5"}
DECISIONS = ("MATCH", "UNSURE", "NONMATCH")
_LOCAL = threading.local()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def select_pairs():
    """Reconstruct gold from dtype=str identifiers, and cross-check cached CSV."""
    import pandas as pd
    from comparar_revisores_exp2 import gold

    table = gold()
    for label, path in ORIGINALS.items():
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["mode"] != "vigilancia" or run.get("errors"):
            raise ValueError("Original run must be complete vigilancia")
        decisions = pd.DataFrame(run["results"])[["pair_id", "decision"]].set_index("pair_id")
        if not decisions.index.is_unique:
            raise ValueError("Duplicate original pair IDs")
        table = table.join(decisions.rename(columns={"decision": label}), how="inner")
    cached = ROOT / "data/exp2_tres_revisores_vigilancia.csv"
    if cached.exists():
        c = pd.read_csv(cached).set_index("pair_id")
        if set([*ORIGINALS, "y", "nota_final"]).issubset(c.columns):
            for col in [*ORIGINALS, "y"]:
                if not c.loc[table.index, col].equals(table[col]):
                    raise ValueError(f"Cached comparison disagrees with reconstructed {col}")
    mask = (table[list(ORIGINALS)] == "NONMATCH").sum(axis=1) >= 2
    selected = table[(table.y == 1) & mask].copy().sort_index()
    selected["revisores_erraram"] = selected.apply(
        lambda row: ";".join(k for k in ORIGINALS if row[k] == "NONMATCH"), axis=1)
    selected["n_revisores_erraram"] = (selected[list(ORIGINALS)] == "NONMATCH").sum(axis=1)
    selected.rename(columns={"nota_final": "nota"})[
        ["nota", "revisores_erraram", "n_revisores_erraram"]
    ].to_csv(ROOT / "data/ablacao_dossie_pares.csv")
    if not 0 < len(selected) <= 100:
        raise ValueError("Expected 1..100 shared-FN pairs")
    fn = table.loc[table.y == 1, list(ORIGINALS)] == "NONMATCH"
    selection_summary = {"common_reviewed_n": len(table), "gold_positive_n": int((table.y == 1).sum()),
                         "original_fn_counts": {k: int(fn[k].sum()) for k in ORIGINALS},
                         "fn_union_at_least_one": int((fn.sum(axis=1) >= 1).sum()),
                         "fn_shared_at_least_two": len(selected),
                         "fn_all_three": int((selected.n_revisores_erraram == 3).sum())}
    save(OUT / "selecao.json", selection_summary)
    print("SELECTION " + json.dumps(selection_summary, ensure_ascii=False), flush=True)
    return selected


class FatalAPIError(Exception):
    """Non-retryable HTTP error; never include response text or credentials."""

    def __init__(self, status):
        self.status = status
        super().__init__(f"HTTP_{status}")


def meter():
    return {"calls": 0, "http_errors": 0, "response_parse_errors": 0, "error_statuses": {}, "input_tokens": 0,
            "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "max_request_input_tokens": 0}


def http_error(stats, status):
    stats["http_errors"] += 1
    key = str(status)
    stats["error_statuses"][key] = stats["error_statuses"].get(key, 0) + 1


def persist_transport(stats):
    path = getattr(_LOCAL, "stats_path", None)
    if path is not None:
        save(path, stats)


class MeteredRequests:
    """Thread-local counters around the unchanged Go transport."""

    def __init__(self, inner):
        self.inner = inner

    def post(self, *args, **kwargs):
        stats = _LOCAL.stats
        stats["calls"] += 1
        persist_transport(stats)
        try:
            response = self.inner.post(*args, **kwargs)
        except Exception:
            http_error(stats, "connection")
            persist_transport(stats)
            raise
        if response.status_code >= 400:
            http_error(stats, response.status_code)
            persist_transport(stats)
            if response.status_code in (400, 401, 402, 403, 404):
                raise FatalAPIError(response.status_code)
        else:
            usage = response.json().get("usage", {})
            stats["input_tokens"] += int(usage.get("prompt_tokens", 0))
            stats["output_tokens"] += int(usage.get("completion_tokens", 0))
            persist_transport(stats)
        return response


class MeteredMessages:
    def __init__(self, inner, stats):
        self.inner, self.stats = inner, stats

    def create(self, **kwargs):
        self.stats["calls"] += 1
        persist_transport(self.stats)
        try:
            result = self.inner.create(**kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", "connection")
            http_error(self.stats, status)
            persist_transport(self.stats)
            if status in (400, 401, 402, 403, 404):
                raise FatalAPIError(status) from None
            raise
        for field in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.stats[field] += int(getattr(result.usage, field, 0) or 0)
        prompt_size = sum(int(getattr(result.usage, f, 0) or 0) for f in
                          ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        self.stats["max_request_input_tokens"] = max(self.stats.get("max_request_input_tokens", 0), prompt_size)
        persist_transport(self.stats)
        return result


def make_reviewer(tag, stats):
    import run_exp2_opencode_go as go
    import run_exp2_anthropic as ant
    if tag == "haiku":
        reviewer = ant.AnthropicReviewer(
            api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com",
            model=MODELS[tag], fallback_model=MODELS[tag], max_tokens=2500,
            timeout_s=120, max_retries=5, retry_delay_s=3)
        client = reviewer._get_client()
        # The imported transport calls only this facade's messages.create.
        class ClientFacade:
            messages = MeteredMessages(client.messages, stats)
        reviewer._client = ClientFacade()
        return reviewer
    return go.llm_review.LLMReviewer(
        api_key=os.environ["OPENCODE_GO_API_KEY"], base_url=go.BASE_URL,
        model=MODELS[tag], fallback_model=MODELS[tag],
        **{**go.GEN_PARAMS, "max_tokens": go.MAX_TOKENS_BY_MODEL[MODELS[tag]]})


class DossierView:
    """Serialization-only view; preserves every non-engine field byte-for-byte."""

    def __init__(self, dossier, remove):
        self.pair_id = dossier.pair_id
        self.payload = ablate(dossier.to_dict()) if remove else dossier.to_dict()

    def to_json(self, **kwargs):
        return json.dumps(self.payload, ensure_ascii=False, **kwargs)


def safe_result(result):
    """No arbitrary provider text is written, including raw/evidence summaries."""
    d = result.to_dict()
    allowed = {"pair_id", "decision", "confidence", "reason_codes", "quality_flags", "protocol",
               "agent_a_decision", "agent_b_decision", "arbiter_decision", "total_latency_s", "total_tokens"}
    return {k: v for k, v in d.items() if k in allowed}


def run_arm(tag, arm, dossiers, pair_ids):
    path = OUT / f"{MODELS[tag]}_{arm}.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    fingerprints = {pid: hashlib.sha256(dossiers[pid].to_json(sort_keys=True).encode()).hexdigest()
                    for pid in pair_ids}
    if any(fingerprints.get(pid) != digest for pid, digest in previous.get("dossier_sha256", {}).items()):
        raise ValueError("Rebuilt dossier changed since checkpoint; do not mix executions")
    stats = previous.get("transport", meter())
    stats_path = OUT / f"{MODELS[tag]}_{arm}_transport.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    _LOCAL.stats = stats
    _LOCAL.model = MODELS[tag]
    _LOCAL.stats_path = stats_path
    output = {"experiment": "exp2_ablacao_dossie", "mode": "vigilancia", "model": MODELS[tag],
              "arm": arm, "removed_fields": REMOVED_FIELDS if arm == "sem_pcal" else [],
              "selected_pair_ids": pair_ids, "results": previous.get("results", []),
              "dossier_sha256": fingerprints,
              "errors": previous.get("errors", []), "transport": stats,
              "elapsed_s": previous.get("elapsed_s", 0),
              "cost_usd": None, "cost_note": "USD billed unavailable; token usage recorded; Go uses subscription quota."}
    done = {r["pair_id"] for r in output["results"]}
    reviewer = make_reviewer(tag, stats)
    base_elapsed, start, consecutive_errors = output["elapsed_s"], time.monotonic(), 0
    for pid in pair_ids:
        if pid in done:
            continue
        try:
            result = reviewer.review(DossierView(dossiers[pid], arm == "sem_pcal"))
            output["results"].append(safe_result(result))
            consecutive_errors = 0
            print(f"RESULT {tag} {arm} {pid} {result.decision} {result.protocol}", flush=True)
        except Exception as exc:
            consecutive_errors += 1
            output["errors"].append({"pair_id": pid, "type": type(exc).__name__,
                                      "status": getattr(exc, "status", None)})
            print(f"ERROR {tag} {arm} {pid} {type(exc).__name__} status={getattr(exc, 'status', None)}", flush=True)
        output["elapsed_s"] = round(base_elapsed + time.monotonic() - start, 2)
        output["n_reviewed"] = len(output["results"])
        output["n_errors"] = len(output["errors"])
        save(path, output)
        if consecutive_errors >= 2 or (output["errors"] and output["errors"][-1]["pair_id"] == pid
                                       and output["errors"][-1]["type"] == "FatalAPIError"):
            break
    return output


def exact_mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)) if n else 1.0


def haiku_cost_estimate(stats):
    # Official standard/global pricing, retrieved 2026-11-02; not an invoice.
    # No cache_control is sent by the original transport (normally zero cache usage).
    if stats.get("max_request_input_tokens", stats["input_tokens"]) > 100_000:
        return None
    return round((stats["input_tokens"] * .10 + stats["output_tokens"] * .50
                  + stats["cache_read_input_tokens"] * .01
                  + stats["cache_creation_input_tokens"] * .125) / 1_000_000, 8)


def audit_outputs(selected):
    """Fail closed on incomplete paired arms, free response text or leaked keys."""
    from gzcmd.llm_review import VALID_REASON_CODES
    expected = set(selected.index)
    result_rows = 0
    allowed = {"pair_id", "decision", "confidence", "reason_codes", "quality_flags", "protocol",
               "agent_a_decision", "agent_b_decision", "arbiter_decision", "total_latency_s", "total_tokens"}
    for tag in ("deepseek", "haiku"):
        for arm in ("controle", "sem_pcal"):
            run = json.loads((OUT / f"{MODELS[tag]}_{arm}.json").read_text(encoding="utf-8"))
            assert set(r["pair_id"] for r in run["results"]) == expected, "Incomplete paired arm"
            assert len(run["results"]) == len(expected), "Duplicate results"
            assert not run["errors"], "Historical review errors need manual inspection"
            assert run["removed_fields"] == (REMOVED_FIELDS if arm == "sem_pcal" else [])
            for r in run["results"]:
                assert set(r) <= allowed, "Unapproved result field"
                assert r["decision"] in DECISIONS
                assert set(r["reason_codes"]) <= VALID_REASON_CODES
                assert r["protocol"] in ("consensus", "arbiter")
                assert all(isinstance(v, bool) for v in r["quality_flags"].values())
                assert not r["quality_flags"].get("pii_leak_detected", False), "Provider flagged possible PII"
            result_rows += len(run["results"])
    files = [Path(__file__), ROOT / "tests/test_exp2_ablacao_dossie.py",
             ROOT / "data/ablacao_dossie_pares.csv", ROOT / "data/ablacao_dossie_resultado.csv",
             ROOT / "data/ablacao_dossie_relatorio.md", *OUT.glob("*.json")]
    # Compare secrets in memory only; neither values nor matching excerpts are emitted.
    secrets = [os.environ.get(key, "") for key in ("OPENCODE_GO_API_KEY", "ANTHROPIC_API_KEY")]
    assert all(secrets), "Cannot audit secret values without loaded environment"
    for path in files:
        content = path.read_text(encoding="utf-8")
        assert not any(secret in content for secret in secrets), "Secret audit failed (value suppressed)"
    audit = {"paired_complete": True, "pairs_per_arm": len(expected), "main_result_rows": result_rows,
             "secret_matches": 0, "files_scanned": len(files), "free_response_fields_saved": False,
             "review_errors": 0, "provider_pii_flags": 0}
    save(OUT / "auditoria.json", audit)
    print("AUDIT " + json.dumps(audit), flush=True)


def summarize(selected):
    import pandas as pd
    rows, summaries = [], []
    for tag in ("deepseek", "haiku"):
        paths = [OUT / f"{MODELS[tag]}_{arm}.json" for arm in ("controle", "sem_pcal")]
        if not all(p.exists() for p in paths):
            continue
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        maps = [{r["pair_id"]: r["decision"] for r in run["results"]} for run in runs]
        ids = sorted(set(selected.index) & maps[0].keys() & maps[1].keys())
        b = c = fn_b = fn_c = agree = original_fn_match = original_fn_not_nonmatch = 0
        transitions = Counter()
        for pid in ids:
            control, ablated, original = maps[0][pid], maps[1][pid], selected.loc[pid, LABELS[tag]]
            b += control != "MATCH" and ablated == "MATCH"
            c += control == "MATCH" and ablated != "MATCH"
            fn_b += control == "NONMATCH" and ablated != "NONMATCH"
            fn_c += control != "NONMATCH" and ablated == "NONMATCH"
            agree += control == original
            original_fn_match += original == "NONMATCH" and ablated == "MATCH"
            original_fn_not_nonmatch += original == "NONMATCH" and ablated != "NONMATCH"
            transitions[f"{control}->{ablated}"] += 1
            rows.append({"pair_id": pid, "modelo": MODELS[tag], "nota": selected.loc[pid, "nota_final"],
                         "original": original, "controle": control, "sem_pcal": ablated,
                         "fn_original_revertido_match": original == "NONMATCH" and ablated == "MATCH",
                         "fn_controle_revertido_match": control == "NONMATCH" and ablated == "MATCH"})
        summary = {"model": MODELS[tag], "n": len(ids), "selected_n": len(selected),
                   "complete": len(ids) == len(selected),
                   "controle": {d: sum(maps[0][p] == d for p in ids) for d in DECISIONS},
                   "sem_pcal": {d: sum(maps[1][p] == d for p in ids) for d in DECISIONS},
                   "transitions": {f"{a}->{b}": transitions[f"{a}->{b}"] for a in DECISIONS for b in DECISIONS},
                   "gained_match": b, "lost_match": c, "mcnemar_match_exact_p": exact_mcnemar(b, c),
                   "fn_to_nonfn": fn_b, "nonfn_to_fn": fn_c, "mcnemar_nonmatch_exact_p": exact_mcnemar(fn_b, fn_c),
                   "original_fn_to_match": original_fn_match,
                   "original_fn_to_match_or_unsure": original_fn_not_nonmatch,
                   "original_fn_n": sum(selected.loc[p, LABELS[tag]] == "NONMATCH" for p in ids),
                   "control_fn_n": sum(maps[0][p] == "NONMATCH" for p in ids),
                   "control_fn_to_match": transitions["NONMATCH->MATCH"],
                   "control_fn_to_unsure": transitions["NONMATCH->UNSURE"],
                   "control_original_agree": agree, "control_original_agreement": agree / len(ids) if ids else None,
                   "transport": {arm: run["transport"] for arm, run in zip(("controle", "sem_pcal"), runs)},
                   "elapsed_s": {arm: run["elapsed_s"] for arm, run in zip(("controle", "sem_pcal"), runs)},
                   "review_errors": sum(len(run["errors"]) for run in runs), "cost_usd": None}
        summary["cost_estimate_usd"] = ({arm: haiku_cost_estimate(run["transport"])
                                          for arm, run in zip(("controle", "sem_pcal"), runs)}
                                         if tag == "haiku" else None)
        summaries.append(summary)
    pd.DataFrame(rows).to_csv(ROOT / "data/ablacao_dossie_resultado.csv", index=False)
    totals = {"calls": sum(m["calls"] for s in summaries for m in s["transport"].values()),
              "http_errors": sum(m["http_errors"] for s in summaries for m in s["transport"].values()),
              "parse_errors": sum(m.get("response_parse_errors", 0) for s in summaries for m in s["transport"].values()),
              "review_errors": sum(s["review_errors"] for s in summaries),
              "note": "Main two models only; optional Kimi probe reported separately."}
    save(OUT / "resumo.json", {"models": summaries, "totals": totals})
    report = ["# M4 — Ablação do dossiê (vigilância)", "",
              f"Seleção: {len(selected)} vínculos ouro classificados NONMATCH por ≥2 revisores; "
              f"{int((selected.n_revisores_erraram == 3).sum())} pelos três.",
              "O tamanho decorre do critério explícito NONMATCH por pelo menos dois revisores; "
              "não foi ampliado para atingir a expectativa aproximada do despacho.",
              "", "## Método e fronteira de informação", "",
              "Gabarito reconstruído do extrato dtype=str, PAR ∈ {1,2}; CSV de comparação conferido com os três JSONs originais. "
              "Pipeline v3/fit_platt e dossiês originais reconstruídos; ambos os braços usam os mesmos prompts, parser e protocolo A+B+árbitro. "
              "Removido o grupo inteiro model_outputs: " + ", ".join(REMOVED_FIELDS) + ". "
              "band, EVR e guardrails também podem denunciar probabilidade/decisão. Subescores, nota e MACD mantidos. "
              "Sem alterar o prompt: instruções sobre campos ausentes permanecem (ablação do dossiê, não do prompt).",
              "", "DeepSeek: T=0, top_k=1, thinking disabled/reasoning_effort none, 800 tokens. "
              "Haiku: sem temperature na API, 2500 tokens, sem pensamento estendido. "
              "Ordem dos braços contrabalanceada por modelo; até quatro tarefas simultâneas. "
              "Retries originais com backoff; HTTP 400/401/402/403/404 interrompem sem insistir. "
              "Checkpoint por par; retomada não repete sucessos. Não se salvam raw, evidência livre, dossiês ou PII.",
              "", "McNemar exato bilateral: endpoint primário MATCH vs demais; secundário NONMATCH vs demais. "
              "UNSURE não é correção estrita, mas deixa de ser FN explícito. Sem ajuste de multiplicidade; análise selecionada em FN compartilhados, não generalizável a todos os pares.",
              "", "## Resultados", ""]
    report += ["| Modelo | N | Controle M/U/N | Sem p_cal M/U/N | FN controle → MATCH | FN controle → UNSURE | McNemar MATCH p | Controle=original |",
               "|---|---:|---|---|---:|---:|---:|---:|"]
    for s in summaries:
        counts = lambda arm: "/".join(str(s[arm][d]) for d in DECISIONS)
        report += [f"| {s['model']} | {s['n']} | {counts('controle')} | {counts('sem_pcal')} | "
                   f"{s['control_fn_to_match']} | {s['control_fn_to_unsure']} | {s['mcnemar_match_exact_p']:.6g} | "
                   f"{s['control_original_agree']}/{s['n']} |"]
    report += ["", "Totais dos dois modelos principais: " + json.dumps(totals, ensure_ascii=False), ""]
    for s in summaries:
        report += [f"### {s['model']}", "", "```json", json.dumps(s, ensure_ascii=False, indent=2), "```", ""]
        report += ["Interpretação: " + (
            "a ablação melhora MATCH no conjunto pareado, compatível com ancoragem nos outputs do motor; não isola p_cal de seus proxies."
            if s["gained_match"] > s["lost_match"] else
            "não há ganho líquido de MATCH no conjunto pareado; não sustenta a hipótese de ancoragem como explicação destes FN."), ""]
    report += ["## Custo, tempo e limites", "",
               "Tokens de todas as respostas HTTP bem-sucedidas (inclusive tentativas com JSON inválido) registrados por braço. "
               "USD efetivamente faturado não é devolvido por estes transportes; Go consome quota de assinatura. "
               "Para Haiku, cost_estimate_usd usa tarifa pública standard/global por milhão: entrada $0.10, saída $0.50, "
               "cache read $0.01, cache write 5m $0.125 (prompts ≤100k, sem cache solicitado). Não é fatura. "
               "Fonte consultada: https://platform.claude.com/docs/en/about-claude/pricing (2026-11-02). "
               "elapsed_s mede tempo de parede por braço, inclui retries; braços paralelos não devem ter seus tempos somados como duração de parede total. "
               "Concordância controle-original mede repetibilidade no subconjunto selecionado, não acurácia geral. Campos ausentes podem induzir UNSURE."]
    selection_path = OUT / "selecao.json"
    if selection_path.exists():
        report += ["", "## Auditoria da seleção", "", "```json", selection_path.read_text(encoding="utf-8"), "```"]
    smoke_path = OUT / "fumaca_inspecao.json"
    if smoke_path.exists():
        inspection = json.loads(smoke_path.read_text(encoding="utf-8")).get("inspection", {})
        report += ["", "## Fumaça prévia e inspeção", "", json.dumps(inspection, ensure_ascii=False),
                   "Os dois pares iniciais por braço foram inspecionados antes da execução completa; sucessos reaproveitados sem novas chamadas."]
    execution_path = OUT / "ultima_execucao.json"
    if execution_path.exists():
        report += ["", "Última execução (retomada após fumaça): " + execution_path.read_text(encoding="utf-8")]
    interruption_path = OUT / "interrupcoes.json"
    if interruption_path.exists():
        report += ["", "## Interrupção operacional e limites da contagem", "",
                   interruption_path.read_text(encoding="utf-8"),
                   "As estatísticas preservadas são um limite inferior das chamadas/tokens, pois a primeira retomada "
                   "foi terminada pelo limite de tempo do executor durante pares em andamento. Os resultados completos já "
                   "checkpointados não foram repetidos; pares incompletos foram retomados. A retomada final usa journaling "
                   "por chamada e executor sem limite de tempo. Isso não altera o teste pareado dos resultados finais, mas "
                   "impede afirmar custo faturado ou número histórico exato de chamadas não checkpointadas."]
    kimi = OUT / f"{MODELS['kimi']}_sem_pcal.json"
    if kimi.exists():
        k = json.loads(kimi.read_text(encoding="utf-8"))
        report += ["", "## Sondagem opcional Kimi K3", "", json.dumps({"n": len(k["results"]), "errors": k["errors"],
                   "transport": k["transport"], "elapsed_s": k["elapsed_s"]}, ensure_ascii=False),
                   "reasoning_effort=low; no máximo três pares, interrupção imediata em HTTP 400/quota; exploratório, sem inferência pareada."]
    (ROOT / "data/ablacao_dossie_relatorio.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for s in summaries:
        print("SUMMARY " + json.dumps(s, ensure_ascii=False), flush=True)
    print("TOTALS " + json.dumps(totals, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--try-kimi", action="store_true")
    parser.add_argument("--arm", choices=("ambos", "controle", "sem_pcal"), default="ambos",
                        help="sem_pcal removes all model_outputs; default runs both paired arms")
    args = parser.parse_args()
    import run_exp2_opencode_go as go
    import run_exp2_anthropic as ant
    import requests
    # Avoid arbitrary API exception bodies in logs, including credential-like text.
    logging.disable(logging.CRITICAL)
    original_extract = go.llm_review._extract_json
    def metered_extract(text):
        try:
            return original_extract(text)
        except (ValueError, KeyError):
            stats = _LOCAL.stats
            stats["response_parse_errors"] = stats.get("response_parse_errors", 0) + 1
            raise
    go.llm_review._extract_json = metered_extract
    ant._extract_json = metered_extract
    selected = select_pairs()
    if args.audit_only:
        audit_outputs(selected)
        return
    if args.prepare_only:
        return
    if args.summarize_only:
        summarize(selected)
        return
    for key in ("ANTHROPIC_API_KEY", "OPENCODE_GO_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"Missing {key} (value not shown)")
    # One thread-local model per arm; original Go wrapper injects its exact fields.
    class DynamicGoTransport:
        def post(self, *a, **kw):
            model = _LOCAL.model
            return go._RequestsWithSession(requests, go._session_id(f"ablacao-{model}"),
                                            go.EXTRA_BY_MODEL[model]).post(*a, **kw)
    go.llm_review.requests = MeteredRequests(DynamicGoTransport())
    df, _ = go.exp2.run_v3(input_csv=str(ROOT / "data/COMPARADORSEMIDENT.csv"),
                           config_path=str(ROOT / "gzcmd/gzcmd_v3_config.yaml"), mode="vigilancia", p_cal="fit_platt")
    dossiers = {d.pair_id: d for d in go.exp2.build_dossiers(df, only_llm_review=True)}
    ids = list(selected.index)
    if not set(ids) <= dossiers.keys():
        raise ValueError("Selected pairs missing from rebuilt LLM_REVIEW dossiers")
    if args.limit:
        if not 1 <= args.limit <= 100:
            raise ValueError("limit must be 1..100")
        ids = ids[:args.limit]
    started = time.monotonic()
    # Counterbalance which arm is submitted first across models.
    jobs = [("deepseek", "controle"), ("haiku", "sem_pcal"),
            ("deepseek", "sem_pcal"), ("haiku", "controle")]
    if args.arm != "ambos":
        jobs = [(tag, arm) for tag, arm in jobs if arm == args.arm]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_arm, tag, arm, dossiers, ids) for tag, arm in jobs]
        for f in as_completed(futures):
            f.result()
    if args.try_kimi:
        run_arm("kimi", "sem_pcal", dossiers, list(selected.index)[:3])
    wall = round(time.monotonic() - started, 2)
    save(OUT / "ultima_execucao.json", {"wall_seconds": wall, "limit": args.limit,
                                       "workers": args.workers, "arms": args.arm,
                                       "note": "resumed successes are not called again"})
    print(f"WALL_SECONDS {wall:.2f}", flush=True)
    summarize(selected)


if __name__ == "__main__":
    main()
