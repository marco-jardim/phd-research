"""Price recorded per-pair token usage; not a provider invoice or a new API call."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PRICES_USD_PER_MILLION = {"input_tokens": .10, "output_tokens": .50,
                          "cache_read_input_tokens": .01, "cache_creation_input_tokens": .125}


def price_usage(usage: dict) -> float:
    if usage.get("max_request_input_tokens", usage["input_tokens"]) > 100_000:
        raise ValueError("Recorded context exceeds the historical price tier")
    values = []
    for field, price in PRICES_USD_PER_MILLION.items():
        count = usage[field]
        if not isinstance(count, int) or count < 0:
            raise ValueError("Token usage must be a nonnegative integer")
        values.append(count * price / 1_000_000)
    return sum(values)


def selected_cost(paths: list[Path], pair_ids: set[str]) -> tuple[dict, list[dict]]:
    costs, sources = {}, []
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        journal = path.with_name(path.stem + "_journal")
        ids = {r["pair_id"] for r in run["results"]}
        chosen = ids & pair_ids
        usage_digest = hashlib.sha256()
        for pid in sorted(chosen):
            if pid in costs:
                raise ValueError("A selected pair is charged twice across cohorts")
            if len(pid) != 16 or any(c not in "0123456789abcdef" for c in pid):
                raise ValueError("Invalid safe pair hash")
            usage_file = journal / f"{pid}.transport.json"
            usage_bytes = usage_file.read_bytes()
            usage_digest.update(pid.encode("ascii") + hashlib.sha256(usage_bytes).digest())
            usage = json.loads(usage_bytes.decode("utf-8"))
            costs[pid] = price_usage(usage)
        sources.append({"source": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "pares_selecionados": len(chosen), "custo_usd_selecionados": sum(costs[pid] for pid in sorted(chosen)),
                        "selected_journal_sha256": usage_digest.hexdigest(),
                        "custo_usd_run_completo_estimado": run.get("cost_estimate_usd")})
    if set(costs) != pair_ids:
        raise ValueError("Recorded token journals do not cover all selected LLM pairs")
    return costs, sources
