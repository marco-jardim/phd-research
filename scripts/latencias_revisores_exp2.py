"""Resume latências históricas do Exp2 sem acessar os reruns em andamento.

Mediana e p90 usam total_latency_s por par (percentil linear, sem amostragem).
A duração do lote e o número de trabalhadores são copiados dos metadados;
não se estima duração paralela pela soma das latências individuais.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("Kimi K2.5", "exp2_llm_sync_vigilancia.json"),
    ("DeepSeek V4.1 Flash", "opencode_go/exp2_llm_sync_vigilancia.json"),
    ("DeepSeek V4.1 Flash", "opencode_go/exp2_llm_sync_confirmacao.json"),
    ("Claude Haiku 5.5", "anthropic/exp2_llm_sync_vigilancia.json"),
    ("Claude Haiku 5.5", "anthropic/exp2_llm_sync_confirmacao.json"),
    ("Kimi K3", "opencode_go_kimi/exp2_llm_sync_confirmacao.json"),
]


def resumir_latencias(data: dict) -> dict:
    """Exige latência válida em todos os resultados persistidos."""
    latencies = np.array([row["total_latency_s"] for row in data["results"]])
    if not len(latencies) or not np.isfinite(latencies).all() or (latencies < 0).any():
        raise ValueError("Latências ausentes, não finitas ou negativas")
    metrics = data.get("metrics", {})
    return {
        "modo": data["mode"],
        "n": len(latencies),
        "mediana_s": float(np.median(latencies)),
        "p90_s": float(np.percentile(latencies, 90, method="linear")),
        "media_s": float(np.mean(latencies)),
        "duracao_lote_s": metrics.get("total_elapsed_s", data.get("elapsed_s")),
        "trabalhadores": data.get("workers"),
    }


def main() -> None:
    rows = []
    for model, relative in SOURCES:
        path = ROOT / "data" / "sprint3b" / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"modelo": model, **resumir_latencias(data),
                     "fonte": path.relative_to(ROOT).as_posix()})
    result = pd.DataFrame(rows)
    output = ROOT / "data" / "exp2_latencias_revisores.csv"
    result.to_csv(output, index=False, float_format="%.6f")
    print(result.to_string(index=False))
    print(f"Artefato: {output}")


if __name__ == "__main__":
    main()
