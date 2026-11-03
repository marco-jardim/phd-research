"""M3 band reliability; explicit fixed fallback for n<10 or undefined estimates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .policy_engine import LLMError


def reference_rates_model(source: str | Path | None) -> str:
    """v4's pooled block includes nondiscriminating DeepSeek; never use it by default."""
    if source is None:
        return "pooled"
    data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if data.get("prompt") == "v4" and data.get("dossie") == "sem_mo":
        if not all("claude-haiku-5-5" in block.get("models", {}) for block in data["by_mode"].values()):
            raise ValueError("v4 reference requires the Haiku model block in every mode")
        return "claude-haiku-5-5"
    return "pooled"


def load_band_rates(
    fixed: dict[str, dict[str, float]], *, mode: str,
    source: str | Path | None = None, model: str | None = None,
) -> dict[str, LLMError]:
    block = {}
    if source is not None:
        data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        model = model or reference_rates_model(source)
        mode_block = data["by_mode"][mode]
        if model == "pooled":
            block = mode_block["pooled"]
        else:
            if model not in mode_block.get("models", {}):
                raise ValueError(f"Missing explicit reviewer block: {mode}/{model}")
            block = mode_block["models"][model]
    estimates = block.get("estimates_by_band", {})
    engine_rates = block.get("error_rates_by_band", {})
    result = {}
    for band, fallback in fixed.items():
        estimate = estimates.get(band, {})
        enough = int(estimate.get("n", 0)) >= 10
        rates = engine_rates.get(band, {}) if enough else {}
        # M3's engine mapping already restores fixed rates for denominator zero.
        fp = rates.get("e_fp", fallback["e_fp"])
        fn = rates.get("e_fn", fallback["e_fn"])
        u = estimate.get("u") if enough else None
        values = [fallback["e_fp"] if fp is None else fp,
                  fallback["e_fn"] if fn is None else fn,
                  fallback.get("e_unsure", 0.0) if u is None else u]
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f"Invalid reviewer probabilities: {mode}/{model}/{band}")
        result[band] = LLMError(*map(float, values))
    return result
