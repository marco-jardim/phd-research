from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class BandDefinition:
    name: str
    min: float | None
    max: float | None
    inclusive_max: bool = False


@dataclass(frozen=True)
class BandsConfig:
    definitions: tuple[BandDefinition, ...]


@dataclass(frozen=True)
class ModePolicyConfig:
    false_positive_cost: float
    false_negative_cost: float
    llm_review_cost: float
    min_auto_match_threshold: float | None
    max_auto_nonmatch_threshold: float | None
    llm_max_calls_per_window: int


@dataclass(frozen=True)
class DecisionPolicyConfig:
    modes: dict[str, ModePolicyConfig]


@dataclass(frozen=True)
class LLMReviewConfig:
    enabled: bool
    error_rates_by_band: dict[str, dict[str, float]]


@dataclass(frozen=True)
class GZCMDConfig:
    version: str
    bands: BandsConfig
    decision_policy: DecisionPolicyConfig
    llm_review: LLMReviewConfig


def _as_mapping(x: Any, *, field: str) -> Mapping[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, Mapping):
        raise TypeError(f"{field} must be a mapping, got {type(x).__name__}")
    return x


def _load_band_definitions(data: Mapping[str, Any]) -> tuple[BandDefinition, ...]:
    raw_defs = _as_mapping(data.get("bands"), field="bands").get("definitions")
    if raw_defs is None:
        raise ValueError("Missing required config: bands.definitions")

    # Support both styles:
    #  - list of mappings with an explicit `name`
    #  - mapping of {band_name: {min/max/...}}
    items: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw_defs, Mapping):
        for name, cfg in raw_defs.items():
            items.append(
                (str(name), _as_mapping(cfg, field=f"bands.definitions[{name}]"))
            )
    elif isinstance(raw_defs, list):
        if not raw_defs:
            raise ValueError("bands.definitions must be non-empty")
        for i, item in enumerate(raw_defs):
            if not isinstance(item, Mapping):
                raise TypeError(f"bands.definitions[{i}] must be a mapping")
            name = str(item.get("name", "")).strip()
            if not name:
                raise ValueError(f"bands.definitions[{i}].name must be non-empty")
            items.append((name, item))
    else:
        raise TypeError("bands.definitions must be a mapping or a list")

    out: list[BandDefinition] = []
    seen: set[str] = set()
    inclusive_max_count = 0

    for i, (name, item) in enumerate(items):
        name = str(name).strip()
        if not name:
            raise ValueError(f"bands.definitions[{i}] band name must be non-empty")
        if name in seen:
            raise ValueError(f"Duplicate band name: {name!r}")
        seen.add(name)

        min_v = item.get("min")
        max_v = item.get("max")
        min_f = None if min_v is None else float(min_v)
        max_f = None if max_v is None else float(max_v)
        inclusive_max = bool(item.get("inclusive_max", False))
        if inclusive_max:
            inclusive_max_count += 1

        if (min_f is not None) and (max_f is not None) and not (min_f < max_f):
            raise ValueError(
                f"Invalid band range for {name!r}: expected min < max, got {min_f} >= {max_f}"
            )

        out.append(
            BandDefinition(
                name=name,
                min=min_f,
                max=max_f,
                inclusive_max=inclusive_max,
            )
        )

    if inclusive_max_count > 1:
        raise ValueError("Only one band may set inclusive_max: true")

    return tuple(out)


def _load_decision_policy(data: Mapping[str, Any]) -> DecisionPolicyConfig:
    modes = _as_mapping(
        _as_mapping(data.get("decision_policy"), field="decision_policy").get("modes"),
        field="decision_policy.modes",
    )
    if not modes:
        raise ValueError("Missing required config: decision_policy.modes")

    parsed: dict[str, ModePolicyConfig] = {}
    for mode_name, mode_cfg in modes.items():
        cfg = _as_mapping(mode_cfg, field=f"decision_policy.modes[{mode_name}]")

        costs = _as_mapping(
            cfg.get("costs"), field=f"decision_policy.modes[{mode_name}].costs"
        )
        auto = _as_mapping(
            cfg.get("auto"), field=f"decision_policy.modes[{mode_name}].auto"
        )
        llm_budget = _as_mapping(
            cfg.get("llm_budget"),
            field=f"decision_policy.modes[{mode_name}].llm_budget",
        )

        parsed[mode_name] = ModePolicyConfig(
            false_positive_cost=float(costs["false_positive"]),
            false_negative_cost=float(costs["false_negative"]),
            llm_review_cost=float(costs["llm_review"]),
            min_auto_match_threshold=(
                None
                if auto.get("min_auto_match_threshold") is None
                else float(auto.get("min_auto_match_threshold"))
            ),
            max_auto_nonmatch_threshold=(
                None
                if auto.get("max_auto_nonmatch_threshold") is None
                else float(auto.get("max_auto_nonmatch_threshold"))
            ),
            llm_max_calls_per_window=int(llm_budget["max_calls_per_window"]),
        )

    return DecisionPolicyConfig(modes=parsed)


def _load_llm_review(data: Mapping[str, Any]) -> LLMReviewConfig:
    llm = _as_mapping(data.get("llm_review"), field="llm_review")
    enabled = bool(llm.get("enabled", False))

    reliability = _as_mapping(llm.get("reliability"), field="llm_review.reliability")
    raw_rates = reliability.get("error_rates_by_band") or {}
    if not isinstance(raw_rates, Mapping):
        raise TypeError("llm_review.reliability.error_rates_by_band must be a mapping")

    rates: dict[str, dict[str, float]] = {}
    for band, rate_cfg in raw_rates.items():
        cfg = _as_mapping(
            rate_cfg, field=f"llm_review.reliability.error_rates_by_band[{band}]"
        )
        rates[str(band)] = {"e_fp": float(cfg["e_fp"]), "e_fn": float(cfg["e_fn"])}

    return LLMReviewConfig(enabled=enabled, error_rates_by_band=rates)


def load_config(path: str | Path) -> GZCMDConfig:
    """Load `gzcmd_v3_config.yaml`-style config into typed dataclasses."""

    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Config root must be a mapping")

    version = str(data.get("version", "")).strip()
    if not version:
        raise ValueError("Missing required config: version")

    bands = BandsConfig(definitions=_load_band_definitions(data))
    decision_policy = _load_decision_policy(data)
    llm_review = _load_llm_review(data)

    return GZCMDConfig(
        version=version,
        bands=bands,
        decision_policy=decision_policy,
        llm_review=llm_review,
    )
