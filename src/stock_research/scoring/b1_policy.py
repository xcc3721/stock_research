#!/usr/bin/env python3
"""B1 comprehensive scoring policies."""

from __future__ import annotations

from typing import Any


LEGACY_LOGIC = "legacy"
V2_LOGIC = "v2_gate_washout_raw_ge_0p2"
V3_LOGIC = "v3_winner"
HOT_UP_LOGIC = "hot_up_use_v2_else_v3"
DEFAULT_B1_LOGIC = V3_LOGIC

ACTIVE_COMPONENT_WEIGHTS: dict[str, float] = {
    "b1_pattern": 0.40,
    "red_fat_green_thin": 0.25,
    "dealer_behavior": 0.20,
    "sb1_bonus": 0.0,
}
LEGACY_COMPONENT_WEIGHTS: dict[str, float] = {
    "b1_pattern": 0.40,
    "red_fat_green_thin": 0.25,
    "dealer_behavior": 0.20,
    "sb1_bonus": 0.10,
}
LEGACY_ELASTICITY_WEIGHT = 0.15
V2_ELASTICITY_WEIGHT = 0.175
WASHOUT_GATE_THRESHOLD = 0.2
V3_EXTREME_MOVE_WEIGHT = 0.20
V3_INTRADAY_RANGE_WEIGHT = 0.10
V3_BIG_MOVE_DENSITY_WEIGHT = 0.15


def extract_b1_params(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    params: dict[str, Any] = {}
    direct = config.get("b1_comprehensive")
    if isinstance(direct, dict):
        direct_params = direct.get("params")
        params.update(direct_params if isinstance(direct_params, dict) else direct)
    nested = config.get("factors", {}).get("b1_comprehensive", {}) if isinstance(config.get("factors"), dict) else {}
    nested_params = nested.get("params") if isinstance(nested, dict) else None
    if isinstance(nested_params, dict):
        params.update(nested_params)
    return params


def resolve_b1_logic(config: dict[str, Any] | None) -> str:
    params = extract_b1_params(config)
    raw_mode = (
        params.get("scoring_mode")
        or params.get("score_mode")
        or params.get("logic")
        or params.get("version")
        or DEFAULT_B1_LOGIC
    )
    mode = str(raw_mode).strip().lower()
    aliases = {
        "old": LEGACY_LOGIC,
        "v1": LEGACY_LOGIC,
        "v1_legacy": LEGACY_LOGIC,
        "teacher": LEGACY_LOGIC,
        "current": LEGACY_LOGIC,
        "legacy": LEGACY_LOGIC,
        "v2": V2_LOGIC,
        "v2_gate_washout_raw_ge_0.2": V2_LOGIC,
        "v2_gate_washout_raw_ge_0p2": V2_LOGIC,
        "v3": V3_LOGIC,
        "v3_winner": V3_LOGIC,
        "hot_up": HOT_UP_LOGIC,
        "hot_up_use_v2_else_v3": HOT_UP_LOGIC,
    }
    return aliases.get(mode, DEFAULT_B1_LOGIC)


def _clip_score(value: float) -> float:
    return float(max(0.0, min(float(value), 1.0)))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_b1_score_payload(
    *,
    component_values: dict[str, float],
    elasticity_parts: dict[str, float],
    washout_score: float,
    b1_logic: str,
    market_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_logic = b1_logic if b1_logic in {LEGACY_LOGIC, V2_LOGIC, V3_LOGIC, HOT_UP_LOGIC} else DEFAULT_B1_LOGIC
    gate_active = bool((market_state or {}).get("is_hot_up"))
    if resolved_logic == HOT_UP_LOGIC:
        selected_branch = V2_LOGIC if gate_active else V3_LOGIC
    else:
        selected_branch = resolved_logic

    elasticity_raw = _to_float(component_values.get("elasticity"))
    if "elasticity" not in component_values:
        elasticity_raw = sum(_to_float(elasticity_parts.get(key)) for key in ("extreme_move", "intraday_range", "big_move_density"))

    fixed_weights = LEGACY_COMPONENT_WEIGHTS if selected_branch == LEGACY_LOGIC else ACTIVE_COMPONENT_WEIGHTS
    base_contributions = {
        name: _to_float(component_values.get(name)) * weight
        for name, weight in fixed_weights.items()
    }
    washout_gate = _to_float(washout_score) >= WASHOUT_GATE_THRESHOLD

    if selected_branch == LEGACY_LOGIC:
        elasticity_contribution = elasticity_raw * LEGACY_ELASTICITY_WEIGHT
        elasticity_weights = {
            "effective": LEGACY_ELASTICITY_WEIGHT,
            "extreme_move": LEGACY_ELASTICITY_WEIGHT,
            "intraday_range": LEGACY_ELASTICITY_WEIGHT,
            "big_move_density": LEGACY_ELASTICITY_WEIGHT,
        }
    elif selected_branch == V2_LOGIC:
        effective_weight = V2_ELASTICITY_WEIGHT if washout_gate else LEGACY_ELASTICITY_WEIGHT
        elasticity_contribution = elasticity_raw * effective_weight
        elasticity_weights = {
            "effective": effective_weight,
            "extreme_move": effective_weight,
            "intraday_range": effective_weight,
            "big_move_density": effective_weight,
        }
    else:
        extreme_move_weight = V3_EXTREME_MOVE_WEIGHT if washout_gate else LEGACY_ELASTICITY_WEIGHT
        extreme_move = _to_float(elasticity_parts.get("extreme_move"))
        intraday_range = _to_float(elasticity_parts.get("intraday_range"))
        big_move_density = _to_float(elasticity_parts.get("big_move_density"))
        elasticity_contribution = (
            extreme_move * extreme_move_weight
            + intraday_range * V3_INTRADAY_RANGE_WEIGHT
            + big_move_density * V3_BIG_MOVE_DENSITY_WEIGHT
        )
        effective_weight = elasticity_contribution / elasticity_raw if elasticity_raw else LEGACY_ELASTICITY_WEIGHT
        elasticity_weights = {
            "effective": float(effective_weight),
            "extreme_move": float(extreme_move_weight),
            "intraday_range": V3_INTRADAY_RANGE_WEIGHT,
            "big_move_density": V3_BIG_MOVE_DENSITY_WEIGHT,
        }

    contributions = {**base_contributions, "elasticity": float(elasticity_contribution)}
    score_raw = float(sum(contributions.values()))
    score_final = _clip_score(score_raw)
    return {
        "score_raw": score_raw,
        "score_final": score_final,
        "clipped": bool(score_raw != score_final),
        "contributions": contributions,
        "elasticity_weights": elasticity_weights,
        "component_weights": {**fixed_weights, "elasticity": float(elasticity_weights["effective"])},
        "b1_logic": resolved_logic,
        "selected_branch": selected_branch,
        "market_gate_active": gate_active if resolved_logic == HOT_UP_LOGIC else False,
        "washout_gate_active": bool(washout_gate),
        "washout_gate_threshold": WASHOUT_GATE_THRESHOLD,
        "market_state": market_state or {},
    }
