#!/usr/bin/env python3
"""
Standalone mechanical anchor score calculator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple


logger = logging.getLogger("explicit_mechanical_anchor")


class ExplicitMechanicalAnchor:
    """Calculate the current mechanical score semantics without ScoringEngine."""

    REQUIRED_FACTORS = (
        "ma_structure",
        "b1_comprehensive",
        "kline_patterns",
        "supply_analysis",
        "risk_penalty",
    )

    def __init__(self, scoring_config: Dict[str, Any]):
        """Initialize the calculator from a scoring_config payload."""
        if not scoring_config:
            raise ValueError("scoring_config cannot be empty")

        factor_configs = scoring_config.get("factors")
        if not isinstance(factor_configs, dict):
            raise ValueError("scoring_config must contain a factors mapping")

        self.factor_weights: Dict[str, float] = {}
        self.factor_neutral: Dict[str, float] = {}
        self.factor_adjustments: Dict[str, List[Dict[str, float]]] = {}
        self.bonus_factors: Set[str] = set()

        for factor_name, factor_config in factor_configs.items():
            if not isinstance(factor_config, dict):
                continue

            if factor_config.get("enabled", False):
                self.factor_weights[factor_name] = float(factor_config.get("weight", 0.0))
            else:
                self.factor_weights[factor_name] = 0.0

            adjustments = factor_config.get("adjustments")
            if isinstance(adjustments, list):
                normalized_adjustments: List[Dict[str, float]] = []
                for item in adjustments:
                    if not isinstance(item, dict):
                        continue
                    try:
                        normalized_adjustments.append(
                            {
                                "min": float(item.get("min", 0.0)),
                                "max": float(item.get("max", 1.0)),
                                "multiplier": float(item.get("multiplier", 1.0)),
                            }
                        )
                    except (TypeError, ValueError):
                        continue
                if normalized_adjustments:
                    self.factor_adjustments[factor_name] = normalized_adjustments

            neutral_val: Optional[float] = None
            if isinstance(factor_config.get("neutral"), (int, float)):
                neutral_val = float(factor_config["neutral"])
            else:
                params = factor_config.get("params", {})
                if isinstance(params, dict) and isinstance(params.get("neutral_score"), (int, float)):
                    neutral_val = float(params["neutral_score"])
            if neutral_val is not None and 0.0 < neutral_val < 1.0:
                self.factor_neutral[factor_name] = neutral_val

            if bool(factor_config.get("bonus", False)):
                self.bonus_factors.add(factor_name)

        missing_factors = [name for name in self.REQUIRED_FACTORS if name not in self.factor_weights]
        if missing_factors:
            raise ValueError(f"scoring_config is missing required factors: {missing_factors}")

        self._bonus_weight_total = sum(
            abs(self.factor_weights.get(name, 0.0)) for name in self.bonus_factors
        )

    def calculate_score(self, factors: Dict[str, float], code: str) -> Dict[str, Any]:
        """Calculate the explicit mechanical score and return trace data."""
        try:
            weighted_scores: Dict[str, float] = {}
            total_weighted_score = 0.0
            total_possible_score = 0.0
            bonus_total_score = 0.0
            factor_traces: Dict[str, Dict[str, float]] = {}

            raw_kline_score = factors.get("kline_patterns", 0.0)
            smart_fusion_trace = self._build_smart_fusion_trace(factors, raw_kline_score)
            needs_smart_fusion = bool(smart_fusion_trace["applied"])

            for factor_name, factor_value in factors.items():
                weight = self.factor_weights.get(factor_name, 0.0)
                raw_value = factor_value
                adjusted_value = raw_value
                centered_value = raw_value
                effective_weight = 0.0
                weighted_score = 0.0

                if weight != 0.0:
                    is_bonus = factor_name in self.bonus_factors
                    adjusted_value = self._apply_adjustment(factor_name, factor_value)
                    centered_value = adjusted_value
                    neutral = self.factor_neutral.get(factor_name)
                    if neutral is not None:
                        centered_value = self._center_factor_value(adjusted_value, neutral)

                    effective_weight = weight

                    if factor_name == "b1_template_similarity" and (
                        adjusted_value is None or adjusted_value <= 0
                    ):
                        effective_weight = 0.0
                        weighted_scores[factor_name] = 0.0
                    elif is_bonus:
                        weighted_score = centered_value * weight
                        weighted_scores[factor_name] = weighted_score
                        bonus_total_score += weighted_score
                    elif factor_name == "risk_penalty":
                        weighted_score = centered_value * weight
                        weighted_scores[factor_name] = weighted_score
                        total_possible_score += abs(weight)
                        total_weighted_score += weighted_score
                    elif factor_name == "kline_patterns" and needs_smart_fusion:
                        adjusted_score, adjusted_weight = self._apply_smart_kline_fusion(
                            adjusted_value, weight, factors
                        )
                        effective_weight = adjusted_weight
                        centered_value = adjusted_score
                        weighted_score = adjusted_score * adjusted_weight
                        weighted_scores[factor_name] = weighted_score
                        total_possible_score += abs(adjusted_weight)
                        total_weighted_score += weighted_score
                        smart_fusion_trace["factor_adjustment"] = {
                            "raw_value": raw_value,
                            "adjusted_value": adjusted_value,
                            "effective_value": adjusted_score,
                            "original_weight": weight,
                            "effective_weight": adjusted_weight,
                            "weighted_score": weighted_score,
                        }
                    else:
                        weighted_score = centered_value * weight
                        weighted_scores[factor_name] = weighted_score
                        total_possible_score += abs(weight)
                        total_weighted_score += weighted_score

                factor_traces[factor_name] = {
                    "raw": float(raw_value),
                    "adjusted": float(adjusted_value),
                    "centered": float(centered_value),
                    "effective_weight": float(effective_weight),
                    "weighted": float(weighted_score),
                }

            if total_possible_score > 0:
                base_score = (total_weighted_score / total_possible_score) * 100.0
            else:
                base_score = 0.0
            base_score = self._clip_score(base_score)

            bonus_scaled = 0.0
            score_after_bonus = base_score
            if bonus_total_score != 0.0:
                denom = total_possible_score if total_possible_score > 0 else self._bonus_weight_total
                if denom > 0:
                    bonus_scaled = (bonus_total_score / denom) * 100.0
                score_after_bonus += bonus_scaled

            score_after_bonus = self._clip_score(score_after_bonus)
            final_score, floor_trace = self._apply_floor_protection(score_after_bonus, factors, code)
            final_score_rounded = round(final_score, 1)

            result: Dict[str, Any] = {
                "code": code,
                "total_score": final_score_rounded,
                "weighted_scores": weighted_scores,
                "smart_fusion_applied": needs_smart_fusion,
                "trace": {
                    "factors": factor_traces,
                    "totals": {
                        "total_weighted_score": total_weighted_score,
                        "total_possible_score": total_possible_score,
                        "base_score": base_score,
                        "bonus_scaled": bonus_scaled,
                        "floor_delta": final_score - score_after_bonus,
                        "protected_score": final_score,
                        "final_score_before_round": final_score,
                        "final_score_rounded": final_score_rounded,
                    },
                    "smart_fusion": smart_fusion_trace,
                    "floor_protection": floor_trace,
                },
            }
            if bonus_scaled != 0.0:
                result["bonus_contribution"] = bonus_scaled
            return result
        except Exception as exc:
            logger.error("failed to calculate explicit mechanical score for %s: %s", code, exc)
            return self._get_empty_score_result(code, str(exc))

    def _build_smart_fusion_trace(
        self, factors: Dict[str, float], kline_score: float
    ) -> Dict[str, Any]:
        """Build the smart fusion decision trace from raw factors."""
        b1_score = factors.get("b1_comprehensive", 0.0)
        risk_score = factors.get("risk_penalty", 0.0)
        confirmation_factors = [
            "pocket_pivot",
            "supply_analysis",
            "pivot_analysis",
            "ma_structure",
        ]
        confirmation_scores = {name: factors.get(name, 0.0) for name in confirmation_factors}
        strong_confirmations = sum(1 for value in confirmation_scores.values() if value >= 0.5)
        applied = self._should_apply_smart_fusion(factors, kline_score)
        return {
            "applied": applied,
            "decision_inputs": {
                "kline_score_raw": kline_score,
                "b1_score_raw": b1_score,
                "risk_score_raw": risk_score,
                "confirmation_scores_raw": confirmation_scores,
                "strong_confirmations": strong_confirmations,
            },
        }

    def _should_apply_smart_fusion(self, factors: Dict[str, float], kline_score: float) -> bool:
        """Return whether raw factors should trigger smart kline fusion."""
        if kline_score > 0.1:
            return False

        b1_score = factors.get("b1_comprehensive", 0.0)
        if b1_score < 0.55:
            return False

        risk_score = factors.get("risk_penalty", 0.0)
        if risk_score > 0.2:
            return False

        other_strong_factors = 0
        confirmation_factors = [
            "pocket_pivot",
            "supply_analysis",
            "pivot_analysis",
            "ma_structure",
        ]
        for factor_name in confirmation_factors:
            score = factors.get(factor_name, 0.0)
            if score >= 0.5:
                other_strong_factors += 1

        return other_strong_factors >= 2

    def _apply_adjustment(self, factor_name: str, value: float) -> float:
        """Apply configured adjustment tiers for a factor."""
        if value is None:
            return 0.0

        tiers = self.factor_adjustments.get(factor_name)
        if not tiers:
            return value

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value

        for tier in tiers:
            min_value = tier.get("min", 0.0)
            max_value = tier.get("max", 1.0)
            multiplier = tier.get("multiplier", 1.0)
            if numeric_value >= min_value and numeric_value < max_value:
                numeric_value *= multiplier
                break

        return max(0.0, min(1.0, numeric_value))

    @staticmethod
    def _center_factor_value(value: float, neutral: float) -> float:
        """Center a factor around its neutral value."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if neutral <= 0.0 or neutral >= 1.0:
            return numeric

        if numeric >= neutral:
            denom = max(1e-6, 1.0 - neutral)
            return min(1.0, (numeric - neutral) / denom)

        denom = max(1e-6, neutral)
        return max(-1.0, (numeric - neutral) / denom)

    def _apply_smart_kline_fusion(
        self, kline_value: float, original_weight: float, all_factors: Dict[str, float]
    ) -> Tuple[float, float]:
        """Apply the current smart fusion mechanics to kline score and weight."""
        other_values = [
            value
            for name, value in all_factors.items()
            if name not in {"kline_patterns", "risk_penalty"} and isinstance(value, (int, float))
        ]
        avg_other_score = sum(other_values) / len(other_values) if other_values else 0.0

        if avg_other_score >= 0.7:
            cover_strength = 0.8
        elif avg_other_score >= 0.5:
            cover_strength = 0.6
        elif avg_other_score >= 0.3:
            cover_strength = 0.4
        else:
            cover_strength = 0.0

        base_score = max(0.15, kline_value)
        cover_bonus = (0.4 - kline_value) * cover_strength
        adjusted_score = min(0.6, base_score + cover_bonus)

        weight_reduction = 0.3 + 0.4 * cover_strength
        adjusted_weight = original_weight * (1 - weight_reduction)
        return adjusted_score, adjusted_weight

    def _apply_floor_protection(
        self, score: float, factors: Dict[str, float], code: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply the current floor protection mechanics and return trace data."""
        b1_score = factors.get("b1_comprehensive", 0.0)
        risk_score = factors.get("risk_penalty", 0.0)
        confirmation_factors = [
            "pocket_pivot",
            "supply_analysis",
            "pivot_analysis",
            "ma_structure",
        ]
        strong_confirmations = sum(
            1 for factor_name in confirmation_factors if factors.get(factor_name, 0.0) >= 0.5
        )

        trace: Dict[str, Any] = {
            "input_score": score,
            "output_score": score,
            "applied": False,
            "decision_inputs": {
                "b1_score_raw": b1_score,
                "risk_score_raw": risk_score,
                "strong_confirmations": strong_confirmations,
            },
        }

        if score >= 55:
            trace["reason"] = "score_above_floor_gate"
            return score, trace

        if b1_score < 0.65 or risk_score > 0.15:
            trace["reason"] = "raw_factor_gate_not_met"
            return score, trace

        if strong_confirmations < 2:
            trace["reason"] = "insufficient_confirmations"
            return score, trace

        floor_bonus = min(10, (b1_score - 0.65) * 30)
        protected_score = min(60, score + floor_bonus)
        trace["floor_bonus"] = floor_bonus
        trace["output_score"] = protected_score

        if protected_score > score:
            logger.debug(
                "explicit floor protection applied for %s: %.1f -> %.1f",
                code,
                score,
                protected_score,
            )
            trace["applied"] = True
            trace["reason"] = "floor_bonus_applied"
            return protected_score, trace

        trace["reason"] = "no_increment_after_floor_bonus"
        return score, trace

    @staticmethod
    def _clip_score(score: float) -> float:
        """Clip a score into the 0-100 range."""
        return max(0.0, min(100.0, score))

    def _get_empty_score_result(self, code: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Return a stable empty result when calculation fails."""
        trace: Dict[str, Any] = {
            "factors": {},
            "totals": {
                "total_weighted_score": 0.0,
                "total_possible_score": 0.0,
                "base_score": 0.0,
                "bonus_scaled": 0.0,
                "floor_delta": 0.0,
                "protected_score": 0.0,
                "final_score_before_round": 0.0,
                "final_score_rounded": 0.0,
            },
            "smart_fusion": {},
            "floor_protection": {},
        }
        if error:
            trace["error"] = error

        return {
            "code": code,
            "total_score": 0.0,
            "weighted_scores": {},
            "smart_fusion_applied": False,
            "trace": trace,
        }
