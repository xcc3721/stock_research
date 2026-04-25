#!/usr/bin/env python3
"""
基础技术指标计算模块
包含均线结构、基础动量等核心技术指标
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# 导入现有的技术指标函数
from stock_research.selectors.indicators import compute_bbi, compute_dif, compute_kdj

logger = logging.getLogger("technical_indicators")


class TechnicalIndicators:
    """
    基础技术指标计算器
    
    提供：
    - 均线结构分析
    - 基础动量计算
    - RSI计算
    - 技术指标工具函数
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化技术指标计算器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
    
    def calc_ma_structure(self, df: pd.DataFrame) -> float:
        """
        B1策略均线结构因子
        MA60环境过滤 + 短期均线企稳信号
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            得分 0.0-1.0，环境+信号综合评分
        """
        try:
            params = self.config.get('ma_structure', {})
            ma_periods = params.get('ma_periods', [5, 10, 20, 60])
            env_ma_period = params.get('env_ma_period', 10)
            env_ma_ratios = params.get('env_ma_ratios', {'best': [0.92, 1.08], 'good': [0.85, 1.15]})
            stabilization_lookback = params.get('stabilization_lookback', 5)

            if len(df) < env_ma_period:
                return 0.0
                
            # 计算各周期均线
            mas = {p: df['close'].rolling(window=p, min_periods=p).mean() for p in ma_periods}
            
            latest_ma_env = mas[env_ma_period].iloc[-1]
            if pd.isna(latest_ma_env):
                return 0.0
            
            current_price = df['close'].iloc[-1]
            
            score = 0.0
            
            # 1. MA环境过滤 (权重60%) - 核心环境因子
            ma60_ratio = current_price / latest_ma_env
            if env_ma_ratios['best'][0] <= ma60_ratio <= env_ma_ratios['best'][1]:
                score += 0.6
            elif env_ma_ratios['good'][0] <= ma60_ratio <= env_ma_ratios['good'][1]:
                score += 0.4
            elif ma60_ratio < env_ma_ratios['good'][0]:
                score += 0.2
            else:
                score += 0.0
            
            # 2. 短期均线企稳信号 (权重40%)
            stabilization_score = self._calc_ma_stabilization(mas[ma_periods[0]], mas[ma_periods[1]], mas[ma_periods[2]], stabilization_lookback)
            score += stabilization_score * 0.4
            
            return min(score, 1.0)
                
        except Exception as e:
            logger.warning("B1均线结构因子计算失败: %s", e)
            return 0.0

    def calc_symmetry_resonance(
        self, df: pd.DataFrame, return_details: bool = False
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        计算对称走势强弱因子。

        思路：
          1. 在回看窗口内寻找显著的波峰/波谷；
          2. 取最近的显著转折点并确定是 V 型(波谷) 还是倒 V 型(波峰)；
          3. 连接前一极值与转折点形成基准线段，沿转折点所在纵轴反射得到理论右侧走势；
          4. 评估实际右侧走势相对于理论走势的偏离，正值代表强势，负值代表弱势；
          5. 归一化后返回 [0, 1] 区间的得分，0.5 视作中性。
        """
        params = self.config.get("symmetry_resonance", {}) or {}
        lookback = int(params.get("lookback", 60))
        smooth_window = int(params.get("smooth_window", 3))
        extrema_order = int(params.get("extrema_order", 3))
        amplitude_pct = float(params.get("amplitude_pct", 0.05))
        min_segment = int(params.get("min_segment", 5))
        neutral_score = float(params.get("neutral_score", 0.5))
        enforce_boundary = bool(params.get("enforce_boundary", False))
        enforce_time_balance = bool(params.get("enforce_time_balance", True))
        require_left_ge_right = bool(params.get("require_left_ge_right", False))
        large_structure_ratio = float(params.get("large_structure_ratio", 2.0))
        structure_level = max(0, int(params.get("structure_level", 0)))
        diff_mode = str(params.get("diff_mode", "latest")).lower()

        debug_info: Dict[str, Any] = {
            "smooth_window": smooth_window,
            "lookback": lookback,
            "start_idx": None,
            "peaks": [],
            "troughs": [],
            "filtered": [],
            "selected": None,
            "large_structure_ratio": large_structure_ratio,
            "structure_level": structure_level,
            "diff_mode": diff_mode,
            "enforce_time_balance": enforce_time_balance,
            "time_balance_skips": [],
            "left_ge_right_skips": [],
            "left_ge_right_adjustments": [],
        }

        def _wrap(value: float) -> Union[float, Tuple[float, Dict[str, Any]]]:
            return (value, debug_info) if return_details else value

        try:
            closes = pd.to_numeric(df["close"], errors="coerce")
        except Exception:
            return _wrap(neutral_score)

        closes = closes.ffill().bfill()
        if "high" in df.columns:
            highs_series = pd.to_numeric(df["high"], errors="coerce").ffill().bfill()
        else:
            highs_series = closes.copy()
        if "low" in df.columns:
            lows_series = pd.to_numeric(df["low"], errors="coerce").ffill().bfill()
        else:
            lows_series = closes.copy()
        if closes.isna().all():
            return _wrap(neutral_score)

        values = closes.to_numpy(dtype=float)
        high_values = highs_series.to_numpy(dtype=float)
        low_values = lows_series.to_numpy(dtype=float)
        if len(values) < max(lookback, min_segment * 2 + extrema_order + 2):
            return _wrap(neutral_score)

        if smooth_window > 1:
            smooth_values = (
                pd.Series(values)
                .rolling(window=smooth_window, min_periods=1)
                .mean()
                .to_numpy()
            )
        else:
            smooth_values = values.copy()

        lookback = min(lookback, len(smooth_values))
        start_idx = len(smooth_values) - lookback
        debug_info["start_idx"] = start_idx
        window_vals = smooth_values[start_idx:]

        def _detect_extrema(series: np.ndarray, order: int) -> tuple[List[int], List[int]]:
            peaks, troughs = [], []
            n = len(series)
            if n == 0:
                return peaks, troughs
            eps = 1e-9
            min_tail = order // 2
            for idx in range(order, n):
                left = idx - order
                if left < 0:
                    continue
                tail_len = n - idx - 1
                if tail_len < min_tail:
                    continue
                right = min(n, idx + order + 1)
                window = series[left:right]
                if window.size == 0 or np.isnan(window).any() or np.isnan(series[idx]):
                    continue
                center = series[idx]

                max_val = np.nanmax(window)
                if np.isfinite(center) and np.isfinite(max_val) and np.isclose(center, max_val, atol=eps):
                    tie_offsets = np.where(np.isclose(window, max_val, atol=eps))[0]
                    if tie_offsets.size == 1:
                        peaks.append(idx)
                    else:
                        tie_abs = [left + int(off) for off in tie_offsets]
                        tie_highs = np.asarray([high_values[abs_idx] for abs_idx in tie_abs], dtype=float)
                        if np.isnan(tie_highs).all():
                            continue
                        max_high = np.nanmax(tie_highs)
                        winners = [
                            abs_idx
                            for abs_idx, high_val in zip(tie_abs, tie_highs)
                            if np.isfinite(high_val) and np.isclose(high_val, max_high, atol=eps)
                        ]
                        if idx in winners and len(winners) == 1:
                            peaks.append(idx)

                min_val = np.nanmin(window)
                if np.isfinite(center) and np.isfinite(min_val) and np.isclose(center, min_val, atol=eps):
                    tie_offsets = np.where(np.isclose(window, min_val, atol=eps))[0]
                    if tie_offsets.size == 1:
                        troughs.append(idx)
                    else:
                        tie_abs = [left + int(off) for off in tie_offsets]
                        tie_lows = np.asarray([low_values[abs_idx] for abs_idx in tie_abs], dtype=float)
                        if np.isnan(tie_lows).all():
                            continue
                        min_low = np.nanmin(tie_lows)
                        winners = [
                            abs_idx
                            for abs_idx, low_val in zip(tie_abs, tie_lows)
                            if np.isfinite(low_val) and np.isclose(low_val, min_low, atol=eps)
                        ]
                        if idx in winners and len(winners) == 1:
                            troughs.append(idx)
            return peaks, troughs

        peaks, troughs = _detect_extrema(window_vals, extrema_order)
        peaks_abs = [start_idx + idx for idx in peaks]
        troughs_abs = [start_idx + idx for idx in troughs]
        debug_info["peaks"] = peaks_abs
        debug_info["troughs"] = troughs_abs
        if not peaks and not troughs:
            return _wrap(neutral_score)

        combined: List[tuple[int, str]] = sorted(
            [(idx, "peak") for idx in peaks] + [(idx, "trough") for idx in troughs],
            key=lambda x: x[0],
        )
        if not combined:
            return _wrap(neutral_score)
        filtered: List[dict[str, Any]] = []
        recent_peaks: List[int] = []
        recent_troughs: List[int] = []

        for rel_idx, tp in combined:
            price = window_vals[rel_idx]
            abs_idx = start_idx + rel_idx
            if tp == "peak":
                candidate_troughs = list(recent_troughs)
                for prev_rel_idx in reversed(candidate_troughs):
                    prev_abs_idx = start_idx + prev_rel_idx
                    prev_price = smooth_values[prev_abs_idx]
                    if prev_price <= 0:
                        continue
                    denom = max(abs(prev_price), 1e-6)
                    amplitude = (price - prev_price) / denom
                    duration = abs_idx - prev_abs_idx
                    if amplitude >= amplitude_pct and duration >= min_segment:
                        entry = {
                            "idx": rel_idx,
                            "abs_idx": abs_idx,
                            "type": "peak",
                            "prev_abs_idx": prev_abs_idx,
                        }
                        filtered.append(entry)
                        debug_info["filtered"].append(
                            {
                                "type": "peak",
                                "abs_idx": abs_idx,
                                "prev_abs_idx": prev_abs_idx,
                            }
                        )
                        break
                recent_peaks.append(rel_idx)
            else:
                candidate_peaks = list(recent_peaks)
                for prev_rel_idx in reversed(candidate_peaks):
                    prev_abs_idx = start_idx + prev_rel_idx
                    prev_price = smooth_values[prev_abs_idx]
                    if prev_price <= 0:
                        continue
                    denom = max(abs(prev_price), 1e-6)
                    amplitude = (prev_price - price) / denom
                    duration = abs_idx - prev_abs_idx
                    if amplitude >= amplitude_pct and duration >= min_segment:
                        entry = {
                            "idx": rel_idx,
                            "abs_idx": abs_idx,
                            "type": "trough",
                            "prev_abs_idx": prev_abs_idx,
                        }
                        filtered.append(entry)
                        debug_info["filtered"].append(
                            {
                                "type": "trough",
                                "abs_idx": abs_idx,
                                "prev_abs_idx": prev_abs_idx,
                            }
                        )
                        break
                recent_troughs.append(rel_idx)

        if not filtered:
            return _wrap(neutral_score)

        tolerance = float(params.get("tolerance", 1e-6))
        def _amplitude_percent(prev_val: float, turn_val: float, t_type: str) -> float:
            denom = max(abs(prev_val), 1e-6)
            if t_type == "peak":
                return max(0.0, (turn_val - prev_val) / denom)
            return max(0.0, (prev_val - turn_val) / denom)

        peak_values = {idx: float(smooth_values[idx]) for idx in peaks_abs}
        trough_values = {idx: float(smooth_values[idx]) for idx in troughs_abs}

        selected_result: Optional[Dict[str, Any]] = None
        fallback_result: Optional[Dict[str, Any]] = None

        for turning in reversed(filtered):
            turn_idx = int(turning["abs_idx"])
            prev_idx = int(turning["prev_abs_idx"])
            turn_type = turning["type"]

            if turn_idx <= prev_idx or turn_idx >= len(smooth_values) - 1:
                continue

            right_available = smooth_values[turn_idx + 1 :]
            if right_available.size == 0:
                continue
            turn_value = float(smooth_values[turn_idx])

            def _build_candidates() -> List[Dict[str, Any]]:
                candidates: List[Dict[str, Any]] = []

                base_prev_value = float(smooth_values[prev_idx])
                base_amp = _amplitude_percent(base_prev_value, turn_value, turn_type)
                base_duration = turn_idx - prev_idx
                base_valid = base_duration > 0 and base_amp >= amplitude_pct and base_duration >= min_segment
                if base_valid:
                    candidates.append(
                        {
                            "prev_idx": prev_idx,
                            "prev_value": base_prev_value,
                            "amplitude": base_amp,
                            "duration": base_duration,
                            "is_large": False,
                            "ratio": 1.0,
                            "source": "base",
                        }
                    )

                baseline_amp = base_amp if base_valid else amplitude_pct

                if large_structure_ratio > 1.0:
                    if turn_type == "trough":
                        source_points = [
                            (idx, peak_values[idx]) for idx in peaks_abs if idx < turn_idx and idx != prev_idx
                        ]
                    else:
                        source_points = [
                            (idx, trough_values[idx]) for idx in troughs_abs if idx < turn_idx and idx != prev_idx
                        ]
                    for idx, value in sorted(source_points, reverse=True):
                        duration = turn_idx - idx
                        if duration < min_segment:
                            continue
                        amp = _amplitude_percent(float(value), turn_value, turn_type)
                        if amp < amplitude_pct:
                            continue
                        ratio = amp / baseline_amp if baseline_amp > 1e-6 else np.inf
                        if ratio >= large_structure_ratio:
                            candidates.append(
                                {
                                    "prev_idx": idx,
                                    "prev_value": float(value),
                                    "amplitude": amp,
                                    "duration": duration,
                                    "is_large": True,
                                    "ratio": float(ratio),
                                    "source": "large",
                                }
                            )

                return candidates

            candidate_pairs = _build_candidates()
            if not candidate_pairs:
                continue

            base_candidates = [c for c in candidate_pairs if not c.get("is_large")]
            large_candidates = sorted(
                [c for c in candidate_pairs if c.get("is_large")],
                key=lambda item: item.get("ratio", 1.0),
            )
            ordered_candidates = base_candidates + large_candidates
            for idx, candidate in enumerate(ordered_candidates):
                candidate["rank"] = idx

            evaluated: List[Dict[str, Any]] = []

            def _expand_for_time_balance(prev_idx_c: int, base_amplitude: float, right_distance: int) -> Optional[Dict[str, Any]]:
                if right_distance <= 0:
                    return None
                if turn_type == "peak":
                    source_indices = [idx for idx in troughs_abs if idx < prev_idx_c]
                else:
                    source_indices = [idx for idx in peaks_abs if idx < prev_idx_c]
                for idx in sorted(source_indices, reverse=True):
                    distance_candidate = turn_idx - idx
                    if distance_candidate < max(min_segment, right_distance):
                        continue
                    prev_value_candidate = float(smooth_values[idx])
                    if not np.isfinite(prev_value_candidate):
                        continue
                    amplitude_candidate = _amplitude_percent(prev_value_candidate, turn_value, turn_type)
                    if amplitude_candidate < amplitude_pct:
                        continue
                    ratio_candidate = (
                        amplitude_candidate / base_amplitude if base_amplitude > 1e-6 else np.inf
                    )
                    return {
                        "prev_idx": idx,
                        "prev_value": prev_value_candidate,
                        "amplitude": amplitude_candidate,
                        "ratio": float(ratio_candidate),
                        "is_large": bool(ratio_candidate >= large_structure_ratio),
                        "source": "time_balance_expand",
                    }
                return None

            for candidate in ordered_candidates:
                candidate_local = dict(candidate)
                prev_idx_c = candidate_local["prev_idx"]
                prev_value = candidate_local["prev_value"]
                distance = turn_idx - prev_idx_c
                if distance <= 0:
                    continue

                right_distance = len(right_available)
                if require_left_ge_right and right_distance > distance:
                    expansion = _expand_for_time_balance(prev_idx_c, candidate_local.get("amplitude", 0.0), right_distance)
                    if expansion:
                        candidate_local.update(expansion)
                        prev_idx_c = candidate_local["prev_idx"]
                        prev_value = candidate_local["prev_value"]
                        distance = turn_idx - prev_idx_c
                        debug_info["left_ge_right_adjustments"].append(
                            {
                                "turn_idx": turn_idx,
                                "prev_idx": prev_idx_c,
                                "distance_left": distance,
                                "distance_right": right_distance,
                            }
                        )
                    else:
                        debug_info["left_ge_right_skips"].append(
                            {
                                "turn_idx": turn_idx,
                                "prev_idx": prev_idx_c,
                                "distance_left": distance,
                                "distance_right": right_distance,
                            }
                        )
                        continue

                if enforce_time_balance and right_distance > distance:
                    debug_info["time_balance_skips"].append(
                        {
                            "turn_idx": turn_idx,
                            "prev_idx": prev_idx_c,
                            "distance_left": distance,
                            "distance_right": right_distance,
                        }
                    )
                    continue

                future_len = min(distance, right_distance)
                if future_len <= 0:
                    continue

                slope = (turn_value - prev_value) / distance
                offsets = np.arange(0, future_len + 1, dtype=float)
                theoretical = turn_value - slope * offsets
                actual_full = smooth_values[turn_idx : turn_idx + future_len + 1]
                if actual_full.size < theoretical.size:
                    continue
                actual_series = np.asarray(actual_full, dtype=float)
                theoretical_right = theoretical[1:]
                actual_right = actual_series[1:]
                if theoretical_right.size == 0 or actual_right.size == 0:
                    continue
                if np.isnan(theoretical_right).all() or np.isnan(actual_right).all():
                    continue

                theoretical_max = np.nanmax(theoretical_right)
                theoretical_min = np.nanmin(theoretical_right)
                actual_max = np.nanmax(actual_right)
                actual_min = np.nanmin(actual_right)

                boundary_violation = (
                    (actual_max > theoretical_max + tolerance)
                    or (actual_min < theoretical_min - tolerance)
                )
                if boundary_violation and enforce_boundary:
                    continue

                denom_value = max(abs(prev_value), 1e-6)
                diff_percent = (actual_right - theoretical_right) / denom_value
                if diff_mode == "mean":
                    try:
                        diff_metric = float(np.nanmean(diff_percent))
                    except Exception:
                        continue
                else:
                    valid = diff_percent[~np.isnan(diff_percent)]
                    if valid.size == 0:
                        continue
                    diff_metric = float(valid[-1])

                amplitude_scale = max(candidate["amplitude"], 1e-6)
                score_raw = diff_metric / amplitude_scale
                score_clipped = float(np.clip(score_raw, -1.0, 1.0))

                future_indices = list(range(turn_idx, turn_idx + future_len + 1))
                evaluated.append(
                    {
                        "rank": candidate["rank"],
                        "score_raw": score_raw,
                        "score_clipped": score_clipped,
                        "turn_idx": turn_idx,
                        "prev_idx": prev_idx_c,
                        "turn_value": float(turn_value),
                        "prev_value": float(prev_value),
                        "slope": float(slope),
                        "future_indices": future_indices,
                        "theoretical": [float(x) for x in theoretical.tolist()],
                        "actual": [float(x) for x in actual_series.tolist()],
                        "boundary_violation": boundary_violation,
                        "type": turn_type,
                        "is_large": bool(candidate.get("is_large")),
                        "amplitude_percent": float(candidate["amplitude"]),
                        "amplitude_ratio": float(candidate.get("ratio", 1.0)),
                        "denominator": float(denom_value),
                        "theoretical_max": float(theoretical_max),
                        "theoretical_min": float(theoretical_min),
                        "diff_metric": diff_metric,
                        "distance_left": int(distance),
                        "distance_right_current": int(right_distance),
                    }
                )

            if not evaluated:
                continue

            evaluated.sort(key=lambda item: item["rank"])
            max_rank_available = evaluated[-1]["rank"]

            if structure_level > max_rank_available:
                if fallback_result is None:
                    fallback_result = evaluated[-1]
                continue

            chosen: Optional[Dict[str, Any]] = None
            for item in evaluated:
                if item["rank"] == structure_level:
                    chosen = item
                    break
            if chosen is None:
                # 结构级别高于某些候选但不高于最大 rank，选择最接近目标级别的更大结构
                larger = [item for item in evaluated if item["rank"] > structure_level]
                chosen = larger[0] if larger else evaluated[-1]

            selected_result = chosen
            break

        if selected_result is None and fallback_result is not None:
            selected_result = fallback_result

        if selected_result is None:
            return _wrap(neutral_score)

        score = selected_result["score_clipped"]
        hi = selected_result["theoretical_max"]
        lo = selected_result["theoretical_min"]
        latest_value = float(smooth_values[-1])
        if enforce_boundary and (latest_value > hi + tolerance or latest_value < lo - tolerance):
            return _wrap(neutral_score)

        final_score = float(neutral_score + (1.0 - neutral_score) * score) if score >= 0 else float(
            neutral_score * (1.0 + score)
        )
        debug_info["selected"] = {
            "type": selected_result["type"],
            "turn_idx": selected_result["turn_idx"],
            "prev_idx": selected_result["prev_idx"],
            "turn_value": selected_result["turn_value"],
            "prev_value": selected_result["prev_value"],
            "slope": selected_result["slope"],
            "future_indices": selected_result["future_indices"],
            "theoretical": selected_result["theoretical"],
            "actual": selected_result["actual"],
            "boundary_violation": selected_result["boundary_violation"],
            "is_large": selected_result["is_large"],
            "amplitude_percent": selected_result["amplitude_percent"],
            "amplitude_ratio": selected_result["amplitude_ratio"],
            "denominator": selected_result["denominator"],
            "structure_rank": selected_result["rank"],
            "diff_metric": selected_result.get("diff_metric"),
            "distance_left": selected_result.get("distance_left"),
            "distance_right_current": selected_result.get("distance_right_current"),
        }
        return _wrap(final_score)
    
    def _calc_ma_stabilization(self, ma5: pd.Series, ma10: pd.Series, ma20: pd.Series, lookback: int = 5) -> float:
        """
        计算短期均线企稳程度
        针对B1策略：放宽短期均线要求，因为B1买点往往出现在短期均线空头排列时
        """
        try:
            params = self.config.get('ma_stabilization', {})
            slope_lookback = params.get('slope_lookback', 3)
            slope_threshold = params.get('slope_threshold', -0.05)  # 放宽从-0.02到-0.05
            crossover_threshold = params.get('crossover_threshold', 0.02)  # 放宽从0.01到0.02

            if len(ma5) < lookback:
                return 0.0
            
            score = 0.0
            
            # 最近lookback天的均线值
            recent_ma5 = ma5.tail(lookback)
            recent_ma10 = ma10.tail(lookback)
            recent_ma20 = ma20.tail(lookback)
            
            # 1. 均线收敛程度 (均线之间距离缩小) - 权重降低
            ma_spread_today = abs(recent_ma5.iloc[-1] - recent_ma20.iloc[-1]) / recent_ma20.iloc[-1]
            ma_spread_lookback_ago = abs(recent_ma5.iloc[0] - recent_ma20.iloc[0]) / recent_ma20.iloc[0]
            
            if ma_spread_today < ma_spread_lookback_ago:  # 均线在收敛
                score += 0.3  # 从0.4降到0.3
            
            # 2. MA5斜率企稳 (放宽标准，允许较大下跌) - 权重提高
            ma5_slope_recent = (recent_ma5.iloc[-1] - recent_ma5.iloc[-slope_lookback]) / recent_ma5.iloc[-slope_lookback]
            if ma5_slope_recent > slope_threshold:  # 放宽斜率阈值
                score += 0.4  # 从0.3提高到0.4
            elif ma5_slope_recent > -0.08:  # 新增：即使下跌也给部分分数
                score += 0.2
            
            # 3. 短期均线排列改善 - 大幅放宽要求
            latest_ma5 = recent_ma5.iloc[-1]
            latest_ma10 = recent_ma10.iloc[-1]
            current_price = recent_ma5.index[-1]  # 使用最新价格位置
            
            if latest_ma5 > latest_ma10:  # MA5上穿MA10，短期转强
                score += 0.3
            elif abs(latest_ma5 - latest_ma10) / latest_ma10 < crossover_threshold:  # 放宽接近阈值
                score += 0.25  # 提高即将金叉的分数
            else:
                # B1策略特殊处理：即使短期均线空头排列，如果斜率放缓也给分
                ma10_slope = (recent_ma10.iloc[-1] - recent_ma10.iloc[-slope_lookback]) / recent_ma10.iloc[-slope_lookback]
                if ma10_slope > -0.03:  # MA10斜率企稳
                    score += 0.15  # 给予基础企稳分数
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("均线企稳计算失败: %s", e)
            return 0.0
    
    def calc_momentum_basic(self, df: pd.DataFrame) -> float:
        """
        B1抄底策略动量因子
        结合了[反转信号]和[点火信号]，能够识别深度反转和发射台模式
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            得分 0.0-1.0，1.0表示完美的抄底或点火信号
        """
        try:
            params = self.config.get('momentum_basic', {})
            lookback_period = params.get('lookback_period', 30)
            reversal_weight = params.get('reversal_weight', 0.5)
            ignition_weight = params.get('ignition_weight', 0.5)

            if len(df) < lookback_period:
                return 0.0
            
            # 1. 计算反转信号得分 (J值钝化 + 底背离 + MACD确认)
            reversal_score = 0
            j_stagnation_score = self._calc_j_stagnation(df)
            reversal_score += j_stagnation_score * 0.4
            bottom_divergence_score = self._calc_bottom_divergence(df)
            reversal_score += bottom_divergence_score * 0.4
            macd_confirm_score = self._calc_macd_confirm(df)
            reversal_score += macd_confirm_score * 0.2
            
            # 2. 计算动量点火得分
            ignition_score = self._calc_momentum_ignition(df)

            # 3. 取两种信号中的最强者
            total_score = max(reversal_score, ignition_score)
            
            return min(total_score, 1.0)
            
        except Exception as e:
            logger.warning("B1动量因子计算失败: %s", e)
            return 0.0
    
    def _calc_j_stagnation(self, df: pd.DataFrame) -> float:
        """
        计算J值钝化程度
        深度超卖区的钝化表明抛压释放充分，是潜在的反转信号
        """
        try:
            params = self.config.get('j_stagnation', {})
            oversold_threshold = params.get('oversold_threshold', 10)
            stagnation_days = params.get('stagnation_days', 3)
            kdj_period = params.get('kdj_period', 9)

            if len(df) < kdj_period + stagnation_days:
                return 0.0
            
            # 计算KDJ
            k, d, j = compute_kdj(df['high'], df['low'], df['close'], kdj_period)
            
            if len(j) < stagnation_days:
                return 0.0
            
            # 检查最近几天J值是否在超卖区钝化
            recent_j = j.tail(stagnation_days)
            
            if all(j_val < oversold_threshold for j_val in recent_j):
                # 所有值都在超卖区，计算钝化程度
                j_variance = np.var(recent_j)
                max_variance_threshold = 25  # 根据实际情况调整
                
                # 方差越小，钝化程度越高
                stagnation_score = max(0, 1 - j_variance / max_variance_threshold)
                
                # 根据超卖深度给予额外分数
                avg_j = np.mean(recent_j)
                if avg_j < 5:  # 极度超卖
                    stagnation_score *= 1.2
                elif avg_j < 3:  # 历史性超卖
                    stagnation_score *= 1.5
                
                return min(stagnation_score, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.debug("J值钝化计算失败: %s", e)
            return 0.0
    
    def _calc_bottom_divergence(self, df: pd.DataFrame) -> float:
        """
        计算底背离信号强度
        价格创新低而指标未创新低，表明下跌动能减弱
        """
        try:
            params = self.config.get('bottom_divergence', {})
            lookback_period = params.get('lookback_period', 20)
            rsi_period = params.get('rsi_period', 9)
            kdj_period = params.get('kdj_period', 9)

            if len(df) < lookback_period:
                return 0.0
            
            recent_data = df.tail(lookback_period)
            
            # 计算技术指标
            rsi = self._compute_rsi(recent_data['close'], rsi_period)
            k, d, j = compute_kdj(recent_data['high'], recent_data['low'], recent_data['close'], kdj_period)
            
            if len(rsi) < 5 or len(j) < 5:
                return 0.0
            
            # 寻找价格和指标的低点
            price_lows = []
            rsi_lows = []
            j_lows = []
            
            close_prices = recent_data['close'].values
            for i in range(2, len(close_prices) - 2):
                # 简单的局部最小值检测
                if (close_prices[i] < close_prices[i-1] and close_prices[i] < close_prices[i-2] and
                    close_prices[i] < close_prices[i+1] and close_prices[i] < close_prices[i+2]):
                    price_lows.append((i, close_prices[i]))
                    if i < len(rsi):
                        rsi_lows.append((i, rsi.iloc[i]))
                    if i < len(j):
                        j_lows.append((i, j.iloc[i]))
            
            # 检查背离
            divergence_score = 0.0
            
            if len(price_lows) >= 2:
                # 比较最近两个低点
                last_price_low = price_lows[-1]
                prev_price_low = price_lows[-2]
                
                if last_price_low[1] < prev_price_low[1]:  # 价格创新低
                    # 检查RSI背离
                    if len(rsi_lows) >= 2:
                        last_rsi_low = rsi_lows[-1]
                        prev_rsi_low = rsi_lows[-2]
                        if last_rsi_low[1] > prev_rsi_low[1]:  # RSI未创新低
                            divergence_score += 0.5
                    
                    # 检查J值背离
                    if len(j_lows) >= 2:
                        last_j_low = j_lows[-1]
                        prev_j_low = j_lows[-2]
                        if last_j_low[1] > prev_j_low[1]:  # J值未创新低
                            divergence_score += 0.5
            
            return min(divergence_score, 1.0)
            
        except Exception as e:
            logger.debug("底背离计算失败: %s", e)
            return 0.0
    
    def _calc_macd_confirm(self, df: pd.DataFrame) -> float:
        """
        计算MACD确认信号
        """
        try:
            params = self.config.get('macd_confirm', {})
            
            if len(df) < 30:
                return 0.0
            
            # 计算MACD
            dif = compute_dif(df['close'])
            
            if len(dif) < 10:
                return 0.0
            
            recent_dif = dif.tail(5)
            
            score = 0.0
            
            # 1. DIF线上穿零轴
            if recent_dif.iloc[-1] > 0 and recent_dif.iloc[-2] <= 0:
                score += 0.5
            
            # 2. DIF线斜率向上
            if len(recent_dif) >= 3:
                if recent_dif.iloc[-1] > recent_dif.iloc[-3]:
                    score += 0.3
            
            # 3. DIF在零轴下方但向上
            if recent_dif.iloc[-1] < 0:
                if recent_dif.iloc[-1] > recent_dif.iloc[-2]:
                    score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("MACD确认计算失败: %s", e)
            return 0.0
    
    def _calc_momentum_ignition(self, df: pd.DataFrame) -> float:
        """
        计算动量点火信号
        """
        try:
            params = self.config.get('momentum_ignition', {})
            
            if len(df) < 20:
                return 0.0
            
            recent_data = df.tail(10)
            
            score = 0.0
            
            # 1. 成交量放大
            avg_volume = df['volume'].tail(20).mean()
            current_volume = recent_data['volume'].iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                score += 0.4
            elif current_volume > avg_volume * 1.2:
                score += 0.2
            
            # 2. 价格突破
            recent_high = recent_data['high'].max()
            prev_high = df['high'].tail(20).iloc[:-5].max()
            
            if recent_high > prev_high:
                score += 0.3
            
            # 3. 斜率改善
            price_slope = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-5]) / recent_data['close'].iloc[-5]
            if price_slope > 0.02:
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("动量点火计算失败: %s", e)
            return 0.0
    
    def _compute_rsi(self, prices: pd.Series, period: int = 9) -> pd.Series:
        """
        计算RSI指标
        
        Args:
            prices: 价格序列
            period: 计算周期
            
        Returns:
            RSI值序列
        """
        try:
            if len(prices) < period + 1:
                return pd.Series()
                
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.debug("RSI计算失败: %s", e)
            return pd.Series()
