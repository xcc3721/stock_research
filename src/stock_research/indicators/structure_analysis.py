#!/usr/bin/env python3
"""
结构化分析模块 (Structure Analysis)
专注于捕捉"脉冲放量 -> 缩量回踩 -> 企稳反转"的N字结构第二笔买点
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger("structure_analysis")


class StructureAnalysis:
    """
    结构化分析器
    
    核心功能：
    1. 资金脉冲记忆 (Fund Pulse Memory): 识别过去一段时间的主力资金介入
    2. 回调缩量比 (Pullback Volume Shrinkage): 评估洗盘的彻底程度
    3. 支撑位偏离度 (Support Deviation): 确认回踩支撑的精准度
    4. 微观企稳信号 (Stabilization): 识别止跌信号
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化结构化分析器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.params = self.config.get('structure_analysis', {})
        
    def calc_structure_score(self, df: pd.DataFrame) -> float:
        """
        计算结构化评分 (异动因子)。
        """
        score, _ = self._calc_structure_score(df, with_parts=False)
        return score

    def calc_structure_score_with_parts(self, df: pd.DataFrame) -> tuple[float, Dict[str, float]]:
        """
        计算结构化评分并返回组成部分，便于回测分析。

        Returns:
            (score, parts) 其中 parts 包含 pulse/pullback/stabilization/support 的得分。
        """
        score, parts = self._calc_structure_score(df, with_parts=True)
        return score, parts

    def _calc_structure_score(self, df: pd.DataFrame, *, with_parts: bool) -> tuple[float, Dict[str, float]]:
        """
        核心计算逻辑，支持返回子得分。
        """
        try:
            if df is None or len(df) < 60:
                return 0.0, {"pulse_score": 0.0, "pullback_score": 0.0, "stabilization_score": 0.0, "support_score": 0.0}
                
            # 获取参数
            params = self.params
            lookback_pulse = params.get('lookback_pulse', 60)
            
            # 1. 检测资金脉冲 (Fund Pulse)
            # 寻找过去 [min_pulse_days, lookback_pulse] 区间内的最大量能脉冲
            recent_data = df.iloc[-lookback_pulse:]
            if len(recent_data) < 30:
                return 0.0, {"pulse_score": 0.0, "pullback_score": 0.0, "stabilization_score": 0.0, "support_score": 0.0}
                
            has_pulse, pulse_score, pulse_info = self._detect_fund_pulse(recent_data)
            
            if not has_pulse:
                return 0.0, {"pulse_score": 0.0, "pullback_score": 0.0, "stabilization_score": 0.0, "support_score": 0.0}

            # 1.1 峰值距离校验 + 最小回撤要求，确保不是“正在冲顶”
            min_peak_ago = self.params.get('min_pulse_days', 5)
            min_retracement_pct = self.params.get('min_retracement_pct', 0.05)
            days_since_peak = pulse_info.get('days_since_peak', 0)
            peak_price = pulse_info.get('peak_price', np.nan)
            peak_high = pulse_info.get('peak_high', peak_price)
            current_price = recent_data['close'].iloc[-1]

            if days_since_peak < min_peak_ago:
                return 0.0, {"pulse_score": pulse_score, "pullback_score": 0.0, "stabilization_score": 0.0, "support_score": 0.0}

            retr_source_price = peak_high if self.params.get('use_high_for_retracement', False) else peak_price
            retracement = (retr_source_price - current_price) / retr_source_price if retr_source_price else 0.0
            if retracement < min_retracement_pct:
                return 0.0, {"pulse_score": pulse_score, "pullback_score": 0.0, "stabilization_score": 0.0, "support_score": 0.0}
            
            # 2. 评估缩量回踩 (Pullback Quality)
            pullback_score = self._analyze_pullback(recent_data, pulse_info)
            
            # 3. 评估支撑和企稳 (Support & Stabilization)
            support_parts = None
            if with_parts:
                support_score, support_parts = self._analyze_support_and_stabilization(recent_data, pulse_info, return_parts=True)
            else:
                support_score = self._analyze_support_and_stabilization(recent_data, pulse_info)
            
            # 4. 综合评分
            # 权重: 脉冲力度(30%) + 回踩质量(40%) + 企稳支撑(30%)
            total_score = (pulse_score * 0.3) + (pullback_score * 0.4) + (support_score * 0.3)
            
            parts = {
                "pulse_score": float(pulse_score),
                "pullback_score": float(pullback_score),
                "stabilization_score": float((support_parts or {}).get("stabilization_score", 0.0) if support_parts is not None else 0.0),
                "support_score": float((support_parts or {}).get("support_score", support_score) if support_parts is not None else support_score),
            }
            if not with_parts:
                return min(total_score, 1.0), parts
            return min(total_score, 1.0), parts
            
        except Exception as e:
            logger.error(f"结构化评分计算失败: {e}")
            parts = {"pulse_score": 0.0, "pullback_score": 0.0, "stabilization_score": 0.0, "support_score": 0.0}
            return 0.0, parts

    def _detect_fund_pulse(self, df: pd.DataFrame) -> Tuple[bool, float, Dict]:
        """
        检测是否存在资金脉冲
        
        逻辑:
        1. 寻找换手率/成交量激增的日子
        2. 股价在脉冲期间有明显上涨
        """
        try:
            # 参数
            vol_multiplier = self.params.get('pulse_vol_multiplier', 2.0)
            price_rise_min = self.params.get('pulse_price_rise', 0.15)  # 脉冲波段涨幅至少15%
            baseline_vol_lookback = int(self.params.get('baseline_vol_lookback', 20))
            
            volumes = df['volume']
            closes = df['close']
            highs = df['high'] if 'high' in df.columns else closes
            lows = df['low'] if 'low' in df.columns else closes
            current_price = closes.iloc[-1]
            
            closes_arr = closes.to_numpy()
            highs_arr = highs.to_numpy()
            lows_arr = lows.to_numpy()
            ma14 = closes.rolling(window=14, min_periods=14).mean()
            ma28 = closes.rolling(window=28, min_periods=28).mean()
            ma57 = closes.rolling(window=57, min_periods=57).mean()
            ma114 = closes.rolling(window=114, min_periods=114).mean()
            long_trend_arr = pd.concat([ma14, ma28, ma57, ma114], axis=1).mean(axis=1).to_numpy()

            prom_pct = self.params.get('peak_prominence_pct', 0.05)
            prom_lookback = int(self.params.get('peak_prominence_lookback', 5))
            min_peak_days = self.params.get('min_pulse_days', 3)
            min_retr = self.params.get('min_retracement_pct', 0.1)
            use_high_retr = bool(self.params.get('use_high_for_retracement', False))

            # 从近到远寻找满足条件的显著高点（不满足则继续向前找）
            for pos in range(len(closes_arr) - 2, 0, -1):
                if not (closes_arr[pos] >= closes_arr[pos - 1] and closes_arr[pos] >= closes_arr[pos + 1]):
                    continue
                # 额外要求：前后各 N 天收盘价最高
                peak_max_window = int(self.params.get('peak_max_window', 3))
                left_max = closes_arr[max(0, pos - peak_max_window): pos + 1].max()
                right_max = closes_arr[pos: min(len(closes_arr), pos + peak_max_window + 1)].max()
                if closes_arr[pos] < left_max or closes_arr[pos] < right_max:
                    continue
                left = max(0, pos - prom_lookback)
                right = min(len(closes_arr) - 1, pos + prom_lookback)
                left_min = closes_arr[left:pos].min(initial=closes_arr[pos])
                right_min = closes_arr[pos + 1:right + 1].min(initial=closes_arr[pos])
                valley = min(left_min, right_min)
                prominence = (closes_arr[pos] - valley) / closes_arr[pos] if closes_arr[pos] else 0.0
                if prominence < prom_pct:
                    continue

                peak_loc = pos
                peak_label = df.index[peak_loc]
                max_price = closes_arr[peak_loc]
                peak_high = highs_arr[peak_loc]

                current_loc = len(df) - 1
                days_since_peak = current_loc - peak_loc
                if days_since_peak < min_peak_days:
                    continue

                retr_source = peak_high if use_high_retr else max_price
                retracement = (retr_source - current_price) / retr_source if retr_source else 0.0
                if retracement < min_retr:
                    continue

                pre_peak_data = df.loc[:peak_label]
                if len(pre_peak_data) < 5:
                    continue
                
                start_price = pre_peak_data['low'].min()
                start_idx = pre_peak_data['low'].idxmin()
                start_loc = df.index.get_loc(start_idx)

                # 起始点距离过长，回退为“峰值向前最后一次收盘跌破长趋势”的位置
                max_start_gap = int(self.params.get('max_start_gap', 20))
                if (peak_loc - start_loc) > max_start_gap:
                    fallback_loc = None
                    for back in range(peak_loc, -1, -1):
                        lt = long_trend_arr[back] if back < len(long_trend_arr) else np.nan
                        if np.isnan(lt):
                            continue
                        if closes_arr[back] < lt:
                            fallback_loc = back
                            break
                    if fallback_loc is not None:
                        start_loc = fallback_loc
                        start_price = lows_arr[fallback_loc]
                pulse_rise = (max_price - start_price) / start_price
                
                if pulse_rise < price_rise_min:
                    continue
                
                baseline_start = max(0, start_loc - baseline_vol_lookback)
                baseline_end = max(0, start_loc)
                baseline_slice = volumes.iloc[baseline_start:baseline_end]
                baseline_vol = baseline_slice.mean()
                if np.isnan(baseline_vol) or baseline_vol <= 0:
                    baseline_vol = volumes.mean()
                if np.isnan(baseline_vol) or baseline_vol <= 0:
                    continue

                start_loc_pulse = max(0, peak_loc - 3)
                end_loc_pulse = min(len(df), peak_loc + 4)
                pulse_vol_avg = volumes.iloc[start_loc_pulse:end_loc_pulse].mean()
                if np.isnan(pulse_vol_avg) or pulse_vol_avg <= 0:
                    continue

                vol_ratio = pulse_vol_avg / baseline_vol

                turnover_ratio = None
                turnover_multiplier = self.params.get('pulse_turnover_multiplier')
                if 'turnover' in df.columns and turnover_multiplier:
                    baseline_turn = df['turnover'].iloc[baseline_start:baseline_end].mean()
                    pulse_turn = df['turnover'].iloc[start_loc_pulse:end_loc_pulse].mean()
                    if baseline_turn and not np.isnan(baseline_turn) and pulse_turn and not np.isnan(pulse_turn):
                        turnover_ratio = pulse_turn / baseline_turn

                passes_volume = vol_ratio >= vol_multiplier
                passes_turnover = turnover_ratio is not None and turnover_ratio >= turnover_multiplier
                if not (passes_volume or passes_turnover):
                    continue
                    
                score_rise = min(pulse_rise / 0.5, 1.0)
                score_vol = min(vol_ratio / 4.0, 1.0)
                pulse_score = (score_rise * 0.6 + score_vol * 0.4)
                
                return True, pulse_score, {
                    'peak_price': max_price,
                    'peak_high': peak_high,
                    'peak_date': peak_label,
                    'peak_loc': peak_loc,
                    'pulse_vol': pulse_vol_avg,
                    'start_price': start_price,
                    'days_since_peak': days_since_peak,
                    'vol_ratio': vol_ratio,
                    'turnover_ratio': turnover_ratio
                }
            
            return False, 0.0, {}
            
        except Exception as e:
            logger.warning(f"资金脉冲检测异常: {e}")
            return False, 0.0, {}

    def _analyze_pullback(self, df: pd.DataFrame, pulse_info: Dict) -> float:
        """
        分析回调质量
        
        逻辑:
        1. 缩量: 当前成交量应显著小于脉冲期 (例如 < 0.5)
        2. 深度: 回调幅度适中 (0.382 - 0.618 黄金分割位最佳)
        """
        try:
            current_vol = df['volume'].iloc[-1]
            current_price = df['close'].iloc[-1]
            pulse_vol = pulse_info['pulse_vol']
            peak_price = pulse_info['peak_price']
            start_price = pulse_info['start_price']
            
            # 1. 量能萎缩评分
            if pulse_vol is None or pulse_vol <= 0:
                return 0.0

            vol_shrink_cfg = self.params.get('vol_shrink_thresholds', {})
            vol_full = vol_shrink_cfg.get('full', 0.4)
            vol_pass = vol_shrink_cfg.get('pass', 0.6)
            vol_weak = vol_shrink_cfg.get('weak', 0.8)

            vol_shrink_ratio = current_vol / pulse_vol
            # 理想: < full (满分), < pass (及格)
            if vol_shrink_ratio < vol_full:
                vol_score = 1.0
            elif vol_shrink_ratio < vol_pass:
                vol_score = 0.7
            elif vol_shrink_ratio < vol_weak:
                vol_score = 0.4
            else:
                vol_score = 0.0
                
            # 2. 回调深度评分
            # 相对脉冲波段的位置
            pulse_range = peak_price - start_price
            if pulse_range <= 0:
                return 0.0
                
            retracement = (peak_price - current_price) / pulse_range

            retrace_cfg = self.params.get('retracement_bands', {})
            retrace_ideal_low = retrace_cfg.get('ideal_low', 0.3)
            retrace_ideal_high = retrace_cfg.get('ideal_high', 0.6)
            retrace_ok_shallow = retrace_cfg.get('ok_shallow', 0.2)
            retrace_ok_deep = retrace_cfg.get('ok_deep', 0.75)

            # 理想回调: ideal_low - ideal_high
            if retrace_ideal_low <= retracement <= retrace_ideal_high:
                depth_score = 1.0
            elif retrace_ok_shallow <= retracement < retrace_ideal_low or retrace_ideal_high < retracement <= retrace_ok_deep:
                depth_score = 0.6
            elif retracement < retrace_ok_shallow: # 回调太浅，可能还没到位
                depth_score = 0.4
            else: # 回调太深
                depth_score = 0.1
                
            return (vol_score * 0.6 + depth_score * 0.4)
            
        except Exception as e:
            logger.warning(f"回调分析异常: {e}")
            return 0.0

    def _analyze_support_and_stabilization(self, df: pd.DataFrame, pulse_info: Dict, *, return_parts: bool = False):
        """
        分析支撑和微观企稳
        
        逻辑:
        1. 支撑: 价格是否在 MA60 或 前期平台 附近
        2. 企稳: 最近3天是否缩量止跌、收小K线
        """
        try:
            closes = df['close']
            volumes = df['volume']
            current_price = closes.iloc[-1]
            
            # 1. 企稳信号 (Stabilization)
            # 检查最近3天
            recent_days = 3
            recent_closes = closes.iloc[-recent_days:]
            recent_opens = df['open'].iloc[-recent_days:]
            recent_highs = df['high'].iloc[-recent_days:]
            recent_lows = df['low'].iloc[-recent_days:]
            
            # A. 波动率收窄 (K线实体变小)
            # 计算平均实体比例
            body_sizes = (recent_closes - recent_opens).abs() / recent_closes
            avg_body_size = body_sizes.mean()
            small_body_threshold = self.params.get('small_body_threshold', 0.02)
            
            is_small_body = avg_body_size < small_body_threshold # 实体小于阈值
            
            # B. 下影线支撑 (至少有一天有下影线)
            lower_shadows = (recent_opens.combine(recent_closes, min) - recent_lows) / recent_closes
            shadow_threshold = self.params.get('shadow_threshold', 0.01)
            has_support_shadow = lower_shadows.max() > shadow_threshold
            
            # C. 不创新低 (或者创新低后拉回)
            # 比较今天收盘和前几天最低
            last_close = recent_closes.iloc[-1]
            min_low_prev = recent_lows.iloc[:-1].min()
            
            is_holding = last_close >= min_low_prev * 0.99
            
            stab_score = 0.0
            if is_small_body: stab_score += 0.4
            if has_support_shadow: stab_score += 0.3
            if is_holding: stab_score += 0.3
            
            # 2. 均线支撑 (Support)
            short_band = self.params.get('support_short_band', {})
            long_band = self.params.get('support_long_band', self.params.get('support_band', {}))
            support_weights = self.params.get('support_weights', {})
            weight_short = support_weights.get('short', 0.5)
            weight_long = support_weights.get('long', 0.5)
            long_break_penalty = self.params.get('long_break_penalty', 0.3)
            long_break_days = int(self.params.get('long_break_days', 2))

            # 短期趋势: EMA(EMA(CLOSE,10),10)
            short_trend_series = closes.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
            short_trend = short_trend_series.iloc[-1]

            # 长期趋势: (MA14+MA28+MA57+MA114)/4
            ma14 = closes.rolling(window=14, min_periods=14).mean()
            ma28 = closes.rolling(window=28, min_periods=28).mean()
            ma57 = closes.rolling(window=57, min_periods=57).mean()
            ma114 = closes.rolling(window=114, min_periods=114).mean()
            long_trend_series = pd.concat([ma14, ma28, ma57, ma114], axis=1).mean(axis=1)
            long_trend = long_trend_series.iloc[-1]

            def _score_support(dist: float, band: Dict[str, float]) -> float:
                band_min = band.get('min', -0.02)
                band_ideal_max = band.get('ideal_max', 0.10)
                band_far_max = band.get('far_max', 0.20)
                if np.isnan(dist):
                    return 0.0
                if band_min <= dist <= band_ideal_max:
                    return 1.0
                if band_ideal_max < dist <= band_far_max:
                    return 0.6
                if dist < band_min:
                    return 0.2
                return 0.4

            short_support = 0.0
            if pd.notna(short_trend) and short_trend > 0:
                short_dist = (current_price - short_trend) / short_trend
                short_support = _score_support(short_dist, short_band)

            long_support = 0.0
            if pd.notna(long_trend) and long_trend > 0:
                long_dist = (current_price - long_trend) / long_trend
                long_support = _score_support(long_dist, long_band)

            support_score = weight_short * short_support + weight_long * long_support

            # 跌破长期趋势惩罚：连续 N 天收盘低于长趋势
            if long_break_days >= 2 and len(long_trend_series.dropna()) >= long_break_days:
                recent_closes = closes.iloc[-long_break_days:]
                recent_long = long_trend_series.iloc[-long_break_days:]
                if (recent_closes < recent_long).all():
                    support_score = max(0.0, support_score - long_break_penalty)

            total_support = (stab_score * 0.6 + support_score * 0.4)
            if return_parts:
                return total_support, {"stabilization_score": float(stab_score), "support_score": float(support_score)}
            return total_support
            
        except Exception as e:
            logger.warning(f"支撑企稳分析异常: {e}")
            return 0.0
