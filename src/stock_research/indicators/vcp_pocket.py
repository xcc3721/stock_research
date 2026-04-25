#!/usr/bin/env python3
"""
VCP压缩和口袋支点分析模块
专门分析波动性压缩和口袋支点形态
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger("vcp_pocket")


class VCPPocketAnalysis:
    """
    VCP压缩和口袋支点分析器
    
    提供：
    - VCP波动性压缩分析
    - 口袋支点检测
    - 成交量突破分析
    - 价格位置分析
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化VCP口袋分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}

    def _get_params(self, factor_name: str) -> Dict[str, Any]:
        factor_cfg = self.config.get("factors", {}).get(factor_name, {})
        if isinstance(factor_cfg, dict):
            params = factor_cfg.get("params", {})
            if isinstance(params, dict) and params:
                return params
        params = self.config.get(factor_name, {})
        if isinstance(params, dict):
            nested = params.get("params", {})
            if isinstance(nested, dict) and nested:
                return nested
            return params
        return {}
    
    def calc_vcp_compression(self, df: pd.DataFrame) -> float:
        """
        VCP压缩因子 - Volatility Contraction Pattern
        识别波动性收缩形态，通常预示着即将发生的突破
        
        VCP特征:
        1. 价格在一个逐渐收缩的通道内震荡
        2. 成交量在收缩过程中逐渐萎缩
        3. 波动幅度逐渐减小，形成压缩态势
        4. 通常在重要均线附近发生
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            得分 0.0-1.0，1.0表示完美的VCP压缩形态
        """
        try:
            params = self._get_params('vcp_compression')
            lookback = params.get('lookback', 30)
            min_compression_days = params.get('min_compression_days', 5)
            
            if len(df) < lookback or lookback < min_compression_days:
                return 0.0
            
            recent_data = df.tail(lookback)
            
            # 1. 波动收缩检测 (30%权重)
            volatility_compression_score = self._detect_volatility_compression(recent_data, params)
            
            # 2. 成交量萎缩检测 (35%权重)
            volume_contraction_score = self._detect_volume_contraction(recent_data, params)
            
            # 3. 价格通道收缩检测 (20%权重)
            price_channel_score = self._detect_price_channel_contraction(recent_data, params)
            
            # 4. 均线支撑确认 (15%权重)
            ma_support_score = self._detect_ma_support_during_compression(recent_data, params)
            
            # 综合评分
            vcp_score = (
                volatility_compression_score * 0.30 +
                volume_contraction_score * 0.35 +
                price_channel_score * 0.20 +
                ma_support_score * 0.15
            )
            
            return min(max(vcp_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning("VCP压缩因子计算失败: %s", e)
            return 0.0
    
    def calc_pocket_pivot(self, df: pd.DataFrame) -> float:
        """
        B1策略口袋妖怪因子 (Pocket Pivot)
        识别10天周期的成交量突破模式，专为抄底设计
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            得分 0.0-1.0，1.0表示完美的口袋支点信号
        """
        try:
            params = self._get_params('pocket_pivot')
            lookback = params.get('lookback', 10)  # 缩短回看期，适配中短线
            candidate_days = params.get('candidate_days', 5)  # 扩大候选期

            if len(df) < 20:
                return 0.0
            
            score = 0.0
            recent_data = df.tail(lookback)  # 最近lookback天数据
            
            # 寻找口袋支点候选日（最近candidate_days天内）
            pivot_candidates = []
            min_history_days = min(8, len(recent_data) - 2)  # 需要足够前期数据，但适配短回看期
            for i in range(len(recent_data) - candidate_days, len(recent_data)):
                if i < min_history_days:  # 确保有足够前期数据对比
                    continue
                
                pivot_score = self._evaluate_pocket_pivot_candidate(recent_data, i)
                if pivot_score > 0:
                    pivot_candidates.append((i, pivot_score))
            
            # 返回最高分的口袋支点信号
            if pivot_candidates:
                max_score = max(candidate[1] for candidate in pivot_candidates)
                return min(max_score, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning("口袋妖怪因子计算失败: %s", e)
            return 0.0
    
    def _evaluate_pocket_pivot_candidate(self, recent_data: pd.DataFrame, pivot_idx: int) -> float:
        """
        评估某一天是否为有效的口袋支点
        
        口袋支点条件（适配B1抄底）：
        1. 当天收盘价高于开盘价（阳线）
        2. 成交量超过前10天下跌日的最高成交量
        3. 价格在相对低位（近期下跌后的反弹）
        4. 符合抄底时机（不追高）
        """
        try:
            params = self.config.get('pocket_pivot_eval', {})
            weights = params.get('weights', {'volume': 0.3, 'position': 0.35, 'support': 0.25, 'timing': 0.1})

            # 检查索引有效性（已在外层循环中检查了最小历史天数）
            if pivot_idx >= len(recent_data):
                return 0.0
            
            pivot_row = recent_data.iloc[pivot_idx]
            
            # 提取关键数据
            pivot_open = pivot_row['open']
            pivot_close = pivot_row['close']
            pivot_high = pivot_row['high']
            pivot_volume = pivot_row['volume']
            pivot_low = pivot_row['low']
            
            # 基础条件：优先阳线，但也接受长下影线的小阴线
            is_bullish = pivot_close > pivot_open
            has_long_lower_shadow = (pivot_open - pivot_low) / pivot_open > 0.02  # 下影线超过2%
            small_bearish = pivot_close < pivot_open and (pivot_open - pivot_close) / pivot_open < 0.01  # 小幅收阴<1%
            
            if not (is_bullish or (has_long_lower_shadow and small_bearish)):
                return 0.0
            
            score = 0.0
            
            # 1. 成交量突破检验
            volume_score = self._check_volume_breakthrough(recent_data, pivot_idx, pivot_volume)
            score += volume_score * weights.get('volume', 0.4)
            
            # 2. 价格位置检验 - 在相对低位
            price_position_score = self._check_price_position_for_bottom(recent_data, pivot_idx, pivot_close)
            score += price_position_score * weights.get('position', 0.3)
            
            # 3. 支撑强度检验
            support_score = self._check_support_strength(recent_data, pivot_idx, pivot_low)
            score += support_score * weights.get('support', 0.2)
            
            # 4. 时机适宜性 - 抄底时机而非追高
            timing_score = self._check_bottom_timing(recent_data, pivot_idx)
            score += timing_score * weights.get('timing', 0.1)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("口袋支点候选评估失败: %s", e)
            return 0.0
    
    def _check_volume_breakthrough(self, recent_data: pd.DataFrame, pivot_idx: int, pivot_volume: float) -> float:
        """
        检查成交量突破情况
        B1抄底专用：成交量需要超过前10天所有下跌日的最高成交量
        """
        try:
            params = self.config.get('volume_breakthrough', {})
            lookback_days = params.get('lookback_days', 8)  # 减少回看天数
            breakthrough_ratios = params.get('breakthrough_ratios', [1.2, 1.5, 2.0])  # 降低阈值

            if pivot_idx < lookback_days:
                return 0.0
            
            # 获取前lookback_days天的数据
            prev_data = recent_data.iloc[pivot_idx - lookback_days:pivot_idx]
            
            # 筛选下跌日
            down_days = prev_data[prev_data['close'] < prev_data['open']]
            
            if len(down_days) == 0:
                return 0.5  # 没有下跌日，给基础分
            
            # 找到下跌日中的最高成交量
            max_down_volume = down_days['volume'].max()
            
            if max_down_volume <= 0:
                return 0.0
            
            # 计算成交量突破比例
            volume_ratio = pivot_volume / max_down_volume
            
            score = 0.0
            if volume_ratio >= breakthrough_ratios[2]:  # 3倍突破
                score = 1.0
            elif volume_ratio >= breakthrough_ratios[1]:  # 2倍突破
                score = 0.8
            elif volume_ratio >= breakthrough_ratios[0]:  # 1.5倍突破
                score = 0.6
            elif volume_ratio >= 1.2:  # 轻微突破
                score = 0.3
            
            return score
            
        except Exception as e:
            logger.debug("成交量突破检查失败: %s", e)
            return 0.0
    
    def _check_price_position_for_bottom(self, recent_data: pd.DataFrame, pivot_idx: int, pivot_close: float) -> float:
        """
        检查价格位置是否适合抄底
        在相对低位反弹而非追高
        """
        try:
            params = self.config.get('price_position', {})
            lookback_days = params.get('lookback_days', 20)
            bottom_percentiles = params.get('bottom_percentiles', [0.2, 0.35, 0.5])

            start_idx = max(0, pivot_idx - lookback_days)
            price_window = recent_data.iloc[start_idx:pivot_idx + 1]
            
            if len(price_window) < 5:
                return 0.0
            
            # 计算价格在区间中的位置
            price_lows = price_window['low']
            price_highs = price_window['high']
            
            price_range_low = price_lows.min()
            price_range_high = price_highs.max()
            
            if price_range_high <= price_range_low:
                return 0.0
            
            # 当前收盘价在价格区间的分位数
            price_percentile = (pivot_close - price_range_low) / (price_range_high - price_range_low)
            
            score = 0.0
            if price_percentile <= bottom_percentiles[0]:  # 在底部20%
                score = 1.0
            elif price_percentile <= bottom_percentiles[1]:  # 在底部35%
                score = 0.8
            elif price_percentile <= bottom_percentiles[2]:  # 在底部50%
                score = 0.5
            elif price_percentile <= 0.7:  # 在底部70%
                score = 0.2
            
            return score
            
        except Exception as e:
            logger.debug("价格位置检查失败: %s", e)
            return 0.0
    
    def _check_support_strength(self, recent_data: pd.DataFrame, pivot_idx: int, pivot_low: float) -> float:
        """
        检查支撑强度
        """
        try:
            params = self.config.get('support_strength', {})
            lookback_days = params.get('lookback_days', 15)
            support_tolerance = params.get('support_tolerance', 0.02)  # 2%容忍度

            start_idx = max(0, pivot_idx - lookback_days)
            price_window = recent_data.iloc[start_idx:pivot_idx]
            
            if len(price_window) < 3:
                return 0.0
            
            score = 0.0
            
            # 1. 检查是否有重要支撑位
            prev_lows = price_window['low']
            support_count = 0
            
            for prev_low in prev_lows:
                if abs(prev_low - pivot_low) / pivot_low <= support_tolerance:
                    support_count += 1
            
            if support_count >= 2:
                score += 0.6  # 多次测试支撑
            elif support_count >= 1:
                score += 0.3  # 有支撑
            
            # 2. 检查是否是近期新低
            recent_low = prev_lows.min()
            if pivot_low <= recent_low * (1 + support_tolerance):
                score += 0.4  # 接近或创新低，支撑重要
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("支撑强度检查失败: %s", e)
            return 0.0
    
    def _check_bottom_timing(self, recent_data: pd.DataFrame, pivot_idx: int) -> float:
        """
        检查抄底时机
        确保不是在追高，而是在相对低位反弹
        """
        try:
            params = self.config.get('bottom_timing', {})
            recent_days = params.get('recent_days', 5)

            if pivot_idx < recent_days:
                return 0.0
            
            # 检查最近几天的价格走势
            recent_window = recent_data.iloc[pivot_idx - recent_days:pivot_idx + 1]
            
            if len(recent_window) < 3:
                return 0.0
            
            closes = recent_window['close'].values
            
            score = 0.0
            
            # 1. 检查是否从低位反弹
            min_close_idx = np.argmin(closes[:-1])  # 排除当前日
            current_idx = len(closes) - 1
            
            if min_close_idx >= current_idx - 3:  # 最低点在最近3天内
                score += 0.6
            elif min_close_idx >= current_idx - recent_days:  # 最低点在观察期内
                score += 0.4
            
            # 2. 检查反弹幅度是否合理（不追高）
            if min_close_idx < current_idx:
                rebound_ratio = (closes[current_idx] - closes[min_close_idx]) / closes[min_close_idx]
                
                if 0.02 <= rebound_ratio <= 0.08:  # 2%-8%的合理反弹
                    score += 0.4
                elif 0.01 <= rebound_ratio <= 0.12:  # 1%-12%的可接受反弹
                    score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("抄底时机检查失败: %s", e)
            return 0.0
    
    # ==================== VCP压缩辅助方法 ====================
    
    def _detect_volatility_compression(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> float:
        """
        检测波动性收缩 - 通过ATR或日内波幅的变化来检测
        """
        try:
            volatility_params = (params or {}).get('volatility_compression', {})
            strong_threshold = volatility_params.get('strong_threshold', 0.5)
            obvious_threshold = volatility_params.get('obvious_threshold', 0.7)
            mild_threshold = volatility_params.get('mild_threshold', 0.85)
            weak_threshold = volatility_params.get('weak_threshold', 0.95)

            daily_ranges = (df['high'] - df['low']) / df['low']
            
            # 将数据分成前后两段，比较波动性变化
            mid_point = len(daily_ranges) // 2
            early_volatility = daily_ranges[:mid_point].mean()
            recent_volatility = daily_ranges[mid_point:].mean()
            
            # 检查波动性是否在收缩
            if early_volatility <= 0:
                return 0.0
            
            compression_ratio = recent_volatility / early_volatility
            
            # 波动性收缩评分：越小表示收缩越明显
            if compression_ratio <= strong_threshold:
                return 1.0  # 强烈收缩
            elif compression_ratio <= obvious_threshold:
                return 0.8  # 明显收缩
            elif compression_ratio <= mild_threshold:
                return 0.6  # 轻微收缩
            elif compression_ratio <= weak_threshold:
                return 0.3  # 很轻微收缩
            else:
                return 0.0  # 没有收缩
                
        except Exception as e:
            logger.debug("波动收缩检测失败: %s", e)
            return 0.0
    
    def _detect_volume_contraction(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> float:
        """
        检测成交量收缩 - 成交量在压缩期间逐渐萎缩
        """
        try:
            volume_params = (params or {}).get('volume_contraction', {})
            strong_threshold = volume_params.get('strong_threshold', 0.4)
            obvious_threshold = volume_params.get('obvious_threshold', 0.6)
            mild_threshold = volume_params.get('mild_threshold', 0.8)
            weak_threshold = volume_params.get('weak_threshold', 0.9)

            volumes = df['volume']
            
            # 计算成交量移动平均，检测趋势
            ma10_volume = volumes.rolling(10).mean()
            ma20_volume = volumes.rolling(20).mean()
            
            if len(ma10_volume) < 20 or len(ma20_volume) < 20:
                return 0.0
            
            # 检查最近成交量相对于前期的萎缩程度
            recent_avg = ma10_volume.tail(10).mean()
            earlier_avg = ma20_volume.iloc[-30:-10].mean() if len(ma20_volume) >= 30 else ma20_volume.iloc[:10].mean()
            
            if earlier_avg <= 0:
                return 0.0
            
            volume_contraction_ratio = recent_avg / earlier_avg
            
            # 成交量萎缩评分
            if volume_contraction_ratio <= strong_threshold:
                return 1.0  # 强烈萎缩
            elif volume_contraction_ratio <= obvious_threshold:
                return 0.8  # 明显萎缩
            elif volume_contraction_ratio <= mild_threshold:
                return 0.6  # 轻微萎缩
            elif volume_contraction_ratio <= weak_threshold:
                return 0.3  # 很轻微萎缩
            else:
                return 0.0  # 没有萎缩
                
        except Exception as e:
            logger.debug("成交量收缩检测失败: %s", e)
            return 0.0
    
    def _detect_price_channel_contraction(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> float:
        """
        检测价格通道收缩 - 高低点之间的价格区间逐渐收窄
        """
        try:
            channel_params = (params or {}).get('price_channel', {})
            min_periods = int(channel_params.get('min_periods', 3))
            strong_compression = channel_params.get('strong_compression', 0.5)
            obvious_compression = channel_params.get('obvious_compression', 0.7)
            mild_compression = channel_params.get('mild_compression', 0.85)

            period_length = len(df) // 3
            if period_length < max(1, min_periods):
                return 0.0
            
            periods = [
                df.iloc[:period_length],
                df.iloc[period_length:2*period_length], 
                df.iloc[2*period_length:]
            ]
            
            # 计算每个时期的价格通道宽度
            channel_widths = []
            for period in periods:
                if len(period) == 0:
                    continue
                high = period['high'].max()
                low = period['low'].min()
                if low > 0:
                    channel_width = (high - low) / low
                    channel_widths.append(channel_width)
            
            if len(channel_widths) < 2:
                return 0.0
            
            # 检查通道是否逐渐收窄
            compression_detected = True
            for i in range(1, len(channel_widths)):
                if channel_widths[i] >= channel_widths[i-1]:
                    compression_detected = False
                    break
            
            if compression_detected:
                # 计算压缩程度
                compression_ratio = channel_widths[-1] / channel_widths[0] if channel_widths[0] > 0 else 1.0
                
                if compression_ratio <= strong_compression:
                    return 1.0  # 强烈压缩
                elif compression_ratio <= obvious_compression:
                    return 0.8  # 明显压缩
                elif compression_ratio <= mild_compression:
                    return 0.5  # 轻微压缩
                else:
                    return 0.2  # 很轻微压缩
            
            return 0.0
            
        except Exception as e:
            logger.debug("价格通道收缩检测失败: %s", e)
            return 0.0
    
    def _detect_ma_support_during_compression(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> float:
        """
        检测压缩期间的均线支撑 - VCP通常在重要均线附近发生
        """
        try:
            current_price = df['close'].iloc[-1]
            ma_params = (params or {}).get('ma_support', {})
            ma10_tolerance = ma_params.get('ma10_tolerance', 0.03)
            ma20_tolerance = ma_params.get('ma20_tolerance', 0.05)
            ma30_tolerance = ma_params.get('ma30_tolerance', 0.05)
            
            # 计算关键均线
            ma10 = df['close'].rolling(10).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            ma30 = df['close'].rolling(30).mean().iloc[-1]
            
            # 检查价格与均线的关系
            support_score = 0.0
            
            # 在MA10附近 (±3%)
            if abs(current_price - ma10) / current_price <= ma10_tolerance:
                support_score += 0.4
            
            # 在MA20附近 (±5%)
            if abs(current_price - ma20) / current_price <= ma20_tolerance:
                support_score += 0.3
            
            # 在MA30附近 (±5%)
            if abs(current_price - ma30) / current_price <= ma30_tolerance:
                support_score += 0.3
            
            return min(support_score, 1.0)
                
        except Exception as e:
            logger.debug("均线支撑检测失败: %s", e)
            return 0.0
