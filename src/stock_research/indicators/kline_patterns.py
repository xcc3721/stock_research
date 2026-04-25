#!/usr/bin/env python3
"""
K线形态分析模块
专门识别各种经典K线反转和突破形态
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger("kline_patterns")


class KLinePatterns:
    """
    K线形态分析器
    
    提供：
    - 锤子线/倒锤子线检测
    - 吞没形态检测  
    - 晨星形态检测
    - 十字星确认
    - 长下影反转线检测
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化K线形态分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}

    def _get_params(self) -> Dict[str, Any]:
        factor_cfg = self.config.get('factors', {}).get('kline_patterns', {})
        if isinstance(factor_cfg, dict):
            params = factor_cfg.get('params', {})
            if isinstance(params, dict) and params:
                return params
        params = self.config.get('kline_patterns', {})
        if isinstance(params, dict):
            nested = params.get('params', {})
            if isinstance(nested, dict) and nested:
                return nested
            return params
        return {}
    
    def calc_kline_patterns(self, df: pd.DataFrame) -> float:
        """
        全面K线形态因子 - 不仅识别底部反转，还识别突破前蓄势和强势整理形态
        
        整合四大类形态：
        1. 经典底部反转 (40%权重) - 锤子线、阳包阴、晨星等
        2. 突破前蓄势 (30%权重) - 缩量横盘、箱体震荡、三角收敛
        3. 强势整理 (20%权重) - 强势股的高位震荡、旗形整理  
        4. 放量突破 (10%权重) - 向上突破确认
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            得分 0.0-1.0，1.0表示出现强烈的形态信号
        """
        try:
            kline_factor_params = self._get_params()
            if not kline_factor_params:
                logger.warning("⚠️  K线形态模块配置缺失，使用默认参数。建议检查 configs.json 中的 scoring_config.factors.kline_patterns.params")
                
            params = kline_factor_params
            
            lookback = params.get('lookback', None)
            if lookback is None:
                lookback = 10
                logger.warning("⚠️  缺少 kline_patterns.lookback 配置，使用默认值 %d", lookback)
            # clamp to 7-10 天，避免过短噪声或过长稀释形态
            try:
                lookback = int(lookback)
            except Exception:
                lookback = 10
            lookback = max(7, min(lookback, 10))
                
            # 权重强制聚焦：仅保留早晨之星≥0.6，其他子因子禁用
            weights = {
                'reversal': 1.0,      # 只看反转信号（内部仅保留早晨之星≥0.6）
                'accumulation': 0.0,
                'consolidation': 0.0,
                'breakout': 0.0
            }

            if len(df) < 15:
                return 0.0
            
            score = 0.0
            recent_data = df.tail(lookback)
            extended_data = df.tail(20)  # 更长期的形态观察
            
            # 1. 经典底部反转形态 (40%权重)
            reversal_score = self._detect_reversal_patterns(recent_data, extended_data)
            score += reversal_score * weights.get('reversal', 0.40)
            
            # 2. 突破前蓄势形态 (30%权重) - 新增核心逻辑
            accumulation_score = self._detect_accumulation_patterns(extended_data)
            score += accumulation_score * weights.get('accumulation', 0.30)
            
            # 3. 强势整理形态 (20%权重) - 新增核心逻辑
            consolidation_score = 0.0  # 禁用强势整理，方向不稳定
            score += consolidation_score * weights.get('consolidation', 0.0)
            
            # 4. 放量突破确认 (10%权重) - 新增核心逻辑
            breakout_score = 0.0  # 禁用放量突破，避免冲顶误判
            score += breakout_score * weights.get('breakout', 0.0)
            
            # 5. 形态互补加成 - 如果多种形态同时出现，给予额外加成
            pattern_count = sum([
                1 if reversal_score >= 0.3 else 0,
                1 if accumulation_score >= 0.3 else 0,
                1 if consolidation_score >= 0.3 else 0,
                1 if breakout_score >= 0.3 else 0
            ])
            
            if pattern_count >= 3:  # 多重形态共振
                score *= 1.25
            elif pattern_count >= 2:  # 双重形态配合
                score *= 1.15
            
            # 6. 防止过低评分 - 即使没有强烈形态，也给基础分
            if score < 0.1:
                # 基础分：只要不是明显的看跌形态，给予基础分0.1-0.3
                basic_score = self._calc_basic_pattern_score(recent_data)
                score = max(score, basic_score)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning("K线形态因子计算失败: %s", e)
            return 0.2  # 容错：给予基础分而不是0分
    
    def _detect_reversal_patterns(
        self,
        recent_data: pd.DataFrame,
        context_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        检测经典底部反转形态 - 整合原有的所有反转形态检测
        """
        try:
            if len(recent_data) < 3:
                return 0.0
            
            pattern_scores = {
                "hammer": self._detect_hammer_pattern(recent_data, context_data),
                "engulfing": self._detect_bullish_engulfing(recent_data),
                "morning_star": self._detect_morning_star(recent_data),
                "doji": self._detect_doji_confirmation(recent_data, context_data),
                "long_shadow": self._detect_long_lower_shadow_reversal(recent_data, context_data),
            }
            configured_weights = self._get_params().get('weights', {})
            if not isinstance(configured_weights, dict):
                configured_weights = {}
            weights = {
                "hammer": float(configured_weights.get("hammer", 0.0) or 0.0),
                "engulfing": float(configured_weights.get("engulfing", 0.0) or 0.0),
                "morning_star": float(configured_weights.get("morning_star", 1.0) or 0.0),
                "doji": float(configured_weights.get("doji", 0.0) or 0.0),
                "long_shadow": float(configured_weights.get("long_shadow", 0.0) or 0.0),
            }
            if sum(abs(value) for value in weights.values()) <= 0.0:
                weights["morning_star"] = 1.0

            weighted_scores = []
            for name, weight in weights.items():
                if weight <= 0.0:
                    continue
                score = pattern_scores.get(name, 0.0)
                if name == "morning_star":
                    score = min(score * 1.2, 1.0) if score >= 0.6 else 0.0
                weighted_scores.append((score, weight))

            total_weight = sum(weight for _, weight in weighted_scores)
            max_reversal_score = (
                sum(score * weight for score, weight in weighted_scores) / total_weight
                if total_weight > 0.0
                else 0.0
            )
            
            return min(max_reversal_score, 1.0)
            
        except Exception as e:
            logger.debug("反转形态检测失败: %s", e)
            return 0.0

    # -------------------- 共用上下文工具 --------------------

    def _has_recent_downtrend(
        self,
        recent_data: pd.DataFrame,
        window: int = 10,
        slope_thresh: float = -0.0005,
        drawdown_thresh: float = -0.02,
    ) -> bool:
        """
        检查最近一段是否处于下行/回撤，避免在高位给出“底部反转”高分
        """
        if len(recent_data) < 3:
            return False
        window = min(window, len(recent_data) - 1)
        closes = recent_data["close"].values
        sub = closes[-(window + 1) : -1]
        if len(sub) < 3 or np.any(~np.isfinite(sub)):
            return False
        base = np.mean(sub)
        if base <= 0:
            return False
        x = np.arange(len(sub))
        slope = np.polyfit(x, sub, 1)[0] / base
        drawdown = (closes[-1] - np.max(sub)) / np.max(sub)
        return slope <= slope_thresh or drawdown <= drawdown_thresh

    def _recent_pullback_range(
        self,
        data: pd.DataFrame,
        window: int = 20,
    ) -> float:
        """
        计算最近收盘相对窗口内最高价的回撤比例（负值代表低于高点）
        """
        if len(data) < window:
            return 0.0
        highs = data["high"].tail(window)
        if highs.empty or highs.max() <= 0:
            return 0.0
        current_close = data["close"].iloc[-1]
        return (current_close - highs.max()) / highs.max()
    
    def _detect_accumulation_patterns(self, extended_data: pd.DataFrame) -> float:
        """
        检测突破前蓄势形态 - 这是新增的核心逻辑，专门捕捉即将爆发的股票
        
        特征：
        1. 缩量横盘 - 成交量萎缩但价格相对稳定
        2. 箱体震荡 - 在一定价格区间内反复震荡
        3. 三角收敛 - 波动幅度逐渐收窄
        4. 底部抬升 - 低点逐步抬高但高点相对稳定
        """
        try:
            if len(extended_data) < 15:
                return 0.0
            
            score = 0.0
            
            # 1. 缩量横盘检测 (30分)
            volume_shrink_score = self._detect_volume_shrinkage_consolidation(extended_data)
            score += volume_shrink_score * 0.3
            
            # 2. 箱体震荡检测 (30分)
            box_pattern_score = self._detect_box_pattern(extended_data)
            score += box_pattern_score * 0.3
            
            # 3. 三角收敛检测 (25分)
            triangle_pattern_score = self._detect_triangle_convergence(extended_data)
            score += triangle_pattern_score * 0.25
            
            # 4. 底部抬升检测 (15分)
            rising_lows_score = self._detect_rising_lows_pattern(extended_data)
            score += rising_lows_score * 0.15
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("蓄势形态检测失败: %s", e)
            return 0.0
    
    def _detect_consolidation_patterns(self, extended_data: pd.DataFrame) -> float:
        """
        检测强势整理形态 - 专门识别强势股的健康回调和整理
        
        特征：
        1. 高位横盘 - 在相对高位进行横盘整理
        2. 旗形整理 - 短期回调后重新向上
        3. 强势回踩 - 快速拉升后的温和回调
        """
        try:
            if len(extended_data) < 10:
                return 0.0
            
            score = 0.0
            
            # 1. 高位横盘检测 (40分)
            high_level_consolidation_score = self._detect_high_level_consolidation(extended_data)
            score += high_level_consolidation_score * 0.4
            
            # 2. 旗形整理检测 (35分)
            flag_pattern_score = self._detect_flag_pattern(extended_data)
            score += flag_pattern_score * 0.35
            
            # 3. 强势回踩检测 (25分)
            strong_pullback_score = self._detect_strong_pullback_pattern(extended_data)
            score += strong_pullback_score * 0.25
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("强势整理检测失败: %s", e)
            return 0.0
    
    def _detect_breakout_patterns(self, recent_data: pd.DataFrame) -> float:
        """
        检测放量突破形态 - 确认向上突破信号
        
        特征：
        1. 价格突破前期高点
        2. 成交量明显放大
        3. 突破后站稳
        """
        try:
            if len(recent_data) < 5:
                return 0.0
            
            # 预检：突破前应有一定收敛/整理，避免在高波动冲顶阶段误判
            pre_slice = recent_data.iloc[-5:-1]
            pre_range = (pre_slice["high"].max() - pre_slice["low"].min()) / max(pre_slice["close"].mean(), 1e-6)
            if pre_range > 0.12:  # 近几日波动过大，视作假突破风险
                return 0.0
            
            score = 0.0
            
            # 1. 价格突破检测 (50分)
            price_breakout_score = self._detect_price_breakout(recent_data)
            score += price_breakout_score * 0.5
            
            # 2. 放量确认检测 (35分)
            volume_breakout_score = self._detect_volume_breakout(recent_data)
            score += volume_breakout_score * 0.35
            
            # 3. 突破站稳检测 (15分)
            breakout_sustain_score = self._detect_breakout_sustain(recent_data)
            score += breakout_sustain_score * 0.15
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("突破形态检测失败: %s", e)
            return 0.0
    
    def _calc_basic_pattern_score(self, recent_data: pd.DataFrame) -> float:
        """
        计算基础形态分 - 防止过低评分的兜底机制
        
        即使没有明显的技术形态，只要不是明显看跌，也给予0.1-0.3的基础分
        """
        try:
            if len(recent_data) < 3:
                return 0.1
            
            score = 0.1  # 基础分
            
            # 检查是否为明显看跌形态
            recent_closes = recent_data['close'].values
            recent_opens = recent_data['open'].values
            
            # 1. 连续阴线惩罚
            consecutive_down = 0
            for i in range(len(recent_closes)):
                if recent_closes[i] < recent_opens[i]:
                    consecutive_down += 1
                else:
                    break
            
            if consecutive_down >= 5:  # 连续5根阴线，明显看跌
                return 0.05
            elif consecutive_down >= 3:  # 连续3根阴线，适度惩罚
                score = 0.08
            
            # 2. 价格趋势检查
            if len(recent_closes) >= 5:
                price_trend = (recent_closes[-1] - recent_closes[-5]) / recent_closes[-5]
                if price_trend > 0.02:  # 近期有上涨
                    score += 0.1
                elif price_trend > -0.02:  # 横盘
                    score += 0.05
            
            # 3. 波动率检查 - 高波动的股票给予更多基础分
            if len(recent_closes) >= 3:
                price_changes = np.diff(recent_closes) / recent_closes[:-1]
                volatility = np.std(price_changes)
                if volatility > 0.05:  # 高波动
                    score += 0.1
                elif volatility > 0.03:  # 中等波动
                    score += 0.05
            
            return min(score, 0.3)
            
        except Exception as e:
            logger.debug("基础形态分计算失败: %s", e)
            return 0.1
    
    def _detect_hammer_pattern(
        self,
        recent_data: pd.DataFrame,
        context_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        检测锤子线和倒锤子线形态
        特征：实体小，下影线长，上影线短或无
        """
        try:
            kline_factor_params = self._get_params()
            params = kline_factor_params.get('hammer_pattern', {})
            
            body_ratio_max = params.get('body_ratio_max', 0.3)
            lower_shadow_min = params.get('lower_shadow_min', 0.5)
            upper_shadow_max = params.get('upper_shadow_max', 0.2)
            perfect_hammer_ratios = params.get('perfect_hammer_ratios', {'lower': 0.7, 'body': 0.15})
            standard_hammer_ratios = params.get('standard_hammer_ratios', {'lower': 0.6, 'body': 0.25})
            bullish_hammer_multiplier = params.get('bullish_hammer_multiplier', 1.2)

            if len(recent_data) < 2:
                return 0.0

            ctx = context_data if context_data is not None else recent_data
            # 需要有近期下行/回撤背景，否则高位长下影不计为看涨反转
            if not self._has_recent_downtrend(ctx, window=12, slope_thresh=-0.0005, drawdown_thresh=-0.015):
                return 0.0
            
            score = 0.0
            
            # 检查最近2天的K线
            for i in range(min(2, len(recent_data))):
                row = recent_data.iloc[-(i+1)]
                
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                if pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
                    continue
                
                full_range = high_price - low_price
                if full_range <= 0:
                    continue
                    
                # 实体、上影线、下影线计算
                body_size = abs(close_price - open_price)
                upper_shadow = high_price - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low_price
                
                # 计算比例
                body_ratio = body_size / full_range
                lower_shadow_ratio = lower_shadow / full_range
                upper_shadow_ratio = upper_shadow / full_range
                
                # 锤子线特征检测
                is_hammer = (
                    body_ratio <= body_ratio_max and
                    lower_shadow_ratio >= lower_shadow_min and
                    upper_shadow_ratio <= upper_shadow_max
                )
                
                if is_hammer:
                    # 计算锤子线强度
                    if (lower_shadow_ratio >= perfect_hammer_ratios['lower'] and 
                        body_ratio <= perfect_hammer_ratios['body']):
                        hammer_strength = 1.0  # 完美锤子线
                    elif (lower_shadow_ratio >= standard_hammer_ratios['lower'] and 
                          body_ratio <= standard_hammer_ratios['body']):
                        hammer_strength = 0.8  # 标准锤子线
                    else:
                        hammer_strength = 0.5  # 一般锤子线
                    
                    # 阳线锤子线加成
                    if close_price > open_price:
                        hammer_strength *= bullish_hammer_multiplier
                    
                    # 位置加成：越靠近最新日期，权重越高
                    position_weight = 1.0 if i == 0 else 0.7
                    
                    score = max(score, hammer_strength * position_weight)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("锤子线检测失败: %s", e)
            return 0.0
    
    def _detect_bullish_engulfing(self, recent_data: pd.DataFrame) -> float:
        """
        检测阳包阴（看涨吞没）形态
        特征：前一天阴线，当天阳线完全包含前一天K线
        """
        try:
            kline_factor_params = self._get_params()
            params = kline_factor_params.get('bullish_engulfing', {})
            
            min_body_ratio = params.get('min_body_ratio', 0.3)
            perfect_engulfing_ratio = params.get('perfect_engulfing_ratio', 1.5)
            standard_engulfing_ratio = params.get('standard_engulfing_ratio', 1.2)
            volume_confirm_ratio = params.get('volume_confirm_ratio', 1.2)

            if len(recent_data) < 2:
                return 0.0
            
            score = 0.0
            
            # 检查连续的K线对
            for i in range(len(recent_data) - 1):
                prev_row = recent_data.iloc[i]
                curr_row = recent_data.iloc[i + 1]
                
                # 前一天：阴线
                prev_open, prev_close = prev_row['open'], prev_row['close']
                prev_high, prev_low = prev_row['high'], prev_row['low']
                
                # 当天：阳线
                curr_open, curr_close = curr_row['open'], curr_row['close']
                curr_high, curr_low = curr_row['high'], curr_row['low']
                
                if pd.isna(prev_open) or pd.isna(curr_open):
                    continue
                
                # 检查基本条件
                prev_is_bearish = prev_close < prev_open  # 前一天阴线
                curr_is_bullish = curr_close > curr_open  # 当天阳线
                
                if not (prev_is_bearish and curr_is_bullish):
                    continue
                
                # 检查吞没条件
                curr_opens_below = curr_open <= prev_close  # 低开
                curr_closes_above = curr_close >= prev_open  # 高收
                
                if not (curr_opens_below and curr_closes_above):
                    continue
                
                # 计算实体大小
                prev_body = abs(prev_open - prev_close)
                curr_body = abs(curr_open - curr_close)
                prev_range = prev_high - prev_low
                curr_range = curr_high - curr_low
                
                # 实体比例检查
                if curr_range > 0 and curr_body / curr_range >= min_body_ratio:
                    # 计算吞没强度
                    if curr_body >= prev_body * perfect_engulfing_ratio:
                        engulfing_strength = 1.0
                    elif curr_body >= prev_body * standard_engulfing_ratio:
                        engulfing_strength = 0.8
                    else:
                        engulfing_strength = 0.5
                    
                    # 成交量确认
                    if ('volume' in prev_row and 'volume' in curr_row and 
                        curr_row['volume'] > prev_row['volume'] * volume_confirm_ratio):
                        engulfing_strength *= 1.2
                    
                    # 位置权重：越靠近最新，权重越高
                    days_from_latest = len(recent_data) - 1 - (i + 1)
                    position_weight = max(0.5, 1.0 - days_from_latest * 0.2)
                    
                    score = max(score, engulfing_strength * position_weight)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("阳包阴检测失败: %s", e)
            return 0.0
    
    def _detect_morning_star(self, recent_data: pd.DataFrame) -> float:
        """
        检测早晨之星（启明星）形态
        特征：三K线组合，第一根阴线，第二根十字星/小实体，第三根阳线
        """
        try:
            kline_factor_params = self._get_params()
            params = kline_factor_params.get('morning_star', {})
            
            min_first_body_ratio = params.get('min_first_body_ratio', 0.4)
            max_middle_body_ratio = params.get('max_middle_body_ratio', 0.3)
            min_third_body_ratio = params.get('min_third_body_ratio', 0.4)
            gap_down_min = params.get('gap_down_min', 0.01)
            gap_up_min = params.get('gap_up_min', 0.01)

            if len(recent_data) < 3:
                return 0.0
            
            score = 0.0
            
            # 检查3K线组合
            for i in range(len(recent_data) - 2):
                first_row = recent_data.iloc[i]
                middle_row = recent_data.iloc[i + 1]
                third_row = recent_data.iloc[i + 2]
                
                # 第一根K线：阴线
                first_open, first_close = first_row['open'], first_row['close']
                first_high, first_low = first_row['high'], first_row['low']
                first_range = first_high - first_low
                
                if first_range <= 0 or first_close >= first_open:
                    continue
                
                first_body = first_open - first_close
                if first_body / first_range < min_first_body_ratio:
                    continue
                
                # 第二根K线：十字星或小实体
                middle_open, middle_close = middle_row['open'], middle_row['close']
                middle_high, middle_low = middle_row['high'], middle_row['low']
                middle_range = middle_high - middle_low
                
                if middle_range <= 0:
                    continue
                
                middle_body = abs(middle_open - middle_close)
                if middle_body / middle_range > max_middle_body_ratio:
                    continue
                
                # 第三根K线：阳线
                third_open, third_close = third_row['open'], third_row['close']
                third_high, third_low = third_row['high'], third_row['low']
                third_range = third_high - third_low
                
                if third_range <= 0 or third_close <= third_open:
                    continue
                
                third_body = third_close - third_open
                if third_body / third_range < min_third_body_ratio:
                    continue
                
                # 检查缺口
                gap_down = (first_close - middle_high) / first_close if first_close > middle_high else 0
                gap_up = (third_low - middle_close) / middle_close if third_low > middle_close else 0
                
                # 计算早晨之星强度
                morning_star_strength = 0.5  # 基础分数
                
                if gap_down >= gap_down_min and gap_up >= gap_up_min:
                    morning_star_strength = 1.0  # 标准早晨之星（有缺口）
                elif gap_down >= gap_down_min or gap_up >= gap_up_min:
                    morning_star_strength = 0.8  # 部分缺口
                
                # 第三根K线强度加成
                if third_body / third_range > 0.6:
                    morning_star_strength *= 1.2
                
                # 位置权重
                days_from_latest = len(recent_data) - 1 - (i + 2)
                position_weight = max(0.5, 1.0 - days_from_latest * 0.2)
                
                score = max(score, morning_star_strength * position_weight)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("早晨之星检测失败: %s", e)
            return 0.0
    
    def _detect_doji_confirmation(
        self,
        recent_data: pd.DataFrame,
        context_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        检测十字星底部确认形态
        特征：十字星后有阳线确认
        """
        try:
            kline_factor_params = self._get_params()
            params = kline_factor_params.get('doji_confirmation', {})
            
            max_doji_body_ratio = params.get('max_doji_body_ratio', 0.1)
            min_confirm_body_ratio = params.get('min_confirm_body_ratio', 0.3)
            min_shadow_ratio = params.get('min_shadow_ratio', 0.3)

            if len(recent_data) < 2:
                return 0.0

            ctx = context_data if context_data is not None else recent_data
            # 仅在近期存在回撤背景下才视为底部十字星确认，避免高位滞涨误判
            if not self._has_recent_downtrend(ctx, window=10, slope_thresh=-0.0005, drawdown_thresh=-0.015):
                return 0.0
            
            score = 0.0
            
            # 检查十字星 + 确认K线组合
            for i in range(len(recent_data) - 1):
                doji_row = recent_data.iloc[i]
                confirm_row = recent_data.iloc[i + 1]
                
                # 十字星检测
                doji_open, doji_close = doji_row['open'], doji_row['close']
                doji_high, doji_low = doji_row['high'], doji_row['low']
                doji_range = doji_high - doji_low
                
                if doji_range <= 0:
                    continue
                
                doji_body = abs(doji_open - doji_close)
                if doji_body / doji_range > max_doji_body_ratio:
                    continue
                
                # 确认K线检测
                confirm_open, confirm_close = confirm_row['open'], confirm_row['close']
                confirm_high, confirm_low = confirm_row['high'], confirm_row['low']
                confirm_range = confirm_high - confirm_low
                
                if confirm_range <= 0 or confirm_close <= confirm_open:
                    continue
                
                confirm_body = confirm_close - confirm_open
                if confirm_body / confirm_range < min_confirm_body_ratio:
                    continue
                
                # 检查确认K线是否向上突破
                if confirm_close <= doji_high:
                    continue
                
                # 计算十字星质量
                upper_shadow = doji_high - max(doji_open, doji_close)
                lower_shadow = min(doji_open, doji_close) - doji_low
                
                doji_strength = 0.5
                if lower_shadow / doji_range >= min_shadow_ratio:
                    doji_strength = 0.8  # 有长下影的十字星
                
                # 位置权重
                days_from_latest = len(recent_data) - 1 - (i + 1)
                position_weight = max(0.5, 1.0 - days_from_latest * 0.3)
                
                score = max(score, doji_strength * position_weight)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("十字星确认检测失败: %s", e)
            return 0.0
    
    def _detect_long_lower_shadow_reversal(
        self,
        recent_data: pd.DataFrame,
        context_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        检测长下影反转线（非锤子线的其他长下影形态）
        """
        try:
            kline_factor_params = self._get_params()
            params = kline_factor_params.get('long_lower_shadow', {})
            
            min_lower_shadow_ratio = params.get('min_lower_shadow_ratio', 0.5)
            max_upper_shadow_ratio = params.get('max_upper_shadow_ratio', 0.3)
            min_body_ratio = params.get('min_body_ratio', 0.2)
            perfect_shadow_ratio = params.get('perfect_shadow_ratio', 0.7)
            bullish_multiplier = params.get('bullish_multiplier', 1.3)

            if len(recent_data) < 1:
                return 0.0

            ctx = context_data if context_data is not None else recent_data
            # 必须处于下跌/回撤环境，避免高位长下影被误判为看涨
            if not self._has_recent_downtrend(ctx, window=12, slope_thresh=-0.0005, drawdown_thresh=-0.02):
                return 0.0
            
            score = 0.0
            
            # 检查最近的K线
            for i in range(min(3, len(recent_data))):
                row = recent_data.iloc[-(i+1)]
                
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                if pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
                    continue
                
                full_range = high_price - low_price
                if full_range <= 0:
                    continue
                
                # 计算各部分比例
                body_size = abs(close_price - open_price)
                upper_shadow = high_price - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low_price
                
                body_ratio = body_size / full_range
                lower_shadow_ratio = lower_shadow / full_range
                upper_shadow_ratio = upper_shadow / full_range
                
                # 长下影反转线特征
                is_long_lower_shadow = (
                    lower_shadow_ratio >= min_lower_shadow_ratio and
                    upper_shadow_ratio <= max_upper_shadow_ratio and
                    body_ratio >= min_body_ratio  # 有一定实体大小
                )
                
                if is_long_lower_shadow:
                    # 计算反转线强度
                    if lower_shadow_ratio >= perfect_shadow_ratio:
                        reversal_strength = 1.0
                    elif lower_shadow_ratio >= 0.6:
                        reversal_strength = 0.8
                    else:
                        reversal_strength = 0.6
                    
                    # 阳线加成
                    if close_price > open_price:
                        reversal_strength *= bullish_multiplier
                    
                    # 位置权重
                    position_weight = 1.0 if i == 0 else max(0.6, 1.0 - i * 0.2)
                    
                    score = max(score, reversal_strength * position_weight)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("长下影反转线检测失败: %s", e)
            return 0.0
    
    # ==================== 新增的核心形态检测方法 ====================
    
    def _detect_volume_shrinkage_consolidation(self, extended_data: pd.DataFrame) -> float:
        """
        检测缩量横盘形态 - 成交量萎缩但价格相对稳定
        """
        try:
            if len(extended_data) < 10:
                return 0.0
            
            # 需要先有适度回撤或整理，不在绝对高点也不深度破位
            pullback = self._recent_pullback_range(extended_data, window=20)
            if pullback > -0.005 or pullback < -0.2:
                return 0.0
            closes_all = extended_data['close'].values
            if len(closes_all) >= 20:
                mid_trend = (closes_all[-1] - closes_all[-20]) / closes_all[-20]
                if mid_trend < -0.05:  # 中期趋势明显向下则跳过“蓄势”判定
                    return 0.0
            
            # 分析成交量变化
            volumes = extended_data['volume'].values
            recent_volume = np.mean(volumes[-5:])  # 最近5天平均成交量
            earlier_volume = np.mean(volumes[-15:-5])  # 前10天平均成交量
            
            # 分析价格稳定性
            closes = extended_data['close'].values
            price_volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
            
            score = 0.0
            
            # 1. 成交量萎缩检查
            if earlier_volume > 0:
                volume_shrink_ratio = recent_volume / earlier_volume
                if volume_shrink_ratio <= 0.5:  # 萎缩50%以上
                    score += 0.6
                elif volume_shrink_ratio <= 0.7:  # 萎缩30%以上
                    score += 0.4
                elif volume_shrink_ratio <= 0.8:  # 萎缩20%以上
                    score += 0.2
            
            # 2. 价格稳定性检查
            if price_volatility <= 0.02:  # 波动率小于2%
                score += 0.4
            elif price_volatility <= 0.03:  # 波动率小于3%
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("缩量横盘检测失败: %s", e)
            return 0.0
    
    def _detect_box_pattern(self, extended_data: pd.DataFrame) -> float:
        """
        检测箱体震荡形态 - 在一定价格区间内反复震荡
        """
        try:
            if len(extended_data) < 10:
                return 0.0
            
            highs = extended_data['high'].values[-15:]
            lows = extended_data['low'].values[-15:]
            
            # 计算支撑位和阻力位
            resistance_level = np.percentile(highs, 85)
            support_level = np.percentile(lows, 15)
            
            if resistance_level <= support_level:
                return 0.0
            
            box_height = (resistance_level - support_level) / support_level
            
            # 检查价格是否在箱体内震荡
            recent_data = extended_data.tail(10)
            in_box_count = 0
            
            for _, row in recent_data.iterrows():
                if support_level <= row['low'] and row['high'] <= resistance_level:
                    in_box_count += 1
                elif (row['low'] <= support_level * 1.02 and 
                      row['high'] >= resistance_level * 0.98):
                    in_box_count += 0.5  # 部分在箱体内
            
            box_ratio = in_box_count / len(recent_data)
            
            score = 0.0
            
            # 1. 箱体完整性
            if box_ratio >= 0.8:  # 80%以上时间在箱体内
                score += 0.6
            elif box_ratio >= 0.6:  # 60%以上时间在箱体内
                score += 0.4
            elif box_ratio >= 0.4:  # 40%以上时间在箱体内
                score += 0.2
            
            # 2. 箱体高度合理性
            if 0.05 <= box_height <= 0.15:  # 5%-15%的箱体高度比较理想
                score += 0.4
            elif 0.03 <= box_height <= 0.20:  # 3%-20%可接受
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("箱体震荡检测失败: %s", e)
            return 0.0
    
    def _detect_triangle_convergence(self, extended_data: pd.DataFrame) -> float:
        """
        检测三角收敛形态 - 波动幅度逐渐收窄
        """
        try:
            if len(extended_data) < 15:
                return 0.0
            
            # 分两段分析波动幅度
            early_data = extended_data.head(8)
            recent_data = extended_data.tail(7)
            
            # 计算平均波动幅度
            def calc_avg_volatility(data):
                daily_ranges = (data['high'] - data['low']) / data['close']
                return np.mean(daily_ranges)
            
            early_volatility = calc_avg_volatility(early_data)
            recent_volatility = calc_avg_volatility(recent_data)
            
            if early_volatility <= 0:
                return 0.0
            
            convergence_ratio = recent_volatility / early_volatility
            
            score = 0.0
            
            # 波动收敛程度
            if convergence_ratio <= 0.5:  # 波动收敛50%以上
                score += 0.8
            elif convergence_ratio <= 0.7:  # 波动收敛30%以上
                score += 0.6
            elif convergence_ratio <= 0.8:  # 波动收敛20%以上
                score += 0.3
            
            # 检查价格是否呈收敛趋势
            highs = extended_data['high'].values
            lows = extended_data['low'].values
            
            # 使用线性回归检查高点下降趋势和低点上升趋势
            x = np.arange(len(highs))
            
            try:
                high_slope = np.polyfit(x, highs, 1)[0]
                low_slope = np.polyfit(x, lows, 1)[0]
                
                # 理想的三角形：高点下降，低点上升
                if high_slope < 0 and low_slope > 0:
                    score += 0.2
                elif high_slope < 0 or low_slope > 0:  # 至少一边收敛
                    score += 0.1
            except:
                pass
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("三角收敛检测失败: %s", e)
            return 0.0
    
    def _detect_rising_lows_pattern(self, extended_data: pd.DataFrame) -> float:
        """
        检测底部抬升形态 - 低点逐步抬高
        """
        try:
            if len(extended_data) < 10:
                return 0.0
            
            lows = extended_data['low'].values
            
            # 寻找最近的3个低点
            recent_lows = []
            for i in range(2, len(lows) - 2):
                if lows[i] <= lows[i-1] and lows[i] <= lows[i+1]:
                    if lows[i] <= lows[i-2] and lows[i] <= lows[i+2]:
                        recent_lows.append((i, lows[i]))
            
            if len(recent_lows) < 2:
                return 0.0
            
            # 取最近的3个低点
            recent_lows = recent_lows[-3:] if len(recent_lows) >= 3 else recent_lows[-2:]
            
            score = 0.0
            
            # 检查低点是否逐步抬高
            if len(recent_lows) >= 2:
                low_trend_positive = True
                for i in range(1, len(recent_lows)):
                    if recent_lows[i][1] <= recent_lows[i-1][1]:
                        low_trend_positive = False
                        break
                
                if low_trend_positive:
                    # 计算抬升幅度
                    total_rise = (recent_lows[-1][1] - recent_lows[0][1]) / recent_lows[0][1]
                    if total_rise >= 0.05:  # 低点抬升5%以上
                        score = 0.8
                    elif total_rise >= 0.02:  # 低点抬升2%以上
                        score = 0.6
                    elif total_rise > 0:  # 有抬升
                        score = 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("底部抬升检测失败: %s", e)
            return 0.0
    
    def _detect_high_level_consolidation(self, extended_data: pd.DataFrame) -> float:
        """
        检测高位横盘形态 - 在相对高位进行横盘整理
        """
        try:
            if len(extended_data) < 15:
                return 0.0
            
            # 判断当前是否在高位
            lookback_data = extended_data.head(15)
            recent_data = extended_data.tail(5)
            
            period_high = lookback_data['high'].max()
            recent_avg_price = recent_data['close'].mean()
            
            if period_high <= 0:
                return 0.0
            
            # 计算价格位置
            price_position = recent_avg_price / period_high
            
            if price_position < 0.85:  # 不在高位
                return 0.0
            
            # 需要有轻微回撤后横盘，避免“冲顶未回落”也给高分
            drawdown = self._recent_pullback_range(extended_data, window=20)
            if drawdown > -0.005 or drawdown < -0.12:
                return 0.0
            
            # 检查横盘特征
            recent_volatility = np.std(recent_data['close']) / recent_avg_price
            
            score = 0.0
            
            # 1. 高位程度
            if price_position >= 0.95:  # 在95%以上高位
                score += 0.6
            elif price_position >= 0.90:  # 在90%以上高位
                score += 0.4
            elif price_position >= 0.85:  # 在85%以上高位
                score += 0.2
            
            # 2. 横盘稳定性
            if recent_volatility <= 0.02:  # 波动小于2%
                score += 0.4
            elif recent_volatility <= 0.03:  # 波动小于3%
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("高位横盘检测失败: %s", e)
            return 0.0
    
    def _detect_flag_pattern(self, extended_data: pd.DataFrame) -> float:
        """
        检测旗形整理形态 - 短期回调后重新向上
        """
        try:
            if len(extended_data) < 10:
                return 0.0
            
            closes = extended_data['close'].values
            
            # 寻找旗杆（前期上涨）和旗面（短期回调）
            flagpole_end = len(closes) - 5  # 假设最近5天是旗面
            flagpole_start = max(0, flagpole_end - 8)  # 前8天是旗杆
            
            if flagpole_start >= flagpole_end:
                return 0.0
            
            # 旗杆分析（前期上涨）
            flagpole_start_price = closes[flagpole_start]
            flagpole_end_price = closes[flagpole_end]
            flagpole_gain = (flagpole_end_price - flagpole_start_price) / flagpole_start_price
            
            # 旗面分析（短期回调）
            flag_start_price = closes[flagpole_end]
            flag_end_price = closes[-1]
            flag_retrace = (flag_start_price - flag_end_price) / flag_start_price
            
            score = 0.0
            
            # 1. 旗杆强度检查
            if flagpole_gain >= 0.10:  # 前期涨幅10%以上
                score += 0.5
            elif flagpole_gain >= 0.05:  # 前期涨幅5%以上
                score += 0.3
            elif flagpole_gain >= 0.02:  # 前期涨幅2%以上
                score += 0.1
            
            # 2. 旗面回调合理性
            if 0.02 <= flag_retrace <= 0.08:  # 回调2%-8%比较理想
                score += 0.4
            elif 0.01 <= flag_retrace <= 0.12:  # 回调1%-12%可接受
                score += 0.2
            
            # 3. 成交量配合（旗面期间成交量萎缩）
            if len(extended_data) >= 10:
                flagpole_volume = extended_data.iloc[flagpole_start:flagpole_end]['volume'].mean()
                flag_volume = extended_data.iloc[flagpole_end:]['volume'].mean()
                
                if flagpole_volume > 0 and flag_volume / flagpole_volume <= 0.7:
                    score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("旗形整理检测失败: %s", e)
            return 0.0
    
    def _detect_strong_pullback_pattern(self, extended_data: pd.DataFrame) -> float:
        """
        检测强势回踩形态 - 快速拉升后的温和回调
        """
        try:
            if len(extended_data) < 8:
                return 0.0
            
            closes = extended_data['close'].values
            
            # 分析快速拉升期和回调期
            pullback_start = len(closes) - 3  # 最近3天是回调期
            rally_start = max(0, pullback_start - 5)  # 前5天是拉升期
            
            if rally_start >= pullback_start:
                return 0.0
            
            # 拉升期分析
            rally_start_price = closes[rally_start]
            rally_end_price = closes[pullback_start]
            rally_gain = (rally_end_price - rally_start_price) / rally_start_price
            
            # 回调期分析
            pullback_start_price = closes[pullback_start]
            current_price = closes[-1]
            pullback_ratio = (pullback_start_price - current_price) / pullback_start_price
            
            score = 0.0
            
            # 1. 前期拉升强度
            if rally_gain >= 0.15:  # 前期涨幅15%以上
                score += 0.6
            elif rally_gain >= 0.08:  # 前期涨幅8%以上
                score += 0.4
            elif rally_gain >= 0.03:  # 前期涨幅3%以上
                score += 0.2
            
            # 2. 回调幅度合理性
            if 0.02 <= pullback_ratio <= 0.06:  # 回调2%-6%比较健康
                score += 0.4
            elif 0.01 <= pullback_ratio <= 0.10:  # 回调1%-10%可接受
                score += 0.2
            elif pullback_ratio <= 0.01:  # 基本没回调，强势
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("强势回踩检测失败: %s", e)
            return 0.0
    
    def _detect_price_breakout(self, recent_data: pd.DataFrame) -> float:
        """
        检测价格突破 - 价格突破前期高点
        """
        try:
            if len(recent_data) < 5:
                return 0.0
            
            current_price = recent_data['close'].iloc[-1]
            recent_high = recent_data['high'].iloc[-5:-1].max()  # 前4天的最高价
            
            if recent_high <= 0:
                return 0.0
            
            breakout_ratio = (current_price - recent_high) / recent_high
            
            score = 0.0
            
            if breakout_ratio >= 0.03:  # 突破3%以上
                score = 1.0
            elif breakout_ratio >= 0.01:  # 突破1%以上
                score = 0.8
            elif breakout_ratio >= 0.005:  # 突破0.5%以上
                score = 0.5
            elif breakout_ratio > 0:  # 有突破
                score = 0.3
            
            return score
            
        except Exception as e:
            logger.debug("价格突破检测失败: %s", e)
            return 0.0
    
    def _detect_volume_breakout(self, recent_data: pd.DataFrame) -> float:
        """
        检测成交量突破 - 成交量明显放大
        """
        try:
            if len(recent_data) < 5:
                return 0.0
            
            current_volume = recent_data['volume'].iloc[-1]
            avg_volume = recent_data['volume'].iloc[-5:-1].mean()
            
            if avg_volume <= 0:
                return 0.0
            
            volume_ratio = current_volume / avg_volume
            
            score = 0.0
            
            if volume_ratio >= 3.0:  # 放量3倍以上
                score = 1.0
            elif volume_ratio >= 2.0:  # 放量2倍以上
                score = 0.8
            elif volume_ratio >= 1.5:  # 放量1.5倍以上
                score = 0.6
            elif volume_ratio >= 1.2:  # 放量1.2倍以上
                score = 0.3
            
            return score
            
        except Exception as e:
            logger.debug("成交量突破检测失败: %s", e)
            return 0.0
    
    def _detect_breakout_sustain(self, recent_data: pd.DataFrame) -> float:
        """
        检测突破站稳 - 突破后是否能够站稳
        """
        try:
            if len(recent_data) < 3:
                return 0.0
            
            # 检查最近几天是否保持在突破位之上
            recent_closes = recent_data['close'].values[-3:]
            recent_lows = recent_data['low'].values[-3:]
            
            # 假设突破位是前期高点
            breakout_level = recent_data['high'].iloc[-5:-2].max() if len(recent_data) >= 5 else recent_data['high'].iloc[0]
            
            if breakout_level <= 0:
                return 0.0
            
            # 检查站稳程度
            sustain_count = 0
            for close in recent_closes:
                if close >= breakout_level * 0.99:  # 允许1%的回落
                    sustain_count += 1
            
            # 检查是否有明显回落
            low_sustain_count = 0
            for low in recent_lows:
                if low >= breakout_level * 0.95:  # 低点不能跌破突破位太多
                    low_sustain_count += 1
            
            score = 0.0
            
            if sustain_count == len(recent_closes) and low_sustain_count == len(recent_lows):
                score = 1.0  # 完美站稳
            elif sustain_count >= len(recent_closes) * 0.8:
                score = 0.7  # 基本站稳
            elif sustain_count >= len(recent_closes) * 0.5:
                score = 0.4  # 勉强站稳
            
            return score
            
        except Exception as e:
            logger.debug("突破站稳检测失败: %s", e)
            return 0.0
