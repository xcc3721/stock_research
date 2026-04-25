#!/usr/bin/env python3
"""
成交量分析模块
专门分析成交量异动、量价配合等信号
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger("volume_analysis")


class VolumeAnalysis:
    """
    成交量分析器
    
    提供：
    - 成交量异动检测
    - 缩量企稳分析
    - 放量确认分析  
    - 量价背离检测
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化成交量分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
    
    def calc_volume_anomaly(self, df: pd.DataFrame) -> float:
        """
        B1策略成交量异动因子
        识别缩量企稳+放量确认的抄底时机
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            得分 0.0-1.0，1.0表示完美的量价配合信号
        """
        try:
            params = self.config.get('volume_anomaly', {})
            lookback = params.get('lookback', 15)
            weights = params.get('weights', {'stabilization': 0.5, 'confirmation': 0.3, 'divergence': 0.2})

            if len(df) < 20:
                return 0.0
            
            score = 0.0
            recent_data = df.tail(lookback)  # 最近lookback天数据
            
            # 1. 缩量企稳检测
            volume_stabilization_score = self._detect_volume_stabilization(recent_data)
            score += volume_stabilization_score * weights.get('stabilization', 0.5)
            
            # 2. 放量确认检测
            volume_confirmation_score = self._detect_volume_confirmation(recent_data)
            score += volume_confirmation_score * weights.get('confirmation', 0.3)
            
            # 3. 量价背离检测
            volume_price_divergence_score = self._detect_volume_price_divergence(recent_data)
            score += volume_price_divergence_score * weights.get('divergence', 0.2)
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning("成交量异动因子计算失败: %s", e)
            return 0.0
    
    def _detect_volume_stabilization(self, recent_data: pd.DataFrame) -> float:
        """
        检测缩量企稳信号
        特征：成交量持续萎缩但价格止跌企稳
        """
        try:
            params = self.config.get('volume_stabilization', {})
            lookback = params.get('lookback', 10)
            early_period = params.get('early_period', 5)
            recent_period = params.get('recent_period', 5)
            shrink_ratios = params.get('shrink_ratios', [0.5, 0.7, 0.8])

            if len(recent_data) < lookback:
                return 0.0
            
            volumes = recent_data['volume'].values
            closes = recent_data['close'].values
            
            # 计算前期平均成交量
            early_avg_volume = np.mean(volumes[:early_period])
            # 计算近期平均成交量
            recent_avg_volume = np.mean(volumes[-recent_period:])
            
            if early_avg_volume <= 0:
                return 0.0
            
            score = 0.0
            
            # 1. 成交量萎缩程度 (60分)
            volume_shrink_ratio = recent_avg_volume / early_avg_volume
            if volume_shrink_ratio < shrink_ratios[0]:
                score += 0.6
            elif volume_shrink_ratio < shrink_ratios[1]:
                score += 0.4
            elif volume_shrink_ratio < shrink_ratios[2]:
                score += 0.2
            
            # 2. 价格企稳程度 (40分)
            recent_prices = closes[-recent_period:]
            price_stability = self._calc_price_stability(recent_prices)
            score += price_stability * 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("缩量企稳检测失败: %s", e)
            return 0.0
    
    def _detect_volume_confirmation(self, recent_data: pd.DataFrame) -> float:
        """
        检测放量确认信号
        特征：在企稳基础上出现放量上涨
        """
        try:
            params = self.config.get('volume_confirmation', {})
            confirmation_period = params.get('confirmation_period', 3)
            volume_multiplier = params.get('volume_multiplier', [1.5, 2.0, 3.0])
            price_rise_min = params.get('price_rise_min', 0.02)

            if len(recent_data) < confirmation_period + 5:
                return 0.0
            
            volumes = recent_data['volume'].values
            closes = recent_data['close'].values
            
            # 计算基准成交量（前期平均）
            base_period = len(volumes) - confirmation_period
            base_avg_volume = np.mean(volumes[:base_period])
            
            if base_avg_volume <= 0:
                return 0.0
            
            score = 0.0
            
            # 检查最近confirmation_period天的放量情况
            recent_volumes = volumes[-confirmation_period:]
            recent_closes = closes[-confirmation_period:]
            
            # 1. 放量程度 (60分)
            max_volume_ratio = np.max(recent_volumes) / base_avg_volume
            avg_volume_ratio = np.mean(recent_volumes) / base_avg_volume
            
            if max_volume_ratio >= volume_multiplier[2] or avg_volume_ratio >= volume_multiplier[1]:
                score += 0.6
            elif max_volume_ratio >= volume_multiplier[1] or avg_volume_ratio >= volume_multiplier[0]:
                score += 0.4
            elif max_volume_ratio >= volume_multiplier[0]:
                score += 0.2
            
            # 2. 价格配合度 (40分)
            price_rise = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            if price_rise >= price_rise_min * 2:
                score += 0.4
            elif price_rise >= price_rise_min:
                score += 0.3
            elif price_rise > 0:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("放量确认检测失败: %s", e)
            return 0.0
    
    def _detect_volume_price_divergence(self, recent_data: pd.DataFrame) -> float:
        """
        检测量价背离信号
        特征：价格下跌但成交量萎缩，表明抛压减弱
        """
        try:
            params = self.config.get('volume_price_divergence', {})
            lookback_period = params.get('lookback_period', 10)

            if len(recent_data) < lookback_period:
                return 0.0
            
            volumes = recent_data['volume'].values
            closes = recent_data['close'].values
            
            score = 0.0
            
            # 寻找价格和成交量的趋势
            # 价格趋势：使用线性回归斜率
            price_indices = np.arange(len(closes))
            price_slope = np.polyfit(price_indices, closes, 1)[0]
            
            # 成交量趋势：使用线性回归斜率
            volume_slope = np.polyfit(price_indices, volumes, 1)[0]
            
            # 检查背离条件
            if price_slope < 0 and volume_slope < 0:  # 价格下跌，成交量萎缩
                # 计算背离强度
                price_decline_rate = abs(price_slope) / np.mean(closes)
                volume_shrink_rate = abs(volume_slope) / np.mean(volumes)
                
                # 量价背离度：成交量萎缩幅度 / 价格下跌幅度
                if price_decline_rate > 0:
                    divergence_ratio = volume_shrink_rate / price_decline_rate
                    
                    if divergence_ratio >= 2.0:  # 强背离
                        score = 1.0
                    elif divergence_ratio >= 1.5:  # 中等背离
                        score = 0.7
                    elif divergence_ratio >= 1.0:  # 轻微背离
                        score = 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("量价背离检测失败: %s", e)
            return 0.0
    
    def _calc_price_stability(self, prices: np.ndarray) -> float:
        """
        计算价格稳定性
        
        Args:
            prices: 价格数组
            
        Returns:
            稳定性得分 0.0-1.0
        """
        try:
            if len(prices) < 3:
                return 0.0
            
            # 1. 计算价格波动率
            price_changes = np.diff(prices) / prices[:-1]
            volatility = np.std(price_changes)
            
            # 2. 计算价格趋势（是否企稳向上）
            price_indices = np.arange(len(prices))
            slope = np.polyfit(price_indices, prices, 1)[0]
            
            # 3. 综合评分
            stability_score = 0.0
            
            # 波动率越小，稳定性越高
            if volatility < 0.02:  # 日波动率小于2%
                stability_score += 0.6
            elif volatility < 0.03:  # 日波动率小于3%
                stability_score += 0.4
            elif volatility < 0.05:  # 日波动率小于5%
                stability_score += 0.2
            
            # 价格趋势向上加分
            if slope > 0:
                stability_score += 0.4
            elif slope >= 0:  # 横盘
                stability_score += 0.2
            
            return min(stability_score, 1.0)
            
        except Exception as e:
            logger.debug("价格稳定性计算失败: %s", e)
            return 0.0