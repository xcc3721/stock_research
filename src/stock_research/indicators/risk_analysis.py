#!/usr/bin/env python3
"""
风险分析模块
专门识别各种风险形态和信号
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger("risk_analysis")


class RiskAnalysis:
    """
    风险分析器
    
    提供：
    - 顶部大风车形态检测
    - 其他风险信号识别
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化风险分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
    
    def calc_risk_penalty(self, df: pd.DataFrame, code: str) -> float:
        """
        风险惩罚因子 (第三阶段实现)
        评估各种风险并给出惩罚分数
        
        当前实现的风险模式：
        1. 顶部大风车形态 - 高风险顶部反转信号
        
        Returns:
            惩罚分数 0.0-1.0，数值越高风险越大，用作负分进行惩罚
        """
        try:
            risk_params = self.config.get('factors', {}).get('risk_penalty', {}).get('params', {})
            if not risk_params:
                logger.warning("⚠️  风险分析模块配置缺失，使用默认参数。建议检查 configs.json 中的 scoring_config.factors.risk_penalty.params")
                
            if len(df) < 60:
                return 0.0
            
            penalty_score = 0.0
            
            # 1. 顶部大风车形态检测
            windmill_score = self._detect_top_windmill_pattern(df, risk_params)
            penalty_score += windmill_score
            
            return min(penalty_score, 1.0)
            
        except Exception as e:
            logger.warning("风险惩罚因子计算失败: %s", e)
            return 0.0
    
    def _detect_top_windmill_pattern(self, df: pd.DataFrame, risk_params: Dict[str, Any]) -> float:
        """
        检测顶部大风车形态 - 高风险的顶部反转信号
        
        正确的大风车特征（必须同时满足）：
        1. 前置Spike：前期必须有显著放量的高点（区间最大量/平均量3倍以上）
        2. K线形态：长上下影线 + 短实体
        3. 量价配合：最好是真阴线，阳线勉强成立（需判断假阴真阳）
        4. 后续连续下跌验证
        
        Returns:
            风险评分 0.0-1.0，1.0表示典型的大风车风险形态
        """
        try:
            # 获取参数
            lookback_days = risk_params.get('lookback_days', 30)  # 缩短观察期
            spike_volume_multiplier = risk_params.get('spike_volume_multiplier', 3.0)  # Spike放量倍数
            min_upper_shadow_ratio = risk_params.get('min_upper_shadow_ratio', 0.4)  # 最小上影线比例
            min_lower_shadow_ratio = risk_params.get('min_lower_shadow_ratio', 0.3)  # 最小下影线比例
            max_body_ratio = risk_params.get('max_body_ratio', 0.3)  # 最大实体比例

            if len(df) < lookback_days:
                return 0.0
            
            window_data = df.tail(lookback_days).copy()
            
            # 第一步：寻找前置Spike（显著放量的高点）
            spike_candidates = self._find_volume_spikes(window_data, spike_volume_multiplier)
            if not spike_candidates:
                return 0.0  # 没有前置Spike，不可能是大风车
            max_high = window_data['high'].max()
            
            # 第二步：在Spike之后寻找大风车K线形态
            risk_score = 0.0
            
            for spike_idx in spike_candidates:
                spike_high = window_data.iloc[spike_idx]['high']
                if not np.isfinite(spike_high) or spike_high < max_high:
                    continue  # Spike当日必须是观察窗口内的最高价
                
                # Spike 当日必须同时满足风车K线形态
                spike_row = window_data.iloc[spike_idx]
                windmill_score = self._evaluate_strict_windmill_kline(
                    spike_row, min_upper_shadow_ratio, min_lower_shadow_ratio, max_body_ratio
                )
                
                if windmill_score > 0.7:  # 严格的风车形态
                    # 使用前一日收盘确认真阴线/假阴真阳
                    if spike_idx > 0:
                        prev_close = window_data.iloc[spike_idx - 1]['close']
                    else:
                        prev_close = spike_row['close']
                    is_true_bearish = spike_row['close'] < prev_close
                    
                    if is_true_bearish:
                        risk_score = max(risk_score, windmill_score)
                    elif spike_row['close'] > spike_row['open']:  # 假阴真阳情况
                        risk_score = max(risk_score, windmill_score * 0.7)
                    else:  # 阳线勉强成立
                        risk_score = max(risk_score, windmill_score * 0.5)
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.debug("顶部大风车检测失败: %s", e)
            return 0.0
    
    def _find_volume_spikes(self, df: pd.DataFrame, volume_multiplier: float) -> List[int]:
        """
        寻找显著放量的Spike点
        
        Args:
            df: K线数据
            volume_multiplier: 放量倍数阈值（相对于平均量）
            
        Returns:
            Spike点的索引列表
        """
        try:
            if len(df) < 10:
                return []
            
            # 计算平均成交量（排除最大量避免异常值影响）
            volumes = df['volume'].values
            sorted_volumes = np.sort(volumes)
            avg_volume = np.mean(sorted_volumes[:-2])  # 排除最大的2个值
            
            spike_indices = []
            max_volume = volumes.max()
            
            for i, volume in enumerate(volumes):
                # 条件1: 成交量显著放大（平均量的3倍以上）
                # 条件2: 是区间内的最大量或接近最大量（90%以上）
                if volume >= avg_volume * volume_multiplier and volume >= max_volume * 0.9:
                    spike_indices.append(i)
            
            return spike_indices
            
        except Exception as e:
            logger.debug("Spike检测失败: %s", e)
            return []
    
    def _evaluate_strict_windmill_kline(self, row: pd.Series, min_upper_shadow: float, 
                                       min_lower_shadow: float, max_body: float) -> float:
        """
        严格评估大风车K线形态
        
        Args:
            row: 单日K线数据
            min_upper_shadow: 最小上影线比例
            min_lower_shadow: 最小下影线比例  
            max_body: 最大实体比例
            
        Returns:
            风车得分 0.0-1.0，必须同时满足上下影线和短实体条件
        """
        try:
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            if pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
                return 0.0
            
            full_range = high_price - low_price
            if full_range <= 0:
                return 0.0
            
            # 计算实体和影线
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            # 计算比例
            body_ratio = body_size / full_range
            upper_shadow_ratio = upper_shadow / full_range
            lower_shadow_ratio = lower_shadow / full_range
            
            # 严格的大风车条件：长上下影线 + 短实体
            has_long_upper_shadow = upper_shadow_ratio >= min_upper_shadow
            has_long_lower_shadow = lower_shadow_ratio >= min_lower_shadow
            has_short_body = body_ratio <= max_body
            
            # 必须同时满足所有条件
            if has_long_upper_shadow and has_long_lower_shadow and has_short_body:
                # 根据影线长度和实体短度计算得分
                shadow_score = (upper_shadow_ratio + lower_shadow_ratio) / 2
                body_score = (max_body - body_ratio) / max_body  # 实体越短得分越高
                
                return min((shadow_score + body_score) / 2, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug("严格大风车K线评估失败: %s", e)
            return 0.0
    
    def _evaluate_windmill_kline(self, row: pd.Series, shadow_threshold: float) -> float:
        """
        评估单根K线的大风车特征
        
        Args:
            row: 单日K线数据
            shadow_threshold: 影线比例阈值
            
        Returns:
            大风车得分 0.0-1.0
        """
        try:
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            if pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
                return 0.0
            
            full_range = high_price - low_price
            if full_range <= 0:
                return 0.0
            
            # 计算实体和影线
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            # 计算比例
            body_ratio = body_size / full_range
            upper_shadow_ratio = upper_shadow / full_range
            lower_shadow_ratio = lower_shadow / full_range
            
            windmill_score = 0.0
            
            # 1. 长上影线大风车
            if upper_shadow_ratio >= shadow_threshold:
                if upper_shadow_ratio >= 0.6:  # 极长上影线
                    windmill_score = max(windmill_score, 1.0)
                elif upper_shadow_ratio >= 0.5:  # 长上影线
                    windmill_score = max(windmill_score, 0.8)
                else:  # 中等上影线
                    windmill_score = max(windmill_score, 0.5)
            
            # 2. 长下影线大风车（顶部反转中也可能出现）
            if lower_shadow_ratio >= shadow_threshold:
                if lower_shadow_ratio >= 0.6:  # 极长下影线
                    windmill_score = max(windmill_score, 0.8)  # 下影线风险稍低
                elif lower_shadow_ratio >= 0.5:  # 长下影线
                    windmill_score = max(windmill_score, 0.6)
                else:  # 中等下影线
                    windmill_score = max(windmill_score, 0.3)
            
            # 3. 十字星大风车（上下都有长影线）
            if (upper_shadow_ratio >= shadow_threshold * 0.8 and 
                lower_shadow_ratio >= shadow_threshold * 0.8 and
                body_ratio <= 0.2):  # 小实体
                windmill_score = max(windmill_score, 0.9)
            
            # 4. 实体大小调整
            if body_ratio <= 0.1:  # 极小实体，风车特征更明显
                windmill_score *= 1.2
            elif body_ratio >= 0.4:  # 大实体，风车特征减弱
                windmill_score *= 0.8
            
            return min(windmill_score, 1.0)
            
        except Exception as e:
            logger.debug("大风车K线评估失败: %s", e)
            return 0.0
