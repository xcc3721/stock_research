#!/usr/bin/env python3
"""
供给分析和枢轴分析模块
专门分析供给压力和枢轴突破信号
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
import logging
from scipy import signal
from .chip_distribution import ChipDistributionAnalysis

logger = logging.getLogger("supply_pivot")


class SupplyPivotAnalysis:
    """
    供给分析和枢轴分析器
    包含：
    - 供给枯竭分析 (缩量测试)
    - 枢轴/支撑回踩确认 (包含跌停/大阴线熔断)
    - 筹码分布集成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chip_analyzer = ChipDistributionAnalysis(config)
    
    def _get_config(self, section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        factor_cfg = self.config.get("factors", {}).get(section, {})
        if isinstance(factor_cfg, dict):
            params = factor_cfg.get("params", {})
            if isinstance(params, dict) and params:
                return params
        params = self.config.get(section, default or {})
        if isinstance(params, dict):
            nested = params.get("params", {})
            if isinstance(nested, dict) and nested:
                return nested
            return params
        return default or {}
    
    def calc_supply_analysis(self, df: pd.DataFrame, stock_code: str = '未知') -> float:
        """
        供给分析综合因子
        
        逻辑重构：
        0. 熔断机制：跌停/大阴线直接置零。
        1. 筹码评分 (40%): 紧贴成本线，获利盘适中。
        2. 缩量评分 (30%): 近期成交量显著萎缩 (VCP特征)。
        3. 支撑回踩评分 (30%): 股价位于支撑位附近。
        """
        try:
            # 0. 熔断机制：检测跌停或放量大阴线
            if self._detect_plunge(df):
                logger.debug(f"股票 {stock_code} 触发跌停/大阴线熔断")
                return 0.0

            # 1. 获取各子因子 (允许返回 None)
            chip_score = self.chip_analyzer.calc_supply_pressure_score(df, stock_code)
            volume_score = self._calc_volume_contraction_score(df)
            pivot_score = self._calc_support_proximity_score(df, stock_code)
            
            # 2. 动态权重聚合
            factors = []
            if chip_score is not None: factors.append((chip_score, 0.4))
            if volume_score is not None: factors.append((volume_score, 0.3))
            if pivot_score is not None: factors.append((pivot_score, 0.3))
            
            if not factors:
                logger.debug(f"股票 {stock_code} 所有供给子因子均无效，返回默认低分")
                return 0.3 # 数据全无，视为高风险，给低分
            
            # 归一化权重
            total_weight = sum(w for s, w in factors)
            final_score = sum(s * w for s, w in factors) / total_weight
            
            # 3. K线形态加分 (长下影，最高加 0.1)
            # 仅在基础分及格 (>0.4) 时生效，避免给垃圾股加分
            if final_score > 0.4:
                final_score = min(final_score + self._calc_candlestick_bonus(df), 1.0)

            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning("股票 %s: 供给分析因子计算失败: %s", stock_code, e)
            return 0.0
    
    def calc_pivot_analysis(self, df: pd.DataFrame, stock_code: str = '未知') -> float:
        """
        枢轴突破分析 (暂时复用支撑逻辑作为基础，不做熔断)
        """
        res = self._calc_support_proximity_score(df, stock_code)
        return res if res is not None else 0.5
    
    def _detect_plunge(self, df: pd.DataFrame) -> bool:
        """检测是否发生暴跌或跌停 (熔断机制)"""
        try:
            if len(df) < 2: return False
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 1. 跌幅 > 7%
            change_pct = (curr['close'] - prev['close']) / prev['close']
            if change_pct < -0.07:
                return True
                
            # 2. 光脚大阴线 (收盘价接近最低价，且实体大)
            if change_pct < -0.05:
                # 下影线长度占比 (防止除零)
                rng = curr['high'] - curr['low']
                if rng <= 0: return True # 一字跌停也算
                
                lower_shadow = (min(curr['open'], curr['close']) - curr['low']) / rng
                if lower_shadow < 0.1: # 几乎无下影
                    return True
                    
            return False
        except Exception:
            return False

    def _calc_candlestick_bonus(self, df: pd.DataFrame) -> float:
        """K线形态加分：长下影线 (金针探底)"""
        try:
            curr = df.iloc[-1]
            rng = curr['high'] - curr['low']
            if not np.isfinite(rng) or rng <= 0:
                return 0.0
            
            body_bottom = min(curr['open'], curr['close'])
            lower_shadow = body_bottom - curr['low']
            
            # 下影线长度占比 > 60%
            if lower_shadow / rng > 0.6:
                return 0.1
            return 0.0
        except Exception:
            return 0.0

    def _calc_volume_contraction_score(self, df: pd.DataFrame) -> Optional[float]:
        """
        计算成交量收缩评分 (异常返回 None)
        调整：对温和放量 (ratio 1.0-1.3) 更宽容，视为启动信号。
        """
        try:
            if len(df) < 20: return None
            
            vol = df['volume']
            vol_3d = vol.tail(3).mean()
            vol_20d = vol.tail(20).mean()
            
            if vol_20d <= 0: return None
            
            ratio = vol_3d / vol_20d
            
            # 评分逻辑 (归一化到 0~1)
            # 极度缩量 (<0.5): 1.0分
            # 正常缩量 (0.5-1.0): 1.0 -> 0.7
            # 温和放量 (1.0-1.3): 0.7 -> 0.5 (启动迹象，不严惩)
            # 剧烈放量 (>1.3): 快速衰减
            
            if ratio <= 0.5:
                score = 1.0
            elif ratio <= 1.0:
                # 线性过渡 1.0 -> 0.7
                score = 1.0 - ((ratio - 0.5) / 0.5) * 0.3
            elif ratio <= 1.3:
                # 温和放量：线性过渡 0.7 -> 0.5
                score = 0.7 - ((ratio - 1.0) / 0.3) * 0.2
            else:
                # 剧烈放量惩罚
                score = max(0.5 - (ratio - 1.3) * 1.0, 0.0)
                
            return min(max(score, 0.0), 1.0)
            
        except Exception:
            return None

    def _calc_support_proximity_score(self, df: pd.DataFrame, stock_code: str) -> Optional[float]:
        """
        计算支撑位接近度评分 (异常返回 None, 无支撑返回低分)
        """
        try:
            params = self._get_config('pivot_analysis')
            lookback_period = int(params.get('lookback_period', 120) or 120)
            pivot_periods = params.get('pivot_periods', [3])
            if isinstance(pivot_periods, list) and pivot_periods:
                pivot_strength = int(pivot_periods[0])
            else:
                pivot_strength = int(params.get('pivot_strength', 3) or 3)
            price_tolerance = float(params.get('price_tolerance', 0.02) or 0.02)
            proximity_threshold = float(params.get('proximity_threshold', 0.02) or 0.02)
            best_distance = float(params.get('best_distance', 0.04) or 0.04)
            fade_distance = float(params.get('fade_distance', 0.15) or 0.15)

            if len(df) < lookback_period: return None
            
            current_price = df['close'].iloc[-1]
            hist_data = df.iloc[:-3]
            pivots_raw = self._find_pivot_points(hist_data.tail(lookback_period), strength=pivot_strength)
            
            if not pivots_raw: return 0.2
            
            supports_raw = [p for t, p, s in pivots_raw if t == 'support']
            if not supports_raw: return 0.2
            
            # 聚类逻辑
            supports_raw.sort()
            support_zones = []
            if supports_raw:
                curr_zone = [supports_raw[0]]
                for i in range(1, len(supports_raw)):
                    if (supports_raw[i] - curr_zone[-1]) / curr_zone[-1] < price_tolerance:
                        curr_zone.append(supports_raw[i])
                    else:
                        support_zones.append(np.mean(curr_zone))
                        curr_zone = [supports_raw[i]]
                support_zones.append(np.mean(curr_zone))
            
            valid_supports = [s for s in support_zones if s <= current_price * (1 + proximity_threshold)]
            
            if not valid_supports:
                return 0.2
            
            nearest_support = max(valid_supports)
            
            if current_price < nearest_support * (1 - proximity_threshold):
                return 0.2
            
            distance = (current_price - nearest_support) / nearest_support
            
            # 平滑评分逻辑 - 修正：奖励“脱离底部”
            if distance < 0:
                # 轻微跌破区间 [-0.02, 0]: 0.6 -> 0.8 (逐步回归)
                return 0.6 + ((distance + proximity_threshold) / proximity_threshold) * 0.2
            elif distance <= proximity_threshold:
                # 紧贴支撑 [0, 0.02]: 0.8 -> 0.9 (观察期/左侧潜伏)
                # 尚未完全脱离危险区，给高分但不给满分
                return 0.8 + (distance / proximity_threshold) * 0.1
            elif distance <= proximity_threshold + 0.04:
                # 确认反弹 [0.02, 0.06]: 0.9 -> 1.0 -> 0.9 (最佳右侧买点)
                # 黄金区间，确认支撑有效且未涨多
                if distance <= best_distance:
                    return 0.9 + ((distance - proximity_threshold) / max(best_distance - proximity_threshold, 1e-6)) * 0.1
                else:
                    return 1.0 - ((distance - best_distance) / max(proximity_threshold + 0.04 - best_distance, 1e-6)) * 0.1
            else:
                # 远离支撑 > 0.06: 0.9 -> 0.4 (线性衰减)
                # 0.06 -> 0.9; 0.15 -> 0.4
                fade_start = proximity_threshold + 0.04
                score = 0.9 - ((distance - fade_start) / max(fade_distance - fade_start, 1e-6)) * 0.5
                return max(score, 0.4)
            
        except Exception:
            return None

    def _find_pivot_points(self, data: pd.DataFrame, strength: int) -> List[Tuple[str, float, int]]:
        """寻找枢轴点 (确保无未来函数)"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            pivots = []
            
            # 阻力位
            peaks, _ = signal.find_peaks(highs, distance=strength)
            for p in peaks:
                pivots.append(('resistance', highs[p], strength))
                
            # 支撑位
            troughs, _ = signal.find_peaks(-lows, distance=strength)
            for t in troughs:
                pivots.append(('support', lows[t], strength))
                
            return pivots
        except Exception:
            return []
            
    # 保留旧接口兼容性
    def _get_pivot_enhancement(self, df: pd.DataFrame) -> float:
        res = self._calc_support_proximity_score(df, "unknown")
        return res if res is not None else 0.5
