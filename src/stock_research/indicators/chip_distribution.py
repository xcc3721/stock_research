#!/usr/bin/env python3
"""
筹码分布分析模块
基于换手率衰减模型追踪市场筹码分布情况
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
import logging
from scipy import signal

logger = logging.getLogger("chip_distribution")


class ChipDistributionAnalysis:
    """
    筹码分布分析器
    
    核心功能：
    - 基于换手率的筹码衰减建模
    - 筹码集中度分析
    - 套牢盘和获利盘比例计算
    - 成本分布评估
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化筹码分布分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
    
    def _get_config(self, section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """获取配置参数的辅助方法"""
        return self.config.get(section, default or {})
    
    def calculate_chip_distribution(self, df: pd.DataFrame, stock_code: str = '未知') -> Dict[str, Any]:
        """
        计算筹码分布
        
        Args:
            df: 股票数据DataFrame，必须包含turnover字段
            stock_code: 股票代码，用于错误日志
            
        Returns:
            筹码分布分析结果
        """
        try:
            params = self._get_config('chip_distribution')
            
            # 参数配置
            price_bins = params.get('price_bins', 100)
            preferred_days = params.get('lookback_days', 120)  # 优选天数
            min_required_days = params.get('min_required_days', 30)  # 最少需要天数
            decay_factor = params.get('decay_factor', 1.0)
            min_turnover = params.get('min_turnover', 0.01)  # 最小换手率0.01%
            
            actual_days = len(df)
            
            # 动态降级策略
            if actual_days < min_required_days:
                # logger.warning("股票 %s: 数据过少，无法计算筹码分布（最少需要%d天，实际%d天）", stock_code, min_required_days, actual_days)
                return self._empty_result()
            elif actual_days < preferred_days:
                lookback_days = actual_days
            else:
                lookback_days = preferred_days
            
            if 'turnover' not in df.columns:
                return self._empty_result()
            
            # 获取分析数据
            recent_data = df.tail(lookback_days).copy()
            current_price = recent_data['close'].iloc[-1]
            
            # 计算筹码分布
            chip_dist = self._build_chip_distribution(recent_data, price_bins, decay_factor, min_turnover)
            
            # 计算关键指标
            result = {
                'chip_distribution': chip_dist,
                'concentration': self._calculate_concentration(chip_dist),
                'trapped_ratio': self._calculate_trapped_ratio(chip_dist, current_price),
                'profit_ratio': self._calculate_profit_ratio(chip_dist, current_price),
                'overhang_0_2': self._calculate_overhang(chip_dist, current_price),
                'avg_cost': self._calculate_average_cost(chip_dist),
                'cost_support_levels': self._find_cost_support_levels(chip_dist),
                'cost_resistance_levels': self._find_cost_resistance_levels(chip_dist),
            }
            
            return result
            
        except Exception as e:
            logger.warning("股票 %s: 筹码分布计算失败: %s", stock_code, e)
            return self._empty_result()
    
    def _build_chip_distribution(self, df: pd.DataFrame, price_bins: int, 
                                 decay_factor: float, min_turnover: float) -> Dict[float, float]:
        """构建筹码分布"""
        try:
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_range = price_max - price_min
            
            if price_range <= 0:
                return {}
            
            chip_dist = {}  # {价格: 筹码量}
            
            for i, row in df.iterrows():
                raw_turnover = row.get('turnover', 0)
                if pd.isna(raw_turnover) or not np.isfinite(raw_turnover):
                    raw_turnover = min_turnover
                
                turnover_rate = max(raw_turnover / 100.0, min_turnover / 100.0)
                
                if pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']):
                    continue
                    
                daily_volume = row['volume']
                if pd.isna(daily_volume) or daily_volume <= 0:
                    continue
                
                adjusted_turnover = turnover_rate * decay_factor
                if pd.isna(adjusted_turnover) or not np.isfinite(adjusted_turnover):
                    adjusted_turnover = min_turnover / 100.0 * decay_factor

                # 衰减现有筹码
                keys_to_remove = []
                for price in chip_dist:
                    chip_dist[price] *= (1 - adjusted_turnover)
                    if chip_dist[price] < daily_volume * 0.0001:
                        keys_to_remove.append(price)
                for k in keys_to_remove:
                    del chip_dist[k]
                
                # 新增筹码
                if row['high'] > row['low']:
                    self._distribute_daily_chips(chip_dist, daily_volume * adjusted_turnover, row['low'], row['high'], row['close'])
                else:
                    chip_dist[row['close']] = chip_dist.get(row['close'], 0) + daily_volume * adjusted_turnover
            
            return chip_dist
            
        except Exception as e:
            return {}
    
    def _distribute_daily_chips(self, chip_dist: Dict[float, float], total_chips: float, low: float, high: float, close: float):
        """分布当日筹码"""
        try:
            if high <= low:
                chip_dist[close] = chip_dist.get(close, 0) + total_chips
                return
            
            # 收盘价权重更高
            close_weight = 0.6
            uniform_weight = 0.4
            
            chip_dist[close] = chip_dist.get(close, 0) + total_chips * close_weight
            
            price_levels = np.linspace(low, high, 10)
            chips_per_level = total_chips * uniform_weight / len(price_levels)
            
            for price in price_levels:
                rounded_price = round(price, 2)
                chip_dist[rounded_price] = chip_dist.get(rounded_price, 0) + chips_per_level
                
        except Exception:
            pass
    
    def _calculate_concentration(self, chip_dist: Dict[float, float]) -> float:
        """计算筹码集中度"""
        try:
            if not chip_dist: return 0.5
            total = sum(chip_dist.values())
            if total <= 0: return 0.5
            
            # 使用加权标准差衡量集中度
            prices = np.array(list(chip_dist.keys()))
            weights = np.array(list(chip_dist.values())) / total
            avg_price = np.sum(prices * weights)
            variance = np.sum(weights * (prices - avg_price)**2)
            std_dev = np.sqrt(variance)
            
            # 归一化：std_dev / avg_price (变异系数 CV)
            cv = std_dev / avg_price if avg_price > 0 else 1.0
            
            # CV越小越集中。通常CV < 0.1 算高度集中。映射到 0-1 分数。
            # 0.05 -> 1.0, 0.2 -> 0.0
            concentration_score = 1.0 - min(max((cv - 0.05) / 0.15, 0.0), 1.0)
            return concentration_score
            
        except Exception:
            return 0.5
    
    def _calculate_trapped_ratio(self, chip_dist: Dict[float, float], current_price: float) -> float:
        try:
            if not chip_dist: return 0.0
            total = sum(chip_dist.values())
            if total <= 0: return 0.0
            trapped = sum(v for k, v in chip_dist.items() if k > current_price)
            return trapped / total
        except Exception:
            return 0.0
    
    def _calculate_profit_ratio(self, chip_dist: Dict[float, float], current_price: float) -> float:
        try:
            if not chip_dist: return 0.0
            total = sum(chip_dist.values())
            if total <= 0: return 0.0
            profit = sum(v for k, v in chip_dist.items() if k < current_price)
            return profit / total
        except Exception:
            return 0.0

    def _calculate_overhang(self, chip_dist: Dict[float, float], current_price: float) -> float:
        """计算近端抛压 (Overhang 0-2%)"""
        try:
            if not chip_dist: return 0.0
            total = sum(chip_dist.values())
            if total <= 0: return 0.0
            overhang = sum(v for k, v in chip_dist.items() if current_price <= k <= current_price * 1.02)
            return overhang / total
        except Exception:
            return 0.0

    def _calculate_average_cost(self, chip_dist: Dict[float, float]) -> float:
        try:
            if not chip_dist: return 0.0
            total = sum(chip_dist.values())
            if total <= 0: return 0.0
            return sum(k * v for k, v in chip_dist.items()) / total
        except Exception:
            return 0.0

    def _weighted_percentile(self, prices: np.ndarray, weights: np.ndarray, q: float) -> float:
        """加权分位数 (0-1)。"""
        order = np.argsort(prices)
        p_sorted = prices[order]
        w_sorted = weights[order]
        cum_w = np.cumsum(w_sorted)
        if cum_w[-1] == 0:
            return float("nan")
        cutoff = q * cum_w[-1]
        idx = np.searchsorted(cum_w, cutoff)
        return float(p_sorted[min(idx, len(p_sorted) - 1)])

    def _chip_shape_metrics(self, chip_dist: Dict[float, float]) -> Tuple[float, float, float, float]:
        """
        形态指标：加权标准差、带宽(IQR)、偏度、峰值数量。
        返回 (wstd, bandwidth, skew, peak_count)
        """
        try:
            if not chip_dist:
                return (float("nan"), float("nan"), float("nan"), 0.0)
            prices = np.array(list(chip_dist.keys()), dtype=float)
            weights = np.array(list(chip_dist.values()), dtype=float)
            total_w = weights.sum()
            if total_w <= 0:
                return (float("nan"), float("nan"), float("nan"), 0.0)

            wmean = float(np.average(prices, weights=weights))
            var = float(np.average((prices - wmean) ** 2, weights=weights))
            wstd = float(np.sqrt(var))
            p25 = self._weighted_percentile(prices, weights, 0.25)
            p75 = self._weighted_percentile(prices, weights, 0.75)
            bandwidth = p75 - p25 if np.isfinite(p75) and np.isfinite(p25) else float("nan")
            norm = wstd if wstd > 0 else 1e-9
            skew = float(np.average(((prices - wmean) / norm) ** 3, weights=weights))
            order = np.argsort(prices)
            peak_count = len(signal.find_peaks(weights[order], prominence=np.percentile(weights, 50))[0])
            return (wstd, bandwidth, skew, float(peak_count))
        except Exception:
            return (float("nan"), float("nan"), float("nan"), 0.0)

    def _find_cost_support_levels(self, chip_dist: Dict[float, float]) -> List[Tuple[float, float]]:
        # 简化实现
        if not chip_dist: return []
        return sorted(chip_dist.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _find_cost_resistance_levels(self, chip_dist: Dict[float, float]) -> List[Tuple[float, float]]:
        return self._find_cost_support_levels(chip_dist) # 简化
        
    def _empty_result(self) -> Dict[str, Any]:
        return {
            'chip_distribution': {}, 'concentration': 0.5,
            'trapped_ratio': 0.0, 'profit_ratio': 0.0, 'overhang_0_2': 0.0,
            'avg_cost': 0.0, 'cost_support_levels': [], 'cost_resistance_levels': [],
        }

    # ==================== 优化的评分接口 ====================
    
    def calc_supply_pressure_score(self, df: pd.DataFrame, stock_code: str = '未知') -> Optional[float]:
        """
        优化后的供给压力评分
        基于最新数据挖掘结论：
        1. 大涨股大部分是微套状态 (套牢盘 60%~85%)。
        2. 黄金坑位置在水下 -2% ~ -8% (中位数 -4.2%)。
        3. 水上股风险远大于机会。
        4. [NEW] 近端抛压 (Overhang 0-2%) 应较小，通常 < 5%。
        5. [NEW] 筹码带宽/标准差越窄越佳，高获利盘需重罚。
        """
        try:
            chip_result = self.calculate_chip_distribution(df, stock_code)
            if not chip_result['chip_distribution']: 
                return None
            
            avg_cost = chip_result['avg_cost']
            current_price = df['close'].iloc[-1]
            trapped_ratio = chip_result['trapped_ratio']
            concentration = chip_result['concentration']
            profit_ratio = chip_result['profit_ratio']
            overhang = chip_result.get('overhang_0_2', 0.0)
            wstd, bandwidth, _, _ = self._chip_shape_metrics(chip_result['chip_distribution'])
            
            # 1. 成本乖离率评分 (Cost Deviation)
            # 更精确：最佳 -8% ~ -3%，-1% 以上快速扣分
            if avg_cost > 0:
                deviation = (current_price - avg_cost) / avg_cost
                
                if deviation > 0:
                    dev_score = max(0.3 * np.exp(-deviation * 40.0), 0.0)
                elif -0.08 <= deviation <= -0.03:
                    # 黄金坑 (-8% ~ -3%)
                    dev_score = 1.0
                elif -0.03 < deviation <= -0.01:
                    dev_score = 0.85 + (deviation + 0.03) / 0.02 * 0.15
                elif -0.01 < deviation <= 0:
                    dev_score = 0.6 + (deviation + 0.01) / 0.01 * -0.2
                else:
                    dev_score = max(1.0 - (abs(deviation) - 0.08) * 5.0, 0.4)
            else:
                dev_score = 0.5
            
            # 2. 套牢盘比例评分 (Trapped Ratio)
            # 目标：0.65~0.82 最优，<0.55 重罚，>0.90 重罚
            if 0.65 <= trapped_ratio <= 0.82:
                trapped_score = 1.0
            elif 0.60 <= trapped_ratio < 0.65:
                trapped_score = 0.8 + (trapped_ratio - 0.60) / 0.05 * 0.2
            elif 0.55 <= trapped_ratio < 0.60:
                trapped_score = 0.4 + (trapped_ratio - 0.55) / 0.05 * 0.4
            elif trapped_ratio < 0.55:
                trapped_score = 0.2
            else:
                trapped_score = max(1.0 - (trapped_ratio - 0.82) / 0.08 * 0.6, 0.1)
            
            # 3. 获利盘比例评分 (Profit Ratio)
            # 目标：0.20~0.30 满分，>0.35 快速扣分
            if profit_ratio < 0.10:
                profit_score = 0.6 + (profit_ratio / 0.10) * 0.2
            elif profit_ratio < 0.20:
                profit_score = 0.8 + (profit_ratio - 0.10) / 0.10 * 0.2
            elif profit_ratio <= 0.30:
                profit_score = 1.0
            elif profit_ratio <= 0.35:
                profit_score = 1.0 - (profit_ratio - 0.30) / 0.05 * 0.8
            else:
                profit_score = max(0.2 - (profit_ratio - 0.35) * 1.0, 0.0)
            
            # 4. 集中度钟形评分
            if 0.75 <= concentration <= 0.85:
                conc_score = 1.0
            elif 0.65 <= concentration < 0.75:
                conc_score = 0.8 + (concentration - 0.65) / 0.10 * 0.2
            elif 0.85 < concentration <= 0.90:
                conc_score = 1.0 - (concentration - 0.85) / 0.05 * 0.3
            elif concentration < 0.65:
                conc_score = 0.6
            else:
                conc_score = 0.4

            # 5. 带宽/标准差评分：越窄越佳
            if not np.isfinite(bandwidth):
                bandwidth_score = 0.5
            elif bandwidth <= 1.5:
                bandwidth_score = 1.0
            elif bandwidth <= 3.0:
                bandwidth_score = 1.0 - (bandwidth - 1.5) / 1.5 * 0.4
            else:
                bandwidth_score = max(0.6 - (bandwidth - 3.0) / 3.0 * 0.6, 0.0)

            # 6. 近端抛压评分 (Overhang)
            if overhang <= 0.08:
                overhang_score = 1.0
            elif overhang <= 0.12:
                overhang_score = 1.0 - (overhang - 0.08) / 0.04 * 0.4
            else:
                overhang_score = max(0.6 - (overhang - 0.12) / 0.08 * 0.6, 0.0)
            
            # 综合评分 (归一化到 0~1)
            # 权重：乖离率(35%) + 套牢盘(20%) + 获利盘(15%) + 带宽(10%) + 抛压(10%) + 集中度(10%)
            final_score = (
                dev_score * 0.35 +
                trapped_score * 0.20 +
                profit_score * 0.15 +
                bandwidth_score * 0.10 +
                overhang_score * 0.10 +
                conc_score * 0.10
            )
            gating = 1.0
            if profit_ratio > 0.40 or trapped_ratio < 0.55:
                gating *= 0.7
            if overhang > 0.15:
                gating *= 0.7
            
            return min(max(final_score * gating, 0.0), 1.0)

        except Exception as e:
            logger.warning("股票 %s: 供给压力评分计算失败: %s", stock_code, e)
            return None

    def calc_chip_concentration_score(self, df: pd.DataFrame, stock_code: str = '未知') -> float:
        try:
            chip_result = self.calculate_chip_distribution(df, stock_code)
            return chip_result['concentration']
        except Exception:
            return 0.5
