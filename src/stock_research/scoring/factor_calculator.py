#!/usr/bin/env python3
"""
多因子量化评分系统 - 重构后的因子计算器
提供统一的因子计算接口，使用模块化架构
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging
from copy import deepcopy

# 导入各个专业模块
from stock_research.indicators import (
    TechnicalIndicators,
    KLinePatterns,
    VolumeAnalysis,
    VCPPocketAnalysis,
    SupplyPivotAnalysis,
    B1Analysis,
    RiskAnalysis,
    StructureAnalysis
)

logger = logging.getLogger("factor_calculator")


class FactorCalculator:
    """
    重构后的多因子计算器 - 使用组合模式
    
    支持的因子：
    - ma_structure: 均线结构因子
    - b1_comprehensive: B1综合因子
    - kline_patterns: K线形态因子
    - pocket_pivot: 口袋支点因子
    - supply_analysis: 供给分析因子
    - vcp_compression: VCP压缩因子
    - pivot_analysis: 枢轴分析因子
    - risk_penalty: 风险惩罚因子
    - structure_analysis: 结构化(异动)因子
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化因子计算器
        
        Args:
            config: 因子配置字典，包含各因子的参数设置
            
        Note:
            配置为空时仍可工作，但建议传入完整的scoring_config以确保参数一致性
        """
        self.config = self._normalize_config(config or {})
        
        # 记录配置状态
        if not config:
            logger.warning("⚠️  因子计算器未收到配置，将使用各模块的内置默认参数")
        else:
            logger.debug("✅ 因子计算器收到配置，参数将从配置文件读取")
        
        # 初始化各个专业模块
        self.technical_indicators = TechnicalIndicators(self.config)
        self.kline_patterns = KLinePatterns(self.config)
        self.volume_analysis = VolumeAnalysis(self.config)
        self.vcp_pocket_analysis = VCPPocketAnalysis(self.config)
        self.supply_pivot_analysis = SupplyPivotAnalysis(self.config)
        self.b1_analysis = B1Analysis(self.config)
        self.risk_analysis = RiskAnalysis(self.config)
        self.structure_analysis = StructureAnalysis(self.config)

        logger.debug("🔧 重构版因子计算器初始化完成")
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将评分配置中的 factor.params 扁平化到顶层，方便各指标模块按老接口读取。
        """
        if not isinstance(config, dict):
            return {}

        normalized = deepcopy(config)
        factors_node = normalized.get('factors', {})
        if not isinstance(factors_node, dict):
            return normalized

        for factor_name, factor_cfg in factors_node.items():
            if not isinstance(factor_cfg, dict):
                continue
            params = factor_cfg.get('params')
            if isinstance(params, dict):
                normalized.setdefault(factor_name, deepcopy(params))
                for sub_key, sub_val in params.items():
                    if isinstance(sub_val, dict) and sub_key not in normalized:
                        normalized[sub_key] = deepcopy(sub_val)
        return normalized
    
    def _factor_enabled(self, factor_name: str) -> bool:
        """
        判断指定因子是否在配置中启用，默认启用。
        """
        factors_cfg = self.config.get('factors')
        if isinstance(factors_cfg, dict):
            cfg = factors_cfg.get(factor_name)
            if isinstance(cfg, dict) and cfg.get('enabled') is False:
                return False
        return True
    
    def calculate_all_factors(self, df: pd.DataFrame, code: str) -> Dict[str, float]:
        """
        计算所有因子的得分
        
        Args:
            df: 股票历史数据DataFrame，包含OHLCV数据
            code: 股票代码
            
        Returns:
            包含所有因子得分的字典，每个因子得分范围为0.0-1.0
        """
        try:
            # 数据验证
            if df is None or len(df) < 20:
                logger.warning("数据不足，无法计算因子: %s", code)
                return self._get_empty_factors()
            
            factors = {}
            
            # 1. 均线结构因子
            if self._factor_enabled('ma_structure'):
                factors['ma_structure'] = self.technical_indicators.calc_ma_structure(df)
            else:
                factors['ma_structure'] = 0.0
            # 1.1 对称走势因子（衡量震荡后的强弱）
            if self._factor_enabled('symmetry_resonance'):
                factors['symmetry_resonance'] = self.technical_indicators.calc_symmetry_resonance(df)
            else:
                factors['symmetry_resonance'] = 0.5
            
            # 2. B1综合因子 (整合了多个B1相关逻辑)
            if self._factor_enabled('b1_comprehensive'):
                factors['b1_comprehensive'] = self.b1_analysis.calc_b1_comprehensive(df)
            else:
                factors['b1_comprehensive'] = 0.0
            # 2.1 B1形态相似度（独立因子，强调高相似）
            if self._factor_enabled('b1_template_similarity'):
                try:
                    factors['b1_template_similarity'] = self.b1_analysis.calc_b1_template_similarity(df)
                    tm = getattr(self.b1_analysis, 'last_template_match', None)
                    if tm and isinstance(tm, dict) and tm.get('desc'):
                        factors['b1_template_best'] = tm['desc']  # 展示用
                except Exception as exc:
                    logger.warning("B1模板相似度计算失败: %s", exc)
                    factors['b1_template_similarity'] = 0.0
            else:
                factors['b1_template_similarity'] = 0.0
            
            # 3. K线形态因子
            if self._factor_enabled('kline_patterns'):
                factors['kline_patterns'] = self.kline_patterns.calc_kline_patterns(df)
            else:
                factors['kline_patterns'] = 0.0
            
            # 4. 口袋支点因子
            if self._factor_enabled('pocket_pivot'):
                factors['pocket_pivot'] = self.vcp_pocket_analysis.calc_pocket_pivot(df)
            else:
                factors['pocket_pivot'] = 0.0
            
            # 5. 供给分析因子
            if self._factor_enabled('supply_analysis'):
                factors['supply_analysis'] = self.supply_pivot_analysis.calc_supply_analysis(df, code)
            else:
                factors['supply_analysis'] = 0.0
            
            # 6. VCP压缩因子
            if self._factor_enabled('vcp_compression'):
                factors['vcp_compression'] = self.vcp_pocket_analysis.calc_vcp_compression(df)
            else:
                factors['vcp_compression'] = 0.0
            
            # 7. 枢轴分析因子
            if self._factor_enabled('pivot_analysis'):
                factors['pivot_analysis'] = self.supply_pivot_analysis.calc_pivot_analysis(df, code)
            else:
                factors['pivot_analysis'] = 0.0
            
            # 8. 风险惩罚因子
            if self._factor_enabled('risk_penalty'):
                factors['risk_penalty'] = self.risk_analysis.calc_risk_penalty(df, code)
            else:
                factors['risk_penalty'] = 0.0
            
            # 9. 结构化(异动)因子
            if self._factor_enabled('structure_analysis'):
                factors['structure_analysis'] = self.structure_analysis.calc_structure_score(df)
            else:
                factors['structure_analysis'] = 0.0
            
            # 以下因子已经整合到其他因子中，保留兼容性
            # factors['momentum_basic'] = self.technical_indicators.calc_momentum_basic(df)  # 已整合到b1_comprehensive
            # factors['volume_anomaly'] = self.volume_analysis.calc_volume_anomaly(df)      # 已整合到b1_comprehensive
            
            logger.debug("因子计算完成: %s", code)
            return factors
            
        except Exception as e:
            logger.error("因子计算失败 %s: %s", code, e)
            return self._get_empty_factors()
    
    def _get_empty_factors(self) -> Dict[str, float]:
        """
        返回空的因子字典（所有因子得分为0）
        """
        return {
            'ma_structure': 0.0,
            'symmetry_resonance': 0.5,
            'b1_comprehensive': 0.0,
            'b1_template_similarity': 0.0,
            'kline_patterns': 0.0,
            'pocket_pivot': 0.0,
            'supply_analysis': 0.0,
            'vcp_compression': 0.0,
            'pivot_analysis': 0.0,
            'risk_penalty': 0.0,
            'structure_analysis': 0.0
        }
    
    # 为了保持向后兼容性，提供一些便捷方法
    def calc_ma_structure(self, df: pd.DataFrame) -> float:
        """均线结构因子计算（兼容性方法）"""
        return self.technical_indicators.calc_ma_structure(df)
    
    def calc_b1_comprehensive(self, df: pd.DataFrame) -> float:
        """B1综合因子计算（兼容性方法）"""
        return self.b1_analysis.calc_b1_comprehensive(df)
    
    def calc_kline_patterns(self, df: pd.DataFrame) -> float:
        """K线形态因子计算（兼容性方法）"""
        return self.kline_patterns.calc_kline_patterns(df)
    
    def calc_pocket_pivot(self, df: pd.DataFrame) -> float:
        """口袋支点因子计算（兼容性方法）"""
        return self.vcp_pocket_analysis.calc_pocket_pivot(df)
    
    def calc_supply_analysis(self, df: pd.DataFrame) -> float:
        """供给分析因子计算（兼容性方法）"""
        return self.supply_pivot_analysis.calc_supply_analysis(df)
    
    def calc_vcp_compression(self, df: pd.DataFrame) -> float:
        """VCP压缩因子计算（兼容性方法）"""
        return self.vcp_pocket_analysis.calc_vcp_compression(df)
    
    def calc_pivot_analysis(self, df: pd.DataFrame) -> float:
        """枢轴分析因子计算（兼容性方法）"""
        return self.supply_pivot_analysis.calc_pivot_analysis(df)
    
    def calc_risk_penalty(self, df: pd.DataFrame, code: str) -> float:
        """风险惩罚因子计算（兼容性方法）"""
        return self.risk_analysis.calc_risk_penalty(df, code)

    def calc_structure_score(self, df: pd.DataFrame) -> float:
        """结构化(异动)因子计算（兼容性方法）"""
        return self.structure_analysis.calc_structure_score(df)
    
    # 其他可能需要的兼容性方法
    def calc_momentum_basic(self, df: pd.DataFrame) -> float:
        """基础动量因子计算（兼容性方法，已整合到B1综合因子中）"""
        return self.technical_indicators.calc_momentum_basic(df)
    
    def calc_volume_anomaly(self, df: pd.DataFrame) -> float:
        """成交量异动因子计算（兼容性方法，已整合到B1综合因子中）"""
        return self.volume_analysis.calc_volume_anomaly(df)
