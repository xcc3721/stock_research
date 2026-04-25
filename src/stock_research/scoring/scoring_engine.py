#!/usr/bin/env python3
"""
多因子量化评分系统 - 评分引擎
提供因子标准化、加权、分级评分功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from enum import Enum

logger = logging.getLogger("scoring_engine")


class TradingAction(Enum):
    """交易操作建议"""
    BUY = "买入"
    WATCH = "观望"  
    AVOID = "不买"


class ScoringEngine:
    """
    多因子评分引擎
    
    功能：
    1. 因子标准化和加权
    2. 综合评分计算 
    3. 评分分级和操作建议
    4. 核心理由生成
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化评分引擎
        
        Args:
            config: 评分配置，包含权重、阈值等参数
            
        Raises:
            ValueError: 当配置为空或缺少必要配置时抛出异常
        """
        if not config:
            raise ValueError("评分引擎配置不能为空！请确保正确加载 configs.json 中的 scoring_config 部分")
            
        if 'factors' not in config:
            raise ValueError("评分配置中缺少 'factors' 配置项！请检查 configs.json 中的 scoring_config.factors")
            
        # 从配置中提取权重，严格依赖配置文件
        self.factor_weights = {}
        self.factor_neutral: Dict[str, float] = {}
        self.factor_display_names: Dict[str, str] = {}
        self.factor_adjustments: Dict[str, List[Dict[str, float]]] = {}
        self.bonus_factors: Set[str] = set()
        for factor_name, factor_config in config.get('factors', {}).items():
            if isinstance(factor_config, dict):
                if factor_config.get('enabled', False):
                    weight = factor_config.get('weight', 0.0)
                    self.factor_weights[factor_name] = weight
                else:
                    # 禁用的因子权重设为0
                    self.factor_weights[factor_name] = 0.0
                adjustments = factor_config.get('adjustments')
                if isinstance(adjustments, list):
                    normalized: List[Dict[str, float]] = []
                    for item in adjustments:
                        if not isinstance(item, dict):
                            continue
                        try:
                            normalized.append({
                                'min': float(item.get('min', 0.0)),
                                'max': float(item.get('max', 1.0)),
                                'multiplier': float(item.get('multiplier', 1.0)),
                            })
                        except (TypeError, ValueError):
                            continue
                    if normalized:
                        self.factor_adjustments[factor_name] = normalized
                # 记录中性值（若配置提供），用于对称加减分因子
                neutral_val: Optional[float] = None
                if isinstance(factor_config.get('neutral'), (int, float)):
                    neutral_val = float(factor_config['neutral'])
                else:
                    params = factor_config.get('params', {}) if isinstance(factor_config.get('params', {}), dict) else {}
                    neutral_param = params.get('neutral_score')
                    if isinstance(neutral_param, (int, float)):
                        neutral_val = float(neutral_param)
                if neutral_val is not None and 0.0 < neutral_val < 1.0:
                    self.factor_neutral[factor_name] = neutral_val
                display_name = factor_config.get('name')
                if isinstance(display_name, str) and display_name:
                    self.factor_display_names[factor_name] = display_name
                else:
                    self.factor_display_names[factor_name] = factor_name
                if bool(factor_config.get('bonus', False)):
                    self.bonus_factors.add(factor_name)
            else:
                logger.warning(f"因子 {factor_name} 配置格式错误，跳过")
        self._bonus_weight_total = sum(
            abs(self.factor_weights.get(name, 0.0)) for name in self.bonus_factors
        )
        
        # 验证必要的因子是否存在
        required_factors = ['ma_structure', 'b1_comprehensive', 'kline_patterns', 'supply_analysis', 'risk_penalty']
        missing_factors = [f for f in required_factors if f not in self.factor_weights]
        if missing_factors:
            raise ValueError(f"配置中缺少必要的因子: {missing_factors}，请检查 configs.json")
            
        # 获取评分阈值
        self.score_thresholds = config.get('thresholds', {})
        if not self.score_thresholds:
            raise ValueError("评分配置中缺少 'thresholds' 配置项！请检查 configs.json 中的 scoring_config.thresholds")
        
        # 调试输出：显示实际使用的权重配置
        enabled_factors = [(name, weight) for name, weight in self.factor_weights.items() if weight != 0.0]
        disabled_factors = [(name, weight) for name, weight in self.factor_weights.items() if weight == 0.0]
        logger.info("🔧 评分引擎初始化完成，严格使用配置文件")
        logger.info("✅ 启用的因子权重: %s", enabled_factors)
        logger.info("❌ 禁用的因子权重: %s", disabled_factors)
        
        # 验证权重合理性
        total_positive_weight = sum(w for w in self.factor_weights.values() if w > 0)
        total_negative_weight = sum(w for w in self.factor_weights.values() if w < 0)
        logger.info(f"⚖️  权重统计: 正向权重={total_positive_weight}, 负向权重={total_negative_weight}")
    
    
    def calculate_score(self, factors: Dict[str, float], code: str) -> Dict[str, Any]:
        """
        计算综合评分 - 包含智能因子融合机制
        
        解决K线因子过度惩罚问题：
        1. 实施"覆盖"机制：当其他动量因子得分极高时，可以部分覆盖K线因子的负面影响
        2. 渐进式惩罚：K线因子低分时不完全按权重惩罚，而是使用渐进式折扣
        3. 保底机制：确保有潜力的股票不会因单一因子过低而被完全排除
        
        Args:
            factors: 各因子得分字典 (0-1标准化值)
            code: 股票代码
            
        Returns:
            评分结果字典，包含总分、操作建议、详情等
        """
        try:
            # 计算加权得分
            weighted_scores = {}
            total_weighted_score = 0.0
            total_possible_score = 0.0
            bonus_total_score = 0.0
            
            # 第一步：检查是否需要智能融合
            kline_score = factors.get('kline_patterns', 0.0)
            needs_smart_fusion = self._should_apply_smart_fusion(factors, kline_score)
            
            for factor_name, factor_value in factors.items():
                weight = self.factor_weights.get(factor_name, 0.0)
                is_bonus = factor_name in self.bonus_factors and weight != 0.0

                if weight != 0.0:
                    factor_value = self._apply_adjustment(factor_name, factor_value)
                    # 稀疏因子：b1_template_similarity 在值<=0时不参与分母，避免稀释其他因子
                    if factor_name == 'b1_template_similarity' and (factor_value is None or factor_value <= 0):
                        weighted_scores[factor_name] = 0.0
                        # 不增加 total_possible_score，不改变 total_weighted_score
                        continue
                    neutral = self.factor_neutral.get(factor_name)
                    if neutral is not None:
                        centered_value = self._center_factor_value(factor_value, neutral)
                    else:
                        centered_value = factor_value

                    if is_bonus:
                        weighted_score = centered_value * weight
                        weighted_scores[factor_name] = weighted_score
                        bonus_total_score += weighted_score
                        continue

                    if factor_name == 'risk_penalty':
                        # 风险因子是负权重，直接相乘
                        weighted_score = centered_value * weight
                        total_possible_score += abs(weight)
                    elif factor_name == 'kline_patterns' and needs_smart_fusion:
                        # 对K线因子应用智能融合逻辑
                        adjusted_score, adjusted_weight = self._apply_smart_kline_fusion(
                            factor_value, weight, factors
                        )
                        weighted_score = adjusted_score * adjusted_weight
                        total_possible_score += abs(adjusted_weight)
                        logger.debug("K线智能融合: %s, 原始=%.2f->调整=%.2f, 权重=%.1f->%.1f", 
                                   code, factor_value, adjusted_score, weight, adjusted_weight)
                    else:
                        # 正向因子，标准化值乘以权重
                        weighted_score = centered_value * weight
                        total_possible_score += abs(weight)
                    
                    weighted_scores[factor_name] = weighted_score
                    total_weighted_score += weighted_score
            
            # 转换为0-100分制
            if total_possible_score > 0:
                base_score = (total_weighted_score / total_possible_score) * 100
            else:
                base_score = 0.0
            base_score = max(0.0, min(100.0, base_score))

            final_score = base_score

            bonus_scaled = 0.0
            if bonus_total_score != 0.0:
                denom = total_possible_score if total_possible_score > 0 else self._bonus_weight_total
                if denom > 0:
                    bonus_scaled = (bonus_total_score / denom) * 100.0
                final_score += bonus_scaled

            final_score = max(0.0, min(100.0, final_score))

            # 第二步：应用保底机制 - 防止有潜力的股票分数过低
            final_score = self._apply_floor_protection(final_score, factors, code)

            
            # 确定操作建议
            action = self._get_trading_action(final_score)
            
            # 生成核心理由
            core_reasons = self._generate_core_reasons(factors, weighted_scores)
            
            result = {
                'code': code,
                'total_score': round(final_score, 1),
                'action': action,
                'action_text': action.value,
                'weighted_scores': weighted_scores,
                'core_reasons': core_reasons,
                'factor_details': self._format_factor_details(factors, weighted_scores),
                'smart_fusion_applied': needs_smart_fusion  # 调试信息
            }
            if bonus_scaled != 0.0:
                result['bonus_contribution'] = bonus_scaled
            
            logger.debug("股票 %s 评分完成: %.1f分 [%s]%s", code, final_score, action.value,
                        " [智能融合]" if needs_smart_fusion else "")
            return result
            
        except Exception as e:
            logger.error("股票 %s 评分计算失败: %s", code, e)
            return self._get_empty_score_result(code)

    
    
    def _should_apply_smart_fusion(self, factors: Dict[str, float], kline_score: float) -> bool:
        """
        判断是否需要应用智能融合机制 - 精确度优先版本
        
        更严格的条件：
        1. K线因子得分极低 (≤0.1) 
        2. 必须有明确的技术共振信号：
           - B1综合因子≥0.6 (核心抄底信号强)
           - 且至少2个其他因子≥0.5 (多重确认)
           - 且风险因子≤0.2 (风险可控)
        """
        if kline_score > 0.1:  # K线分数不是极低，不需要融合
            return False
        
        # 必要条件：B1综合信号必须强劲
        b1_score = factors.get('b1_comprehensive', 0.0)
        if b1_score < 0.55:  # 微调：0.6→0.55，适配历史有效案例
            return False
        
        # 必要条件：风险必须可控
        risk_score = factors.get('risk_penalty', 0.0)
        if risk_score > 0.2:
            return False
        
        # 检查其他因子的确认信号
        other_strong_factors = 0
        confirmation_factors = ['pocket_pivot', 'supply_analysis', 'pivot_analysis', 'ma_structure']
        
        for factor_name in confirmation_factors:
            score = factors.get(factor_name, 0.0)
            if score >= 0.5:  # 强确认信号
                other_strong_factors += 1
        
        # 严格融合条件：至少2个其他因子≥0.5
        return other_strong_factors >= 2

    def _apply_adjustment(self, factor_name: str, value: float) -> float:
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
            min_v = tier.get('min', 0.0)
            max_v = tier.get('max', 1.0)
            multiplier = tier.get('multiplier', 1.0)
            if numeric_value >= min_v and numeric_value < max_v:
                numeric_value *= multiplier
                break

        return max(0.0, min(1.0, numeric_value))

    @staticmethod
    def _center_factor_value(value: float, neutral: float) -> float:
        """
        将因子值围绕中性点进行对称映射，使得：
        - value == neutral 时返回 0
        - value > neutral 时返回 (0, 1]
        - value < neutral 时返回 [-1, 0)
        """
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if neutral <= 0.0 or neutral >= 1.0:
            return numeric

        if numeric >= neutral:
            denom = max(1e-6, 1.0 - neutral)
            return min(1.0, (numeric - neutral) / denom)
        else:
            denom = max(1e-6, neutral)
            return max(-1.0, (numeric - neutral) / denom)
    
    def _apply_smart_kline_fusion(self, kline_value: float, original_weight: float, 
                                 all_factors: Dict[str, float]) -> Tuple[float, float]:
        """
        应用K线因子智能融合
        
        策略：
        1. 如果其他因子得分很高，给予K线因子"基础分"而非0分
        2. 降低K线因子的权重，避免过度惩罚
        3. 根据其他因子的强度来调整覆盖程度
        """
        # 计算其他因子的平均表现
        other_factors = {
            k: v for k, v in all_factors.items()
            if k not in ['kline_patterns', 'risk_penalty'] and isinstance(v, (int, float))
        }
        avg_other_score = np.mean(list(other_factors.values())) if other_factors else 0.0
        
        # 计算覆盖强度 (基于其他因子的表现)
        if avg_other_score >= 0.7:
            cover_strength = 0.8  # 强覆盖
        elif avg_other_score >= 0.5:
            cover_strength = 0.6  # 中等覆盖
        elif avg_other_score >= 0.3:
            cover_strength = 0.4  # 轻微覆盖
        else:
            cover_strength = 0.0  # 不覆盖
        
        # 调整K线得分：原分数 + 覆盖分数
        base_score = max(0.15, kline_value)  # 给予基础分，最低0.15
        cover_bonus = (0.4 - kline_value) * cover_strength  # 覆盖奖励
        adjusted_score = min(0.6, base_score + cover_bonus)  # 调整后分数，上限0.6
        
        # 调整权重：降低K线因子权重，减少过度惩罚
        weight_reduction = 0.3 + 0.4 * cover_strength  # 权重减少30%-70%
        adjusted_weight = original_weight * (1 - weight_reduction)
        
        return adjusted_score, adjusted_weight
    
    def _apply_floor_protection(self, score: float, factors: Dict[str, float], code: str) -> float:
        """
        应用保底保护机制 - 精确度优先版本
        
        更严格的条件：只保护真正有强烈技术共振的股票
        """
        if score >= 55:  # 分数不低，不需要保护
            return score
        
        # 更严格的保底条件：
        # 1. B1综合因子必须≥0.65 (非常强的核心信号，微调适配历史案例)
        # 2. 至少2个其他因子≥0.5 (强烈多重确认，降低门槛)
        # 3. 风险因子≤0.15 (风险很低)
        
        b1_score = factors.get('b1_comprehensive', 0.0)
        risk_score = factors.get('risk_penalty', 0.0)
        
        if b1_score < 0.65 or risk_score > 0.15:  # 微调：0.7→0.65
            return score
        
        # 检查其他因子的强确认
        other_factors = ['pocket_pivot', 'supply_analysis', 'pivot_analysis', 'ma_structure']
        strong_confirmations = sum(1 for f in other_factors 
                                 if factors.get(f, 0.0) >= 0.5)
        
        # 只有在有2个以上强确认的情况下才保底
        if strong_confirmations >= 2:  # 微调：3→2，降低门槛
            # 给予适度保底奖励，确保进入观望区间
            floor_bonus = min(10, (b1_score - 0.65) * 30)  # 最多10分奖励，调整基准
            protected_score = min(60, score + floor_bonus)  # 保底后最高60分
            
            if protected_score > score:
                logger.debug("精准保底保护生效: %s, %.1f->%.1f (B1=%.2f, 确认数=%d)", 
                           code, score, protected_score, b1_score, strong_confirmations)
                return protected_score
        
        return score
    
    def _get_trading_action(self, score: float) -> TradingAction:
        """根据评分确定交易操作建议"""
        if score >= self.score_thresholds['buy']:
            return TradingAction.BUY
        elif score >= self.score_thresholds['watch']:
            return TradingAction.WATCH
        else:
            return TradingAction.AVOID
    
    def _generate_core_reasons(self, factors: Dict[str, float], 
                             weighted_scores: Dict[str, float]) -> str:
        """
        生成核心理由文本
        基于得分最高的几个因子自动生成描述
        """
        reasons = []
        
        # 按权重得分排序，找出贡献最大的因子
        positive_factors = [(name, score) for name, score in weighted_scores.items() 
                          if score > 0 and name != 'risk_penalty']
        positive_factors.sort(key=lambda x: x[1], reverse=True)
        
        # 生成正向理由 - 基于原始得分判断是否有意义，按加权得分排序
        for factor_name, weighted_score in positive_factors:  # 按加权得分排序
            original_score = factors[factor_name]
            # 使用原始得分判断是否显示理由
            if original_score > 0.15:  # 原始得分阈值：15%以上才有意义
                reason = self._get_factor_reason_text(factor_name, original_score)
                if reason:
                    reasons.append(reason)
        
        # 添加模板形态最优匹配（仅当形态相似度因子真正贡献分数时）
        try:
            tmpl_desc = factors.get('b1_template_best')  # 来自B1Analysis的元信息
            template_weighted_score = weighted_scores.get('b1_template_similarity', 0)
            # 只有当形态相似度真正贡献分数时才显示描述
            if isinstance(tmpl_desc, str) and tmpl_desc.strip() and template_weighted_score > 0:
                reasons.insert(0, tmpl_desc.strip())
        except Exception:
            pass

        # 添加风险提示
        risk_score = weighted_scores.get('risk_penalty', 0)
        risk_factor_value = factors.get('risk_penalty', 0)
        if risk_score < -2:  # 降低阈值，提高敏感度
            if risk_factor_value > 0.5:
                reasons.append("大风车风险")
            elif risk_factor_value > 0.2:
                reasons.append("有技术风险")
            else:
                reasons.append("轻微风险")
        
        return "；".join(reasons) if reasons else "技术指标一般"
    
    def _get_factor_reason_text(self, factor_name: str, factor_value: float) -> str:
        """获取单个因子的理由文本"""
        display_name = self.factor_display_names.get(factor_name, factor_name)
        # B1因子阈值调整：权重高但阈值要合理
        if factor_name == 'b1_comprehensive' and factor_value < 0.3:
            return ""
        # 其他因子保持原有门槛
        elif factor_name != 'b1_comprehensive' and factor_value <= 0.1:
            return ""
            
        reason_texts = {
            'ma_structure': "均线呈多头排列" if factor_value > 0.8 else "均线结构良好",
            'b1_comprehensive': (
                "B1综合信号强劲" if factor_value > 0.8 
                else "B1形态良好" if factor_value > 0.6 
                else "B1信号确认" if factor_value > 0.4
                else "B1形态初现"  # 0.3-0.4区间
            ),
            'b1_template_similarity': None,  # 由详细模板描述替代，避免重复
            'symmetry_resonance': (
                "对称结构右侧强势" if factor_value > 0.7
                else "对称走势与镜像基本一致" if factor_value > 0.5
                else "右侧走势偏弱"
            ),
            'kline_patterns': "关键反转形态确认" if factor_value > 0.8 else "K线形态支持",
            'pocket_pivot': "口袋妖怪成立" if factor_value > 0.8 else "口袋支点形成",
            'supply_analysis': "上方供给较轻" if factor_value > 0.7 else "供给压力适中",
            'vcp_compression': "完美VCP压缩" if factor_value > 0.8 else "波动性收缩",
            'pivot_analysis': "关键枢轴支撑确认" if factor_value > 0.8 else "枢轴位技术配合",
            'structure_analysis': (
                f"{display_name}强势" if factor_value > 0.8
                else f"{display_name}良好" if factor_value > 0.6
                else f"{display_name}初现"
            ),
            'risk_penalty': "发现顶部大风车风险" if factor_value > 0.5 else ("有技术风险" if factor_value > 0.2 else "风险可控")
        }
        
        reason = reason_texts.get(factor_name, f"{display_name}表现良好")
        return reason if reason is not None else ""  # 处理显式设置为None的情况
    
    def _format_factor_details(self, factors: Dict[str, float], 
                             weighted_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """格式化因子详情用于展示"""
        details = []

        preferred_order = [
            'ma_structure',
            'b1_comprehensive',
            'kline_patterns',
            'symmetry_resonance',
            'pocket_pivot',
            'supply_analysis',
            'vcp_compression',
            'pivot_analysis',
            'risk_penalty',
            'b1_template_similarity',
        ]
        ordered_factors = []
        seen = set()
        for name in preferred_order:
            if name in factors:
                ordered_factors.append(name)
                seen.add(name)
        for name in factors:
            if name not in seen:
                ordered_factors.append(name)

        for factor_name in ordered_factors:
            factor_value = factors.get(factor_name)
            weight = self.factor_weights.get(factor_name, 0.0)
            weighted_score = weighted_scores.get(factor_name, 0.0)
            
            if weight != 0.0:  # 只显示有权重的因子
                item = {
                    'name': self.factor_display_names.get(factor_name, factor_name),
                    'raw_value': factor_value,
                    'weight': weight,
                    'weighted_score': weighted_score,
                    'enabled': True,
                    'status': self._get_factor_status(factor_value, factor_name)
                }
                # 为模板相似度补充说明
                if factor_name == 'b1_template_similarity':
                    desc = factors.get('b1_template_best')
                    if isinstance(desc, str) and desc:
                        item['note'] = desc
                details.append(item)
        
        return details
    
    def _get_factor_status(self, value: float, factor_name: str) -> str:
        """获取因子状态描述"""
        if factor_name == 'risk_penalty':
            if value > 0.8:
                return "风险较高"
            elif value > 0.5: 
                return "存在风险"
            else:
                return "风险可控"
        else:
            if value > 0.8:
                return "优秀"
            elif value > 0.5:
                return "良好"
            elif value > 0.2:
                return "一般"
            else:
                return "较差"
    
    def _get_empty_score_result(self, code: str) -> Dict[str, Any]:
        """返回空的评分结果"""
        return {
            'code': code,
            'total_score': 0.0,
            'action': TradingAction.AVOID,
            'action_text': TradingAction.AVOID.value,
            'weighted_scores': {},
            'core_reasons': "数据不足，无法评分",
            'factor_details': []
        }
    
    def batch_score(self, factors_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        批量评分
        
        Args:
            factors_dict: {code: factors} 字典
            
        Returns:
            {code: score_result} 字典
        """
        results = {}
        for code, factors in factors_dict.items():
            results[code] = self.calculate_score(factors, code)
        
        return results
    
    def get_top_picks(self, score_results: Dict[str, Dict[str, Any]], 
                     min_score: float = None, top_n: int = None) -> List[Dict[str, Any]]:
        """
        获取评分最高的股票
        
        Args:
            score_results: 评分结果字典
            min_score: 最低分数要求
            top_n: 返回前N只股票
            
        Returns:
            排序后的评分结果列表
        """
        results_list = list(score_results.values())
        
        # 按评分排序
        results_list.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 过滤最低分数
        if min_score is not None:
            results_list = [r for r in results_list if r['total_score'] >= min_score]
        
        # 限制返回数量
        if top_n is not None:
            results_list = results_list[:top_n]
        
        return results_list


class ShortTermScoringEngine:
    """
    短期多窗口评分引擎
    
    专为短期预测（1、3、5天）设计的评分系统：
    - 支持窗口特化的因子权重配置
    - 实现多窗口融合评分算法
    - 提供智能操作建议和持仓周期
    """
    
    def __init__(self, short_term_config: Dict[str, Any]):
        """
        初始化短期评分引擎
        
        Args:
            short_term_config: 短期评分配置，包含window_configs等
        """
        if not short_term_config.get('enabled', False):
            raise ValueError("短期评分配置未启用！请检查 configs.json 中的 short_term_scoring_config.enabled")
        
        self.window_configs = short_term_config.get('window_configs', {})
        self.fusion_config = short_term_config.get('fusion_config', {})
        
        # 验证配置完整性
        required_windows = ['1day', '3day', '5day']
        missing_windows = [w for w in required_windows if w not in self.window_configs]
        if missing_windows:
            raise ValueError(f"短期评分配置中缺少预测窗口: {missing_windows}")
        
        # 创建各窗口的评分引擎
        self.window_engines = {}
        for window, config in self.window_configs.items():
            try:
                temp_config = {
                    'factors': config['factors'],
                    'thresholds': config['thresholds']
                }
                self.window_engines[window] = ScoringEngine(temp_config)
                logger.info(f"✅ 短期评分引擎: {window} 窗口初始化完成")
            except Exception as e:
                logger.error(f"❌ 短期评分引擎: {window} 窗口初始化失败: {e}")
                raise
        
        # 融合配置
        self.base_weights = self.fusion_config.get('base_weights', [0.30, 0.40, 0.30])
        self.consistency_config = self.fusion_config.get('consistency_scoring', {})
        self.final_thresholds = self.fusion_config.get('final_thresholds', {})
        
        logger.info("🔧 短期多窗口评分引擎初始化完成")
        logger.info(f"⚖️  窗口权重配置: 1天={self.base_weights[0]}, 3天={self.base_weights[1]}, 5天={self.base_weights[2]}")
    
    def calculate_window_score(self, factors: Dict[str, float], window: str, code: str) -> Dict[str, Any]:
        """计算指定窗口的评分"""
        if window not in self.window_engines:
            raise ValueError(f"不支持的预测窗口: {window}")
        
        engine = self.window_engines[window]
        return engine.calculate_score(factors, code)
    
    def calculate_fusion_score(self, factors: Dict[str, float], code: str) -> Dict[str, Any]:
        """
        计算多窗口融合评分
        
        Args:
            factors: 标准化因子得分字典
            code: 股票代码
            
        Returns:
            融合评分结果，包含各窗口分数、融合逻辑、操作建议等
        """
        # 计算各窗口评分
        window_results = {}
        window_scores = []
        
        windows = ['1day', '3day', '5day']
        for i, window in enumerate(windows):
            try:
                result = self.calculate_window_score(factors, window, code)
                window_results[window] = result
                window_scores.append(result['total_score'])
            except Exception as e:
                logger.warning(f"窗口 {window} 评分失败: {e}")
                window_results[window] = {'total_score': 50.0, 'error': str(e)}
                window_scores.append(50.0)
        
        # 基础加权融合
        base_score = sum(score * weight for score, weight in zip(window_scores, self.base_weights))
        
        # 一致性分析
        consistency_analysis = self._analyze_consistency(window_scores)
        
        # 最终评分
        final_score = base_score + consistency_analysis['bonus'] - consistency_analysis['penalty']
        final_score = max(0.0, min(100.0, final_score))
        
        # 操作建议
        action_analysis = self._generate_action_advice(final_score, window_scores)
        
        # 核心理由生成
        core_reasons = self._generate_core_reasons(window_results, consistency_analysis)
        
        return {
            'code': code,
            'window_scores': dict(zip(windows, window_scores)),
            'window_results': window_results,
            'base_score': base_score,
            'consistency_analysis': consistency_analysis,
            'final_score': final_score,
            'action': action_analysis['action'],
            'action_text': action_analysis['action_text'],
            'recommended_holding': action_analysis['holding_period'],
            'risk_level': action_analysis['risk_level'],
            'core_reasons': core_reasons,
            'score_breakdown': {
                'base': base_score,
                'consistency_bonus': consistency_analysis['bonus'],
                'discord_penalty': consistency_analysis['penalty'],
                'final': final_score
            }
        }
    
    def _analyze_consistency(self, window_scores: List[float]) -> Dict[str, Any]:
        """分析窗口评分一致性"""
        # 一致性阈值
        excellent_threshold = 70.0
        good_threshold = 55.0
        fair_threshold = 45.0
        
        # 统计各档位窗口数量
        excellent_count = sum(1 for score in window_scores if score >= excellent_threshold)
        good_count = sum(1 for score in window_scores if score >= good_threshold)
        fair_count = sum(1 for score in window_scores if score >= fair_threshold)
        
        # 一致性奖励
        bonus = 0
        if excellent_count == 3:
            bonus = self.consistency_config.get('all_excellent_bonus', 15)
            consistency_type = "全面优秀"
        elif excellent_count >= 2:
            bonus = self.consistency_config.get('all_good_bonus', 10)
            consistency_type = "多数优秀"
        elif good_count >= 2:
            bonus = self.consistency_config.get('two_good_bonus', 5)
            consistency_type = "多数良好"
        else:
            consistency_type = "表现分散"
        
        # 不一致惩罚
        score_std = np.std(window_scores)
        penalty = 0
        if score_std > 25:
            penalty = min(score_std * 0.2, 10)
            discord_level = "严重分歧"
        elif score_std > 15:
            penalty = min(score_std * 0.1, 5)
            discord_level = "轻微分歧"
        else:
            discord_level = "相对一致"
        
        return {
            'excellent_count': excellent_count,
            'good_count': good_count,
            'fair_count': fair_count,
            'consistency_type': consistency_type,
            'bonus': bonus,
            'penalty': penalty,
            'score_std': score_std,
            'discord_level': discord_level
        }
    
    def _generate_action_advice(self, final_score: float, window_scores: List[float]) -> Dict[str, Any]:
        """生成操作建议"""
        # 基础操作建议
        if final_score >= self.final_thresholds.get('strong_buy', 75):
            action = TradingAction.BUY
            action_text = "强烈买入"
            holding_period = "1-2天"
            risk_level = "中高风险"
        elif final_score >= self.final_thresholds.get('buy', 65):
            action = TradingAction.BUY
            action_text = "买入"
            holding_period = "3-5天"
            risk_level = "中等风险"
        elif final_score >= self.final_thresholds.get('watch', 50):
            action = TradingAction.WATCH
            action_text = "观望"
            holding_period = "关注后续"
            risk_level = "低风险"
        else:
            action = TradingAction.AVOID
            action_text = "避免"
            holding_period = "不建议"
            risk_level = "高风险"
        
        # 特殊情况调整
        day1_score, day3_score, day5_score = window_scores
        
        # 如果1天窗口极高但5天窗口较低，提示快进快出
        if day1_score >= 75 and day5_score < 50:
            holding_period = "1天快进快出"
            risk_level = "高风险短线"
        
        # 如果5天窗口很高，建议延长持仓
        elif day5_score >= 70:
            if holding_period == "3-5天":
                holding_period = "5-7天"
        
        return {
            'action': action,
            'action_text': action_text,
            'holding_period': holding_period,
            'risk_level': risk_level
        }
    
    def _generate_core_reasons(self, window_results: Dict[str, Any], consistency_analysis: Dict[str, Any]) -> List[str]:
        """生成核心理由"""
        reasons = []
        
        # 各窗口核心理由
        for window, result in window_results.items():
            score = result.get('total_score', 0)
            window_name = window.replace('day', '天')
            
            if score >= 70:
                reasons.append(f"✅ {window_name}预测优秀({score:.1f}分)")
            elif score >= 55:
                reasons.append(f"📊 {window_name}预测良好({score:.1f}分)")
            elif score < 40:
                reasons.append(f"⚠️ {window_name}预测偏弱({score:.1f}分)")
        
        # 一致性分析
        if consistency_analysis['consistency_type'] != "表现分散":
            reasons.append(f"🎯 多窗口{consistency_analysis['consistency_type']}")
        
        if consistency_analysis['discord_level'] == "严重分歧":
            reasons.append(f"⚠️ 预测窗口间存在{consistency_analysis['discord_level']}")
        
        return reasons[:5]  # 最多返回5条理由
    
    def batch_score(self, factors_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """批量评分"""
        results = {}
        for code, factors in factors_dict.items():
            try:
                results[code] = self.calculate_fusion_score(factors, code)
            except Exception as e:
                logger.error(f"股票 {code} 短期评分失败: {e}")
                results[code] = {
                    'code': code,
                    'error': str(e),
                    'final_score': 0,
                    'action_text': '评分失败'
                }
        return results

