#!/usr/bin/env python3
"""
B1综合分析模块
专门分析B1形态、红肥绿瘦、庄家行为等综合信号
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging

# 导入现有的技术指标函数
from stock_research.selectors.indicators import compute_bbi, compute_dif, compute_kdj
from stock_research.scoring.b1_market_gate import B1MarketGateResolver
from stock_research.scoring.b1_policy import HOT_UP_LOGIC, build_b1_score_payload, resolve_b1_logic

logger = logging.getLogger("b1_analysis")


class B1Analysis:
    """
    B1综合分析器
    
    提供：
    - B1形态识别
    - 红肥绿瘦量价配合分析
    - 庄家行为分析
    - SB1连续形态检测
    - 高波动弹性分析
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化B1分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        # 模板数据缓存（避免重复IO）
        self._template_df_cache: Dict[str, pd.DataFrame] = {}
        self._template_feat_cache: Dict[str, Dict[str, Any]] = {}
        self._target_feat_cache: Dict[tuple, Dict[str, pd.Series]] = {}
        self._warned_window_lengths: set = set()
        self._market_gate_resolver = B1MarketGateResolver(self.config)
    
    def calc_b1_comprehensive(self, df: pd.DataFrame) -> float:
        """
        B1综合因子 - 整合所有B1相关逻辑，消除重叠
        
        整合内容：
        1. 原B1形态识别 (40%权重)
        2. 红肥绿瘦量价配合 (25%权重)  
        3. 大开大合弹性评分 (15%权重)
        4. 庄家行为分析 (20%权重) - 洗盘(正分) vs 出货(负分)
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            得分 0.0-1.0，1.0表示完美的B1综合信号
        """
        try:
            if len(df) < 60:  # 需要足够的历史数据
                return 0.0
            
            # 重置模板匹配中间结果（供上层引用）
            self.last_template_match = None
            
            # 1. 核心B1形态识别 (40%权重)
            b1_pattern_score = self._detect_enhanced_b1_pattern(df)
            
            # 2. "红肥绿瘦"量价配合 (25%权重)
            volume_pattern_score = self._detect_red_fat_green_thin(df)
            
            # 3. 大开大合弹性评分
            elasticity_parts = self._detect_high_volatility_elasticity_parts(df)
            elasticity_score = float(sum(elasticity_parts.values()))
            
            # 4. 庄家行为分析 (20%权重) - 可能是负分
            dealer_behavior_score = self._analyze_dealer_behavior(df)

            # 5. SB1连续形态加分 (额外奖励)
            sb1_bonus = self._detect_sb1_pattern(df)

            # 模板相似度不在此加分，改为独立因子避免双重计算
            b1_logic = resolve_b1_logic(self.config)
            market_state = self._market_gate_resolver.resolve_for_frame(df) if b1_logic == HOT_UP_LOGIC else None
            payload = build_b1_score_payload(
                component_values={
                    "b1_pattern": b1_pattern_score,
                    "red_fat_green_thin": volume_pattern_score,
                    "elasticity": elasticity_score,
                    "dealer_behavior": dealer_behavior_score,
                    "sb1_bonus": sb1_bonus,
                },
                elasticity_parts=elasticity_parts,
                washout_score=self._detect_dealer_washout(df),
                b1_logic=b1_logic,
                market_state=market_state,
            )
            return float(payload["score_final"])
            
        except Exception as e:
            logger.warning("B1综合因子计算失败: %s", e)
            return 0.0

    def calc_b1_template_similarity(self, df: pd.DataFrame) -> float:
        """作为独立因子暴露B1形态库相似度（0-1）。

        逻辑统一：
        - 从单一配置源读取 template_matching（优先 b1_template_similarity，其次 b1_comprehensive）
        - 使用与可视化一致的匹配实现与MA60策略
        - 相似度映射可配置（默认“软阈值 + gamma压缩”）：
          floor=0.65, ceil=0.90, gamma=2.0；即 sim<=floor→0；
          线性归一到[0,1]后做 y=x^gamma，缓解0.7~0.8过敏问题。

        配置缺失：记录warning并抛出异常，便于上层显式处理。
        """
        # 首先尝试从b1_template_similarity自身的配置读取
        tm_conf = (
            self.config.get('factors', {})
            .get('b1_template_similarity', {})
            .get('params', {})
            .get('template_matching', {})
        )
        # 如果没有，则从b1_comprehensive读取（向后兼容）
        if not tm_conf:
            tm_conf = (
                self.config.get('factors', {})
                .get('b1_comprehensive', {})
                .get('params', {})
                .get('template_matching', {})
            )
        if not tm_conf:
            logger.warning("b1_template_similarity 缺少 template_matching 配置")
            raise ValueError("缺少模板匹配配置：factors.[b1_template_similarity|b1_comprehensive].params.template_matching")

        sim, meta = self._match_b1_templates(df, tm_conf)
        self.last_template_match = meta

        # 门槛与强化映射：<0.7直接归零，>=0.7线性到1.0
        if sim <= 0.7:
            return 0.0
        mapped = (sim - 0.7) / 0.3  # 0.7->0, 1.0->1
        return float(max(0.0, min(1.0, mapped)))

    def compute_template_similarity(self, df: pd.DataFrame, tm_conf: Dict[str, Any], return_details: bool = True) -> Dict[str, Any]:
        """权威模板相似度计算入口（可视化/评分统一复用）。

        Returns:
            dict: {
              'raw_sim': float,        # MA60调整前的融合相似度
              'sim': float,            # MA60调整后的相似度（与评分因子一致）
              'mapped': float,         # 应用0.7阈值线性映射后的值
              'p_sim': float, 'v_sim': float, 'alpha': float,
              'ma_note': str|None, 'ma_factor': float,
              'best': {... 详情元数据 ...}
              可选：用于绘图的序列与日期
            }
        配置缺失或无可用模板时：warning + 抛异常。
        """
        if not isinstance(tm_conf, dict) or not tm_conf:
            logger.warning("模板相似度计算缺少 tm_conf 配置")
            raise ValueError("模板相似度计算缺少 tm_conf 配置")

        templates = tm_conf.get('templates') or []
        if not templates:
            logger.warning("模板相似度计算未提供任何模板")
            raise ValueError("模板相似度计算未提供任何模板：template_matching.templates 为空")

        sim, meta = self._match_b1_templates(df, tm_conf)
        if not meta:
            logger.warning("未能匹配到有效模板（窗口可能不足或数据不完整）")
            raise RuntimeError("未能匹配到有效模板")

        raw_sim = float(meta.get('raw_sim', sim))
        adjusted_sim = float(sim)
        mapped = 0.0 if adjusted_sim <= 0.7 else min(1.0, max(0.0, (adjusted_sim - 0.7) / 0.3))

        result = {
            'raw_sim': raw_sim,
            'sim': adjusted_sim,
            'mapped': mapped,
            'p_sim': float(meta.get('p_sim', 0.0)),
            'v_sim': float(meta.get('v_sim', 0.0)),
            'alpha': float(meta.get('alpha', tm_conf.get('alpha', 0.7))),
            'ma_note': meta.get('ma_note'),
            'ma_factor': float(meta.get('ma_factor', 1.0)),
            'best': meta,
        }

        if return_details:
            # 附带可视化所需序列（若存在）
            for k in ['g_p', 't_p', 'g_v', 't_v', 'target_close', 'target_dates', 'template_close', 'template_dates']:
                if k in meta:
                    result[k] = meta[k]
        return result

    def _map_similarity(self, sim: float, tm_conf: Dict[str, Any]) -> float:
        """将原始相似度映射到[0,1]，支持配置化。

        支持配置（位于 template_matching.mapping）：
          - type: 'soft'|'linear'（默认'soft'）
          - floor: 底阈，默认0.65
          - ceil: 顶阈，默认0.90
          - gamma: 软映射指数（>1压缩敏感度），默认2.0

        说明：
          - 当type='linear'时，仅做线性：(sim-floor)/(ceil-floor)
          - 当type='soft'时，先线性归一，再做 y = x^gamma，减小0.7~0.8区间的敏感度
        """
        try:
            mapping = {}
            if isinstance(tm_conf, dict):
                mapping = (tm_conf.get('mapping') or {}) if isinstance(tm_conf.get('mapping'), dict) else {}
            mtype = str(mapping.get('type', 'soft')).lower()
            floor = float(mapping.get('floor', 0.65))
            ceil = float(mapping.get('ceil', 0.90))
            gamma = float(mapping.get('gamma', 2.0))
            if ceil <= floor:
                ceil = floor + 1e-6
            if sim <= floor:
                return 0.0
            x = (float(sim) - floor) / (ceil - floor)
            if x >= 1.0:
                return 1.0
            x = max(0.0, min(1.0, x))
            if mtype == 'linear' or gamma <= 0:
                return x
            # soft 模式：gamma>1 压缩中低段敏感度
            return float(x ** gamma)
        except Exception:
            # 回退旧映射：<=0.7→0，>0.7线性到1
            if sim <= 0.7:
                return 0.0
            return min(1.0, max(0.0, (float(sim) - 0.7) / 0.3))

    def _match_b1_templates(self, df: pd.DataFrame, tm_conf: Dict[str, Any]):
        """
        将标的最近一段窗口与模板形态库进行相似度匹配。

        相似度 = alpha*价格序列相关度 + (1-alpha)*成交量序列相关度（可关）

        Returns:
            (score_0_1, meta)
            meta = {"name": 模板名, "similarity": 0~1, "desc": "形态 X 相似度 90%"}
        """
        if df is None or len(df) < 10:
            return 0.0, None

        templates = tm_conf.get('templates', []) or []
        if not templates:
            return 0.0, None

        # 读取数据目录（由上层在 scoring_config 中注入 data_dir）
        data_dir = self.config.get('data_dir') or self.config.get('daily_data_dir') or './daily_data'

        # 读取特征配置（保持默认行为）
        feat_cfg = tm_conf.get('feat') or {}
        price_mode = str(feat_cfg.get('price', 'logret_z')).lower()  # 'logret_z'|'pctret_z'|'ma60_rel_z'
        vol_mode = str(feat_cfg.get('volume', 'signed')).lower()    # 'signed'|'plain'|'off'

        # 合成特征：价格用标准化特征，量用窗口内zscore
        def _feat_price(series: pd.Series) -> pd.Series:
            try:
                s = series.astype(float)
                if price_mode == 'pctret_z':
                    ret = s.pct_change().dropna()
                    std = ret.std(ddof=0)
                    if std == 0 or np.isnan(std):
                        return pd.Series(np.zeros(len(ret)), index=ret.index)
                    return (ret - ret.mean()) / (std + 1e-9)
                elif price_mode == 'ma60_rel_z':
                    ma = s.rolling(60).mean()
                    rel = (s / (ma + 1e-9) - 1.0)
                    rel = rel.dropna()
                    std = rel.std(ddof=0)
                    if std == 0 or np.isnan(std) or len(rel) < 5:
                        return pd.Series(np.zeros(len(rel)), index=rel.index)
                    return (rel - rel.mean()) / (std + 1e-9)
                else:  # 默认 logret_z
                    ret = np.log(s).diff().dropna()
                    std = ret.std(ddof=0)
                    if std == 0 or np.isnan(std):
                        return pd.Series(np.zeros(len(ret)), index=ret.index)
                    return (ret - ret.mean()) / (std + 1e-9)
            except Exception:
                return pd.Series(dtype=float)

        def _feat_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
            vol = volume.replace(0, np.nan).ffill().bfill()
            vol = vol.reindex(close.index)
            # 标准化量能
            vstd = vol.std(ddof=0)
            if vstd == 0 or np.isnan(vstd):
                vol_z = pd.Series(np.zeros(len(vol)), index=vol.index)
            else:
                vol_z = (vol - vol.mean()) / (vstd + 1e-9)
            if vol_mode == 'off':
                return pd.Series(np.zeros(len(vol_z)), index=vol_z.index)
            if vol_mode == 'plain':
                return vol_z
            # 默认 signed：乘以当日收益方向（与价格特征一致的收益定义）
            if price_mode == 'pctret_z':
                vr = close.pct_change().dropna()
            elif price_mode == 'ma60_rel_z':
                # 用对数收益方向或百分比方向均可，这里沿用log收益方向，稳定性好
                vr = np.log(close).diff().dropna()
            else:
                vr = np.log(close).diff().dropna()
            vol_z = vol_z.reindex(vr.index).dropna()
            return vol_z * np.sign(vr)

        # 始终启用“有符号量能”以区分放量上涨/下跌
        use_volume = (vol_mode != 'off')
        alpha = float(tm_conf.get('alpha', 0.7))  # 价格权重
        sim_variant = str(tm_conf.get('sim_variant', 'pearson')).lower()  # pearson|spearman|hybrid
        hybrid_w = float(tm_conf.get('hybrid_w', 0.7))  # hybrid时价格/量能相似中Pearson占比
        disable_ma60 = bool(tm_conf.get('disable_ma60', False))

        def _corr01(a: pd.Series, b: pd.Series, method: str = 'pearson') -> float:
            try:
                x = a.values
                y = b.values
                if len(x) != len(y) or len(x) < 2:
                    return 0.0
                if method == 'pearson':
                    r = float(np.corrcoef(x, y)[0, 1])
                elif method == 'spearman':
                    # 简易spearman：对rank后做pearson
                    xr = pd.Series(x).rank(method='average').values
                    yr = pd.Series(y).rank(method='average').values
                    r = float(np.corrcoef(xr, yr)[0, 1])
                else:  # fallback
                    r = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(r):
                    r = 0.0
                return float((r + 1.0) / 2.0)
            except Exception:
                return 0.0

        # 预处理模板特征并收集长度，使用缓存以避免重复计算
        template_feats = []
        lengths = []
        from pathlib import Path
        for item in templates:
            try:
                name = item.get('name') or '形态'
                ticker = str(item['ticker'])
                start = pd.to_datetime(item['start'])
                end = pd.to_datetime(item['end'])

                csv_path = Path(data_dir) / f"{ticker}.csv"
                if not csv_path.exists():
                    logger.warning("模板CSV缺失: %s", csv_path)
                    continue
                cache_key_df = str(csv_path)
                if cache_key_df in self._template_df_cache:
                    tdf = self._template_df_cache[cache_key_df]
                else:
                    tdf = pd.read_csv(csv_path, parse_dates=['date']).sort_values('date')
                    self._template_df_cache[cache_key_df] = tdf
                tseg = tdf[(tdf['date'] >= start) & (tdf['date'] <= end)].copy()
                if len(tseg) < 8:
                    logger.warning("模板片段过短: %s %s~%s 长度=%d", ticker, start.date(), end.date(), len(tseg))
                    continue

                feat_key = f"{ticker}|{start.date()}|{end.date()}"
                cached = self._template_feat_cache.get(feat_key)
                if cached is None:
                    t_price = _feat_price(tseg['close'])
                    if len(t_price) < 5:
                        continue
                    t_vol = _feat_volume(tseg['close'], tseg['volume']).reindex(t_price.index, method='nearest') if use_volume else None
                    L = len(t_price)
                    cached = {
                        "L": L,
                        "price": t_price.tail(L),
                        "vol": (t_vol.tail(L) if t_vol is not None else None),
                        # 附带可视化与元信息：确保后续不再依赖外部重建
                        "template_start": start,
                        "template_end": end,
                        "template_close": tseg['close'].tail(L).copy(),
                        "template_dates": tseg['date'].tail(L).copy() if 'date' in tseg.columns else None,
                    }
                    self._template_feat_cache[feat_key] = cached
                template_feats.append({
                    "name": name,
                    "ticker": ticker,
                    "L": cached["L"],
                    "price": cached["price"],
                    "vol": cached["vol"] if use_volume else None,
                    "template_start": cached.get("template_start", start),
                    "template_end": cached.get("template_end", end),
                    "template_close": cached.get("template_close", tseg['close'].tail(cached["L"]).copy()),
                    "template_dates": cached.get("template_dates", (tseg['date'].tail(cached["L"]).copy() if 'date' in tseg.columns else None)),
                })
                lengths.append(cached["L"])
            except Exception as err:
                logger.warning("模板预处理异常: %s (%s~%s): %s", item.get('ticker'), item.get('start'), item.get('end'), err)
                continue

        if not template_feats:
            return 0.0, None

        # 快速跳过：若目标历史长度小于最小模板窗口，直接返回
        try:
            Lmin = min(lengths)
            last_date = df['date'].iloc[-1] if 'date' in df.columns else None
            if len(df) < Lmin + 1:
                warn_key = (getattr(last_date, 'value', None), Lmin)
                if warn_key not in self._warned_window_lengths:
                    logger.debug("目标窗口长度不足: 需要 %d 实际 %d", Lmin + 1, len(df))
                    self._warned_window_lengths.add(warn_key)
                return 0.0, None
        except Exception:
            pass

        # 取目标尾部与模板长度一致的窗口
        best = {"name": None, "similarity": 0.0}

        # 分组按窗口长度，批量计算（PyTorch加速pearson/hybrid）
        use_torch = bool(self.config.get('use_torch', True))
        can_torch = False
        if sim_variant in ('pearson', 'hybrid') and use_torch:
            try:
                import torch  # noqa: F401
                can_torch = True
            except Exception:
                can_torch = False

        from collections import defaultdict
        by_len = defaultdict(list)
        for idx, tf in enumerate(template_feats):
            by_len[int(tf['L'])].append(idx)

        import math
        for L, idxs in by_len.items():
            try:
                # 目标尾部窗口特征（复用缓存）
                last_date = df['date'].iloc[-1] if 'date' in df.columns else None
                key = (getattr(last_date, 'value', None), L)
                cached_target = self._target_feat_cache.get(key)
                if cached_target is None:
                    sub = df.tail(L + 1).copy()
                    if len(sub) < L + 1:
                        warn_key = key
                        if warn_key not in self._warned_window_lengths:
                            logger.debug("目标窗口长度不足: 需要 %d 实际 %d", L + 1, len(sub))
                            self._warned_window_lengths.add(warn_key)
                        continue
                    g_price = _feat_price(sub['close']).tail(L)
                    g_vol = _feat_volume(sub['close'], sub['volume']).tail(L) if use_volume else None
                    cached_target = {"price": g_price, "vol": g_vol}
                    self._target_feat_cache[key] = cached_target
                else:
                    g_price = cached_target['price']
                    g_vol = cached_target['vol']

                # 长期趋势调整因子（原MA60逻辑升级为多均线平均）
                ma_factor, ma_note = 1.0, None
                if not disable_ma60:
                    try:
                        trend_windows = (14, 28, 57, 114)
                        rolling_parts = []
                        for win in trend_windows:
                            ma_series = df['close'].rolling(window=win, min_periods=win).mean()
                            rolling_parts.append(ma_series)
                        if not rolling_parts:
                            raise ValueError("no trend windows available")
                        long_trend_full = sum(rolling_parts) / float(len(rolling_parts))
                        sub_close = df['close'].tail(L + 1)
                        sub_trend = long_trend_full.reindex(sub_close.index)
                        valid = sub_trend.notna()
                        cross_flag = False
                        if valid.sum() >= 2:
                            sc = sub_close[valid]
                            sm = sub_trend[valid]
                            below = (sc.shift(1) < sm.shift(1))
                            above = (sc >= sm)
                            if below.any() and above.any():
                                cross_flag = ((below) & (above)).any()
                            dist = (sc - sm).abs() / (sm + 1e-9)
                            med_dist = float(dist.median()) if len(dist) else 0.0
                        else:
                            med_dist = 0.05
                        require_cross = bool((tm_conf.get('ma60') or {}).get('require_cross', False))
                        if require_cross and not cross_flag:
                            ma_factor, ma_note = 0.0, '无长期趋势金叉(严格)'
                        elif cross_flag:
                            ma_factor, ma_note = 1.0, '长期趋势金叉'
                        else:
                            if med_dist <= 0.03:
                                ma_factor, ma_note = 0.95, f'接近长期趋势(~{int(med_dist*100)}%)'
                            elif med_dist <= 0.06:
                                ma_factor, ma_note = 0.90, f'接近长期趋势(~{int(med_dist*100)}%)'
                            elif med_dist <= 0.10:
                                ma_factor, ma_note = 0.80, f'偏离长期趋势(~{int(med_dist*100)}%)'
                            else:
                                ma_factor, ma_note = 0.70, f'远离长期趋势(~{int(med_dist*100)}%)'
                    except Exception:
                        ma_factor, ma_note = 1.0, None

                # 构建模板矩阵
                t_prices = []
                t_vols = []
                meta = []
                for idx in idxs:
                    tf = template_feats[idx]
                    tp = tf['price'].tail(L).values.astype(float)
                    t_prices.append(tp)
                    if use_volume and tf['vol'] is not None:
                        tv = tf['vol'].tail(L).values.astype(float)
                        t_vols.append(tv)
                    else:
                        t_vols.append(None)
                    meta.append(tf)

                # 价格相似度批量
                def corr01_batch(X: np.ndarray, y: np.ndarray) -> np.ndarray:
                    if can_torch:
                        import torch
                        tx = torch.from_numpy(X)
                        ty = torch.from_numpy(y)
                        # 标准化
                        tx = tx - tx.mean(dim=1, keepdim=True)
                        ty = ty - ty.mean()
                        denom_x = torch.linalg.norm(tx, dim=1) + 1e-12
                        denom_y = torch.linalg.norm(ty) + 1e-12
                        corr = (tx @ ty) / (denom_x * denom_y)
                        corr = torch.nan_to_num(corr, nan=0.0)
                        return ((corr + 1.0) / 2.0).cpu().numpy()
                    # numpy 回退
                    Xc = X - X.mean(axis=1, keepdims=True)
                    yc = y - y.mean()
                    denom_x = np.linalg.norm(Xc, axis=1) + 1e-12
                    denom_y = np.linalg.norm(yc) + 1e-12
                    corr = (Xc @ yc) / (denom_x * denom_y)
                    corr = np.nan_to_num(corr, nan=0.0)
                    return (corr + 1.0) / 2.0

                Xp = np.vstack(t_prices) if t_prices else np.zeros((0, L), dtype=float)
                yp = g_price.tail(L).values.astype(float)
                p_sims = corr01_batch(Xp, yp) if len(Xp) else np.array([])

                # 量能相似度批量（若启用）
                if use_volume and g_vol is not None and any(v is not None for v in t_vols):
                    gv = g_vol.tail(L).values.astype(float)
                    # 将缺失的vol用0.5中性值
                    Xv_list = [v if v is not None else np.full(L, 0.0, dtype=float) for v in t_vols]
                    Xv = np.vstack(Xv_list)
                    v_sims_p = corr01_batch(Xv, gv)
                    if sim_variant == 'hybrid':
                        # 简化：hybrid 这里沿用 pearson（Spearman 代价高，必要时再补）
                        v_sims = v_sims_p
                    else:
                        v_sims = v_sims_p
                else:
                    v_sims = np.zeros(len(p_sims), dtype=float)

                # 融合 + MA60 调整
                sims = (alpha * p_sims + (1 - alpha) * v_sims) if use_volume else p_sims
                sims = sims * float(ma_factor)

                # 取最优模板
                if sims.size:
                    j = int(np.argmax(sims))
                    sim = float(sims[j])
                    tf = meta[j]
                    if sim > best['similarity']:
                        best = {
                            "name": tf['name'],
                            "ticker": tf['ticker'],
                            "L": int(L),
                            "similarity": sim,
                            "raw_sim": float(alpha * p_sims[j] + (1 - alpha) * (v_sims[j] if use_volume else 0.0)),
                            "p_sim": float(p_sims[j]),
                            "v_sim": float(v_sims[j] if use_volume else 0.0),
                            "alpha": float(alpha),
                            "ma_note": ma_note,
                            "ma_factor": float(ma_factor),
                            "g_p": g_price.copy() if isinstance(g_price, pd.Series) else None,
                            "t_p": tf['price'].tail(L).copy(),
                            "g_v": (g_vol.copy() if isinstance(g_vol, pd.Series) else None),
                            "t_v": (tf['vol'].tail(L).copy() if tf['vol'] is not None else None),
                            "target_close": df['close'].tail(L).copy(),
                            "target_dates": df['date'].tail(L).copy() if 'date' in df.columns else None,
                            "template_close": tf.get('template_close'),
                            "template_dates": tf.get('template_dates'),
                            "template_start": tf.get('template_start'),
                            "template_end": tf.get('template_end'),
                        }
            except Exception as err:
                logger.warning("模板匹配异常: L=%s: %s", L, err)
                continue

        if best["name"] is None:
            return 0.0, None

        sim = max(0.0, min(1.0, best["similarity"]))
        note = f" + {best.get('ma_note')}" if best.get('ma_note') else ""
        meta = {
            "name": best.get("name"),
            "ticker": best.get("ticker"),
            "L": best.get("L"),
            "similarity": sim,
            "raw_sim": best.get("raw_sim", sim),
            "p_sim": best.get("p_sim"),
            "v_sim": best.get("v_sim"),
            "alpha": best.get("alpha"),
            "ma_note": best.get("ma_note"),
            "ma_factor": best.get("ma_factor", 1.0),
            "desc": f"{best['name']} 相似度 {int(round(sim * 100))}%{note}",
            # 可视化所需（若存在）
            "g_p": best.get('g_p'),
            "t_p": best.get('t_p'),
            "g_v": best.get('g_v'),
            "t_v": best.get('t_v'),
            "target_close": best.get('target_close'),
            "template_close": best.get('template_close'),
            "target_dates": best.get('target_dates'),
            "template_dates": best.get('template_dates'),
            "template_start": best.get('template_start'),
            "template_end": best.get('template_end'),
        }
        return sim, meta
    
    def _detect_enhanced_b1_pattern(self, df: pd.DataFrame) -> float:
        """
        增强版B1形态检测
        结合KDJ超卖、价格位置、成交量等多维度判断
        """
        try:
            params = self.config.get('enhanced_b1', {})
            
            if len(df) < 30:
                return 0.0
            
            score = 0.0
            recent_data = df.tail(20)
            
            # 1. KDJ超卖信号 (30分) - 使用selector的J值判断逻辑
            try:
                kdj_result = compute_kdj(df)
                if isinstance(kdj_result, pd.DataFrame) and 'J' in kdj_result.columns:
                    current_j = kdj_result['J'].iloc[-1]
                    
                    # 从configs.json获取各selector的J值阈值 - 带默认值保护和警告
                    b1_params = self.config.get('factors', {}).get('b1_comprehensive', {}).get('params', {})
                    
                    # 检查配置完整性并给出警告
                    if not b1_params:
                        logger.warning("⚠️  B1综合因子配置缺失，使用默认参数。建议检查 configs.json 中的 scoring_config.factors.b1_comprehensive.params")
                    
                    bbi_j_threshold = b1_params.get('j_threshold_bbi', None)
                    if bbi_j_threshold is None:
                        bbi_j_threshold = 10
                        logger.warning("⚠️  缺少 j_threshold_bbi 配置，使用默认值 %d", bbi_j_threshold)
                        
                    sb1_j_threshold = b1_params.get('j_threshold_sb1', None)
                    if sb1_j_threshold is None:
                        sb1_j_threshold = 5
                        logger.warning("⚠️  缺少 j_threshold_sb1 配置，使用默认值 %d", sb1_j_threshold)
                        
                    bbi_j_q_threshold = b1_params.get('j_q_threshold_bbi', None)
                    if bbi_j_q_threshold is None:
                        bbi_j_q_threshold = 0.05
                        logger.warning("⚠️  缺少 j_q_threshold_bbi 配置，使用默认值 %.3f", bbi_j_q_threshold)
                        
                    sb1_j_q_threshold = b1_params.get('j_q_threshold_sb1', None)
                    if sb1_j_q_threshold is None:
                        sb1_j_q_threshold = 0.10
                        logger.warning("⚠️  缺少 j_q_threshold_sb1 配置，使用默认值 %.3f", sb1_j_q_threshold)
                    
                    # 计算历史J值分位数 (使用较宽的时间窗口)
                    max_window = 60
                    j_window = kdj_result['J'].tail(max_window).dropna()
                    
                    # 分别检查BBIKDJSelector和SuperB1的条件
                    meets_bbi_condition = False
                    meets_sb1_condition = False
                    
                    if not j_window.empty:
                        bbi_j_quantile = float(j_window.quantile(bbi_j_q_threshold))
                        sb1_j_quantile = float(j_window.quantile(sb1_j_q_threshold))
                        
                        # BBIKDJSelector条件: J < threshold 或 J <= quantile
                        meets_bbi_condition = (current_j < bbi_j_threshold or current_j <= bbi_j_quantile)
                        
                        # SuperB1Selector条件: J < threshold 或 J <= quantile  
                        meets_sb1_condition = (current_j < sb1_j_threshold or current_j <= sb1_j_quantile)
                    
                    # 根据满足的条件给分
                    if meets_sb1_condition and meets_bbi_condition:  # 满足两个条件
                        score += 0.3
                    elif meets_sb1_condition or meets_bbi_condition:  # 满足任一条件
                        score += 0.25
                    elif current_j <= 20:  # 适度放宽的兜底条件
                        score += 0.15
                    elif current_j <= 30:  # 轻微超卖也给基础分
                        score += 0.1
            except Exception as e:
                logger.debug("J值计算失败: %s", e)
            
            # 2. 价格回调幅度判断 (25分) - 基于放量拉升的起点和高点
            pullback_score = self._analyze_pullback_from_volume_surge(df)
            score += pullback_score
            
            # 3. B1成交量模式检测 (20分) - 先Spike后萎缩
            score += self._detect_b1_volume_pattern(df, recent_data)
            
            # 4. 波动率收敛 (15分) - 平滑评分
            recent_volatility = recent_data['close'].pct_change().std()
            earlier_volatility = df['close'].tail(60).head(40).pct_change().std()
            
            if earlier_volatility > 0:
                volatility_ratio = recent_volatility / earlier_volatility
                
                # 平滑评分机制
                if volatility_ratio <= 0.5:  # 强烈收敛
                    volatility_score = 0.15  # 满分
                elif volatility_ratio <= 0.9:  # 平滑区间：0.5-0.9
                    # 线性递减从0.15到0.05
                    volatility_score = 0.15 - (volatility_ratio - 0.5) / (0.9 - 0.5) * (0.15 - 0.05)
                elif volatility_ratio <= 1.2:  # 轻微收敛：0.9-1.2
                    # 继续线性递减从0.05到0.02
                    volatility_score = 0.05 - (volatility_ratio - 0.9) / (1.2 - 0.9) * (0.05 - 0.02)
                else:  # 无收敛或发散
                    volatility_score = 0.0
                
                score += volatility_score
            
            # 5. 连续阴线后企稳 (10分) - 修复逻辑
            stabilization_score = self._detect_stabilization_after_consecutive_bearish(recent_data)
            score += stabilization_score
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("增强B1形态检测失败: %s", e)
            return 0.0
    
    def _detect_b1_volume_pattern(self, df: pd.DataFrame, recent_data: pd.DataFrame) -> float:
        """
        检测B1典型成交量模式：先有成交量Spike，然后萎缩
        """
        try:
            if len(df) < 60:
                return 0.0
            
            score = 0.0
            
            # 寻找前期的成交量Spike（60天内）
            volume_window = df.tail(60)
            volume_avg = volume_window['volume'].mean()
            volume_std = volume_window['volume'].std()
            
            # 寻找显著的成交量Spike（超过均值+2倍标准差）
            spike_threshold = volume_avg + 2 * volume_std if volume_std > 0 else volume_avg * 2
            spike_days = volume_window[volume_window['volume'] > spike_threshold]
            
            if len(spike_days) > 0:
                # 找到最近的Spike日期
                latest_spike_date = spike_days.index[-1]
                
                # 检查Spike后是否有萎缩
                spike_idx = volume_window.index.get_loc(latest_spike_date)
                post_spike_data = volume_window.iloc[spike_idx+1:]
                
                if len(post_spike_data) >= 5:
                    # 计算Spike后的成交量萎缩
                    post_spike_avg = post_spike_data['volume'].tail(5).mean()
                    spike_volume = volume_window.loc[latest_spike_date, 'volume']
                    
                    if spike_volume > 0:
                        contraction_ratio = post_spike_avg / spike_volume
                        
                        # 根据萎缩程度给分
                        if contraction_ratio <= 0.3:  # 萎缩70%以上
                            score += 0.2  # 满分
                        elif contraction_ratio <= 0.5:  # 萎缩50%以上
                            score += 0.15
                        elif contraction_ratio <= 0.7:  # 萎缩30%以上
                            score += 0.1
                        
                        # 检查Spike的强度
                        spike_ratio = spike_volume / volume_avg if volume_avg > 0 else 0
                        if spike_ratio >= 3.0:  # Spike是平均量的3倍以上
                            score += 0.05  # 额外加分
            
            return min(score, 0.2)  # 最高20分
            
        except Exception as e:
            logger.debug("B1成交量模式检测失败: %s", e)
            return 0.0
    
    def _analyze_pullback_from_volume_surge(self, df: pd.DataFrame) -> float:
        """
        分析基于放量拉升的价格回调幅度
        
        逻辑：
        1. 检测上一波放量拉升的起点和高点
        2. 当前价格在这个范围的中间点以下 → 0.6合格
        3. 当前价格在这个范围的0.3以下 → 满分
        4. 如果Spike后已经出现过低点在中间点以下 → 也给合格
        """
        try:
            if len(df) < 60:
                return 0.0
            
            lookback_window = df.tail(60)
            current_price = df['close'].iloc[-1]
            
            # 1. 寻找最近的放量拉升段
            volume_surge_info = self._find_latest_volume_surge(lookback_window)
            
            if volume_surge_info is None:
                return 0.1  # 找不到放量拉升段，给基础分
            
            surge_start_price, surge_high_price, surge_end_idx = volume_surge_info
            
            if surge_high_price <= surge_start_price:
                return 0.1  # 价格范围无效
            
            # 2. 计算关键位置
            price_range = surge_high_price - surge_start_price
            mid_point = surge_start_price + price_range * 0.5  # 中间点
            deep_pullback_level = surge_start_price + price_range * 0.3  # 30%位置
            
            score = 0.0
            
            # 3. 检查当前价格位置 - 平滑评分机制
            price_position_ratio = (current_price - surge_start_price) / price_range
            
            if price_position_ratio <= 0.3:  # 在30%以下
                score += 0.25  # 满分
            elif price_position_ratio <= 0.6:  # 在60%以下（放宽中点）
                # 平滑评分：30%-60%之间线性递减
                smooth_score = 0.25 - (price_position_ratio - 0.3) / (0.6 - 0.3) * (0.25 - 0.18)
                score += smooth_score
            elif price_position_ratio <= 0.8:  # 在80%以下
                # 继续平滑评分：60%-80%之间
                smooth_score = 0.18 - (price_position_ratio - 0.6) / (0.8 - 0.6) * (0.18 - 0.1)
                score += smooth_score
            elif price_position_ratio <= 1.0:  # 在100%以下
                # 最后段平滑：80%-100%之间
                smooth_score = 0.1 - (price_position_ratio - 0.8) / (1.0 - 0.8) * 0.05
                score += max(smooth_score, 0.05)  # 最低给5分
            else:
                score += 0.05  # 超出区间仍给基础分
            
            # 4. 检查Spike后是否出现过深度回调
            post_surge_data = lookback_window.iloc[surge_end_idx+1:]
            if len(post_surge_data) >= 3:
                post_surge_low = post_surge_data['low'].min()
                if post_surge_low <= mid_point:
                    score += 0.08  # 额外加分：已经出现过深度回调
            
            return min(score, 0.25)
            
        except Exception as e:
            logger.debug("价格回调分析失败: %s", e)
            return 0.1
    
    def _find_latest_volume_surge(self, df: pd.DataFrame):
        """
        寻找最近的放量拉升段
        
        Returns:
            tuple: (起点价格, 高点价格, 结束索引) 或 None
        """
        try:
            if len(df) < 20:
                return None
            
            # 计算成交量异常阈值
            volume_mean = df['volume'].mean()
            volume_std = df['volume'].std()
            volume_threshold = volume_mean + 1.5 * volume_std  # 成交量突增阈值
            
            # 寻找成交量异常日
            volume_spikes = df[df['volume'] > volume_threshold]
            
            if len(volume_spikes) == 0:
                return None
            
            # 找到最近的成交量异常日
            latest_spike_idx = df.index.get_loc(volume_spikes.index[-1])
            
            # 从成交量异常日向前寻找拉升起点 (成交量相对平静的起始点)
            start_idx = max(0, latest_spike_idx - 15)  # 向前最多看15天
            
            # 寻找起点：成交量相对较小且价格相对较低的点
            search_window = df.iloc[start_idx:latest_spike_idx+1]
            
            if len(search_window) < 3:
                return None
            
            # 找到成交量最小的几天中价格最低的点作为起点
            volume_threshold_low = search_window['volume'].quantile(0.3)  # 成交量30%分位
            low_volume_days = search_window[search_window['volume'] <= volume_threshold_low]
            
            if len(low_volume_days) == 0:
                surge_start_idx = start_idx
            else:
                surge_start_idx = df.index.get_loc(low_volume_days['low'].idxmin())
            
            # 从异常日向后寻找高点 (价格峰值)
            end_search_idx = min(len(df) - 1, latest_spike_idx + 10)  # 向后最多看10天
            surge_window = df.iloc[surge_start_idx:end_search_idx+1]
            
            surge_high_idx = df.index.get_loc(surge_window['high'].idxmax())
            
            # 提取关键价格
            surge_start_price = df.iloc[surge_start_idx]['close']
            surge_high_price = df.iloc[surge_high_idx]['high']
            
            return (surge_start_price, surge_high_price, surge_high_idx)
            
        except Exception as e:
            logger.debug("寻找放量拉升段失败: %s", e)
            return None
    
    def _detect_stabilization_after_consecutive_bearish(self, recent_data: pd.DataFrame) -> float:
        """
        检测连续阴线调整后的企稳信号
        
        逻辑：
        1. 在最近的数据中寻找连续阴线段
        2. 检查阴线段后是否出现企稳信号（阳线、缩量、价格企稳等）
        3. 重点关注调整后的企稳，而不是要求当前日必须在连续阴线中
        """
        try:
            if len(recent_data) < 5:
                return 0.0
            
            data_array = recent_data.copy()
            data_array['is_bearish'] = data_array['close'] < data_array['open']  # 阴线标记
            
            max_score = 0.0
            
            # 从近到远搜索连续阴线段
            for start_idx in range(len(data_array) - 2, 2, -1):  # 至少留2天看企稳
                # 向前查找连续阴线
                consecutive_bearish = 0
                search_idx = start_idx
                
                while search_idx >= 0 and data_array.iloc[search_idx]['is_bearish']:
                    consecutive_bearish += 1
                    search_idx -= 1
                
                # 如果找到足够的连续阴线
                if consecutive_bearish >= 2:
                    # 检查阴线段后的企稳信号
                    stabilization_score = self._evaluate_stabilization_signals(
                        data_array, start_idx + 1, consecutive_bearish
                    )
                    max_score = max(max_score, stabilization_score)
                    
                    # 如果已经找到高分，可以提前结束
                    if max_score >= 0.1:
                        break
            
            return min(max_score, 0.1)
            
        except Exception as e:
            logger.debug("连续阴线企稳检测失败: %s", e)
            return 0.0
    
    def _evaluate_stabilization_signals(self, data_array: pd.DataFrame, 
                                       after_bearish_idx: int, consecutive_count: int) -> float:
        """
        评估连续阴线后的企稳信号强度
        
        Args:
            data_array: 数据数组
            after_bearish_idx: 连续阴线结束后的索引
            consecutive_count: 连续阴线数量
        """
        try:
            if after_bearish_idx >= len(data_array):
                return 0.0
            
            score = 0.0
            
            # 基础分：根据连续阴线数量
            if consecutive_count >= 4:
                base_score = 0.06  # 4根以上阴线调整后企稳
            elif consecutive_count >= 3:
                base_score = 0.05  # 3根阴线调整后企稳
            elif consecutive_count >= 2:
                base_score = 0.03  # 2根阴线调整后企稳
            else:
                return 0.0
            
            score += base_score
            
            # 检查企稳信号的强度
            post_bearish_data = data_array.iloc[after_bearish_idx:]
            
            if len(post_bearish_data) >= 1:
                # 1. 立即企稳：阴线后第一天是阳线
                first_day = post_bearish_data.iloc[0]
                if not first_day['is_bearish']:  # 阳线
                    score += 0.02
                
                # 2. 成交量配合：企稳时成交量相对温和
                if len(post_bearish_data) >= 2:
                    recent_volumes = post_bearish_data['volume'].head(2).values
                    bearish_volumes = data_array.iloc[after_bearish_idx-consecutive_count:after_bearish_idx]['volume'].values
                    
                    if len(bearish_volumes) > 0 and len(recent_volumes) > 0:
                        bearish_avg_vol = bearish_volumes.mean()
                        recent_avg_vol = recent_volumes.mean()
                        
                        # 企稳期成交量适度萎缩是好信号
                        if bearish_avg_vol > 0 and recent_avg_vol / bearish_avg_vol <= 0.8:
                            score += 0.01
                
                # 3. 价格企稳：没有继续大幅下跌
                if len(post_bearish_data) >= 2:
                    bearish_low = data_array.iloc[after_bearish_idx-consecutive_count:after_bearish_idx]['low'].min()
                    post_low = post_bearish_data.head(2)['low'].min()
                    
                    # 企稳期没有创新低
                    if post_low >= bearish_low * 0.98:  # 允许2%的波动
                        score += 0.02
            
            return min(score, 0.1)
            
        except Exception as e:
            logger.debug("企稳信号评估失败: %s", e)
            return 0.0
    
    def _detect_red_fat_green_thin(self, df: pd.DataFrame) -> float:
        """
        检测"红肥绿瘦"量价配合形态
        """
        try:
            params = self.config.get('red_fat_green_thin', {})
            lookback = params.get('lookback', 20)
            
            if len(df) < lookback:
                return 0.0
            
            recent_data = df.tail(lookback)
            score = 0.0
            
            # 分离红绿K线
            red_days = recent_data[recent_data['close'] > recent_data['open']]  # 阳线
            green_days = recent_data[recent_data['close'] <= recent_data['open']]  # 阴线
            
            if len(red_days) == 0 or len(green_days) == 0:
                return 0.0
            
            # 1. 成交量对比 (60分)
            red_avg_volume = red_days['volume'].mean()
            green_avg_volume = green_days['volume'].mean()
            
            if green_avg_volume > 0:
                volume_ratio = red_avg_volume / green_avg_volume
                if volume_ratio >= 2.0:  # 红肥绿瘦明显
                    score += 0.6
                elif volume_ratio >= 1.5:  # 红肥绿瘦
                    score += 0.4
                elif volume_ratio >= 1.2:  # 轻微红肥绿瘦
                    score += 0.2
            
            # 2. 实体大小对比 (40分)
            red_body_avg = (red_days['close'] - red_days['open']).mean()
            green_body_avg = (green_days['open'] - green_days['close']).mean()
            
            if green_body_avg > 0:
                body_ratio = red_body_avg / green_body_avg
                if body_ratio >= 1.5:  # 红实体明显大于绿实体
                    score += 0.4
                elif body_ratio >= 1.2:  # 红实体大于绿实体
                    score += 0.25
                elif body_ratio >= 1.0:  # 红实体不小于绿实体
                    score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("红肥绿瘦检测失败: %s", e)
            return 0.0
    
    def _detect_high_volatility_elasticity(self, df: pd.DataFrame) -> float:
        """
        检测大开大合高波动弹性 - 重点检测涨跌停板等极端波动
        """
        return min(float(sum(self._detect_high_volatility_elasticity_parts(df).values())), 1.0)

    def _detect_high_volatility_elasticity_parts(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        拆解大开大合高波动弹性子项。
        """
        try:
            if len(df) < 30:
                return {
                    "extreme_move": 0.0,
                    "intraday_range": 0.0,
                    "big_move_density": 0.0,
                }
            
            recent_data = df.tail(60)  # 扩大观察窗口到60天，捕获更多历史极端波动
            
            # 1. 涨跌停板检测 (60分) - 核心优化
            extreme_move_count = 0
            # 修复：使用shift方法正确计算前一日收盘价
            recent_data_copy = recent_data.copy()
            recent_data_copy['prev_close'] = recent_data_copy['close'].shift(1)
            recent_data_copy['daily_return'] = (recent_data_copy['close'] - recent_data_copy['prev_close']) / recent_data_copy['prev_close']
            
            for _, row in recent_data_copy.iterrows():
                if not pd.isna(row['daily_return']):
                    # 检测±9%以上的极端波动（接近涨跌停）
                    if abs(row['daily_return']) >= 0.09:
                        extreme_move_count += 1
            
            # 根据极端波动天数给分
            extreme_move_score = 0.0
            if extreme_move_count >= 5:  # 5根以上涨跌停板
                extreme_move_score = 0.6  # 满分
            elif extreme_move_count >= 3:  # 3-4根
                extreme_move_score = 0.4
            elif extreme_move_count >= 1:  # 1-2根
                extreme_move_score = 0.3  # 基础分
            
            # 2. 日内振幅分析 (25分)
            daily_ranges = (recent_data['high'] - recent_data['low']) / recent_data['close']
            avg_range = daily_ranges.mean()
            
            intraday_range_score = 0.0
            if avg_range >= 0.08:  # 日均振幅8%以上
                intraday_range_score = 0.25
            elif avg_range >= 0.06:  # 日均振幅6%以上
                intraday_range_score = 0.15
            elif avg_range >= 0.04:  # 日均振幅4%以上
                intraday_range_score = 0.1
            
            # 3. 波动密集度 (15分)
            big_move_days = len(daily_ranges[daily_ranges >= 0.07])  # 振幅7%以上
            big_move_ratio = big_move_days / len(daily_ranges)
            
            big_move_density_score = 0.0
            if big_move_ratio >= 0.4:  # 40%以上为大波动日
                big_move_density_score = 0.15
            elif big_move_ratio >= 0.25:  # 25%以上为大波动日
                big_move_density_score = 0.1
            elif big_move_ratio >= 0.15:  # 15%以上为大波动日
                big_move_density_score = 0.05
            
            return {
                "extreme_move": float(extreme_move_score),
                "intraday_range": float(intraday_range_score),
                "big_move_density": float(big_move_density_score),
            }
            
        except Exception as e:
            logger.debug("高波动弹性检测失败: %s", e)
            return {
                "extreme_move": 0.0,
                "intraday_range": 0.0,
                "big_move_density": 0.0,
            }
    
    def _analyze_dealer_behavior(self, df: pd.DataFrame) -> float:
        """
        分析庄家行为（洗盘 vs 出货）
        """
        try:
            # 整合洗盘和出货检测
            washout_score = self._detect_dealer_washout(df)
            distribution_penalty = self._detect_dealer_distribution(df)
            
            # 洗盘是正面信号，出货是负面信号
            net_score = washout_score - distribution_penalty
            
            return max(-1.0, min(net_score, 1.0))
            
        except Exception as e:
            logger.debug("庄家行为分析失败: %s", e)
            return 0.0
    
    def _detect_dealer_washout(self, df: pd.DataFrame) -> float:
        """
        检测庄家洗盘行为
        """
        try:
            if len(df) < 30:
                return 0.0
            
            score = 0.0
            recent_data = df.tail(15)
            
            # 1. 长上影线洗盘 (30分)
            for _, row in recent_data.iterrows():
                upper_shadow = row['high'] - max(row['open'], row['close'])
                full_range = row['high'] - row['low']
                
                if full_range > 0:
                    upper_shadow_ratio = upper_shadow / full_range
                    if upper_shadow_ratio >= 0.5:  # 长上影线
                        score += 0.05  # 每根长上影线加5分，最多30分
            
            score = min(score, 0.3)
            
            # 2. 快速下跌后企稳 (40分)
            if len(df) >= 10:
                pre_crash = df.tail(20).head(10)
                post_crash = recent_data
                
                pre_avg = pre_crash['close'].mean()
                crash_low = post_crash['low'].min()
                current_price = recent_data['close'].iloc[-1]
                
                if pre_avg > 0:
                    crash_depth = (pre_avg - crash_low) / pre_avg
                    recovery_ratio = (current_price - crash_low) / (pre_avg - crash_low) if pre_avg > crash_low else 0
                    
                    if crash_depth >= 0.15 and recovery_ratio >= 0.3:  # 深跌15%后恢复30%
                        score += 0.4
                    elif crash_depth >= 0.1 and recovery_ratio >= 0.2:  # 跌10%后恢复20%
                        score += 0.25
            
            # 3. 缩量洗盘 (30分)
            if len(recent_data) >= 10:
                recent_volume = recent_data['volume'].tail(5).mean()
                earlier_volume = recent_data['volume'].head(5).mean()
                
                if earlier_volume > 0:
                    volume_shrink = recent_volume / earlier_volume
                    if volume_shrink <= 0.5:  # 成交量萎缩50%
                        score += 0.3
                    elif volume_shrink <= 0.7:  # 成交量萎缩30%
                        score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("庄家洗盘检测失败: %s", e)
            return 0.0
    
    def _detect_dealer_distribution(self, df: pd.DataFrame) -> float:
        """
        检测庄家出货行为
        """
        try:
            if len(df) < 20:
                return 0.0
            
            score = 0.0
            recent_data = df.tail(15)
            
            # 1. 高位放量滞涨 (50分)
            price_window = df.tail(60)
            current_price = recent_data['close'].iloc[-1]
            price_high = price_window['high'].max()
            
            if price_high > 0:
                price_position = current_price / price_high
                if price_position >= 0.8:  # 在高位80%以上
                    recent_volume = recent_data['volume'].tail(5).mean()
                    earlier_volume = df['volume'].tail(30).head(20).mean()
                    
                    if earlier_volume > 0:
                        volume_ratio = recent_volume / earlier_volume
                        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-5]) / recent_data['close'].iloc[-5]
                        
                        if volume_ratio >= 1.5 and abs(price_change) <= 0.03:  # 放量但价格滞涨
                            score += 0.5
                        elif volume_ratio >= 1.2 and abs(price_change) <= 0.05:
                            score += 0.3
            
            # 2. 连续放量阴线 (30分)
            consecutive_volume_down = 0
            volumes = recent_data['volume'].values
            opens = recent_data['open'].values
            closes = recent_data['close'].values
            
            avg_volume = df['volume'].tail(30).mean()
            
            for i in range(len(volumes) - 1, -1, -1):
                if closes[i] < opens[i] and volumes[i] > avg_volume * 1.2:  # 放量阴线
                    consecutive_volume_down += 1
                else:
                    break
            
            if consecutive_volume_down >= 3:
                score += 0.3
            elif consecutive_volume_down >= 2:
                score += 0.2
            
            # 3. 破位下跌 (20分)
            if len(df) >= 60:
                support_level = df.tail(60)['low'].quantile(0.2)  # 近60天的20分位低点
                recent_low = recent_data['low'].min()
                
                if recent_low < support_level:
                    score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.debug("庄家出货检测失败: %s", e)
            return 0.0
    
    def _detect_sb1_pattern(self, df: pd.DataFrame) -> float:
        """
        检测SB1连续形态（Super B1）
        """
        try:
            if len(df) < 40:
                return 0.0
            
            score = 0.0
            
            # 检查是否有多个B1信号
            # 这里简化实现，检查多个时间窗口的B1特征
            windows = [df.tail(20), df.tail(30).head(20), df.tail(40).head(20)]
            b1_signals = 0
            
            for window in windows:
                if len(window) >= 10:
                    # 简化的B1检测逻辑
                    try:
                        k, d, j = compute_kdj(window['high'], window['low'], window['close'])
                        if len(j) > 0 and j.iloc[-1] <= 20:  # J值超卖
                            price_position = (window['close'].iloc[-1] - window['low'].min()) / (window['high'].max() - window['low'].min())
                            if price_position <= 0.3:  # 在底部30%
                                b1_signals += 1
                    except:
                        pass
            
            if b1_signals >= 2:  # 多个窗口都有B1信号
                score = 0.5
            elif b1_signals >= 1:
                score = 0.2
            
            return score
            
        except Exception as e:
            logger.debug("SB1形态检测失败: %s", e)
            return 0.0
