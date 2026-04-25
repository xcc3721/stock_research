from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from stock_research.selectors.indicators import (
    bbi_deriv_uptrend,
    compute_bbi,
    compute_dif,
    compute_kdj,
)


@dataclass(frozen=True)
class BBIKDJSelectorConfig:
    j_threshold: float = 13.0
    bbi_min_window: int = 20
    max_window: int = 120
    price_range_pct: float = 10.0
    bbi_q_threshold: float = 0.3
    j_q_threshold: float = 0.1


class BBIKDJSelector:
    def __init__(self, config: BBIKDJSelectorConfig | None = None) -> None:
        self.config = config or BBIKDJSelectorConfig()

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        work = hist.copy()
        work["BBI"] = compute_bbi(work)

        window = work.tail(self.config.max_window)
        high = float(window["close"].max())
        low = float(window["close"].min())
        if low <= 0.0 or (high / low - 1.0) > float(self.config.price_range_pct):
            return False

        if not bbi_deriv_uptrend(
            work["BBI"],
            min_window=int(self.config.bbi_min_window),
            max_window=int(self.config.max_window),
            q_threshold=float(self.config.bbi_q_threshold),
        ):
            return False

        kdj = compute_kdj(work)
        j_today = float(kdj.iloc[-1]["J"])
        j_window = kdj["J"].tail(int(self.config.max_window)).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(float(self.config.j_q_threshold)))
        if not (j_today < float(self.config.j_threshold) or j_today <= j_quantile):
            return False

        work["DIF"] = compute_dif(work)
        return float(work["DIF"].iloc[-1]) > 0.0

    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        picks: List[str] = []
        for code, frame in data.items():
            hist = frame[frame["date"] <= date]
            if hist.empty:
                continue
            hist = hist.tail(int(self.config.max_window) + 20)
            if self._passes_filters(hist):
                picks.append(str(code))
        return sorted(picks)
