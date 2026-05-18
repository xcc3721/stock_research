from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
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
        # Use precomputed indicators if available; otherwise compute on the fly
        if "BBI" not in hist.columns:
            hist = hist.copy()
            hist["BBI"] = compute_bbi(hist)
        if "J" not in hist.columns:
            hist = hist.copy()
            kdj = compute_kdj(hist)
            hist["J"] = kdj["J"]
        if "DIF" not in hist.columns:
            hist = hist.copy()
            hist["DIF"] = compute_dif(hist)

        close_vals = hist["close"].values
        n = len(close_vals)
        window_len = min(n, self.config.max_window)
        window_closes = close_vals[-window_len:]
        high = float(window_closes.max())
        low = float(window_closes.min())
        if low <= 0.0 or (high / low - 1.0) > float(self.config.price_range_pct):
            return False

        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=int(self.config.bbi_min_window),
            max_window=int(self.config.max_window),
            q_threshold=float(self.config.bbi_q_threshold),
        ):
            return False

        j_vals = hist["J"].values
        j_today = float(j_vals[-1])
        j_window = j_vals[-window_len:]
        j_quantile = float(np.quantile(j_window, float(self.config.j_q_threshold)))
        if not (j_today < float(self.config.j_threshold) or j_today <= j_quantile):
            return False

        dif_vals = hist["DIF"].values
        return float(dif_vals[-1]) > 0.0

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
