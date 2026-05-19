from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    if df.empty:
        return df.assign(K=np.nan, D=np.nan, J=np.nan)

    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100.0

    rsv_vals = rsv.to_numpy(dtype=float)
    k = np.zeros_like(rsv_vals, dtype=float)
    d = np.zeros_like(rsv_vals, dtype=float)
    k[0] = 50.0
    d[0] = 50.0
    for idx in range(1, len(rsv_vals)):
        k[idx] = 2.0 / 3.0 * k[idx - 1] + 1.0 / 3.0 * rsv_vals[idx]
        d[idx] = 2.0 / 3.0 * d[idx - 1] + 1.0 / 3.0 * k[idx]
    j = 3.0 * k - 2.0 * d
    return df.assign(K=k, D=d, J=j)


def compute_bbi(df: pd.DataFrame) -> pd.Series:
    ma3 = df["close"].rolling(3).mean()
    ma6 = df["close"].rolling(6).mean()
    ma12 = df["close"].rolling(12).mean()
    ma24 = df["close"].rolling(24).mean()
    return (ma3 + ma6 + ma12 + ma24) / 4.0


def compute_dif(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def bbi_deriv_uptrend(
    bbi: pd.Series,
    *,
    min_window: int,
    max_window: int | None = None,
    q_threshold: float = 0.0,
) -> bool:
    if not 0.0 <= q_threshold <= 1.0:
        raise ValueError("q_threshold 必须位于 [0, 1] 区间内")

    clean = bbi.dropna()
    if len(clean) < min_window:
        return False

    vals = clean.values
    n = len(vals)
    longest = min(n, max_window or n)
    for window in range(longest, min_window - 1, -1):
        segment = vals[-window:]
        norm = segment / segment[0]
        diffs = np.diff(norm)
        if np.quantile(diffs, q_threshold) >= 0.0:
            return True
    return False
