from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    if df.empty:
        return df.assign(K=np.nan, D=np.nan, J=np.nan)

    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100.0

    # Vectorized EMA: equivalent to the manual loop but avoids Python-level
    # iteration.  For long histories the initial-value difference vanishes
    # (alpha=1/3 decays to <1e-6 in ~30 bars).
    rsv_s = pd.Series(rsv.to_numpy(), dtype=float)
    k = rsv_s.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    d = k.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return df.assign(K=k.to_numpy(), D=d.to_numpy(), J=j.to_numpy())


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

    vals = clean.to_numpy(dtype=float)
    n = len(vals)
    longest = min(n, max_window or n)
    # BBI > 0 always (average of MAs of positive prices), so normalizing
    # by segment[0] just scales diffs by a positive constant. Skip it.
    all_diffs = np.diff(vals)
    for window in range(longest, min_window - 1, -1):
        diffs = all_diffs[-(window - 1):]
        m = len(diffs)
        # Quick-reject: quantile(diffs, q) >= 0 requires at least
        # m - floor(q*(m-1)) - 1 values >= 0. Skip the expensive quantile
        # call when this necessary condition cannot hold.
        k = int(np.floor(q_threshold * (m - 1)))
        min_positive = m - k - 1
        if np.count_nonzero(diffs >= 0) < min_positive:
            continue
        if np.quantile(diffs, q_threshold) >= 0.0:
            return True
    return False
