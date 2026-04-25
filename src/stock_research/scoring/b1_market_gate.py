#!/usr/bin/env python3
"""Optional external market-state gate resolver for B1 scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.scoring.b1_policy import extract_b1_params


LOGGER = logging.getLogger("b1_market_gate")
ROOT = Path(__file__).resolve().parents[3]


class B1MarketGateResolver:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.params = extract_b1_params(self.config)
        self._frame: pd.DataFrame | None = None
        self._load_failed = False
        self._warned = False

    def resolve_for_frame(self, df: pd.DataFrame | None) -> dict[str, Any]:
        trade_date = self._resolve_trade_date(df)
        if trade_date is None:
            return {"available": False, "is_hot_up": False, "reason": "missing_trade_date"}
        return self.resolve_for_date(trade_date)

    def resolve_for_date(self, trade_date: Any) -> dict[str, Any]:
        date_value = pd.to_datetime(trade_date, errors="coerce")
        if pd.isna(date_value):
            return {"available": False, "is_hot_up": False, "reason": "invalid_trade_date"}
        frame = self._load_frame()
        if frame is None or frame.empty:
            return {"available": False, "is_hot_up": False, "reason": "market_gate_disabled"}
        normalized_date = pd.Timestamp(date_value).normalize()
        matched = frame.loc[frame["date"] <= normalized_date].tail(1)
        if matched.empty:
            return {
                "available": False,
                "is_hot_up": False,
                "trade_date": normalized_date.strftime("%Y-%m-%d"),
                "reason": "before_first_market_gate_date",
            }
        row = matched.iloc[0]
        return {
            "available": True,
            "is_hot_up": bool(row["is_hot_up"]),
            "trade_date": normalized_date.strftime("%Y-%m-%d"),
            "market_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "heat_regime": str(row.get("heat_regime", "")),
        }

    def _resolve_trade_date(self, df: pd.DataFrame | None) -> pd.Timestamp | None:
        if df is None or len(df) == 0:
            return None
        if "date" in df.columns:
            values = pd.to_datetime(df["date"], errors="coerce").dropna()
            if not values.empty:
                return pd.Timestamp(values.iloc[-1]).normalize()
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            return pd.Timestamp(df.index[-1]).normalize()
        return None

    def _load_frame(self) -> pd.DataFrame | None:
        if self._frame is not None:
            return self._frame
        if self._load_failed:
            return None
        path = self._resolve_gate_path()
        if path is None:
            self._load_failed = True
            return None
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            self._load_failed = True
            self._warn_once("market gate 文件读取失败 %s: %s", path, exc)
            return None

        gate_cfg = self._gate_config()
        date_col = str(gate_cfg.get("date_col", "date"))
        active_col = str(gate_cfg.get("active_col", "is_hot_up"))
        regime_col = str(gate_cfg.get("regime_col", "heat_regime"))
        required = {date_col, active_col}
        if required.difference(frame.columns):
            self._load_failed = True
            self._warn_once("market gate 文件缺少必要列: %s", sorted(required.difference(frame.columns)))
            return None

        output = pd.DataFrame()
        output["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
        output["is_hot_up"] = frame[active_col].map(self._to_bool)
        output["heat_regime"] = frame[regime_col].astype(str) if regime_col in frame.columns else ""
        output = output.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        self._frame = output
        return self._frame

    def _gate_config(self) -> dict[str, Any]:
        gate_cfg = self.params.get("market_gate")
        return gate_cfg if isinstance(gate_cfg, dict) else {}

    def _resolve_gate_path(self) -> Path | None:
        raw_path = self._gate_config().get("path")
        if not raw_path:
            return None
        path = Path(str(raw_path)).expanduser()
        if path.is_absolute():
            return path
        root_path = (ROOT / path).resolve()
        if root_path.exists():
            return root_path
        data_dir = self.config.get("data_dir")
        if data_dir:
            data_path = (Path(str(data_dir)).expanduser() / path).resolve()
            if data_path.exists():
                return data_path
        return root_path

    def _warn_once(self, message: str, *args: Any) -> None:
        if not self._warned:
            LOGGER.warning(message, *args)
            self._warned = True

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"1", "true", "yes", "y", "on", "hot", "hot_up"}
