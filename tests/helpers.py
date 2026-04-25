from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_default_scoring_config() -> dict[str, Any]:
    payload = json.loads((REPO_ROOT / "configs" / "runtime.json").read_text(encoding="utf-8"))
    scoring_config = payload.get("scoring_config")
    if not isinstance(scoring_config, dict):
        raise ValueError("缺少默认 scoring_config")
    return scoring_config


def make_price_frame(
    *,
    start_date: str,
    periods: int,
    base_price: float,
    drift: float,
    phase: float,
    amplitude: float,
) -> pd.DataFrame:
    dates = pd.bdate_range(start_date, periods=periods)
    rows: list[dict[str, Any]] = []
    for index, trade_date in enumerate(dates):
        close = base_price + drift * index + amplitude * math.sin(index / 9.0 + phase)
        open_price = close * 0.99
        rows.append(
            {
                "date": trade_date.date().isoformat(),
                "open": open_price,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": 1_000_000 + index * 1_000,
                "amount": (1_000_000 + index * 1_000) * close,
            }
        )
    return pd.DataFrame(rows)


def write_price_csv(data_dir: Path, code: str, frame: pd.DataFrame) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"{code}.csv"
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def make_report_payload(trade_date: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_id": f"bbikdj_{trade_date.replace('-', '')}",
        "generated_at": "2026-04-24T00:00:00Z",
        "data": {
            "config_path": "configs/runtime.json",
            "summary": {
                "trade_date": trade_date,
                "selected_stocks": len(rows),
                "universe_size": len(rows),
                "strategy": "BBIKDJSelector",
            },
            "scoring_results": {
                "trade_date": trade_date,
                "strategy": "BBIKDJSelector",
                "batch_results": rows,
            },
        },
    }
