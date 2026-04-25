from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import pandas as pd

from tests.helpers import REPO_ROOT, load_default_scoring_config, make_price_frame, write_price_csv


@pytest.fixture()
def synthetic_workspace(tmp_path: Path) -> dict[str, Any]:
    data_dir = tmp_path / "daily_data"
    reports_dir = tmp_path / "reports"
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "000001": make_price_frame(
            start_date="2025-01-01",
            periods=160,
            base_price=10.0,
            drift=0.030,
            phase=0.0,
            amplitude=0.12,
        ),
        "000002": make_price_frame(
            start_date="2025-01-01",
            periods=160,
            base_price=9.5,
            drift=0.028,
            phase=0.5,
            amplitude=0.10,
        ),
        "000003": make_price_frame(
            start_date="2025-01-01",
            periods=160,
            base_price=8.8,
            drift=0.022,
            phase=1.0,
            amplitude=0.05,
        ),
    }
    for code, frame in frames.items():
        write_price_csv(data_dir, code, frame)

    runtime_config = {
        "selector": {
            "j_threshold": 200,
            "bbi_min_window": 10,
            "max_window": 60,
            "price_range_pct": 20,
            "bbi_q_threshold": 0.0,
            "j_q_threshold": 1.0,
        },
        "scoring_config": load_default_scoring_config(),
    }
    runtime_config_path = config_dir / "runtime.test.json"
    runtime_config_path.write_text(json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "root": tmp_path,
        "data_dir": data_dir,
        "reports_dir": reports_dir,
        "config_dir": config_dir,
        "runtime_config_path": runtime_config_path,
        "a2_config_path": REPO_ROOT / "configs" / "a2" / "flat_v1_stable_penalty_overlay.json",
        "trade_dates": pd.bdate_range("2025-01-01", periods=160),
    }
