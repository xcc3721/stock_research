from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from stock_research.backtest.standard_execution import (
    StandardExecutionConfig,
    main as backtest_main,
    run_standard_execution,
)
from tests.helpers import make_report_payload, write_price_csv


def test_standard_execution_cli_accepts_report_dir(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    price_dir = tmp_path / "daily_data"
    output_dir = tmp_path / "backtest"
    report_dir.mkdir(parents=True, exist_ok=True)

    for trade_date, score_a, score_b in [
        ("2025-01-02", 0.80, 0.60),
        ("2025-01-03", 0.70, 0.90),
        ("2025-01-06", 0.85, 0.65),
    ]:
        rows = [
            {
                "code": "000001",
                "name": "000001",
                "final_score": 60.0,
                "weighted_scores": {"ma_structure": 1.0},
                "factors": {"ma_structure": 0.6},
                "result": {"total_score": 60.0, "action_text": "买入", "weighted_scores": {"ma_structure": 1.0}, "factors": {"ma_structure": 0.6}},
                "A2_flat": {"score": score_a},
                "a2_flat_v1_score": score_a,
            },
            {
                "code": "000002",
                "name": "000002",
                "final_score": 55.0,
                "weighted_scores": {"ma_structure": 0.8},
                "factors": {"ma_structure": 0.5},
                "result": {"total_score": 55.0, "action_text": "买入", "weighted_scores": {"ma_structure": 0.8}, "factors": {"ma_structure": 0.5}},
                "A2_flat": {"score": score_b},
                "a2_flat_v1_score": score_b,
            },
        ]
        payload = make_report_payload(trade_date, rows)
        (report_dir / f"bbikdj_{trade_date.replace('-', '')}.report_data.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    price_frame_a = pd.DataFrame(
        [
            {"date": "2025-01-02", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1},
            {"date": "2025-01-03", "open": 10.1, "high": 10.4, "low": 10.0, "close": 10.3},
            {"date": "2025-01-06", "open": 10.3, "high": 10.5, "low": 10.2, "close": 10.4},
            {"date": "2025-01-07", "open": 10.4, "high": 10.8, "low": 10.3, "close": 10.7},
            {"date": "2025-01-08", "open": 10.7, "high": 11.0, "low": 10.6, "close": 10.9},
        ]
    )
    price_frame_b = pd.DataFrame(
        [
            {"date": "2025-01-02", "open": 20.0, "high": 20.2, "low": 19.8, "close": 19.9},
            {"date": "2025-01-03", "open": 19.9, "high": 20.0, "low": 19.6, "close": 19.7},
            {"date": "2025-01-06", "open": 19.7, "high": 19.8, "low": 19.2, "close": 19.4},
            {"date": "2025-01-07", "open": 19.4, "high": 19.5, "low": 19.0, "close": 19.1},
            {"date": "2025-01-08", "open": 19.1, "high": 19.2, "low": 18.8, "close": 18.9},
        ]
    )
    write_price_csv(price_dir, "000001", price_frame_a)
    write_price_csv(price_dir, "000002", price_frame_b)

    old_argv = sys.argv
    sys.argv = [
        "stock-research-backtest",
        "--report-dir",
        str(report_dir),
        "--price-data-dir",
        str(price_dir),
        "--output-dir",
        str(output_dir),
        "--score-col",
        "a2_flat_v1_score",
        "--horizon",
        "2",
        "--top-n",
        "1",
        "--cost-bps",
        "0",
    ]
    try:
        assert backtest_main() == 0
    finally:
        sys.argv = old_argv

    predictions = pd.read_csv(output_dir / "predictions.csv", encoding="utf-8-sig")
    summary = pd.read_csv(output_dir / "summary.csv", encoding="utf-8-sig")
    trades = pd.read_csv(output_dir / "trades.csv", encoding="utf-8-sig")
    skipped = pd.read_csv(output_dir / "skipped_trades.csv", encoding="utf-8-sig")
    nav = pd.read_csv(output_dir / "daily_nav.csv", encoding="utf-8-sig")

    assert not predictions.empty
    assert int(summary.loc[0, "trade_count"]) >= 1
    assert "skipped_count" in summary.columns
    assert not trades.empty
    assert set(skipped.columns) >= {"report_date", "intended_entry_date", "code", "reason"}
    assert not nav.empty


def test_standard_execution_uses_next_trading_day_for_friday_signal(tmp_path: Path) -> None:
    price_dir = tmp_path / "daily_data"
    output_dir = tmp_path / "backtest"
    predictions_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"report_date": "2025-01-03", "code": "000001", "name": "A", "a2_flat_v1_score": 0.9},
        ]
    ).to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    write_price_csv(
        price_dir,
        "000001",
        pd.DataFrame(
            [
                {"date": "2025-01-03", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
                {"date": "2025-01-06", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1},
                {"date": "2025-01-07", "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2},
            ]
        ),
    )

    run_standard_execution(
        StandardExecutionConfig(
            predictions_csv=predictions_csv,
            price_data_dir=price_dir,
            output_dir=output_dir,
            score_col="a2_flat_v1_score",
            horizon=2,
            top_n=1,
            cost_bps=0,
            start_date="2025-01-03",
            end_date="2025-01-07",
        )
    )

    trades = pd.read_csv(output_dir / "trades.csv", encoding="utf-8-sig")
    assert trades.loc[0, "report_date"] == "2025-01-03"
    assert trades.loc[0, "entry_date"] == "2025-01-06"


def test_standard_execution_allows_same_code_overlap(tmp_path: Path) -> None:
    price_dir = tmp_path / "daily_data"
    output_dir = tmp_path / "backtest"
    predictions_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"report_date": "2025-01-02", "code": "000001", "name": "A", "a2_flat_v1_score": 0.9},
            {"report_date": "2025-01-02", "code": "000002", "name": "B", "a2_flat_v1_score": 0.8},
            {"report_date": "2025-01-03", "code": "000001", "name": "A", "a2_flat_v1_score": 0.95},
            {"report_date": "2025-01-03", "code": "000002", "name": "B", "a2_flat_v1_score": 0.7},
        ]
    ).to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    prices = pd.DataFrame(
        [
            {"date": "2025-01-02", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
            {"date": "2025-01-03", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0},
            {"date": "2025-01-06", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1},
            {"date": "2025-01-07", "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2},
        ]
    )
    write_price_csv(price_dir, "000001", prices)
    write_price_csv(price_dir, "000002", prices)

    run_standard_execution(
        StandardExecutionConfig(
            predictions_csv=predictions_csv,
            price_data_dir=price_dir,
            output_dir=output_dir,
            score_col="a2_flat_v1_score",
            horizon=3,
            top_n=1,
            cost_bps=0,
            start_date="2025-01-02",
            end_date="2025-01-07",
        )
    )

    trades = pd.read_csv(output_dir / "trades.csv", encoding="utf-8-sig")
    assert trades["code"].astype(str).str.zfill(6).tolist() == ["000001", "000001"]


def test_standard_execution_carries_exposure_until_next_available_exit_price(tmp_path: Path) -> None:
    price_dir = tmp_path / "daily_data"
    output_dir = tmp_path / "backtest"
    predictions_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"report_date": "2025-01-02", "code": "000001", "name": "A", "a2_flat_v1_score": 0.9},
            {"report_date": "2025-01-02", "code": "000002", "name": "B", "a2_flat_v1_score": 0.1},
        ]
    ).to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    write_price_csv(
        price_dir,
        "000001",
        pd.DataFrame(
            [
                {"date": "2025-01-02", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
                {"date": "2025-01-03", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
                {"date": "2025-01-07", "open": 10.0, "high": 10.4, "low": 9.9, "close": 10.3},
            ]
        ),
    )
    write_price_csv(
        price_dir,
        "000002",
        pd.DataFrame(
            [
                {"date": "2025-01-02", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
                {"date": "2025-01-03", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
                {"date": "2025-01-06", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
                {"date": "2025-01-07", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
            ]
        ),
    )

    run_standard_execution(
        StandardExecutionConfig(
            predictions_csv=predictions_csv,
            price_data_dir=price_dir,
            output_dir=output_dir,
            score_col="a2_flat_v1_score",
            horizon=2,
            top_n=1,
            cost_bps=0,
            start_date="2025-01-02",
            end_date="2025-01-07",
        )
    )

    nav = pd.read_csv(output_dir / "daily_nav.csv", encoding="utf-8-sig")
    trades = pd.read_csv(output_dir / "trades.csv", encoding="utf-8-sig")

    assert nav["date"].tolist() == ["2025-01-03", "2025-01-06", "2025-01-07"]
    assert nav.loc[nav["date"] == "2025-01-06", "exposure"].iloc[0] > 0
    assert trades.loc[0, "exit_date"] == "2025-01-07"


def test_standard_execution_records_skipped_missing_entry_price(tmp_path: Path) -> None:
    price_dir = tmp_path / "daily_data"
    output_dir = tmp_path / "backtest"
    predictions_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"report_date": "2025-01-02", "code": "000001", "name": "A", "a2_flat_v1_score": 0.9},
            {"report_date": "2025-01-02", "code": "000002", "name": "B", "a2_flat_v1_score": 0.1},
        ]
    ).to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    write_price_csv(
        price_dir,
        "000001",
        pd.DataFrame(
            [
                {"date": "2025-01-02", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
                {"date": "2025-01-06", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0},
            ]
        ),
    )
    write_price_csv(
        price_dir,
        "000002",
        pd.DataFrame(
            [
                {"date": "2025-01-02", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
                {"date": "2025-01-03", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
                {"date": "2025-01-06", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0},
            ]
        ),
    )

    run_standard_execution(
        StandardExecutionConfig(
            predictions_csv=predictions_csv,
            price_data_dir=price_dir,
            output_dir=output_dir,
            score_col="a2_flat_v1_score",
            horizon=2,
            top_n=1,
            cost_bps=0,
            start_date="2025-01-02",
            end_date="2025-01-06",
        )
    )

    summary = pd.read_csv(output_dir / "summary.csv", encoding="utf-8-sig")
    skipped = pd.read_csv(output_dir / "skipped_trades.csv", encoding="utf-8-sig")

    assert int(summary.loc[0, "skipped_count"]) == 1
    assert str(skipped.loc[0, "code"]).zfill(6) == "000001"
    assert skipped.loc[0, "reason"] == "missing_entry_price"


def test_standard_execution_skips_entry_on_final_nav_day(tmp_path: Path) -> None:
    price_dir = tmp_path / "daily_data"
    output_dir = tmp_path / "backtest"
    predictions_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"report_date": "2025-01-02", "code": "000001", "name": "A", "a2_flat_v1_score": 0.9},
        ]
    ).to_csv(predictions_csv, index=False, encoding="utf-8-sig")
    write_price_csv(
        price_dir,
        "000001",
        pd.DataFrame(
            [
                {"date": "2025-01-02", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
                {"date": "2025-01-03", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1},
            ]
        ),
    )

    run_standard_execution(
        StandardExecutionConfig(
            predictions_csv=predictions_csv,
            price_data_dir=price_dir,
            output_dir=output_dir,
            score_col="a2_flat_v1_score",
            horizon=2,
            top_n=1,
            cost_bps=0,
            start_date="2025-01-02",
            end_date="2025-01-03",
        )
    )

    summary = pd.read_csv(output_dir / "summary.csv", encoding="utf-8-sig")
    trades = pd.read_csv(output_dir / "trades.csv", encoding="utf-8-sig")
    skipped = pd.read_csv(output_dir / "skipped_trades.csv", encoding="utf-8-sig")

    assert int(summary.loc[0, "trade_count"]) == 0
    assert int(summary.loc[0, "skipped_count"]) == 1
    assert trades.empty
    assert skipped.loc[0, "reason"] == "entry_on_or_after_nav_end"
