from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stock_research.data.update_daily_data import update_daily_data
from stock_research.data.update_daily_data import _fetch_tushare_history
from stock_research.data.update_daily_data import _to_tushare_code


def test_update_daily_data_merges_existing_rows(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "daily_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame(
        [
            {"date": "2025-01-01", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 100, "amount": 1010},
            {"date": "2025-01-02", "open": 10.1, "high": 10.3, "low": 9.9, "close": 10.2, "volume": 120, "amount": 1224},
        ]
    )
    existing.to_csv(data_dir / "000001.csv", index=False, encoding="utf-8-sig")

    monkeypatch.setattr("stock_research.data.update_daily_data._resolve_tushare_token", lambda env_file: "test-token")
    monkeypatch.setattr("stock_research.data.update_daily_data._build_tushare_pro_client", lambda token: object())

    def fake_fetch(*, pro_client, code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        assert code == "000001"
        assert adjust == "qfq"
        return pd.DataFrame(
            [
                {"date": "2025-01-02", "open": 10.1, "high": 10.4, "low": 10.0, "close": 10.25, "volume": 130, "amount": 1332.5},
                {"date": "2025-01-03", "open": 10.2, "high": 10.5, "low": 10.1, "close": 10.4, "volume": 140, "amount": 1456},
                {"date": "2025-01-06", "open": 10.3, "high": 10.6, "low": 10.2, "close": 10.5, "volume": 150, "amount": 1575},
            ]
        )

    monkeypatch.setattr("stock_research.data.update_daily_data._fetch_tushare_history", fake_fetch)

    summaries = update_daily_data(
        data_dir=data_dir,
        codes=["000001"],
        start_date=None,
        end_date="2025-01-06",
        adjust="qfq",
    )

    updated = pd.read_csv(data_dir / "000001.csv", encoding="utf-8-sig")
    assert len(summaries) == 1
    assert summaries[0]["existing_rows"] == 2
    assert summaries[0]["fetched_rows"] == 3
    assert summaries[0]["rows"] == 4
    assert updated["date"].tolist() == ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]
    assert float(updated.loc[updated["date"] == "2025-01-02", "close"].iloc[0]) == 10.25


def test_update_daily_data_can_bootstrap_listed_codes(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "daily_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2025-01-01", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1, "amount": 1},
        ]
    ).to_csv(data_dir / "999999.csv", index=False, encoding="utf-8-sig")

    monkeypatch.setattr("stock_research.data.update_daily_data._resolve_tushare_token", lambda env_file: "test-token")
    monkeypatch.setattr("stock_research.data.update_daily_data._build_tushare_pro_client", lambda token: object())
    monkeypatch.setattr("stock_research.data.update_daily_data._fetch_listed_a_codes", lambda *, pro_client: ["000001", "000002"])

    def fake_fetch(*, pro_client, code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"date": "2025-01-02", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 100, "amount": 1000},
            ]
        )

    monkeypatch.setattr("stock_research.data.update_daily_data._fetch_tushare_history", fake_fetch)

    summaries = update_daily_data(
        data_dir=data_dir,
        codes=None,
        start_date="2025-01-01",
        end_date="2025-01-02",
        adjust="qfq",
        bootstrap_listed=True,
    )

    assert len(summaries) == 2
    assert (data_dir / "000001.csv").exists()
    assert (data_dir / "000002.csv").exists()
    assert summaries[0]["code"] == "000001"


def test_update_daily_data_continues_after_single_code_failure(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "daily_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("stock_research.data.update_daily_data._resolve_tushare_token", lambda env_file: "test-token")
    monkeypatch.setattr("stock_research.data.update_daily_data._build_tushare_pro_client", lambda token: object())

    def fake_fetch(*, pro_client, code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        if code == "000002":
            raise RuntimeError("rate limited")
        return pd.DataFrame(
            [
                {"date": "2025-01-02", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 100, "amount": 1000},
            ]
        )

    monkeypatch.setattr("stock_research.data.update_daily_data._fetch_tushare_history", fake_fetch)

    summaries = update_daily_data(
        data_dir=data_dir,
        codes=["000001", "000002", "000003"],
        start_date="2025-01-01",
        end_date="2025-01-02",
        adjust="qfq",
    )

    assert [row["status"] for row in summaries] == ["updated", "failed", "updated"]
    assert (data_dir / "000001.csv").exists()
    assert not (data_dir / "000002.csv").exists()
    assert (data_dir / "000003.csv").exists()


def test_update_daily_data_fail_fast_raises(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "daily_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("stock_research.data.update_daily_data._resolve_tushare_token", lambda env_file: "test-token")
    monkeypatch.setattr("stock_research.data.update_daily_data._build_tushare_pro_client", lambda token: object())

    def fake_fetch(*, pro_client, code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr("stock_research.data.update_daily_data._fetch_tushare_history", fake_fetch)

    with pytest.raises(RuntimeError, match="boom"):
        update_daily_data(
            data_dir=data_dir,
            codes=["000001"],
            start_date="2025-01-01",
            end_date="2025-01-02",
            adjust="qfq",
            fail_fast=True,
        )


def test_fetch_tushare_history_uses_supported_pro_bar_api_argument(monkeypatch) -> None:
    calls: dict[str, object] = {}
    pro_client = object()

    def fake_pro_bar(**kwargs):
        calls.update(kwargs)
        if "pro_api" in kwargs:
            raise TypeError("pro_bar() got an unexpected keyword argument 'pro_api'")
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20250102",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "vol": 1000,
                    "amount": 10100,
                },
            ]
        )

    monkeypatch.setattr("stock_research.data.update_daily_data.ts.pro_bar", fake_pro_bar)

    frame = _fetch_tushare_history(
        pro_client=pro_client,
        code="000001",
        start_date="2025-01-01",
        end_date="2025-01-02",
        adjust="qfq",
    )

    assert calls["api"] is pro_client
    assert calls["ts_code"] == "000001.SZ"
    assert calls["adj"] == "qfq"
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02"]
    assert frame["volume"].tolist() == [1000]


def test_to_tushare_code_maps_bj_920_before_sh() -> None:
    assert _to_tushare_code("920000") == "920000.BJ"
    assert _to_tushare_code("830799") == "830799.BJ"
    assert _to_tushare_code("900901") == "900901.SH"
