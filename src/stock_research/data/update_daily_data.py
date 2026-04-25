from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import tushare as ts

from stock_research.utils.paths import display_path


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


def _normalize_code(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if "." in text:
        text = text.split(".", 1)[0]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text


def _to_tushare_code(code: str) -> str:
    normalized = _normalize_code(code)
    if not normalized:
        raise ValueError(f"非法股票代码: {code}")
    if normalized.startswith(("4", "8")) or normalized.startswith("920"):
        return f"{normalized}.BJ"
    if normalized.startswith(("0", "1", "2", "3")):
        return f"{normalized}.SZ"
    if normalized.startswith(("5", "6", "9")):
        return f"{normalized}.SH"
    raise ValueError(f"无法识别交易所后缀: {code}")


def _codes_from_dir(data_dir: Path) -> list[str]:
    return sorted({_normalize_code(path.stem) for path in data_dir.glob("*.csv") if _normalize_code(path.stem)})


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DAILY_COLUMNS)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).copy()


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _resolve_tushare_token(env_file: Path | None) -> str:
    token = str(os.environ.get("TUSHARE_TOKEN", "")).strip()
    if token:
        return token
    resolved_env_file = env_file or DEFAULT_ENV_FILE
    if resolved_env_file.exists():
        token = str(_load_env_file(resolved_env_file).get("TUSHARE_TOKEN", "")).strip()
        if token:
            return token
    raise ValueError(
        f"缺少 TUSHARE_TOKEN。请在环境变量中设置，或在 {display_path(resolved_env_file)} 写入 TUSHARE_TOKEN=your_token"
    )


def _build_tushare_pro_client(token: str):
    ts.set_token(token)
    return ts.pro_api(token)


def _normalize_adjust(value: str) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"", "none", "raw", "bfq"}:
        return None
    if text not in {"qfq", "hfq"}:
        raise ValueError(f"不支持的复权类型: {value}")
    return text


def _fetch_tushare_history(
    *,
    pro_client,
    code: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    frame = ts.pro_bar(
        ts_code=_to_tushare_code(code),
        api=pro_client,
        asset="E",
        freq="D",
        adj=_normalize_adjust(adjust),
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    renamed = frame.rename(columns={"trade_date": "date", "vol": "volume"})
    for column in DAILY_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    renamed["date"] = pd.to_datetime(renamed["date"], format="%Y%m%d", errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return (
        renamed[DAILY_COLUMNS]
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def _fetch_listed_a_codes(*, pro_client) -> list[str]:
    frame = pro_client.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,market,list_date",
    )
    if frame is None or frame.empty:
        raise ValueError("Tushare stock_basic 未返回任何上市股票")
    if "symbol" not in frame.columns:
        raise ValueError("Tushare stock_basic 返回结果缺少 symbol 列")
    codes = sorted({_normalize_code(value) for value in frame["symbol"].tolist() if _normalize_code(value)})
    if not codes:
        raise ValueError("Tushare stock_basic 未解析到任何可用股票代码")
    return codes


def update_one_code(
    *,
    pro_client,
    code: str,
    data_dir: Path,
    start_date: str | None,
    end_date: str,
    adjust: str,
) -> dict[str, object]:
    target = data_dir / f"{code}.csv"
    existing = _read_existing(target)
    fetch_start = start_date
    if fetch_start is None and not existing.empty:
        fetch_start = (existing["date"].max() - pd.Timedelta(days=5)).date().isoformat()
    if fetch_start is None:
        fetch_start = "2018-01-01"

    fetched = _fetch_tushare_history(
        pro_client=pro_client,
        code=code,
        start_date=fetch_start,
        end_date=end_date,
        adjust=adjust,
    )
    merged = pd.concat([existing, fetched], ignore_index=True)
    merged["date"] = pd.to_datetime(merged.get("date"), errors="coerce").dt.normalize()
    merged = merged.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    data_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target, index=False, encoding="utf-8-sig")
    return {
        "code": code,
        "status": "updated",
        "rows": int(len(merged)),
        "existing_rows": int(len(existing)),
        "fetched_rows": int(len(fetched)),
        "last_date": merged["date"].max().date().isoformat() if not merged.empty else "",
    }


def update_daily_data(
    *,
    data_dir: Path,
    codes: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str = "qfq",
    bootstrap_listed: bool = False,
    env_file: Path | None = None,
    pro_client: Any | None = None,
    fail_fast: bool = False,
) -> list[dict[str, object]]:
    if pro_client is None:
        token = _resolve_tushare_token(env_file)
        pro_client = _build_tushare_pro_client(token)

    normalized_codes = sorted({_normalize_code(code) for code in (codes or []) if _normalize_code(code)})
    if not normalized_codes and bootstrap_listed:
        normalized_codes = _fetch_listed_a_codes(pro_client=pro_client)
        LOGGER.info("通过 Tushare stock_basic 自举上市股票列表: count=%d", len(normalized_codes))
    if not normalized_codes:
        normalized_codes = _codes_from_dir(data_dir)
    if not normalized_codes:
        raise ValueError("未提供 codes，且 data_dir 下不存在可更新的 csv 文件；如需从零开始，请传 --bootstrap-listed")

    resolved_end_date = end_date or datetime.now().date().isoformat()
    summaries: list[dict[str, object]] = []
    for code in normalized_codes:
        LOGGER.info("通过 Tushare 更新日线数据: %s", code)
        try:
            summaries.append(
                update_one_code(
                    pro_client=pro_client,
                    code=code,
                    data_dir=data_dir,
                    start_date=start_date,
                    end_date=resolved_end_date,
                    adjust=adjust,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            LOGGER.warning("跳过更新失败股票: code=%s error=%s", code, exc)
            summaries.append({"code": code, "status": "failed", "error": str(exc)})
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Update local daily_data/*.csv via Tushare")
    parser.add_argument("--data-dir", default="daily_data")
    parser.add_argument("--codes", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--bootstrap-listed", action="store_true", help="未传 --codes 时，通过 Tushare stock_basic 更新当前全部上市股票")
    parser.add_argument("--env-file", default="", help="可选：显式指定 .env 路径；默认读取仓库根目录 .env")
    parser.add_argument("--fail-fast", action="store_true", help="任一股票更新失败时立即中断；默认记录失败并继续")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    codes = [token.strip() for token in str(args.codes).split(",") if token.strip()]
    env_file = Path(args.env_file).expanduser().resolve() if str(args.env_file).strip() else None
    token = _resolve_tushare_token(env_file)
    pro_client = _build_tushare_pro_client(token)
    summaries = update_daily_data(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        codes=codes or None,
        start_date=str(args.start_date).strip() or None,
        end_date=str(args.end_date).strip() or None,
        adjust=str(args.adjust).strip() or "qfq",
        bootstrap_listed=bool(args.bootstrap_listed),
        env_file=env_file,
        pro_client=pro_client,
        fail_fast=bool(args.fail_fast),
    )
    success_count = sum(1 for row in summaries if row.get("status") == "updated")
    failed_count = sum(1 for row in summaries if row.get("status") == "failed")
    LOGGER.info("完成日线更新: success=%d failed=%d total=%d", success_count, failed_count, len(summaries))
    for row in summaries:
        LOGGER.info("结果: %s", row)
    if failed_count > 0 and success_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
