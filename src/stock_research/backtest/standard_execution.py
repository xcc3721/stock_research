from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from stock_research.utils.paths import display_path


LOGGER = logging.getLogger(__name__)
TRADE_COLUMNS = [
    "report_date",
    "entry_date",
    "exit_date",
    "code",
    "name",
    "score",
    "weight",
    "entry_price",
    "exit_price",
    "gross_return",
    "net_return",
    "holding_days",
]
SKIPPED_COLUMNS = [
    "report_date",
    "intended_entry_date",
    "code",
    "name",
    "score",
    "reason",
    "detail",
]
DAILY_NAV_COLUMNS = ["date", "daily_return", "turnover", "exposure"]
SUMMARY_FIELD_LABELS = {
    "total_return": "总收益",
    "annualized_return": "年化收益",
    "max_drawdown": "最大回撤",
    "trade_win_rate": "交易胜率",
    "average_trade_return": "单笔均值",
    "average_exposure": "平均仓位",
    "average_turnover": "平均换手",
    "total_cost": "总成本",
}
SUMMARY_NUMBER_LABELS = {
    "trade_count": "交易数",
    "skipped_count": "跳过信号",
    "sample_days": "样本日",
}


@dataclass(frozen=True)
class StandardExecutionConfig:
    predictions_csv: Path
    price_data_dir: Path
    output_dir: Path
    score_col: str
    horizon: int = 10
    top_n: int = 1
    cost_bps: float = 10.0
    start_date: str = ""
    end_date: str = ""


def _normalize_code(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text


def _read_price_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
    for column in ["open", "close"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)


def _format_float(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "N/A"
    return f"{float(numeric):.{digits}f}"


def _format_int(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "0"
    return str(int(numeric))


def _format_percent(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "N/A"
    return f"{float(numeric) * 100.0:.2f}%"


def format_backtest_statistics(summary_csv: Path) -> str:
    summary = pd.read_csv(summary_csv, encoding="utf-8-sig", low_memory=False)
    if summary.empty:
        return "回测结果统计: summary.csv 为空"
    row = summary.iloc[0].to_dict()
    period = f"{row.get('start_date', '')} -> {row.get('end_date', '')}"
    lines = [
        "回测结果统计:",
        f"  区间: {period}, 样本日: {_format_int(row.get('sample_days'))}",
        (
            "  收益: "
            f"{SUMMARY_FIELD_LABELS['total_return']} {_format_percent(row.get('total_return'))}, "
            f"{SUMMARY_FIELD_LABELS['annualized_return']} {_format_percent(row.get('annualized_return'))}, "
            f"{SUMMARY_FIELD_LABELS['max_drawdown']} {_format_percent(row.get('max_drawdown'))}, "
            f"Sharpe {_format_float(row.get('sharpe'))}"
        ),
        (
            "  交易: "
            f"{SUMMARY_NUMBER_LABELS['trade_count']} {_format_int(row.get('trade_count'))}, "
            f"{SUMMARY_NUMBER_LABELS['skipped_count']} {_format_int(row.get('skipped_count'))}, "
            f"{SUMMARY_FIELD_LABELS['trade_win_rate']} {_format_percent(row.get('trade_win_rate'))}, "
            f"{SUMMARY_FIELD_LABELS['average_trade_return']} {_format_percent(row.get('average_trade_return'))}"
        ),
        (
            "  暴露: "
            f"{SUMMARY_FIELD_LABELS['average_exposure']} {_format_percent(row.get('average_exposure'))}, "
            f"{SUMMARY_FIELD_LABELS['average_turnover']} {_format_percent(row.get('average_turnover'))}, "
            f"{SUMMARY_FIELD_LABELS['total_cost']} {_format_percent(row.get('total_cost'))}"
        ),
    ]
    if "average_holding_days" in row:
        lines.append(f"  持仓: 平均持仓天数 {_format_float(row.get('average_holding_days'))}")
    return "\n".join(lines)


def _load_price_map(data_dir: Path, codes: Iterable[str]) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    frames: dict[str, pd.DataFrame] = {}
    all_dates: set[pd.Timestamp] = set()
    for code in sorted({_normalize_code(code) for code in codes if _normalize_code(code)}):
        path = data_dir / f"{code}.csv"
        if not path.exists():
            continue
        frame = _read_price_frame(path)
        if frame.empty:
            continue
        frames[code] = frame
        for value in pd.DatetimeIndex(frame["date"].dropna().unique()):
            all_dates.add(pd.Timestamp(value).normalize())
    return frames, pd.DatetimeIndex(sorted(all_dates))


def build_prediction_frame(report_dir: Path, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = ["report_date", "code", "name", score_col]
    for path in sorted(report_dir.glob("*.report_data.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("data", {})
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        scoring = data.get("scoring_results", {}) if isinstance(data, dict) else {}
        trade_date = summary.get("trade_date") or scoring.get("trade_date")
        batch_results = scoring.get("batch_results") if isinstance(scoring, dict) else None
        if not isinstance(batch_results, list):
            continue
        for row in batch_results:
            if not isinstance(row, dict):
                continue
            code = _normalize_code(row.get("code"))
            if not code:
                continue
            score = row.get(score_col)
            if score is None:
                nested_a2 = row.get("A2_flat")
                if isinstance(nested_a2, dict) and score_col in {"a2_flat_v1_score", "score"}:
                    score = nested_a2.get("score")
                elif isinstance(row.get("result"), dict) and score_col == "total_score":
                    score = row["result"].get("total_score")
                elif score_col == "final_score":
                    score = row.get("final_score")
            rows.append(
                {
                    "report_date": trade_date,
                    "code": code,
                    "name": str(row.get("name", "") or ""),
                    score_col: pd.to_numeric(score, errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].map(_normalize_code)
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    return (
        frame.dropna(subset=["report_date", "code", score_col])[columns]
        .sort_values(["report_date", score_col, "code"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def export_prediction_frame(report_dir: Path, output_csv: Path, score_col: str) -> Path:
    frame = build_prediction_frame(report_dir, score_col)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv


def resolve_predictions_csv(
    *,
    predictions_csv: Path | None,
    report_dir: Path | None,
    output_dir: Path,
    score_col: str,
) -> Path:
    if predictions_csv is not None:
        return predictions_csv
    if report_dir is None:
        raise ValueError("predictions_csv 与 report_dir 不能同时为空")
    exported = output_dir / "predictions.csv"
    return export_prediction_frame(report_dir, exported, score_col)


def _append_trade_row(trade_rows: list[dict[str, object]], position: dict[str, object], exit_date: pd.Timestamp) -> None:
    trade_rows.append(
        {
            "report_date": pd.Timestamp(position["report_date"]),
            "entry_date": pd.Timestamp(position["entry_date"]),
            "exit_date": pd.Timestamp(exit_date),
            "code": str(position["code"]),
            "name": str(position["name"]),
            "score": float(position["score"]),
            "weight": float(position["weight"]),
            "entry_price": float(position["entry_price"]),
            "exit_price": float(position["exit_price"]),
            "gross_return": float(position["exit_price"]) / float(position["entry_price"]) - 1.0,
            "net_return": float(position["trade_nav"] - 1.0),
            "holding_days": int((pd.Timestamp(exit_date) - pd.Timestamp(position["entry_date"])).days + 1),
        }
    )


def _append_skipped_row(
    skipped_rows: list[dict[str, object]],
    *,
    report_date: pd.Timestamp,
    intended_entry_date: pd.Timestamp | None,
    code: str,
    name: str,
    score: float,
    reason: str,
    detail: str = "",
) -> None:
    skipped_rows.append(
        {
            "report_date": pd.Timestamp(report_date),
            "intended_entry_date": pd.Timestamp(intended_entry_date) if intended_entry_date is not None else "",
            "code": _normalize_code(code),
            "name": name,
            "score": float(score),
            "reason": reason,
            "detail": detail,
        }
    )


def _append_skipped_picks(
    skipped_rows: list[dict[str, object]],
    *,
    group_map: dict[pd.Timestamp, pd.DataFrame],
    report_date: pd.Timestamp,
    top_n: int,
    score_col: str,
    intended_entry_date: pd.Timestamp | None,
    reason: str,
    detail: str = "",
) -> None:
    picks = group_map[report_date].head(int(top_n)).copy()
    for row in picks.itertuples(index=False):
        _append_skipped_row(
            skipped_rows,
            report_date=report_date,
            intended_entry_date=intended_entry_date,
            code=str(row.code),
            name=str(row.name),
            score=float(getattr(row, score_col)),
            reason=reason,
            detail=detail,
        )


def run_standard_execution(config: StandardExecutionConfig) -> dict[str, Path]:
    predictions = pd.read_csv(config.predictions_csv, encoding="utf-8-sig", low_memory=False)
    predictions["report_date"] = pd.to_datetime(predictions.get("report_date"), errors="coerce").dt.normalize()
    predictions["code"] = predictions.get("code", pd.Series(dtype="object")).map(_normalize_code)
    predictions[config.score_col] = pd.to_numeric(predictions.get(config.score_col), errors="coerce")
    predictions = predictions.dropna(subset=["report_date", "code", config.score_col]).copy()
    if str(config.start_date).strip():
        predictions = predictions[predictions["report_date"] >= pd.Timestamp(config.start_date).normalize()]
    if str(config.end_date).strip():
        predictions = predictions[predictions["report_date"] <= pd.Timestamp(config.end_date).normalize()]
    if predictions.empty:
        raise ValueError("predictions 为空")

    price_map, calendar = _load_price_map(config.price_data_dir, predictions["code"].tolist())
    if not price_map:
        raise ValueError("未加载到价格数据")
    date_pos = {pd.Timestamp(value): idx for idx, value in enumerate(calendar)}
    cost_rate = float(config.cost_bps) / 10000.0
    trade_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    positions: list[dict[str, object]] = []
    group_map = {
        pd.Timestamp(date): group.sort_values([config.score_col, "code"], ascending=[False, True]).copy()
        for date, group in predictions.groupby("report_date", sort=True)
    }
    requested_end_date = (
        pd.Timestamp(config.end_date).normalize()
        if str(config.end_date).strip()
        else pd.Timestamp(predictions["report_date"].max()).normalize()
    )
    valid_end_dates = [pd.Timestamp(value) for value in calendar if pd.Timestamp(value) <= requested_end_date]
    if not valid_end_dates:
        raise ValueError("回测结束日期早于可用价格日历")
    nav_end_date = valid_end_dates[-1]
    entry_report_map: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for report_date in sorted(group_map):
        if report_date not in date_pos:
            _append_skipped_picks(
                skipped_rows,
                group_map=group_map,
                report_date=report_date,
                top_n=int(config.top_n),
                score_col=config.score_col,
                intended_entry_date=None,
                reason="report_date_not_in_price_calendar",
            )
            continue
        entry_idx = date_pos[report_date] + 1
        if entry_idx >= len(calendar):
            _append_skipped_picks(
                skipped_rows,
                group_map=group_map,
                report_date=report_date,
                top_n=int(config.top_n),
                score_col=config.score_col,
                intended_entry_date=None,
                reason="no_next_trading_day",
            )
            continue
        entry_date = pd.Timestamp(calendar[entry_idx])
        if entry_date >= nav_end_date:
            _append_skipped_picks(
                skipped_rows,
                group_map=group_map,
                report_date=report_date,
                top_n=int(config.top_n),
                score_col=config.score_col,
                intended_entry_date=entry_date,
                reason="entry_on_or_after_nav_end",
                detail=f"nav_end_date={nav_end_date.date().isoformat()}",
            )
            continue
        entry_report_map.setdefault(entry_date, []).append(report_date)
    if not entry_report_map and not skipped_rows:
        raise ValueError("回测区间内没有可入场信号")
    if entry_report_map:
        used_dates = [
            pd.Timestamp(value)
            for value in calendar
            if min(entry_report_map) <= pd.Timestamp(value) <= nav_end_date
        ]
    else:
        used_dates = []

    for current_date in used_dates:
        current_ts = pd.Timestamp(current_date)
        day_return = 0.0
        turnover = 0.0
        exposure = 0.0

        remaining: list[dict[str, object]] = []
        for position in positions:
            price_frame = price_map.get(str(position["code"]))
            if price_frame is None:
                exposure += float(position["weight"])
                remaining.append(position)
                continue
            current_row = price_frame.loc[price_frame["date"] == current_ts]
            if current_row.empty:
                exposure += float(position["weight"])
                remaining.append(position)
                continue
            close_price = float(current_row.iloc[-1]["close"])
            component_return = close_price / float(position["last_close"]) - 1.0
            should_exit = current_ts >= pd.Timestamp(position["exit_date"])
            if should_exit:
                component_return -= cost_rate
                turnover += float(position["weight"])
                position["exit_price"] = close_price
            day_return += float(position["weight"]) * component_return
            exposure += float(position["weight"])
            position["trade_nav"] *= 1.0 + component_return
            position["last_close"] = close_price
            if should_exit:
                _append_trade_row(trade_rows, position, current_ts)
                continue
            remaining.append(position)
        positions = remaining

        for report_date in entry_report_map.get(current_ts, []):
            picks = group_map[report_date].head(int(config.top_n)).copy()
            if picks.empty:
                continue
            weight = 1.0 / float(int(config.horizon) * max(1, len(picks)))
            for row in picks.itertuples(index=False):
                code = str(row.code)
                price_frame = price_map.get(code)
                if price_frame is None:
                    _append_skipped_row(
                        skipped_rows,
                        report_date=report_date,
                        intended_entry_date=current_ts,
                        code=code,
                        name=str(row.name),
                        score=float(getattr(row, config.score_col)),
                        reason="missing_price_data",
                    )
                    continue
                entry_idx = date_pos[report_date] + 1
                exit_idx = date_pos[report_date] + int(config.horizon)
                entry_date = pd.Timestamp(calendar[entry_idx])
                exit_date = pd.Timestamp(calendar[min(exit_idx, len(calendar) - 1)])
                entry_row = price_frame.loc[price_frame["date"] == entry_date]
                current_row = price_frame.loc[price_frame["date"] == current_ts]
                if entry_row.empty:
                    _append_skipped_row(
                        skipped_rows,
                        report_date=report_date,
                        intended_entry_date=entry_date,
                        code=code,
                        name=str(row.name),
                        score=float(getattr(row, config.score_col)),
                        reason="missing_entry_price",
                    )
                    continue
                if current_row.empty:
                    _append_skipped_row(
                        skipped_rows,
                        report_date=report_date,
                        intended_entry_date=entry_date,
                        code=code,
                        name=str(row.name),
                        score=float(getattr(row, config.score_col)),
                        reason="missing_current_price",
                    )
                    continue
                entry_open = float(entry_row.iloc[-1]["open"])
                current_close = float(current_row.iloc[-1]["close"])
                trade_nav = 1.0 + (current_close / entry_open - 1.0 - cost_rate)
                day_return += weight * (current_close / entry_open - 1.0 - cost_rate)
                turnover += weight
                exposure += weight
                positions.append(
                    {
                        "report_date": report_date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "code": code,
                        "name": str(row.name),
                        "score": float(getattr(row, config.score_col)),
                        "weight": weight,
                        "entry_price": entry_open,
                        "last_close": current_close,
                        "exit_price": np.nan,
                        "trade_nav": trade_nav,
                    }
                )

        if current_ts == nav_end_date and positions:
            marked_positions: list[dict[str, object]] = []
            for position in positions:
                price_frame = price_map.get(str(position["code"]))
                current_row = (
                    price_frame.loc[price_frame["date"] == current_ts]
                    if price_frame is not None
                    else pd.DataFrame()
                )
                close_price = (
                    float(current_row.iloc[-1]["close"])
                    if not current_row.empty
                    else float(position["last_close"])
                )
                position["exit_price"] = close_price
                position["last_close"] = close_price
                position["trade_nav"] *= 1.0 - cost_rate
                day_return -= float(position["weight"]) * cost_rate
                turnover += float(position["weight"])
                _append_trade_row(trade_rows, position, current_ts)
            positions = marked_positions

        daily_rows.append({"date": current_ts, "daily_return": day_return, "turnover": turnover, "exposure": exposure})

    daily_nav = pd.DataFrame(daily_rows, columns=DAILY_NAV_COLUMNS)
    daily_nav["nav"] = (1.0 + daily_nav["daily_return"]).cumprod()
    daily_nav["peak_nav"] = daily_nav["nav"].cummax()
    daily_nav["drawdown"] = daily_nav["nav"] / daily_nav["peak_nav"] - 1.0

    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    if not trades.empty:
        trades = trades.sort_values(["report_date", "score"], ascending=[True, False]).reset_index(drop=True)
    skipped = pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS)
    if not skipped.empty:
        skipped = skipped.sort_values(["report_date", "score", "code"], ascending=[True, False, True]).reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "score_col": config.score_col,
                "trade_count": int(len(trades)),
                "skipped_count": int(len(skipped)),
                "total_return": float(daily_nav["nav"].iloc[-1] - 1.0) if not daily_nav.empty else 0.0,
                "annualized_return": (
                    float(daily_nav["nav"].iloc[-1] ** (252.0 / max(len(daily_nav), 1)) - 1.0)
                    if not daily_nav.empty
                    else 0.0
                ),
                "sharpe": (
                    float(daily_nav["daily_return"].mean() / daily_nav["daily_return"].std(ddof=1) * np.sqrt(252.0))
                    if len(daily_nav) > 1 and float(daily_nav["daily_return"].std(ddof=1)) > 0.0
                    else np.nan
                ),
                "max_drawdown": float(daily_nav["drawdown"].min()) if not daily_nav.empty else 0.0,
                "trade_win_rate": float((trades["net_return"] > 0).mean()) if not trades.empty else np.nan,
                "average_trade_return": float(trades["net_return"].mean()) if not trades.empty else np.nan,
                "average_holding_days": float(trades["holding_days"].mean()) if not trades.empty else np.nan,
                "average_exposure": float(daily_nav["exposure"].mean()) if not daily_nav.empty else np.nan,
                "average_turnover": float(daily_nav["turnover"].mean()) if not daily_nav.empty else np.nan,
                "total_cost": float(daily_nav["turnover"].sum() * cost_rate),
                "sample_days": int(len(daily_nav)),
                "start_date": str(daily_nav["date"].min().date()) if not daily_nav.empty else "",
                "end_date": str(daily_nav["date"].max().date()) if not daily_nav.empty else "",
            }
        ]
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "summary.csv"
    trades_path = config.output_dir / "trades.csv"
    skipped_path = config.output_dir / "skipped_trades.csv"
    nav_path = config.output_dir / "daily_nav.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    skipped.to_csv(skipped_path, index=False, encoding="utf-8-sig")
    daily_nav.to_csv(nav_path, index=False, encoding="utf-8-sig")
    return {"summary": summary_path, "trades": trades_path, "skipped_trades": skipped_path, "daily_nav": nav_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixed-horizon execution backtest from generated reports")
    parser.add_argument("--predictions-csv", default="")
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--price-data-dir", default="daily_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-col", default="a2_flat_v1_score")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    output_dir = Path(args.output_dir).expanduser().resolve()
    predictions_csv = (
        Path(args.predictions_csv).expanduser().resolve()
        if str(args.predictions_csv).strip()
        else None
    )
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if str(args.report_dir).strip()
        else None
    )
    resolved_predictions_csv = resolve_predictions_csv(
        predictions_csv=predictions_csv,
        report_dir=report_dir,
        output_dir=output_dir,
        score_col=str(args.score_col),
    )
    outputs = run_standard_execution(
        StandardExecutionConfig(
            predictions_csv=resolved_predictions_csv,
            price_data_dir=Path(args.price_data_dir).expanduser().resolve(),
            output_dir=output_dir,
            score_col=str(args.score_col),
            horizon=int(args.horizon),
            top_n=int(args.top_n),
            cost_bps=float(args.cost_bps),
            start_date=str(args.start_date),
            end_date=str(args.end_date),
        )
    )
    if report_dir is not None and predictions_csv is None:
        LOGGER.info("已从 report_dir 导出 predictions: %s", display_path(resolved_predictions_csv))
    LOGGER.info("完成标准回测: %s", {key: display_path(value) for key, value in outputs.items()})
    LOGGER.info("%s", format_backtest_statistics(outputs["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
