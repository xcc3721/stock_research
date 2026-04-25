from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.a2.enrich_report_with_a2_flat import enrich_report_with_a2_flat
from stock_research.backtest.standard_execution import (
    StandardExecutionConfig,
    run_standard_execution,
    resolve_predictions_csv,
)
from stock_research.data.update_daily_data import update_daily_data
from stock_research.reporting.simple_report import (
    _resolve_trade_dates,
    generate_reports,
    load_universe,
    resolve_code_filter,
)
from stock_research.utils.paths import display_path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class A2PipelineConfig:
    data_dir: Path
    output_root: Path
    runtime_config_path: Path
    a2_config_path: Path
    trade_date: str = ""
    start_date: str = ""
    end_date: str = ""
    report_limit: int = 0
    codes: tuple[str, ...] = ()
    codes_file: Path | None = None
    update_daily_data_first: bool = False
    update_start_date: str = ""
    update_end_date: str = ""
    update_adjust: str = "qfq"
    skip_a2: bool = False
    skip_backtest: bool = False
    score_col: str = "a2_flat_v1_score"
    horizon: int = 10
    top_n: int = 1
    cost_bps: float = 10.0
    candidate_pool_top_n: int = 30
    candidate_pool_sort: str = "total_score"
    candidate_pool_csv: Path | None = None


def _trade_date_from_report(path: Path) -> pd.Timestamp:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    scoring = data.get("scoring_results", {}) if isinstance(data, dict) else {}
    trade_date = summary.get("trade_date") or scoring.get("trade_date")
    timestamp = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"报告缺少 trade_date: {path}")
    return pd.Timestamp(timestamp).normalize()


def _sorted_report_files(report_dir: Path) -> list[Path]:
    files = sorted(report_dir.glob("*.report_data.json"))
    return sorted(files, key=_trade_date_from_report)


def _serialize_config(config: A2PipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = display_path(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def run_a2_pipeline(config: A2PipelineConfig) -> dict[str, Any]:
    output_root = config.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir = output_root / "reports"
    backtest_dir = output_root / "backtest"
    manifest_path = output_root / "run_manifest.json"

    allowed_codes = resolve_code_filter(codes=config.codes, codes_file=config.codes_file)
    if config.update_daily_data_first:
        update_summaries = update_daily_data(
            data_dir=config.data_dir,
            codes=sorted(allowed_codes) if allowed_codes is not None else None,
            start_date=config.update_start_date or None,
            end_date=config.update_end_date or None,
            adjust=config.update_adjust,
        )
    else:
        update_summaries = []

    universe = load_universe(config.data_dir, allowed_codes=allowed_codes)
    if not universe:
        raise ValueError(f"未加载到任何行情数据: {config.data_dir}")
    trade_dates = _resolve_trade_dates(
        universe=universe,
        trade_date=config.trade_date or None,
        start_date=config.start_date or None,
        end_date=config.end_date or None,
    )
    if int(config.report_limit) > 0:
        trade_dates = trade_dates[-int(config.report_limit):]
    if not trade_dates:
        raise ValueError("未解析到任何 trade_dates")

    generated_reports = generate_reports(
        data_dir=config.data_dir,
        output_dir=report_dir,
        config_path=config.runtime_config_path,
        trade_dates=trade_dates,
        allowed_codes=allowed_codes,
    )
    report_files = sorted(generated_reports, key=_trade_date_from_report)

    a2_rows: list[dict[str, Any]] = []
    if not config.skip_a2:
        for report_json in report_files:
            summary = enrich_report_with_a2_flat(
                report_json=report_json,
                config_path=config.a2_config_path,
                candidate_pool_top_n=int(config.candidate_pool_top_n),
                candidate_pool_sort=str(config.candidate_pool_sort),
                candidate_pool_csv=config.candidate_pool_csv,
            )
            a2_rows.append(
                {
                    "report_json": display_path(report_json),
                    "trade_date": summary.get("trade_date", ""),
                    "rows": int(summary.get("rows", 0) or 0),
                    "top1_code": summary.get("top1", {}).get("code", "") if isinstance(summary.get("top1"), dict) else "",
                    "score_col": summary.get("score_col", config.score_col),
                }
            )

    if config.skip_backtest:
        backtest_outputs: dict[str, str] = {}
        predictions_csv = None
    else:
        predictions_csv = resolve_predictions_csv(
            predictions_csv=None,
            report_dir=report_dir,
            output_dir=backtest_dir,
            score_col=config.score_col,
        )
        backtest_outputs = {
            key: display_path(value)
            for key, value in run_standard_execution(
                StandardExecutionConfig(
                    predictions_csv=predictions_csv,
                    price_data_dir=config.data_dir,
                    output_dir=backtest_dir,
                    score_col=config.score_col,
                    horizon=int(config.horizon),
                    top_n=int(config.top_n),
                    cost_bps=float(config.cost_bps),
                    start_date=config.start_date,
                    end_date=config.end_date,
                )
            ).items()
        }

    manifest = {
        "pipeline": "a2_report_backtest",
        "config": _serialize_config(config),
        "resolved": {
            "output_root": display_path(output_root),
            "report_dir": display_path(report_dir),
            "backtest_dir": display_path(backtest_dir),
            "codes_count": int(len(allowed_codes)) if allowed_codes is not None else int(len(universe)),
            "trade_dates_count": int(len(trade_dates)),
            "first_trade_date": trade_dates[0].date().isoformat(),
            "last_trade_date": trade_dates[-1].date().isoformat(),
        },
        "update_daily_data": update_summaries,
        "generated_reports": [display_path(path) for path in report_files],
        "a2_flat": a2_rows,
        "predictions_csv": display_path(predictions_csv) if predictions_csv is not None else "",
        "backtest_outputs": backtest_outputs,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("A2 pipeline 完成: reports=%d output_root=%s", len(report_files), display_path(output_root))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the end-to-end A2 report and backtest pipeline")
    parser.add_argument("--data-dir", default="daily_data")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--runtime-config-path", default="configs/runtime.json")
    parser.add_argument("--a2-config-path", default="configs/a2/flat_v1_stable_penalty_overlay.json")
    parser.add_argument("--trade-date", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--report-limit", type=int, default=0)
    parser.add_argument("--codes", default="")
    parser.add_argument("--codes-file", default="")
    parser.add_argument("--update-daily-data-first", action="store_true")
    parser.add_argument("--update-start-date", default="")
    parser.add_argument("--update-end-date", default="")
    parser.add_argument("--update-adjust", default="qfq")
    parser.add_argument("--skip-a2", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--score-col", default="a2_flat_v1_score")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--candidate-pool-top-n", type=int, default=30)
    parser.add_argument(
        "--candidate-pool-sort",
        default="total_score",
        choices=("total_score", "report_order", "report_score", "canonical_mechanical_anchor_score"),
    )
    parser.add_argument("--candidate-pool-csv", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    codes_file = Path(args.codes_file).expanduser().resolve() if str(args.codes_file).strip() else None
    candidate_pool_csv = Path(args.candidate_pool_csv).expanduser().resolve() if str(args.candidate_pool_csv).strip() else None
    manifest = run_a2_pipeline(
        A2PipelineConfig(
            data_dir=Path(args.data_dir).expanduser().resolve(),
            output_root=Path(args.output_root).expanduser().resolve(),
            runtime_config_path=Path(args.runtime_config_path).expanduser().resolve(),
            a2_config_path=Path(args.a2_config_path).expanduser().resolve(),
            trade_date=str(args.trade_date).strip(),
            start_date=str(args.start_date).strip(),
            end_date=str(args.end_date).strip(),
            report_limit=int(args.report_limit),
            codes=tuple(token.strip() for token in str(args.codes).split(",") if token.strip()),
            codes_file=codes_file,
            update_daily_data_first=bool(args.update_daily_data_first),
            update_start_date=str(args.update_start_date).strip(),
            update_end_date=str(args.update_end_date).strip(),
            update_adjust=str(args.update_adjust).strip() or "qfq",
            skip_a2=bool(args.skip_a2),
            skip_backtest=bool(args.skip_backtest),
            score_col=str(args.score_col).strip(),
            horizon=int(args.horizon),
            top_n=int(args.top_n),
            cost_bps=float(args.cost_bps),
            candidate_pool_top_n=int(args.candidate_pool_top_n),
            candidate_pool_sort=str(args.candidate_pool_sort),
            candidate_pool_csv=candidate_pool_csv,
        )
    )
    LOGGER.info("pipeline manifest: %s", json.dumps(manifest["resolved"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
