from __future__ import annotations

import argparse
import gc
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.a2.enrich_report_with_a2_flat import enrich_report_with_a2_flat
from stock_research.reporting.simple_report import (
    _resolve_trade_dates,
    generate_one_report,
    load_universe,
    load_runtime_config,
    resolve_code_filter,
)
from stock_research.utils.paths import display_path


LOGGER = logging.getLogger(__name__)
DEFAULT_RUNTIME_CONFIG_PATH = "configs/runtime.json"
DEFAULT_A2_CONFIG_PATH = "configs/a2/flat_v1_stable_penalty_overlay.json"
_WORKER_CONTEXT: dict[str, Any] = {}


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


def _init_a2_report_worker(context: dict[str, Any]) -> None:
    data_dir = str(context["data_dir"])
    output_dir = str(context["output_dir"])
    runtime_config_path = str(context["runtime_config_path"])
    a2_config_path = str(context["a2_config_path"])
    allowed_codes = context["allowed_codes"]
    candidate_pool_top_n = int(context["candidate_pool_top_n"])
    candidate_pool_sort = str(context["candidate_pool_sort"])
    candidate_pool_csv = str(context["candidate_pool_csv"])
    resolved_allowed_codes = set(allowed_codes) if allowed_codes is not None else None
    data_path = Path(data_dir)
    runtime_path = Path(runtime_config_path)
    universe = load_universe(data_path, allowed_codes=resolved_allowed_codes)
    if not universe:
        raise ValueError(f"未加载到任何行情数据: {data_path}")
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "output_dir": Path(output_dir),
            "runtime_config_path": runtime_path,
            "runtime_config": load_runtime_config(runtime_path),
            "universe": universe,
            "a2_config_path": Path(a2_config_path),
            "candidate_pool_top_n": int(candidate_pool_top_n),
            "candidate_pool_sort": str(candidate_pool_sort),
            "candidate_pool_csv": Path(candidate_pool_csv) if str(candidate_pool_csv).strip() else None,
        }
    )


def _generate_one_a2_report_worker(trade_date: pd.Timestamp) -> dict[str, Any]:
    if not _WORKER_CONTEXT:
        raise RuntimeError("A2 report worker 未初始化")
    trade_ts = pd.Timestamp(trade_date).normalize()
    output_dir: Path = _WORKER_CONTEXT["output_dir"]
    report_json = output_dir / f"bbikdj_{trade_ts.strftime('%Y%m%d')}.report_data.json"
    payload = generate_one_report(
        trade_date=trade_ts,
        universe=_WORKER_CONTEXT["universe"],
        runtime_config=_WORKER_CONTEXT["runtime_config"],
        config_path=_WORKER_CONTEXT["runtime_config_path"],
    )
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = enrich_report_with_a2_flat(
        report_json=report_json,
        config_path=_WORKER_CONTEXT["a2_config_path"],
        candidate_pool_top_n=int(_WORKER_CONTEXT["candidate_pool_top_n"]),
        candidate_pool_sort=str(_WORKER_CONTEXT["candidate_pool_sort"]),
        candidate_pool_csv=_WORKER_CONTEXT["candidate_pool_csv"],
    )
    return {
        "report_json": str(report_json),
        "trade_date": summary.get("trade_date", trade_ts.date().isoformat()),
        "rows": int(summary.get("rows", 0) or 0),
        "top1_code": summary.get("top1", {}).get("code", "") if isinstance(summary.get("top1"), dict) else "",
        "score_col": summary.get("score_col", "a2_flat_v1_score"),
    }


def generate_a2_reports(
    *,
    data_dir: Path,
    output_dir: Path,
    runtime_config_path: Path,
    a2_config_path: Path,
    trade_date: str = "",
    start_date: str = "",
    end_date: str = "",
    report_limit: int = 0,
    codes: tuple[str, ...] = (),
    codes_file: Path | None = None,
    candidate_pool_top_n: int = 30,
    candidate_pool_sort: str = "total_score",
    candidate_pool_csv: Path | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    allowed_codes = resolve_code_filter(codes=codes, codes_file=codes_file)
    universe = load_universe(data_dir, allowed_codes=allowed_codes)
    if not universe:
        raise ValueError(f"未加载到任何行情数据: {data_dir}")

    trade_dates = _resolve_trade_dates(
        universe=universe,
        trade_date=trade_date or None,
        start_date=start_date or None,
        end_date=end_date or None,
    )
    if int(report_limit) > 0:
        trade_dates = trade_dates[-int(report_limit):]
    if not trade_dates:
        raise ValueError("未解析到任何 trade_dates")

    codes_count = int(len(allowed_codes)) if allowed_codes is not None else int(len(universe))
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = min(max(1, int(workers)), len(trade_dates))
    allowed_codes_tuple = tuple(sorted(allowed_codes)) if allowed_codes is not None else None
    del universe
    gc.collect()

    worker_kwargs = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "runtime_config_path": str(runtime_config_path),
        "a2_config_path": str(a2_config_path),
        "allowed_codes": allowed_codes_tuple,
        "candidate_pool_top_n": int(candidate_pool_top_n),
        "candidate_pool_sort": str(candidate_pool_sort),
        "candidate_pool_csv": str(candidate_pool_csv) if candidate_pool_csv is not None else "",
    }
    if worker_count == 1:
        _init_a2_report_worker(worker_kwargs)
        a2_rows = [_generate_one_a2_report_worker(trade_date) for trade_date in trade_dates]
        _WORKER_CONTEXT.clear()
    else:
        chunksize = max(1, len(trade_dates) // (worker_count * 4))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_a2_report_worker,
            initargs=(worker_kwargs,),
        ) as executor:
            a2_rows = list(executor.map(_generate_one_a2_report_worker, trade_dates, chunksize=chunksize))

    a2_rows = sorted(a2_rows, key=lambda row: str(row.get("trade_date", "")))
    report_files = [Path(str(row["report_json"])) for row in a2_rows]
    report_files = sorted(report_files, key=_trade_date_from_report)
    for row in a2_rows:
        row["report_json"] = display_path(Path(str(row["report_json"])))

    manifest = {
        "report_dir": display_path(output_dir),
        "data_dir": display_path(data_dir),
        "runtime_config_path": display_path(runtime_config_path),
        "a2_config_path": display_path(a2_config_path),
        "codes_count": codes_count,
        "trade_dates_count": int(len(trade_dates)),
        "first_trade_date": trade_dates[0].date().isoformat(),
        "last_trade_date": trade_dates[-1].date().isoformat(),
        "workers": int(worker_count),
        "candidate_pool": {
            "top_n": int(candidate_pool_top_n),
            "sort": str(candidate_pool_sort),
            "csv": display_path(candidate_pool_csv) if candidate_pool_csv is not None else "",
        },
        "reports": [display_path(path) for path in report_files],
        "a2_flat": a2_rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JSON reports with A2 scores")
    parser.add_argument("--data-dir", default="daily_data")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--runtime-config-path", default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--a2-config-path", default=DEFAULT_A2_CONFIG_PATH)
    parser.add_argument("--trade-date", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--report-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="并行生成 report 的进程数，默认 4")
    parser.add_argument("--codes", default="")
    parser.add_argument("--codes-file", default="")
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
    manifest = generate_a2_reports(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        runtime_config_path=Path(args.runtime_config_path).expanduser().resolve(),
        a2_config_path=Path(args.a2_config_path).expanduser().resolve(),
        trade_date=str(args.trade_date).strip(),
        start_date=str(args.start_date).strip(),
        end_date=str(args.end_date).strip(),
        report_limit=int(args.report_limit),
        codes=tuple(token.strip() for token in str(args.codes).split(",") if token.strip()),
        codes_file=codes_file,
        candidate_pool_top_n=int(args.candidate_pool_top_n),
        candidate_pool_sort=str(args.candidate_pool_sort),
        candidate_pool_csv=candidate_pool_csv,
        workers=int(args.workers),
    )
    LOGGER.info("完成生成 A2 reports: %s", json.dumps({key: manifest[key] for key in ("report_dir", "trade_dates_count", "workers")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
