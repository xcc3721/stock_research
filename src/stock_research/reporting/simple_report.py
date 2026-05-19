from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stock_research.scoring.factor_calculator import FactorCalculator
from stock_research.scoring.scoring_engine import ScoringEngine
from stock_research.selectors.bbikdj import BBIKDJSelector, BBIKDJSelectorConfig
from stock_research.utils.paths import display_path


LOGGER = logging.getLogger(__name__)

FACTOR_NAME_MAP = {
    "ma_structure": "均线结构",
    "b1_comprehensive": "B1综合因子",
    "symmetry_resonance": "对称走势",
    "kline_patterns": "关键K线",
    "pocket_pivot": "口袋妖怪",
    "vcp_compression": "VCP压缩",
    "supply_analysis": "供给分析",
    "pivot_analysis": "枢轴分析",
    "risk_penalty": "风险控制",
    "structure_analysis": "异动",
    "b1_template_similarity": "B1形态相似度",
}


@dataclass(frozen=True)
class RuntimeConfig:
    selector: BBIKDJSelectorConfig
    scoring_config: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置必须是 JSON object: {path}")
    return payload


def _normalize_code(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text


def resolve_code_filter(
    *,
    codes: Iterable[str] | None = None,
    codes_file: Path | None = None,
) -> set[str] | None:
    resolved: set[str] = set()
    for code in codes or []:
        normalized = _normalize_code(code)
        if normalized:
            resolved.add(normalized)
    if codes_file is not None:
        for line in codes_file.read_text(encoding="utf-8").splitlines():
            normalized = _normalize_code(line)
            if normalized:
                resolved.add(normalized)
    return resolved or None


def _read_price_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if "volume" not in frame.columns:
        frame["volume"] = pd.NA
    if "amount" not in frame.columns:
        frame["amount"] = pd.NA
    return frame


def load_runtime_config(path: Path) -> RuntimeConfig:
    payload = _load_json(path)
    selector_payload = payload.get("selector", {})
    scoring_config = payload.get("scoring_config")
    if not isinstance(scoring_config, dict):
        raise ValueError("配置缺少 scoring_config")
    return RuntimeConfig(
        selector=BBIKDJSelectorConfig(**selector_payload),
        scoring_config=scoring_config,
    )


def load_universe(
    data_dir: Path,
    *,
    allowed_codes: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        code = _normalize_code(csv_path.stem)
        if not code:
            continue
        if allowed_codes is not None and code not in allowed_codes:
            continue
        frame = _read_price_frame(csv_path)
        if frame.empty:
            continue
        frames[code] = frame
    return frames


def available_trade_dates(universe: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    # _read_price_frame already normalizes dates; avoid per-cell Timestamp creation
    dates = set()
    for frame in universe.values():
        dates.update(frame["date"].dropna().unique())
    return sorted(pd.to_datetime(list(dates)))


def build_factor_details(
    *,
    factors: dict[str, float],
    weighted_scores: dict[str, float],
) -> list[dict[str, Any]]:
    details = []
    for key, raw_value in factors.items():
        details.append(
            {
                "name": FACTOR_NAME_MAP.get(key, key),
                "factor_key": key,
                "raw_value": float(raw_value),
                "weighted_score": float(weighted_scores.get(key, 0.0)),
            }
        )
    return details


def build_batch_row(
    *,
    code: str,
    hist: pd.DataFrame,
    factor_calculator: FactorCalculator,
    scoring_engine: ScoringEngine,
) -> dict[str, Any]:
    factors = factor_calculator.calculate_all_factors(hist, code)
    score_result = scoring_engine.calculate_score(factors, code)
    weighted_scores = score_result.get("weighted_scores", {})
    if not isinstance(weighted_scores, dict):
        weighted_scores = {}
    return {
        "code": code,
        "name": code,
        "selected": True,
        "action_text": str(score_result.get("action_text", "买入")),
        "final_score": float(score_result.get("total_score", 0.0)),
        "weighted_scores": {
            str(key): float(value)
            for key, value in weighted_scores.items()
            if pd.notna(value)
        },
        "factors": {str(key): float(value) for key, value in factors.items() if pd.notna(value)},
        "factor_details": build_factor_details(
            factors={str(key): float(value) for key, value in factors.items() if pd.notna(value)},
            weighted_scores={str(key): float(value) for key, value in weighted_scores.items() if pd.notna(value)},
        ),
        "result": {
            "total_score": float(score_result.get("total_score", 0.0)),
            "action_text": str(score_result.get("action_text", "买入")),
            "weighted_scores": {
                str(key): float(value)
                for key, value in weighted_scores.items()
                if pd.notna(value)
            },
            "factors": {str(key): float(value) for key, value in factors.items() if pd.notna(value)},
        },
    }


def build_report_payload(
    *,
    trade_date: pd.Timestamp,
    config_path: Path,
    picks: list[dict[str, Any]],
    universe_size: int,
) -> dict[str, Any]:
    trade_date_text = trade_date.date().isoformat()
    return {
        "schema_version": 1,
        "report_id": f"bbikdj_{trade_date.strftime('%Y%m%d')}",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data": {
            "config_path": display_path(config_path),
            "summary": {
                "trade_date": trade_date_text,
                "selected_stocks": int(len(picks)),
                "universe_size": int(universe_size),
                "strategy": "BBIKDJSelector",
            },
            "scoring_results": {
                "trade_date": trade_date_text,
                "strategy": "BBIKDJSelector",
                "batch_results": picks,
            },
        },
    }


def generate_one_report(
    *,
    trade_date: pd.Timestamp,
    universe: dict[str, pd.DataFrame],
    runtime_config: RuntimeConfig,
    config_path: Path,
) -> dict[str, Any]:
    selector = BBIKDJSelector(runtime_config.selector)
    factor_calculator = FactorCalculator(runtime_config.scoring_config)
    scoring_engine = ScoringEngine(runtime_config.scoring_config)
    selected_codes = selector.select(trade_date, universe)
    picks: list[dict[str, Any]] = []
    for code in selected_codes:
        hist = universe[code]
        hist = hist[hist["date"] <= trade_date].tail(max(int(runtime_config.selector.max_window) + 30, 260))
        if hist.empty:
            continue
        picks.append(
            build_batch_row(
                code=code,
                hist=hist,
                factor_calculator=factor_calculator,
                scoring_engine=scoring_engine,
            )
        )
    picks.sort(key=lambda row: (float(row.get("final_score", 0.0)), str(row.get("code", ""))), reverse=True)
    return build_report_payload(
        trade_date=trade_date,
        config_path=config_path,
        picks=picks,
        universe_size=len(universe),
    )


def generate_reports(
    *,
    data_dir: Path,
    output_dir: Path,
    config_path: Path,
    trade_dates: list[pd.Timestamp],
    allowed_codes: set[str] | None = None,
) -> list[Path]:
    runtime_config = load_runtime_config(config_path)
    universe = load_universe(data_dir, allowed_codes=allowed_codes)
    if not universe:
        raise ValueError(f"data_dir 下没有可用 csv: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for trade_date in trade_dates:
        payload = generate_one_report(
            trade_date=trade_date,
            universe=universe,
            runtime_config=runtime_config,
            config_path=config_path,
        )
        target = output_dir / f"bbikdj_{trade_date.strftime('%Y%m%d')}.report_data.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(target)
    return written


def _resolve_trade_dates(
    *,
    universe: dict[str, pd.DataFrame],
    trade_date: str | None,
    start_date: str | None,
    end_date: str | None,
) -> list[pd.Timestamp]:
    all_dates = available_trade_dates(universe)
    if not all_dates:
        return []
    if trade_date:
        target = pd.Timestamp(trade_date).normalize()
        return [target]
    start = pd.Timestamp(start_date).normalize() if start_date else all_dates[0]
    end = pd.Timestamp(end_date).normalize() if end_date else all_dates[-1]
    return [value for value in all_dates if start <= value <= end]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simplified report_data.json for BBIKDJSelector")
    parser.add_argument("--data-dir", default="daily_data")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--config-path", default="configs/runtime.json")
    parser.add_argument("--trade-date", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--codes", default="")
    parser.add_argument("--codes-file", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    codes_file = Path(args.codes_file).expanduser().resolve() if str(args.codes_file).strip() else None
    allowed_codes = resolve_code_filter(
        codes=[token.strip() for token in str(args.codes).split(",") if token.strip()],
        codes_file=codes_file,
    )
    universe = load_universe(data_dir, allowed_codes=allowed_codes)
    dates = _resolve_trade_dates(
        universe=universe,
        trade_date=str(args.trade_date).strip() or None,
        start_date=str(args.start_date).strip() or None,
        end_date=str(args.end_date).strip() or None,
    )
    written = generate_reports(
        data_dir=data_dir,
        output_dir=output_dir,
        config_path=config_path,
        trade_dates=dates,
        allowed_codes=allowed_codes,
    )
    LOGGER.info("完成生成 report_data.json: %d", len(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
