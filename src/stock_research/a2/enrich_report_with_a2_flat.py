from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.scoring.explicit_mechanical_anchor import ExplicitMechanicalAnchor
from stock_research.utils.paths import display_path


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_A2_CONFIG_PATH = REPO_ROOT / "configs" / "a2" / "flat_v1_stable_penalty_overlay.json"
BREAKOUT_CLUSTER_KEYS = (
    "ma_structure",
    "vcp_compression",
    "symmetry_resonance",
    "pocket_pivot",
)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 结构非法: {path}")
    return payload


def _load_scoring_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    config = payload.get("scoring_config", payload)
    if not isinstance(config, dict):
        raise ValueError(f"评分配置非法: {path}")
    return config


def _extract_embedded_mechanical_config(config: dict[str, Any]) -> dict[str, Any]:
    anchor = config.get("mechanical_anchor")
    if not isinstance(anchor, dict):
        raise ValueError("A2 配置缺少 mechanical_anchor；如需使用外部配置，请传 --mechanical-config-path")
    scoring_config = anchor.get("scoring_config", anchor)
    if not isinstance(scoring_config, dict):
        raise ValueError("A2 mechanical_anchor.scoring_config 必须是 JSON object")
    factors = scoring_config.get("factors")
    if not isinstance(factors, dict) or not factors:
        raise ValueError("A2 mechanical_anchor.scoring_config 缺少 factors")
    return scoring_config


def _resolve_mechanical_config(
    *,
    a2_config: dict[str, Any],
    mechanical_config_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mechanical_config_path is not None:
        return _load_scoring_config(mechanical_config_path), {
            "source": "external_path",
            "path": display_path(mechanical_config_path),
        }
    return _extract_embedded_mechanical_config(a2_config), {
        "source": "a2_config",
        "config_key": "mechanical_anchor.scoring_config",
    }


def _load_a2_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    components = config.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"A2 配置缺少 components: {path}")
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("A2 component 必须是 JSON object")
        for key in ("name", "source_col", "weight"):
            if key not in component:
                raise ValueError(f"A2 component 缺少字段: {key}")
    return config


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _json_safe_data(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_data(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_data(item) for item in value]
    return value


def _normalize_code(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text


def _extract_trade_date(payload: dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, dict):
        return ""
    summary = data.get("summary")
    if isinstance(summary, dict):
        trade_date = summary.get("trade_date")
        if trade_date:
            return str(trade_date)[:10]
    scoring_results = data.get("scoring_results")
    if isinstance(scoring_results, dict):
        trade_date = scoring_results.get("trade_date")
        if trade_date:
            return str(trade_date)[:10]
    return ""


def _extract_batch_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("report data 非法")
    scoring_results = data.setdefault("scoring_results", {})
    if not isinstance(scoring_results, dict):
        raise ValueError("report scoring_results 非法")
    batch_results = scoring_results.get("batch_results")
    if not isinstance(batch_results, list):
        raise ValueError("report batch_results 非法")
    return [row for row in batch_results if isinstance(row, dict)]


def _extract_factors(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("factors")
    if not isinstance(raw, dict):
        result = row.get("result")
        raw = result.get("factors") if isinstance(result, dict) else {}
    factors: dict[str, float] = {}
    if not isinstance(raw, dict):
        return factors
    for key, value in raw.items():
        number = _safe_float(value)
        if number is not None:
            factors[str(key)] = number
    return factors


def _breakout_cluster_score(factors: dict[str, float], weighted_scores: dict[str, Any]) -> float:
    total = 0.0
    for key in BREAKOUT_CLUSTER_KEYS:
        weighted_value = _safe_float(weighted_scores.get(key))
        if weighted_value is not None:
            total += weighted_value
            continue
        total += float(factors.get(key, 0.0) or 0.0)
    return float(total)


def _extract_report_score(row: dict[str, Any]) -> float | None:
    for key in ("final_score", "score", "total_score"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    result = row.get("result")
    if isinstance(result, dict):
        for key in ("final_score", "score", "total_score"):
            value = _safe_float(result.get(key))
            if value is not None:
                return value
    return None


def _extract_total_score_tiebreak(report_score: float | None, canonical_score: float | None) -> tuple[float, str]:
    if report_score is not None:
        return float(report_score), "report_total_score"
    return float(canonical_score or 0.0), "canonical_mechanical_anchor_score_fallback"


def _rank_pct(
    frame: pd.DataFrame,
    *,
    source_col: str,
    rank_method: str,
    higher_is_better: bool,
) -> pd.Series:
    values = pd.to_numeric(frame[source_col], errors="coerce")
    return values.rank(method=rank_method, pct=True, ascending=bool(higher_is_better)).fillna(0.0)


def _build_source_frame(
    *,
    batch_results: list[dict[str, Any]],
    mechanical_engine: ExplicitMechanicalAnchor,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(batch_results):
        code = _normalize_code(row.get("code"))
        factors = _extract_factors(row)
        canonical_result = mechanical_engine.calculate_score(factors, code) if factors else {}
        weighted_scores = canonical_result.get("weighted_scores")
        if not isinstance(weighted_scores, dict):
            weighted_scores = {}
        canonical_score = _safe_float(canonical_result.get("total_score"))
        report_score = _extract_report_score(row)
        tiebreak_score, tiebreak_source = _extract_total_score_tiebreak(report_score, canonical_score)
        rows.append(
            {
                "row_index": int(row_index),
                "code": code,
                "name": str(row.get("name") or row.get("stock_name") or ""),
                "report_score": float(report_score or 0.0),
                "report_total_score": float(report_score or 0.0),
                "canonical_mechanical_anchor_score": float(canonical_score or 0.0),
                "breakout_cluster_score": _breakout_cluster_score(factors, weighted_scores),
                "vcp_compression__weighted": float(_safe_float(weighted_scores.get("vcp_compression")) or 0.0),
                "ma_structure__directional": float(_safe_float(factors.get("ma_structure")) or 0.0),
                "kline_patterns__weighted": float(_safe_float(weighted_scores.get("kline_patterns")) or 0.0),
                "pivot_analysis__weighted": float(_safe_float(weighted_scores.get("pivot_analysis")) or 0.0),
                "a2_flat_total_score_tiebreak": float(tiebreak_score),
                "a2_flat_tiebreak_source": tiebreak_source,
            }
        )
    return pd.DataFrame(rows)


def _load_candidate_pool_codes(
    *,
    candidate_pool_csv: Path,
    trade_date: str,
) -> set[str]:
    frame = pd.read_csv(candidate_pool_csv, dtype={"code": str}, low_memory=False)
    date_col = "report_date" if "report_date" in frame.columns else "trade_date" if "trade_date" in frame.columns else ""
    if not date_col or "code" not in frame.columns:
        raise ValueError(f"候选池 CSV 必须包含 report_date/trade_date 与 code 列: {candidate_pool_csv}")
    dates = pd.to_datetime(frame[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    day = frame.loc[dates == trade_date, "code"].map(_normalize_code)
    return {code for code in day.tolist() if code}


def _apply_candidate_pool(
    *,
    source_frame: pd.DataFrame,
    trade_date: str,
    candidate_pool_top_n: int,
    candidate_pool_sort: str,
    candidate_pool_csv: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": False,
        "mode": "all",
        "input_rows": int(len(source_frame)),
        "scored_rows": int(len(source_frame)),
        "excluded_rows": 0,
    }
    if source_frame.empty:
        return source_frame.copy(), summary

    filtered = source_frame.copy()
    if candidate_pool_csv is not None:
        pool_codes = _load_candidate_pool_codes(candidate_pool_csv=candidate_pool_csv, trade_date=trade_date)
        report_codes = set(filtered["code"].astype(str))
        filtered = filtered.loc[filtered["code"].astype(str).isin(pool_codes)].copy()
        summary.update(
            {
                "enabled": True,
                "mode": "csv",
                "candidate_pool_csv": display_path(candidate_pool_csv),
                "csv_codes_for_date": int(len(pool_codes)),
                "csv_codes_matched": int(len(report_codes & pool_codes)),
                "csv_codes_missing_from_report": int(len(pool_codes - report_codes)),
            }
        )
    elif candidate_pool_top_n > 0:
        sort = str(candidate_pool_sort or "total_score")
        if sort in {"total_score", "report_score"}:
            filtered = filtered.sort_values(["report_total_score", "code"], ascending=[False, True], kind="mergesort")
        elif sort == "canonical_mechanical_anchor_score":
            filtered = filtered.sort_values(
                ["canonical_mechanical_anchor_score", "report_total_score", "code"],
                ascending=[False, False, True],
                kind="mergesort",
            )
        elif sort == "report_order":
            filtered = filtered.sort_values("row_index", ascending=True, kind="mergesort")
        else:
            raise ValueError(f"未知 candidate_pool_sort: {candidate_pool_sort}")
        filtered = filtered.head(int(candidate_pool_top_n)).copy()
        summary.update(
            {
                "enabled": True,
                "mode": "top_n",
                "top_n": int(candidate_pool_top_n),
                "sort": sort,
            }
        )

    filtered = filtered.sort_values("row_index", ignore_index=True)
    summary["scored_rows"] = int(len(filtered))
    summary["excluded_rows"] = int(len(source_frame) - len(filtered))
    return filtered, summary


def _attach_a2_scores(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"components": [], "abs_weight_sum": 0.0}
    components = config["components"]
    rank_method = str(config.get("rank_method", "average"))
    score_col = str(config.get("score_col", "a2_flat_v1_score"))
    output = frame.copy()
    weighted_score = pd.Series(0.0, index=output.index, dtype="float64")
    abs_weight_sum = 0.0
    component_summary: list[dict[str, Any]] = []

    for component in components:
        name = str(component["name"])
        source_col = str(component["source_col"])
        if source_col not in output.columns:
            raise ValueError(f"缺少 A2 源列: {source_col}")
        weight = float(component["weight"])
        higher_is_better = bool(component.get("higher_is_better", True))
        rank_col = f"{score_col}__{name}"
        rank = _rank_pct(
            output,
            source_col=source_col,
            rank_method=rank_method,
            higher_is_better=higher_is_better,
        )
        output[rank_col] = rank
        weighted_score = weighted_score + rank * weight
        abs_weight_sum += abs(weight)
        component_summary.append(
            {
                "name": name,
                "source_col": source_col,
                "rank_col": rank_col,
                "weight": weight,
                "higher_is_better": higher_is_better,
                "nonnull_rank_rows": int(rank.notna().sum()),
            }
        )

    if abs_weight_sum <= 0.0:
        raise ValueError("A2 components 权重绝对值之和必须大于 0")
    output[score_col] = weighted_score / abs_weight_sum

    tie_break_summary: dict[str, Any] | None = None
    tie_break = config.get("tie_break")
    if isinstance(tie_break, dict):
        tie_source_col = str(tie_break.get("source_col", "")).strip()
        tie_epsilon = float(tie_break.get("epsilon", 0.0) or 0.0)
        if tie_source_col and tie_source_col in output.columns and tie_epsilon > 0.0:
            tie_rank = _rank_pct(
                output,
                source_col=tie_source_col,
                rank_method=rank_method,
                higher_is_better=bool(tie_break.get("higher_is_better", True)),
            )
            output[score_col] = output[score_col] + tie_rank * tie_epsilon
            tie_break_summary = {
                "source_col": tie_source_col,
                "epsilon": tie_epsilon,
                "higher_is_better": bool(tie_break.get("higher_is_better", True)),
                "nonnull_rank_rows": int(tie_rank.notna().sum()),
            }

    output = output.sort_values([score_col, "a2_flat_total_score_tiebreak", "code"], ascending=[False, False, True], kind="mergesort")
    output["a2_flat_rank"] = range(1, len(output) + 1)
    output = output.sort_values("row_index", ignore_index=True)
    return output, {
        "components": component_summary,
        "abs_weight_sum": float(abs_weight_sum),
        "tie_break": tie_break_summary,
        "score_col": score_col,
        "rank_method": rank_method,
        "normalization": config.get("normalization", "sum_abs_weights"),
    }


def _summary_top_list(frame: pd.DataFrame, score_col: str, limit: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    ordered = frame.sort_values([score_col, "a2_flat_total_score_tiebreak", "code"], ascending=[False, False, True], kind="mergesort")
    rows: list[dict[str, Any]] = []
    for row in ordered.head(limit).itertuples(index=False):
        rows.append(
            {
                "code": _normalize_code(getattr(row, "code", "")),
                "name": str(getattr(row, "name", "") or ""),
                "score": _safe_float(getattr(row, score_col, None)),
                "rank": _safe_int(getattr(row, "a2_flat_rank", None)),
                "canonical_mechanical_anchor_score": _safe_float(getattr(row, "canonical_mechanical_anchor_score", None)),
                "breakout_cluster_score": _safe_float(getattr(row, "breakout_cluster_score", None)),
                "report_total_score": _safe_float(getattr(row, "a2_flat_total_score_tiebreak", None)),
            }
        )
    return rows


def _merge_scores_into_report(
    *,
    payload: dict[str, Any],
    scored_frame: pd.DataFrame,
    score_col: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    batch_results = _extract_batch_results(payload)
    row_map = {int(row.row_index): row for row in scored_frame.itertuples(index=False)}
    component_rank_cols = [column for column in scored_frame.columns if column.startswith(f"{score_col}__")]
    candidate_pool = summary.get("candidate_pool")
    clear_excluded = isinstance(candidate_pool, dict) and bool(candidate_pool.get("enabled"))
    for row_index, row in enumerate(batch_results):
        scored = row_map.get(row_index)
        if scored is None:
            if clear_excluded:
                row.pop("A2_flat", None)
                row.pop(score_col, None)
                model_scores = row.get("model_scores")
                if isinstance(model_scores, dict):
                    model_scores.pop("a2_flat", None)
                    if model_scores:
                        row["model_scores"] = model_scores
                    else:
                        row.pop("model_scores", None)
            continue
        component_values = {
            "report_total_score": _safe_float(getattr(scored, "report_total_score", None)),
            "canonical_mechanical_anchor_score": _safe_float(getattr(scored, "canonical_mechanical_anchor_score", None)),
            "breakout_cluster_score": _safe_float(getattr(scored, "breakout_cluster_score", None)),
            "vcp_compression__weighted": _safe_float(getattr(scored, "vcp_compression__weighted", None)),
            "ma_structure__directional": _safe_float(getattr(scored, "ma_structure__directional", None)),
            "kline_patterns__weighted": _safe_float(getattr(scored, "kline_patterns__weighted", None)),
            "pivot_analysis__weighted": _safe_float(getattr(scored, "pivot_analysis__weighted", None)),
            "a2_flat_total_score_tiebreak": _safe_float(getattr(scored, "a2_flat_total_score_tiebreak", None)),
        }
        component_ranks = {
            column.replace(f"{score_col}__", ""): _safe_float(getattr(scored, column, None))
            for column in component_rank_cols
        }
        score_payload = {
            "model_key": "a2_flat",
            "strategy_id": str(summary.get("strategy_id", "A2_flat_v1")),
            "score_col": score_col,
            "score": _safe_float(getattr(scored, score_col, None)),
            "rank": _safe_int(getattr(scored, "a2_flat_rank", None)),
            "is_top1": bool(int(getattr(scored, "a2_flat_rank", 0) or 0) == 1),
            "component_values": component_values,
            "component_ranks": component_ranks,
            "tie_break_source": str(getattr(scored, "a2_flat_tiebreak_source", "") or ""),
        }
        row["A2_flat"] = score_payload
        row[score_col] = score_payload["score"]
        model_scores = row.get("model_scores")
        if not isinstance(model_scores, dict):
            model_scores = {}
        model_scores["a2_flat"] = score_payload
        row["model_scores"] = model_scores

    data = payload.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("report data 非法")
    data.pop("selected_stocks", None)
    scoring_results = data.setdefault("scoring_results", {})
    if isinstance(scoring_results, dict):
        scoring_results.pop("selected_stocks", None)
        scoring_results["a2_flat_summary"] = summary
        model_summary = scoring_results.get("model_scores_summary")
        if not isinstance(model_summary, dict):
            model_summary = {}
        model_summary["a2_flat"] = summary
        scoring_results["model_scores_summary"] = model_summary
    data["A2_flat"] = summary
    payload["data"] = data
    return payload


def _build_summary(
    *,
    report_json: Path,
    trade_date: str,
    config_path: Path,
    mechanical_config_path: Path | None,
    mechanical_config_source: dict[str, Any],
    config: dict[str, Any],
    scored_frame: pd.DataFrame,
    attach_summary: dict[str, Any],
) -> dict[str, Any]:
    score_col = str(attach_summary.get("score_col", config.get("score_col", "a2_flat_v1_score")))
    tiebreak_counts: dict[str, int] = {}
    if "a2_flat_tiebreak_source" in scored_frame.columns:
        for source, count in scored_frame["a2_flat_tiebreak_source"].value_counts(dropna=False).items():
            tiebreak_counts[str(source)] = int(count)
    score_series = pd.to_numeric(scored_frame[score_col], errors="coerce") if score_col in scored_frame.columns else pd.Series(dtype=float)
    top5 = _summary_top_list(scored_frame, score_col, 5)
    return {
        "model_key": "a2_flat",
        "strategy_id": str(config.get("strategy_id", "A2_flat_v1")),
        "description": str(config.get("description", "")),
        "trade_date": trade_date,
        "report_json": display_path(report_json),
        "a2_flat_config_path": display_path(config_path),
        "mechanical_config_path": display_path(mechanical_config_path) if mechanical_config_path is not None else None,
        "mechanical_config_source": mechanical_config_source,
        "score_col": score_col,
        "rows": int(len(scored_frame)),
        "rank_method": str(attach_summary.get("rank_method", config.get("rank_method", "average"))),
        "normalization": attach_summary.get("normalization", config.get("normalization", "sum_abs_weights")),
        "abs_weight_sum": _safe_float(attach_summary.get("abs_weight_sum")),
        "tie_break": attach_summary.get("tie_break"),
        "candidate_pool": attach_summary.get("candidate_pool"),
        "tie_break_source_counts": tiebreak_counts,
        "components": attach_summary.get("components", []),
        "score_min": _safe_float(score_series.min()) if not score_series.empty else None,
        "score_max": _safe_float(score_series.max()) if not score_series.empty else None,
        "top1": top5[0] if top5 else {},
        "top5": top5,
    }


def enrich_report_with_a2_flat(
    *,
    report_json: Path,
    config_path: Path,
    mechanical_config_path: Path | None = None,
    candidate_pool_top_n: int = 30,
    candidate_pool_sort: str = "total_score",
    candidate_pool_csv: Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(report_json)
    batch_results = _extract_batch_results(payload)
    trade_date = _extract_trade_date(payload)
    a2_config = _load_a2_config(config_path)
    mechanical_config, mechanical_config_source = _resolve_mechanical_config(
        a2_config=a2_config,
        mechanical_config_path=mechanical_config_path,
    )
    mechanical_engine = ExplicitMechanicalAnchor(mechanical_config)
    source_frame = _build_source_frame(batch_results=batch_results, mechanical_engine=mechanical_engine)
    pooled_frame, candidate_pool_summary = _apply_candidate_pool(
        source_frame=source_frame,
        trade_date=trade_date,
        candidate_pool_top_n=int(candidate_pool_top_n),
        candidate_pool_sort=candidate_pool_sort,
        candidate_pool_csv=candidate_pool_csv,
    )
    scored_frame, attach_summary = _attach_a2_scores(pooled_frame, a2_config)
    attach_summary["candidate_pool"] = candidate_pool_summary
    score_col = str(attach_summary.get("score_col", a2_config.get("score_col", "a2_flat_v1_score")))
    summary = _build_summary(
        report_json=report_json,
        trade_date=trade_date,
        config_path=config_path,
        mechanical_config_path=mechanical_config_path,
        mechanical_config_source=mechanical_config_source,
        config=a2_config,
        scored_frame=scored_frame,
        attach_summary=attach_summary,
    )
    summary = _json_safe_data(summary)
    merged = _merge_scores_into_report(
        payload=payload,
        scored_frame=scored_frame,
        score_col=score_col,
        summary=summary,
    )
    report_json.write_text(json.dumps(_json_safe_data(merged), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Enrich one report_data.json with A2-flat v1 score")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--price-data-dir", default="daily_data")
    parser.add_argument("--config-path", default=str(DEFAULT_A2_CONFIG_PATH))
    parser.add_argument(
        "--mechanical-config-path",
        default="",
        help="可选：覆盖 A2 配置内嵌的 mechanical_anchor.scoring_config",
    )
    parser.add_argument("--candidate-pool-top-n", type=int, default=30, help="A2 打分前仅保留报告候选池前 N；0 表示不裁剪")
    parser.add_argument(
        "--candidate-pool-sort",
        default="total_score",
        choices=("total_score", "report_order", "report_score", "canonical_mechanical_anchor_score"),
        help="candidate-pool-top-n 的排序来源；默认按原始 total/final score",
    )
    parser.add_argument("--candidate-pool-csv", default="", help="按外部 CSV 的 report_date/code 精确约束 A2 候选池")
    args = parser.parse_args()

    report_json = _resolve_repo_path(args.report_json)
    config_path = _resolve_repo_path(args.config_path)
    mechanical_config_path = (
        _resolve_repo_path(args.mechanical_config_path)
        if str(args.mechanical_config_path or "").strip()
        else None
    )
    candidate_pool_csv = _resolve_repo_path(args.candidate_pool_csv) if str(args.candidate_pool_csv or "").strip() else None
    summary = enrich_report_with_a2_flat(
        report_json=report_json,
        config_path=config_path,
        mechanical_config_path=mechanical_config_path,
        candidate_pool_top_n=int(args.candidate_pool_top_n),
        candidate_pool_sort=str(args.candidate_pool_sort),
        candidate_pool_csv=candidate_pool_csv,
    )
    LOGGER.info(
        "A2-flat enrich 完成: report=%s trade_date=%s rows=%d top1=%s",
        report_json.name,
        summary.get("trade_date", ""),
        int(summary.get("rows", 0) or 0),
        summary.get("top1", {}).get("code") if isinstance(summary.get("top1"), dict) else "",
    )
    LOGGER.info("A2-flat summary: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
