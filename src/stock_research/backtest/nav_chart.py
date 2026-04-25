from __future__ import annotations

import argparse
import html
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_research.utils.paths import display_path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NavChartConfig:
    daily_nav_csv: Path
    output_svg: Path
    title: str = "Backtest NAV"
    width: int = 960
    height: int = 560


def resolve_daily_nav_csv(*, backtest_dir: Path | None, daily_nav_csv: Path | None) -> Path:
    if daily_nav_csv is not None:
        return daily_nav_csv
    if backtest_dir is None:
        raise ValueError("需要传入 --backtest-dir 或 --daily-nav-csv")
    return backtest_dir / "daily_nav.csv"


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _format_nav(value: float) -> str:
    return f"{value:.3f}"


def _clean_nav_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "date" not in frame.columns or "nav" not in frame.columns:
        raise ValueError(f"daily_nav 缺少 date/nav 列: {path}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    if "drawdown" not in frame.columns:
        frame["peak_nav"] = frame["nav"].cummax()
        frame["drawdown"] = frame["nav"] / frame["peak_nav"] - 1.0
    frame["drawdown"] = pd.to_numeric(frame["drawdown"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"daily_nav 为空: {path}")
    return frame


def _scale(value: float, *, value_min: float, value_max: float, pixel_min: float, pixel_max: float) -> float:
    if value_max == value_min:
        return (pixel_min + pixel_max) / 2.0
    ratio = (value - value_min) / (value_max - value_min)
    return pixel_max - ratio * (pixel_max - pixel_min)


def _x_at(index: int, count: int, left: float, right: float) -> float:
    if count <= 1:
        return (left + right) / 2.0
    return left + (right - left) * index / float(count - 1)


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _date_ticks(frame: pd.DataFrame, limit: int = 6) -> list[int]:
    count = len(frame)
    if count <= limit:
        return list(range(count))
    step = max(1, round((count - 1) / float(limit - 1)))
    ticks = list(range(0, count, step))
    if ticks[-1] != count - 1:
        ticks.append(count - 1)
    return ticks


def render_nav_chart_svg(config: NavChartConfig) -> str:
    frame = _clean_nav_frame(config.daily_nav_csv)
    width = max(520, int(config.width))
    height = max(360, int(config.height))
    left = 76.0
    right = width - 44.0
    nav_top = 76.0
    nav_bottom = height * 0.67
    dd_top = nav_bottom + 52.0
    dd_bottom = height - 74.0
    nav_values = frame["nav"].astype(float).tolist()
    drawdowns = frame["drawdown"].astype(float).tolist()
    nav_min = min(min(nav_values), 1.0)
    nav_max = max(max(nav_values), 1.0)
    if nav_max == nav_min:
        nav_min -= 0.02
        nav_max += 0.02
    padding = max((nav_max - nav_min) * 0.08, 0.01)
    nav_min -= padding
    nav_max += padding
    dd_min = min(min(drawdowns), 0.0)
    if dd_min == 0.0:
        dd_min = -0.01

    nav_points = [
        (
            _x_at(index, len(frame), left, right),
            _scale(value, value_min=nav_min, value_max=nav_max, pixel_min=nav_top, pixel_max=nav_bottom),
        )
        for index, value in enumerate(nav_values)
    ]
    baseline_y = _scale(1.0, value_min=nav_min, value_max=nav_max, pixel_min=nav_top, pixel_max=nav_bottom)
    drawdown_bars = []
    for index, value in enumerate(drawdowns):
        x = _x_at(index, len(frame), left, right)
        y = dd_top + (0.0 - float(value)) / (0.0 - dd_min) * (dd_bottom - dd_top)
        bar_width = max(1.0, min(6.0, (right - left) / max(len(frame), 1) * 0.72))
        drawdown_bars.append(
            f'<rect x="{x - bar_width / 2:.2f}" y="{dd_top:.2f}" width="{bar_width:.2f}" '
            f'height="{max(0.0, y - dd_top):.2f}" fill="#d86f45" opacity="0.55" />'
        )

    grid_lines: list[str] = []
    axis_labels: list[str] = []
    for tick in range(5):
        value = nav_min + (nav_max - nav_min) * tick / 4.0
        y = _scale(value, value_min=nav_min, value_max=nav_max, pixel_min=nav_top, pixel_max=nav_bottom)
        grid_lines.append(f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}" stroke="#e6e2dc" />')
        axis_labels.append(
            f'<text x="{left - 12:.2f}" y="{y + 4:.2f}" text-anchor="end" class="axis-label">{html.escape(_format_nav(value))}</text>'
        )
    for index in _date_ticks(frame):
        x = _x_at(index, len(frame), left, right)
        label = pd.Timestamp(frame.loc[index, "date"]).date().isoformat()
        grid_lines.append(f'<line x1="{x:.2f}" y1="{nav_top:.2f}" x2="{x:.2f}" y2="{dd_bottom:.2f}" stroke="#f0ede8" />')
        axis_labels.append(
            f'<text x="{x:.2f}" y="{height - 36:.2f}" text-anchor="middle" class="axis-label">{html.escape(label)}</text>'
        )
    dd_label_y = dd_top + (0.0 - dd_min) / (0.0 - dd_min) * (dd_bottom - dd_top)
    axis_labels.append(
        f'<text x="{left - 12:.2f}" y="{dd_top + 4:.2f}" text-anchor="end" class="axis-label">0.00%</text>'
    )
    axis_labels.append(
        f'<text x="{left - 12:.2f}" y="{dd_label_y + 4:.2f}" text-anchor="end" class="axis-label">{html.escape(_format_percent(dd_min))}</text>'
    )

    title = html.escape(config.title)
    start_date = pd.Timestamp(frame["date"].iloc[0]).date().isoformat()
    end_date = pd.Timestamp(frame["date"].iloc[-1]).date().isoformat()
    final_nav = float(nav_values[-1])
    total_return = final_nav - 1.0
    max_drawdown = min(drawdowns)
    points_text = _polyline(nav_points)
    summary = (
        f"{html.escape(start_date)} -> {html.escape(end_date)}  "
        f"NAV {html.escape(_format_nav(final_nav))}  "
        f"Return {html.escape(_format_percent(total_return))}  "
        f"Max DD {html.escape(_format_percent(max_drawdown))}"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{summary}</desc>
  <style>
    .title {{ font: 700 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #25211c; }}
    .subtitle {{ font: 500 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #6f6860; }}
    .axis-label {{ font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #7b746b; }}
    .section-label {{ font: 600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4f4941; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" fill="#fbfaf7" />
  <text x="{left:.2f}" y="36" class="title">{title}</text>
  <text x="{left:.2f}" y="58" class="subtitle">{summary}</text>
  <text x="{left:.2f}" y="{nav_top - 14:.2f}" class="section-label">NAV</text>
  <text x="{left:.2f}" y="{dd_top - 16:.2f}" class="section-label">Drawdown</text>
  {"".join(grid_lines)}
  <line x1="{left:.2f}" y1="{baseline_y:.2f}" x2="{right:.2f}" y2="{baseline_y:.2f}" stroke="#9c9288" stroke-dasharray="4 5" />
  <line x1="{left:.2f}" y1="{nav_bottom:.2f}" x2="{right:.2f}" y2="{nav_bottom:.2f}" stroke="#c9c1b8" />
  <line x1="{left:.2f}" y1="{dd_top:.2f}" x2="{right:.2f}" y2="{dd_top:.2f}" stroke="#c9c1b8" />
  {"".join(drawdown_bars)}
  <polyline points="{points_text}" fill="none" stroke="#1e6f68" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round" />
  {"".join(axis_labels)}
</svg>
"""


def build_nav_chart(config: NavChartConfig) -> Path:
    svg = render_nav_chart_svg(config)
    config.output_svg.parent.mkdir(parents=True, exist_ok=True)
    config.output_svg.write_text(svg, encoding="utf-8")
    return config.output_svg


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an SVG NAV curve from backtest daily_nav.csv")
    parser.add_argument("--backtest-dir", default="")
    parser.add_argument("--daily-nav-csv", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--title", default="Backtest NAV")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=560)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    backtest_dir = Path(args.backtest_dir).expanduser().resolve() if str(args.backtest_dir).strip() else None
    daily_nav_csv = Path(args.daily_nav_csv).expanduser().resolve() if str(args.daily_nav_csv).strip() else None
    resolved_daily_nav_csv = resolve_daily_nav_csv(backtest_dir=backtest_dir, daily_nav_csv=daily_nav_csv)
    if str(args.output).strip():
        output_svg = Path(args.output).expanduser().resolve()
    elif backtest_dir is not None:
        output_svg = backtest_dir / "nav_curve.svg"
    else:
        output_svg = resolved_daily_nav_csv.with_name("nav_curve.svg")
    output = build_nav_chart(
        NavChartConfig(
            daily_nav_csv=resolved_daily_nav_csv,
            output_svg=output_svg,
            title=str(args.title),
            width=int(args.width),
            height=int(args.height),
        )
    )
    LOGGER.info("完成净值曲线图: %s", display_path(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
