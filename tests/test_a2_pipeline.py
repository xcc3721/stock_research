from __future__ import annotations

import json
from pathlib import Path

from stock_research.pipeline.a2_pipeline import A2PipelineConfig, run_a2_pipeline


def test_run_a2_pipeline_smoke(synthetic_workspace: dict[str, object]) -> None:
    trade_dates = synthetic_workspace["trade_dates"]
    start_date = trade_dates[-25].date().isoformat()
    end_date = trade_dates[-1].date().isoformat()
    output_root = Path(synthetic_workspace["root"]) / "pipeline_run"

    manifest = run_a2_pipeline(
        A2PipelineConfig(
            data_dir=Path(synthetic_workspace["data_dir"]),
            output_root=output_root,
            runtime_config_path=Path(synthetic_workspace["runtime_config_path"]),
            a2_config_path=Path(synthetic_workspace["a2_config_path"]),
            start_date=start_date,
            end_date=end_date,
            report_limit=20,
            score_col="a2_flat_v1_score",
            horizon=2,
            top_n=1,
            cost_bps=0.0,
            candidate_pool_top_n=0,
        )
    )

    manifest_path = output_root / "run_manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["resolved"]["trade_dates_count"] == 20
    assert saved["resolved"]["trade_dates_count"] == 20
    assert len(saved["generated_reports"]) == 20
    assert saved["a2_flat"]
    assert Path(saved["predictions_csv"]).exists()
    assert Path(saved["backtest_outputs"]["summary"]).exists()
    assert Path(saved["backtest_outputs"]["trades"]).exists()
    assert Path(saved["backtest_outputs"]["skipped_trades"]).exists()
    assert Path(saved["backtest_outputs"]["daily_nav"]).exists()
