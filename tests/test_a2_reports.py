from __future__ import annotations

import json
from pathlib import Path

from stock_research.reporting.a2_reports import generate_a2_reports


def test_generate_a2_reports_writes_reports_with_a2_scores(synthetic_workspace: dict[str, object]) -> None:
    trade_dates = synthetic_workspace["trade_dates"]
    output_dir = Path(synthetic_workspace["root"]) / "a2_reports"

    manifest = generate_a2_reports(
        data_dir=Path(synthetic_workspace["data_dir"]),
        output_dir=output_dir,
        runtime_config_path=Path(synthetic_workspace["runtime_config_path"]),
        a2_config_path=Path(synthetic_workspace["a2_config_path"]),
        start_date=trade_dates[-5].date().isoformat(),
        end_date=trade_dates[-1].date().isoformat(),
        candidate_pool_top_n=2,
        works=2,
    )

    manifest_path = output_dir / "manifest.json"
    report_paths = sorted(output_dir.glob("*.report_data.json"))

    assert manifest_path.exists()
    assert manifest["trade_dates_count"] == 5
    assert manifest["works"] == 2
    assert len(report_paths) == 5
    assert len(manifest["a2_flat"]) == 5

    payload = json.loads(report_paths[-1].read_text(encoding="utf-8"))
    data = payload["data"]
    batch_results = data["scoring_results"]["batch_results"]

    assert "A2_flat" in data
    assert "selected_stocks" not in data
    assert "selected_stocks" not in data["scoring_results"]
    assert any("A2_flat" in row for row in batch_results)
    assert any("a2_flat_v1_score" in row for row in batch_results)
