from __future__ import annotations

import json
from pathlib import Path

from stock_research.reporting.simple_report import generate_reports


def test_generate_simple_report_outputs_required_fields(synthetic_workspace: dict[str, object]) -> None:
    trade_date = synthetic_workspace["trade_dates"][-1]
    written = generate_reports(
        data_dir=Path(synthetic_workspace["data_dir"]),
        output_dir=Path(synthetic_workspace["reports_dir"]),
        config_path=Path(synthetic_workspace["runtime_config_path"]),
        trade_dates=[trade_date],
    )

    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    summary = payload["data"]["summary"]
    batch_results = payload["data"]["scoring_results"]["batch_results"]

    assert "selected_stocks" not in payload["data"]
    assert "selected_stocks" not in payload["data"]["scoring_results"]
    assert summary["trade_date"] == trade_date.date().isoformat()
    assert summary["selected_stocks"] >= 1
    assert summary["strategy"] == "BBIKDJSelector"
    assert batch_results

    row = batch_results[0]
    assert row["selected"] is True
    assert {"code", "name", "final_score", "weighted_scores", "factors", "factor_details", "result"} <= set(row)
    assert isinstance(row["weighted_scores"], dict)
    assert isinstance(row["factors"], dict)
    assert isinstance(row["factor_details"], list)
    assert row["result"]["action_text"] == row["action_text"]
