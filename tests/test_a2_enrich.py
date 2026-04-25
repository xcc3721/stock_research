from __future__ import annotations

import json
from pathlib import Path

from stock_research.a2.enrich_report_with_a2_flat import enrich_report_with_a2_flat
from stock_research.reporting.simple_report import generate_reports
from tests.helpers import make_report_payload


def test_a2_can_enrich_generated_reports(synthetic_workspace: dict[str, object]) -> None:
    trade_dates = list(synthetic_workspace["trade_dates"][-30:])
    written = generate_reports(
        data_dir=Path(synthetic_workspace["data_dir"]),
        output_dir=Path(synthetic_workspace["reports_dir"]),
        config_path=Path(synthetic_workspace["runtime_config_path"]),
        trade_dates=trade_dates,
    )

    latest_report = written[-1]
    a2_summary = enrich_report_with_a2_flat(
        report_json=latest_report,
        config_path=Path(synthetic_workspace["a2_config_path"]),
        candidate_pool_top_n=2,
    )
    assert a2_summary["trade_date"] == trade_dates[-1].date().isoformat()
    assert int(a2_summary["rows"]) >= 1
    assert a2_summary["top1"]
    assert a2_summary["candidate_pool"]["mode"] == "top_n"
    assert int(a2_summary["candidate_pool"]["top_n"]) == 2
    assert a2_summary["tie_break_source_counts"]

    payload = json.loads(latest_report.read_text(encoding="utf-8"))
    data = payload["data"]
    batch_results = data["scoring_results"]["batch_results"]

    assert "A2_flat" in data
    assert "selected_stocks" not in data
    assert "selected_stocks" not in data["scoring_results"]
    assert batch_results
    assert any(row.get("A2_flat") for row in batch_results)
    assert any("a2_flat_v1_score" in row for row in batch_results)
    assert any(
        (
            row.get("A2_flat", {}).get("component_values", {}).get("report_total_score") is not None
            and row.get("A2_flat", {}).get("tie_break_source") == "report_total_score"
        )
        for row in batch_results
        if isinstance(row.get("A2_flat"), dict)
    )


def test_a2_enrich_removes_legacy_duplicate_selected_lists(
    tmp_path: Path,
    synthetic_workspace: dict[str, object],
) -> None:
    rows = [
        {
            "code": "000001",
            "name": "A",
            "final_score": 60.0,
            "weighted_scores": {"ma_structure": 1.0},
            "factors": {"ma_structure": 1.0},
            "result": {
                "total_score": 60.0,
                "weighted_scores": {"ma_structure": 1.0},
                "factors": {"ma_structure": 1.0},
            },
        }
    ]
    payload = make_report_payload("2025-01-02", rows)
    payload["data"]["selected_stocks"] = list(rows)
    payload["data"]["scoring_results"]["selected_stocks"] = list(rows)
    report_json = tmp_path / "legacy.report_data.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    enrich_report_with_a2_flat(
        report_json=report_json,
        config_path=Path(synthetic_workspace["a2_config_path"]),
        candidate_pool_top_n=0,
    )

    data = json.loads(report_json.read_text(encoding="utf-8"))["data"]
    assert "selected_stocks" not in data
    assert "selected_stocks" not in data["scoring_results"]
    assert len(data["scoring_results"]["batch_results"]) == 1
