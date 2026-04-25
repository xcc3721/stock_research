from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_research.scoring.b1_market_gate import B1MarketGateResolver
from stock_research.scoring.b1_policy import V3_LOGIC, resolve_b1_logic


def test_b1_default_logic_is_v3() -> None:
    assert resolve_b1_logic({}) == V3_LOGIC


def test_b1_market_gate_can_read_user_supplied_state_csv(tmp_path: Path) -> None:
    gate_path = tmp_path / "market_state.csv"
    pd.DataFrame(
        [
            {"date": "2025-01-02", "active": 0, "regime": "neutral"},
            {"date": "2025-01-03", "active": 1, "regime": "hot_up"},
        ]
    ).to_csv(gate_path, index=False, encoding="utf-8-sig")

    resolver = B1MarketGateResolver(
        {
            "b1_comprehensive": {
                "params": {
                    "market_gate": {
                        "path": str(gate_path),
                        "active_col": "active",
                        "regime_col": "regime",
                    }
                }
            }
        }
    )

    state = resolver.resolve_for_date("2025-01-06")

    assert state["available"] is True
    assert state["is_hot_up"] is True
    assert state["market_date"] == "2025-01-03"
    assert state["heat_regime"] == "hot_up"
