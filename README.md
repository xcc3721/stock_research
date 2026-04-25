# stock_research

`stock_research` 是一个基于本地 A 股日线数据的选股研究工具。它提供三件事：

1. 通过 Tushare 更新本地 `daily_data/*.csv`
2. 生成包含 BBIKDJ 选股结果、子因子明细和 A2 排序结果的 JSON 报告
3. 基于报告目录执行固定持有期回测

项目不包含 HTML 渲染和行情数据文件。默认配置只保留两份：

- `configs/runtime.json`
- `configs/a2/flat_v1_stable_penalty_overlay.json`

## Install

```bash
cd stock_research
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

配置 Tushare Token：

```bash
cp .env.example .env
```

在 `.env` 中写入：

```bash
TUSHARE_TOKEN=your_tushare_token_here
```

## Data

日线数据放在：

```text
daily_data/<code>.csv
```

必要列：

- `date`
- `open`
- `high`
- `low`
- `close`

推荐列：

- `volume`
- `amount`

## Usage

### 1. 更新日线数据

更新已有 `daily_data/*.csv`：

```bash
stock-research-update-data --data-dir daily_data
```

更新指定股票：

```bash
stock-research-update-data --data-dir daily_data --codes 000001,000002
```

需要按 Tushare 当前上市股票列表全量更新时：

```bash
stock-research-update-data --data-dir daily_data --bootstrap-listed
```

`--bootstrap-listed` 会拉取当前全部上市 A 股，首次运行耗时较长，也可能受 Tushare 频率限制影响。默认遇到单只股票更新失败会记录 warning 并继续；如需严格中断：

```bash
stock-research-update-data --data-dir daily_data --bootstrap-listed --fail-fast
```

### 2. 生成 A2 报告

```bash
stock-research-generate-reports \
  --data-dir daily_data \
  --output-dir reports \
  --start-date 2026-04-01 \
  --end-date 2026-04-24
```

报告文件会写入：

```text
reports/*.report_data.json
reports/manifest.json
```

每个 `report_data.json` 包含：

- `data.summary.trade_date`
- `data.summary.selected_stocks` 和 `data.summary.universe_size`
- `data.scoring_results.batch_results`，这是唯一的候选股明细列表
- 候选股的 `factors`、`weighted_scores`、`final_score`
- 进入 A2 候选池股票的 `A2_flat`

默认逻辑是先按原始 `total_score` 保留前 `30` 只，再计算 A2 排序。可以调整：

```bash
stock-research-generate-reports \
  --data-dir daily_data \
  --output-dir reports \
  --report-limit 20 \
  --candidate-pool-top-n 30 \
  --candidate-pool-sort total_score
```

### 3. 回测

```bash
stock-research-backtest \
  --report-dir reports \
  --price-data-dir daily_data \
  --output-dir backtest/a2_flat_v1 \
  --score-col a2_flat_v1_score \
  --horizon 10 \
  --top-n 1
```

输出文件：

- `backtest/a2_flat_v1/predictions.csv`
- `backtest/a2_flat_v1/summary.csv`
- `backtest/a2_flat_v1/trades.csv`
- `backtest/a2_flat_v1/skipped_trades.csv`
- `backtest/a2_flat_v1/daily_nav.csv`

## B1 Scoring

B1 综合因子默认使用 `v3_winner` 口径。高级用户如需接入自己的市场状态开关，可以在 `scoring_config.factors.b1_comprehensive.params` 中配置：

```json
{
  "scoring_mode": "hot_up_use_v2_else_v3",
  "market_gate": {
    "path": "data/market_state.csv",
    "date_col": "date",
    "active_col": "is_hot_up",
    "regime_col": "heat_regime"
  }
}
```

`market_gate.path` 指向用户自行准备的本地 CSV。默认配置不启用该扩展。

## Development

运行测试：

```bash
pytest -q
```

## License

本项目代码使用 MIT License，详见 `LICENSE`。

MIT License 只覆盖本仓库代码和文档，不覆盖用户通过 Tushare 或其他数据源获取的行情数据。用户需要自行遵守数据源服务条款，并自行承担数据使用、研究结论和交易决策风险。

## Notes

- `daily_data/*.csv` 通过 Tushare 接口拉取
- 回测按每日信号独立开仓，同一标的已有持仓时不额外去重
- 回测遇到持仓标的缺失价格日时，会沿用上一收盘价保留 exposure，并在下一个可用价格日退出
- 回测不会在最后一个净值日开新仓；无法入场的信号会写入 `skipped_trades.csv`
- 报告真实交易日读取 `data.summary.trade_date`，不要用文件名推断交易日
