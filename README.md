# stock_research

`stock_research` 是一个基于本地 A 股日线数据的选股研究工具，支持 Tushare 数据更新、A2 JSON 报告生成、固定持有期回测和净值曲线输出。

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

必要列：`date`、`open`、`high`、`low`、`close`。

推荐列：`volume`、`amount`。

## Usage

### 1. 更新日线数据

首次使用建议先按 Tushare 当前上市股票列表初始化并全量更新：

```bash
stock-research-update-data --data-dir daily_data --bootstrap-listed
```

`--bootstrap-listed` 首次运行耗时较长，也可能受 Tushare 频率限制影响。默认遇到单只股票更新失败会记录 warning 并继续；如需遇错中断，可加 `--fail-fast`。

已有 `daily_data/*.csv` 后，可以只增量更新本地已有股票：

```bash
stock-research-update-data --data-dir daily_data
```

也可以更新指定股票：

```bash
stock-research-update-data --data-dir daily_data --codes 000001,000002
```

### 2. 生成 A2 报告

```bash
stock-research-generate-reports \
  --data-dir daily_data \
  --output-dir reports \
  --start-date 2026-04-01 \
  --end-date 2026-04-24 \
  --workers 4
```

报告文件会写入：

```text
reports/*.report_data.json
reports/manifest.json
```

每个 `report_data.json` 包含：

- `data.summary.trade_date`
- `data.summary.selected_stocks` 和 `data.summary.universe_size`
- `data.scoring_results.batch_results`，包含候选股明细
- 候选股的 `factors`、`weighted_scores`、`final_score`
- 进入 A2 候选池股票的 `A2_flat`

`--workers` 控制并行进程数，默认 `4`。

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

命令结束时会在终端打印核心统计。

### 4. 输出净值曲线

从回测输出目录生成 SVG：

```bash
stock-research-plot-nav \
  --backtest-dir backtest/a2_flat_v1 \
  --output backtest/a2_flat_v1/nav_curve.svg \
  --title "A2 Flat V1 NAV"
```

## Development

运行测试：

```bash
pytest -q
```

## License

本项目代码使用 MIT License，详见 `LICENSE`。

MIT License 只覆盖本仓库代码和文档，不覆盖用户通过 Tushare 或其他数据源获取的行情数据。用户需要自行遵守数据源服务条款，并自行承担数据使用、研究结论和交易决策风险。
