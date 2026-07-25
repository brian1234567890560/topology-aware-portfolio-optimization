# Results

No final empirical performance claim is committed yet.

The notebook can generate the following outputs after a fixed experiment is run:

| Output | File |
|---|---|
| Daily out-of-sample strategy returns | `portfolio_outputs/daily_log_returns.csv` |
| Performance metrics | `portfolio_outputs/performance_metrics.csv` |
| Full-model weights and turnover | `portfolio_outputs/full_model_weights.csv` |
| Topological regime history | `portfolio_outputs/topology_history.csv` |
| Latest cluster assignments | `portfolio_outputs/latest_clusters.csv` |

## Reporting template

Every committed experiment should record:

- execution date;
- Git commit;
- data source and download date;
- asset universe;
- sample period;
- training and TDA windows;
- rebalance interval;
- transaction-cost assumption;
- clustering parameters;
- optimizer parameters;
- random seed;
- failed or dropped assets.

## Required tables

1. Full-period performance comparison.
2. Crisis and non-crisis subperiod comparison.
3. Turnover and cost comparison.
4. Component ablation table.
5. Sensitivity analysis.
6. Bootstrap confidence intervals.

## Required figures

1. Growth of one dollar.
2. Drawdowns.
3. Topological persistence and regime score.
4. Portfolio weights through time.
5. Efficient-frontier diagnostic.
6. Distribution-distance heatmap and cluster assignments.

Results should not be copied into this page until their complete configuration and source CSVs are committed.

