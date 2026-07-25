# Validation and Limitations

This project is a research prototype. A mathematically sophisticated pipeline can still overfit, leak information, or fail after costs. This page separates what is implemented from what has been scientifically established.

## What is currently implemented

- Causal trailing data slices.
- Adjusted-price download with local CSV fallback.
- Robust market-state scaling.
- $H_1$ persistent homology.
- Robust and moving-block-bootstrap persistence thresholds.
- Wasserstein and MMD distance matrices.
- K-medoids clustering and representative selection.
- Long-only efficient-frontier diagnostic.
- Regime-aware constrained optimization.
- Walk-forward benchmark comparison.
- Transaction-cost deduction.
- Exportable result tables and diagnostic plots.

## What has not yet been established

- That persistent homology predicts crashes.
- That the topological score improves allocations out of sample.
- That Wasserstein/MMD clustering improves performance.
- That the selected parameters generalize across universes and periods.
- That the strategy survives realistic bid-ask spreads, slippage, taxes, and liquidity constraints.
- That the descriptive frontier comparison satisfies a formal mean-variance spanning test.
- That reported improvements are statistically distinguishable from sampling variation.

## Known implementation cautions

### 1. Return convention

The pipeline models log returns, but some formulas and transaction-cost operations are easiest to interpret with simple returns. The current cumulative-wealth calculation correctly exponentiates cumulative log returns. Before a final study, all cost and holding-period equations should use one explicitly documented convention.

### 2. Turnover convention

The notebook computes

```python
np.abs(new_weights - old_weights).sum()
```

and deducts costs from that full $L^1$ change. Many finance texts define one-way turnover as half of that value. The final experiment must choose one convention and apply it consistently in the code, equations, and reported tables.

### 3. Mean estimation error

Sample means are extremely noisy. Because the objective includes estimated expected return, small changes in the training sample may cause unstable weights. Possible comparisons include:

- minimum variance;
- mean shrinkage;
- Black-Litterman;
- robust or distributionally robust objectives.

These should be tested as separate optimizer variants.

### 4. Topological interpretation

Longer $H_1$ persistence means that loop structure survives over more distance thresholds. It is not automatically a crash magnitude, probability, or causal warning signal. The score should be described as a market-geometry or regime indicator until predictive evidence exists.

### 5. Threshold selection

The robust threshold is fast but heuristic. The moving-block bootstrap better respects serial dependence but is computationally more expensive. Threshold settings must be selected inside training data, not after examining full-period results.

### 6. Asset-universe bias

The example uses currently available ETFs. This can create survivorship and availability bias if conclusions are generalized to historical stock selection. A larger study needs a historically valid universe.

### 7. Yahoo Finance data

Yahoo Finance is convenient for research but is not an institutional data source. Corporate actions, missing values, symbol changes, and download revisions should be checked against a second source before strong conclusions.

### 8. K-medoids approximation

The current clustering is distribution-aware K-medoids, not true Wasserstein K-means with Wasserstein barycenters. Claims should use the implemented method's name.

### 9. Spanning interpretation

The notebook compares estimated long-only efficient frontiers. It does not implement the complete Huberman-Kandel or related statistical spanning test.

### 10. Multiple testing

Trying many thresholds, kernels, universes, windows, and cost assumptions can produce a favorable result by chance. The final study needs a frozen primary specification and separately labeled exploratory tests.

## Required validation sequence

### Phase A — Software checks

1. Run a synthetic-data smoke test.
2. Verify distance matrices are finite, symmetric, and have zero diagonals.
3. Verify cluster labels and representative selection for edge cases.
4. Verify every weight vector is feasible and sums to one.
5. Verify holding returns begin only after their estimation window.
6. Verify output files reproduce plotted values.

### Phase B — Baseline experiment

Freeze:

- asset universe;
- data range;
- training window;
- TDA window;
- rebalance interval;
- transaction cost;
- cluster count;
- all optimizer parameters.

Then compare the full model with all-asset MVO, equal weight, and SPY.

### Phase C — Ablations

Run:

| Experiment | TDA | Wasserstein | MMD | Clustering |
|---|---:|---:|---:|---:|
| Conventional MVO | No | No | No | No |
| TDA only | Yes | No | No | No |
| Wasserstein selection | No | Yes | No | Yes |
| MMD selection | No | No | Yes | Yes |
| Distributional selection | No | Yes | Yes | Yes |
| Full model | Yes | Yes | Yes | Yes |

Only one component should change at a time when estimating its marginal contribution.

### Phase D — Robustness

Test:

- multiple market periods;
- pre-crisis, crisis, and post-crisis subperiods;
- alternate universes;
- different rebalance frequencies;
- higher transaction costs;
- different training and topology windows;
- robust versus bootstrap persistence thresholds.

### Phase E — Statistical uncertainty

Use moving-block-bootstrap confidence intervals for performance differences. When comparing Sharpe ratios or forecasting claims, use tests appropriate for dependent financial returns and correct for multiple comparisons.

### Phase F — Nested tuning

At outer rebalance $t$:

1. use only data before $t$;
2. split the training history internally;
3. select hyperparameters on inner folds;
4. refit on the full outer training window;
5. apply once to the unseen outer holding interval.

This is the appropriate path for selecting $\alpha$, $K$, $\tau$, $\gamma_0$, turnover penalty, window lengths, and weight caps.

## Minimum evidence for a positive conclusion

A positive conclusion should require:

- improved prespecified risk-adjusted performance after costs;
- no unacceptable increase in drawdown or turnover;
- consistent direction across multiple periods;
- positive ablation evidence;
- uncertainty intervals that do not make the result trivial;
- no look-ahead leakage;
- complete reproducibility from a clean environment.

Until then, the correct conclusion is that the project provides an implemented and testable research hypothesis.

