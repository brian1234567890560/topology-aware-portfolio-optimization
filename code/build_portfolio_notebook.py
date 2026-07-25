import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "notebooks" / "research" / "topology_aware_portfolio_optimization.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


cells = [
    md(
        r"""
        # Topology-aware, distribution-aware portfolio optimization

        This notebook implements the current project flow:

        \[
        \text{prices}
        \rightarrow \text{causal returns}
        \rightarrow
        \begin{cases}
        \text{persistent homology regime signal}\\
        \text{Wasserstein/MMD asset geometry}
        \end{cases}
        \rightarrow \text{distributional clustering}
        \rightarrow \text{representative assets}
        \rightarrow \text{spanning diagnostics}
        \rightarrow \text{regime-aware weights}
        \rightarrow \text{walk-forward evaluation}.
        \]

        The notebook is a research prototype, not financial advice. It deliberately uses
        only information available before each rebalance and charges transaction costs.
        """
    ),
    md(
        r"""
        ## Why Python is the recommended language

        **Use Python as the main research language.** It has the strongest single ecosystem
        for this exact combination: `pandas`/`NumPy` for time series, `SciPy` for
        Wasserstein distance and optimization, `ripser` for persistent homology,
        scikit-learn utilities, and Jupyter for mathematical experiments.

        A sensible later architecture is:

        - **Python:** research, topology, clustering, optimization, and backtesting.
        - **SQL:** persistent market/experiment data once the project grows.
        - **C++ or Rust:** only later, if profiling identifies a real production bottleneck.

        Rewriting the research pipeline in C++ now would slow iteration without improving
        the mathematical validity of the model.
        """
    ),
    md(
        """
        ## 0. Install the two nonstandard packages

        Run this once per new environment, then restart the kernel if Jupyter requests it.
        """
    ),
    code(
        """
        %pip install -q yfinance ripser
        """
    ),
    md("## 1. Imports and reproducibility"),
    code(
        """
        from __future__ import annotations

        from dataclasses import dataclass
        from pathlib import Path
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import yfinance as yf

        from IPython.display import display
        from ripser import ripser
        from scipy.optimize import minimize
        from scipy.spatial.distance import pdist
        from scipy.stats import wasserstein_distance

        warnings.filterwarnings("ignore", category=FutureWarning)
        sns.set_theme(style="whitegrid", context="notebook")
        np.random.seed(42)
        """
    ),
    md(
        r"""
        ## 2. Experiment configuration

        Start with `FAST_MODE=True`. After the notebook completes once, change it to
        `False` for more assets, more bootstrap samples, and a longer test.

        All window slices below have the form `returns.iloc[t-lookback:t]`; therefore day
        \(t\) is never used to construct weights that are applied on day \(t\).
        """
    ),
    code(
        """
        @dataclass(frozen=True)
        class Config:
            tickers: tuple[str, ...] = (
                "SPY", "QQQ", "IWM", "EFA", "EEM",
                "TLT", "IEF", "GLD", "VNQ",
                "XLE", "XLF", "XLV",
            )
            market_state_tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA")
            baseline_assets: tuple[str, ...] = ("SPY", "TLT", "GLD")
            benchmark: str = "SPY"
            start: str = "2016-01-01"
            end: str | None = None
            # Optional CSV fallback: first column must be Date; remaining columns are tickers.
            local_price_csv: str | None = None

            annualization: int = 252
            train_window: int = 504
            rebalance_every: int = 21
            tda_window: int = 60
            n_clusters: int = 4
            representatives_per_cluster: int = 1

            wasserstein_weight: float = 0.65
            mmd_weight: float = 0.35
            mmd_max_samples: int = 160

            # "robust" is quick. "bootstrap" follows a null-resampling threshold.
            tda_threshold_mode: str = "robust"
            tda_noise_mad_multiplier: float = 2.5
            tda_bootstrap_samples: int = 20
            tda_bootstrap_quantile: float = 0.95
            tda_block_length: int = 5

            base_risk_aversion: float = 8.0
            regime_risk_multiplier: float = 0.75
            max_weight_normal: float = 0.40
            max_weight_stress: float = 0.25
            turnover_penalty: float = 0.002
            transaction_cost_bps: float = 10.0
            covariance_ridge: float = 1e-6

            output_dir: str = "portfolio_outputs"
            fast_mode: bool = True

        CFG = Config()
        CFG
        """
    ),
    md("## 3. Download adjusted prices and construct log returns"),
    code(
        """
        def download_adjusted_close(
            tickers: tuple[str, ...],
            start: str,
            end: str | None,
            local_price_csv: str | None = None,
        ) -> pd.DataFrame:
            if local_price_csv:
                prices = pd.read_csv(local_price_csv, index_col=0, parse_dates=True)
                prices = prices.loc[:, [x for x in tickers if x in prices.columns]]
            else:
                raw = yf.download(
                    list(tickers),
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    group_by="column",
                )
                if raw.empty:
                    raise RuntimeError(
                        "No data downloaded. Check internet access/tickers, or set "
                        "CFG.local_price_csv to a local adjusted-price CSV."
                    )

                if isinstance(raw.columns, pd.MultiIndex):
                    if "Close" in raw.columns.get_level_values(0):
                        prices = raw["Close"].copy()
                    elif "Close" in raw.columns.get_level_values(1):
                        prices = raw.xs("Close", axis=1, level=1).copy()
                    else:
                        raise KeyError("Could not find adjusted Close prices in yfinance output.")
                else:
                    prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

            prices = prices.sort_index().replace([np.inf, -np.inf], np.nan)
            missing_share = prices.isna().mean()
            keep = missing_share[missing_share <= 0.05].index
            dropped = sorted(set(tickers) - set(keep))
            if dropped:
                print("Dropped for >5% missing observations:", dropped)

            prices = prices.loc[:, keep].ffill(limit=3).dropna()
            return prices


        prices = download_adjusted_close(
            CFG.tickers, CFG.start, CFG.end, CFG.local_price_csv
        )
        returns = np.log(prices).diff().dropna()

        required = set(CFG.market_state_tickers) | set(CFG.baseline_assets) | {CFG.benchmark}
        missing_required = sorted(required - set(returns.columns))
        if missing_required:
            raise ValueError(f"Required assets missing after cleaning: {missing_required}")

        print(f"Prices: {prices.index.min().date()} to {prices.index.max().date()}")
        print(f"Observations: {len(returns):,}; assets: {returns.shape[1]}")
        display(prices.tail())
        """
    ),
    code(
        """
        normalized_prices = prices / prices.iloc[0]
        ax = normalized_prices.plot(figsize=(13, 6), lw=1.2, alpha=0.85)
        ax.set(title="Growth of $1 before portfolio construction", ylabel="Normalized value")
        plt.show()
        """
    ),
    md(
        r"""
        ## 4. Persistent-homology branch

        Each trading day is a point

        \[
        x_t=(r_{\mathrm{SPY},t},r_{\mathrm{QQQ},t},
             r_{\mathrm{IWM},t},r_{\mathrm{EFA},t})\in\mathbb R^4.
        \]

        A trailing window creates a point cloud \(X_t\). We robustly scale its coordinates,
        build a Vietoris–Rips filtration, and compute first homology \(H_1\). A feature
        born at \(b_j\) and dying at \(d_j\) has lifetime

        \[
        \ell_j=d_j-b_j.
        \]

        Short-lived loops are removed as topological noise. The raw returns are **not**
        deleted. Surviving lifetimes generate a causal market-regime score.
        """
    ),
    code(
        """
        def robust_scale_frame(frame: pd.DataFrame, eps: float = 1e-12) -> np.ndarray:
            x = frame.to_numpy(dtype=float)
            center = np.median(x, axis=0)
            mad = np.median(np.abs(x - center), axis=0)
            scale = 1.4826 * mad
            fallback = np.std(x, axis=0, ddof=1)
            scale = np.where(scale > eps, scale, np.where(fallback > eps, fallback, 1.0))
            return (x - center) / scale


        def h1_lifetimes(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            diagram = ripser(point_cloud, maxdim=1)["dgms"][1]
            if diagram.size == 0:
                return diagram.reshape(0, 2), np.array([], dtype=float)
            diagram = diagram[np.isfinite(diagram[:, 1])]
            lifetimes = diagram[:, 1] - diagram[:, 0]
            return diagram, lifetimes


        def moving_block_resample(
            x: np.ndarray, block_length: int, rng: np.random.Generator
        ) -> np.ndarray:
            n = len(x)
            starts = rng.integers(0, max(n - block_length + 1, 1), size=int(np.ceil(n / block_length)))
            blocks = [x[s : min(s + block_length, n)] for s in starts]
            return np.vstack(blocks)[:n]


        def persistence_threshold(
            point_cloud: np.ndarray,
            lifetimes: np.ndarray,
            mode: str,
            rng: np.random.Generator,
        ) -> float:
            if lifetimes.size == 0:
                return np.inf

            if mode == "robust":
                med = np.median(lifetimes)
                mad = np.median(np.abs(lifetimes - med))
                return float(med + CFG.tda_noise_mad_multiplier * 1.4826 * mad)

            if mode == "bootstrap":
                null_maxima = []
                for _ in range(CFG.tda_bootstrap_samples):
                    xb = moving_block_resample(point_cloud, CFG.tda_block_length, rng)
                    _, lb = h1_lifetimes(xb)
                    null_maxima.append(lb.max(initial=0.0))
                return float(np.quantile(null_maxima, CFG.tda_bootstrap_quantile))

            raise ValueError("mode must be 'robust' or 'bootstrap'")


        def topology_summary(
            market_window: pd.DataFrame,
            mode: str = "robust",
            seed: int = 42,
        ) -> dict:
            x = robust_scale_frame(market_window)
            diagram, lifetimes = h1_lifetimes(x)
            threshold = persistence_threshold(
                x, lifetimes, mode=mode, rng=np.random.default_rng(seed)
            )
            keep = lifetimes > threshold
            surviving = lifetimes[keep]
            return {
                "diagram": diagram,
                "lifetimes": lifetimes,
                "threshold": threshold,
                "surviving_lifetimes": surviving,
                "n_surviving": int(keep.sum()),
                "total_persistence_l1": float(surviving.sum()),
                "total_persistence_l2": float(np.sqrt(np.square(surviving).sum())),
            }


        def expanding_robust_z(values: list[float], min_history: int = 6) -> float:
            if len(values) <= min_history:
                return 0.0
            history = np.asarray(values[:-1], dtype=float)
            current = float(values[-1])
            center = np.median(history)
            mad = np.median(np.abs(history - center))
            scale = 1.4826 * mad
            if scale < 1e-12:
                scale = np.std(history, ddof=1)
            return 0.0 if scale < 1e-12 else float((current - center) / scale)


        latest_market_window = returns.loc[:, CFG.market_state_tickers].iloc[-CFG.tda_window :]
        latest_topology = topology_summary(
            latest_market_window, mode=CFG.tda_threshold_mode
        )
        {k: v for k, v in latest_topology.items() if k not in {"diagram", "lifetimes", "surviving_lifetimes"}}
        """
    ),
    code(
        """
        def plot_persistence_diagram(summary: dict, title: str) -> None:
            diagram = summary["diagram"]
            threshold = summary["threshold"]
            fig, ax = plt.subplots(figsize=(7, 6))
            if len(diagram):
                lifetimes = diagram[:, 1] - diagram[:, 0]
                signal = lifetimes > threshold
                ax.scatter(
                    diagram[~signal, 0], diagram[~signal, 1],
                    c="0.7", s=35, label="short-lived / filtered"
                )
                ax.scatter(
                    diagram[signal, 0], diagram[signal, 1],
                    c="crimson", s=55, label="persistent signal"
                )
                upper = float(diagram.max()) * 1.05
            else:
                upper = 1.0
            ax.plot([0, upper], [0, upper], "k--", lw=1, label="birth = death")
            ax.set(
                xlim=(0, upper), ylim=(0, upper),
                xlabel="Birth $b$", ylabel="Death $d$", title=title
            )
            ax.legend()
            plt.show()


        plot_persistence_diagram(
            latest_topology,
            f"Latest $H_1$ persistence diagram ({latest_market_window.index[-1].date()})",
        )
        """
    ),
    md(
        r"""
        ## 5. Wasserstein/MMD asset geometry

        For assets \(i\) and \(j\), the one-dimensional Wasserstein distance compares
        their complete empirical return distributions:

        \[
        W_1(P_i,P_j)=\int_0^1|F_i^{-1}(u)-F_j^{-1}(u)|\,du.
        \]

        MMD compares kernel mean embeddings:

        \[
        \mathrm{MMD}^2(P_i,P_j)
        =\mathbb E[k(X,X')]+\mathbb E[k(Y,Y')]-2\mathbb E[k(X,Y)].
        \]

        The two distance matrices are normalized and combined. K-medoids is used instead
        of ordinary K-means because it accepts a precomputed non-Euclidean distance
        matrix and its cluster center is an actual asset.
        """
    ),
    code(
        """
        def rbf_mmd_distance(
            x: np.ndarray,
            y: np.ndarray,
            max_samples: int,
            rng: np.random.Generator,
        ) -> float:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if len(x) > max_samples:
                x = rng.choice(x, size=max_samples, replace=False)
            if len(y) > max_samples:
                y = rng.choice(y, size=max_samples, replace=False)

            pooled = np.concatenate([x, y])[:, None]
            pairwise = pdist(pooled, metric="euclidean")
            positive = pairwise[pairwise > 0]
            bandwidth = np.median(positive) if positive.size else 1.0
            gamma = 1.0 / (2.0 * max(bandwidth, 1e-12) ** 2)

            kxx = np.exp(-gamma * (x[:, None] - x[None, :]) ** 2).mean()
            kyy = np.exp(-gamma * (y[:, None] - y[None, :]) ** 2).mean()
            kxy = np.exp(-gamma * (x[:, None] - y[None, :]) ** 2).mean()
            return float(np.sqrt(max(kxx + kyy - 2.0 * kxy, 0.0)))


        def pairwise_distribution_distances(
            train_returns: pd.DataFrame, seed: int = 42
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            names = list(train_returns.columns)
            n = len(names)
            dw = np.zeros((n, n), dtype=float)
            dm = np.zeros((n, n), dtype=float)
            rng = np.random.default_rng(seed)

            for i in range(n):
                xi = train_returns.iloc[:, i].dropna().to_numpy()
                for j in range(i + 1, n):
                    xj = train_returns.iloc[:, j].dropna().to_numpy()
                    dw[i, j] = dw[j, i] = wasserstein_distance(xi, xj)
                    dm[i, j] = dm[j, i] = rbf_mmd_distance(
                        xi, xj, CFG.mmd_max_samples, rng
                    )

            def normalize_distance(d: np.ndarray) -> np.ndarray:
                upper = d[np.triu_indices_from(d, k=1)]
                positive = upper[upper > 0]
                scale = np.median(positive) if positive.size else 1.0
                return d / max(scale, 1e-12)

            combined = (
                CFG.wasserstein_weight * normalize_distance(dw)
                + CFG.mmd_weight * normalize_distance(dm)
            )
            return (
                pd.DataFrame(dw, index=names, columns=names),
                pd.DataFrame(dm, index=names, columns=names),
                pd.DataFrame(combined, index=names, columns=names),
            )


        def k_medoids(
            distance: pd.DataFrame, n_clusters: int, max_iter: int = 100
        ) -> tuple[pd.Series, list[str]]:
            names = list(distance.index)
            d = distance.to_numpy()
            n = len(names)
            k = min(max(1, n_clusters), n)

            medoids = [int(np.argmin(d.sum(axis=1)))]
            while len(medoids) < k:
                nearest = d[:, medoids].min(axis=1)
                nearest[medoids] = -np.inf
                medoids.append(int(np.argmax(nearest)))

            for _ in range(max_iter):
                labels = np.argmin(d[:, medoids], axis=1)
                new_medoids = []
                for cluster in range(k):
                    members = np.where(labels == cluster)[0]
                    if len(members) == 0:
                        candidates = [i for i in range(n) if i not in new_medoids]
                        new_medoids.append(max(candidates, key=lambda i: d[i, medoids].min()))
                    else:
                        within = d[np.ix_(members, members)]
                        new_medoids.append(int(members[np.argmin(within.sum(axis=1))]))
                if new_medoids == medoids:
                    break
                medoids = new_medoids

            labels = np.argmin(d[:, medoids], axis=1)
            return pd.Series(labels, index=names, name="cluster"), [names[i] for i in medoids]


        def select_representatives(
            train_returns: pd.DataFrame,
            distance: pd.DataFrame,
            labels: pd.Series,
            per_cluster: int,
        ) -> list[str]:
            annual_mean = train_returns.mean() * CFG.annualization
            annual_vol = train_returns.std(ddof=1) * np.sqrt(CFG.annualization)
            sharpe = annual_mean / annual_vol.replace(0, np.nan)
            selected: list[str] = []

            for cluster in sorted(labels.unique()):
                members = labels.index[labels == cluster].tolist()
                centrality = distance.loc[members, members].mean(axis=1)
                centrality_z = (centrality - centrality.mean()) / (centrality.std(ddof=0) + 1e-12)
                sharpe_z = (sharpe.loc[members] - sharpe.loc[members].mean()) / (
                    sharpe.loc[members].std(ddof=0) + 1e-12
                )
                # Prefer a central distribution, with a modest quality tilt.
                score = -centrality_z + 0.25 * sharpe_z.fillna(0.0)
                selected.extend(score.nlargest(min(per_cluster, len(members))).index.tolist())

            return sorted(set(selected))
        """
    ),
    md("### Current-window clustering and representative assets"),
    code(
        """
        current_train = returns.iloc[-CFG.train_window :]
        current_w, current_mmd, current_distance = pairwise_distribution_distances(current_train)
        current_labels, current_medoids = k_medoids(current_distance, CFG.n_clusters)
        current_selected = select_representatives(
            current_train,
            current_distance,
            current_labels,
            CFG.representatives_per_cluster,
        )

        cluster_table = pd.DataFrame({
            "cluster": current_labels,
            "is_medoid": current_labels.index.isin(current_medoids),
            "selected": current_labels.index.isin(current_selected),
        }).sort_values(["cluster", "selected", "is_medoid"], ascending=[True, False, False])
        display(cluster_table)

        plt.figure(figsize=(10, 8))
        sns.heatmap(current_distance, cmap="viridis", square=True)
        plt.title("Combined normalized Wasserstein/MMD distance")
        plt.show()
        """
    ),
    md(
        r"""
        ## 6. Mean–variance spanning diagnostic

        This section asks whether the selected asset set moves the attainable long-only
        frontier relative to a simple baseline set. For target return \(r_\star\),

        \[
        \min_w w^\top\Sigma w
        \quad\text{s.t.}\quad
        w^\top\mu\ge r_\star,\quad \mathbf 1^\top w=1,\quad w\ge0.
        \]

        This is an **economic frontier diagnostic**, not the formal Huberman–Kandel
        statistical spanning test. The walk-forward comparison later is the more
        important out-of-sample evidence.
        """
    ),
    code(
        """
        def annual_moments(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
            mu = frame.mean() * CFG.annualization
            cov = frame.cov() * CFG.annualization
            cov = cov + np.eye(len(cov)) * CFG.covariance_ridge
            return mu, cov


        def minimum_volatility_for_target(
            mu: pd.Series, cov: pd.DataFrame, target_return: float
        ) -> float:
            n = len(mu)

            def objective(w):
                return float(w @ cov.to_numpy() @ w)

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "ineq", "fun": lambda w: float(w @ mu.to_numpy() - target_return)},
            ]
            result = minimize(
                objective,
                x0=np.repeat(1.0 / n, n),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            return np.sqrt(max(result.fun, 0.0)) if result.success else np.nan


        def efficient_frontier(
            train_returns: pd.DataFrame, targets: np.ndarray
        ) -> pd.Series:
            mu, cov = annual_moments(train_returns)
            values = [minimum_volatility_for_target(mu, cov, target) for target in targets]
            return pd.Series(values, index=targets)


        baseline = [x for x in CFG.baseline_assets if x in current_train.columns]
        expanded = sorted(set(baseline) | set(current_selected))
        mu_base = current_train[baseline].mean() * CFG.annualization
        mu_full = current_train[expanded].mean() * CFG.annualization
        lower = max(float(mu_base.min()), float(mu_full.min()))
        upper = min(float(mu_base.max()), float(mu_full.max()))
        targets = np.linspace(lower, upper, 25)

        frontier_base = efficient_frontier(current_train[baseline], targets)
        frontier_full = efficient_frontier(current_train[expanded], targets)
        spanning_table = pd.DataFrame({
            "baseline_vol": frontier_base,
            "expanded_vol": frontier_full,
            "vol_reduction": frontier_base - frontier_full,
        })
        display(spanning_table.describe().loc[["mean", "min", "max"]])

        plt.figure(figsize=(8, 6))
        plt.plot(frontier_base, targets, label=f"Baseline: {baseline}", lw=2)
        plt.plot(frontier_full, targets, label=f"Expanded: {expanded}", lw=2)
        plt.xlabel("Annualized volatility")
        plt.ylabel("Annualized expected return")
        plt.title("Current-window long-only efficient frontiers")
        plt.legend()
        plt.show()
        """
    ),
    md(
        r"""
        ## 7. Regime-aware optimizer

        For the selected assets, solve

        \[
        \min_w\left[
        -\hat\mu_t^\top w
        +\gamma_t w^\top\hat\Sigma_t w
        +\eta\lVert w-w_{t-1}\rVert_1
        \right],
        \]

        where

        \[
        \gamma_t=\gamma_0\left(1+\alpha\max(R_t,0)\right).
        \]

        When the topological regime score \(R_t\) is high, the optimizer becomes more
        risk-averse and uses a smaller maximum asset weight.
        """
    ),
    code(
        """
        def shrink_covariance(sample_cov: np.ndarray, strength: float) -> np.ndarray:
            diagonal_target = np.diag(np.diag(sample_cov))
            shrunk = (1.0 - strength) * sample_cov + strength * diagonal_target
            return shrunk + np.eye(len(sample_cov)) * CFG.covariance_ridge


        def optimize_weights(
            train_returns: pd.DataFrame,
            previous_weights: pd.Series | None,
            regime_z: float,
        ) -> pd.Series:
            assets = list(train_returns.columns)
            n = len(assets)
            mu = train_returns.mean().to_numpy() * CFG.annualization
            sample_cov = train_returns.cov().to_numpy() * CFG.annualization

            stress = max(float(regime_z), 0.0)
            shrinkage = float(np.clip(0.10 + 0.10 * stress, 0.10, 0.60))
            cov = shrink_covariance(sample_cov, shrinkage)
            gamma = CFG.base_risk_aversion * (1.0 + CFG.regime_risk_multiplier * stress)
            max_weight = CFG.max_weight_stress if stress >= 1.0 else CFG.max_weight_normal
            max_weight = max(max_weight, 1.0 / n)

            if previous_weights is None:
                previous = np.repeat(1.0 / n, n)
            else:
                previous = previous_weights.reindex(assets).fillna(0.0).to_numpy()
                if previous.sum() <= 1e-12:
                    previous = np.repeat(1.0 / n, n)
                else:
                    previous = previous / previous.sum()

            def objective(w):
                expected_return = float(w @ mu)
                variance = float(w @ cov @ w)
                turnover = float(np.abs(w - previous).sum())
                return -expected_return + gamma * variance + CFG.turnover_penalty * turnover

            result = minimize(
                objective,
                x0=np.clip(previous, 0.0, max_weight),
                method="SLSQP",
                bounds=[(0.0, max_weight)] * n,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                options={"maxiter": 1000, "ftol": 1e-12},
            )

            if not result.success:
                print("Optimizer fallback:", result.message)
                weights = np.repeat(1.0 / n, n)
            else:
                weights = np.clip(result.x, 0.0, None)
                weights = weights / weights.sum()
            return pd.Series(weights, index=assets, name="weight")


        latest_weights = optimize_weights(
            current_train[current_selected],
            previous_weights=None,
            regime_z=0.0,
        )
        display(latest_weights.sort_values(ascending=False).to_frame())
        """
    ),
    md(
        """
        ## 8. Strict walk-forward backtest

        At each rebalance:

        1. Use only the trailing training window.
        2. Compute the Wasserstein/MMD geometry and select representatives.
        3. Compute the trailing topological score.
        4. Estimate moments and solve for new weights.
        5. Apply those weights to the next holding period.
        6. Deduct turnover-dependent transaction costs.

        Four strategies are evaluated:

        - **Full model:** topology + clustering + regime-aware optimization.
        - **All-asset MVO:** conventional optimizer without topology or selection.
        - **Equal weight:** all available assets.
        - **SPY:** simple benchmark.
        """
    ),
    code(
        """
        def run_walk_forward(
            all_returns: pd.DataFrame,
            use_tda: bool,
            use_clustering: bool,
            equal_weight: bool = False,
            label: str = "strategy",
        ) -> dict:
            start_t = max(CFG.train_window, CFG.tda_window)
            if CFG.fast_mode:
                start_t = max(start_t, len(all_returns) - 5 * CFG.annualization)

            portfolio_chunks = []
            weight_rows = []
            topology_rows = []
            previous_full = pd.Series(0.0, index=all_returns.columns)
            topology_history: list[float] = []

            rebalance_points = list(range(start_t, len(all_returns), CFG.rebalance_every))
            for step_number, t in enumerate(rebalance_points):
                train = all_returns.iloc[t - CFG.train_window : t]
                hold = all_returns.iloc[t : min(t + CFG.rebalance_every, len(all_returns))]
                if hold.empty:
                    continue

                if use_tda:
                    market_window = all_returns.loc[
                        :, CFG.market_state_tickers
                    ].iloc[t - CFG.tda_window : t]
                    summary = topology_summary(
                        market_window,
                        mode=CFG.tda_threshold_mode,
                        seed=42 + step_number,
                    )
                    topology_history.append(summary["total_persistence_l2"])
                    regime_z = expanding_robust_z(topology_history)
                else:
                    summary = {
                        "total_persistence_l2": 0.0,
                        "n_surviving": 0,
                        "threshold": np.nan,
                    }
                    regime_z = 0.0

                if use_clustering:
                    _, _, distance = pairwise_distribution_distances(
                        train, seed=42 + step_number
                    )
                    labels, _ = k_medoids(distance, CFG.n_clusters)
                    selected = select_representatives(
                        train,
                        distance,
                        labels,
                        CFG.representatives_per_cluster,
                    )
                else:
                    selected = list(train.columns)

                if equal_weight:
                    selected_weights = pd.Series(
                        1.0 / len(selected), index=selected, name="weight"
                    )
                else:
                    selected_weights = optimize_weights(
                        train[selected],
                        previous_weights=previous_full,
                        regime_z=regime_z,
                    )

                full_weights = pd.Series(0.0, index=all_returns.columns)
                full_weights.loc[selected_weights.index] = selected_weights
                turnover = float(np.abs(full_weights - previous_full).sum())

                realized = hold @ full_weights
                realized = realized.copy()
                realized.iloc[0] -= turnover * CFG.transaction_cost_bps / 10_000.0
                portfolio_chunks.append(realized.rename(label))

                row = full_weights.to_dict()
                row.update({
                    "date": hold.index[0],
                    "turnover": turnover,
                    "regime_z": regime_z,
                    "n_selected": len(selected),
                })
                weight_rows.append(row)
                topology_rows.append({
                    "date": hold.index[0],
                    "total_persistence_l2": summary["total_persistence_l2"],
                    "n_surviving": summary["n_surviving"],
                    "threshold": summary["threshold"],
                    "regime_z": regime_z,
                })
                previous_full = full_weights

            strategy_returns = pd.concat(portfolio_chunks).sort_index()
            weights = pd.DataFrame(weight_rows).set_index("date").sort_index()
            topology = pd.DataFrame(topology_rows).set_index("date").sort_index()
            return {"returns": strategy_returns, "weights": weights, "topology": topology}
        """
    ),
    md("### Run the strategies"),
    code(
        """
        full_model = run_walk_forward(
            returns, use_tda=True, use_clustering=True, label="Full model"
        )
        conventional_mvo = run_walk_forward(
            returns, use_tda=False, use_clustering=False, label="All-asset MVO"
        )
        equal_weight = run_walk_forward(
            returns,
            use_tda=False,
            use_clustering=False,
            equal_weight=True,
            label="Equal weight",
        )

        common_start = max(
            full_model["returns"].index.min(),
            conventional_mvo["returns"].index.min(),
            equal_weight["returns"].index.min(),
        )
        strategy_returns = pd.concat(
            [
                full_model["returns"],
                conventional_mvo["returns"],
                equal_weight["returns"],
                returns.loc[common_start:, CFG.benchmark].rename(CFG.benchmark),
            ],
            axis=1,
        ).dropna()
        strategy_returns.tail()
        """
    ),
    md("## 9. Performance, drawdown, turnover, and regime diagnostics"),
    code(
        """
        def max_drawdown(series: pd.Series) -> float:
            wealth = np.exp(series.cumsum())
            running_peak = wealth.cummax().clip(lower=1.0)
            drawdown = wealth / running_peak - 1.0
            return float(drawdown.min())


        def performance_table(
            log_return_frame: pd.DataFrame,
            turnover_by_strategy: dict[str, pd.Series] | None = None,
        ) -> pd.DataFrame:
            rows = {}
            for name, x in log_return_frame.items():
                years = len(x) / CFG.annualization
                total_growth = float(np.exp(x.sum()))
                cagr = total_growth ** (1.0 / years) - 1.0
                vol = float(x.std(ddof=1) * np.sqrt(CFG.annualization))
                annual_return = float(x.mean() * CFG.annualization)
                sharpe = annual_return / vol if vol > 0 else np.nan
                downside = x[x < 0].std(ddof=1) * np.sqrt(CFG.annualization)
                sortino = annual_return / downside if downside > 0 else np.nan
                rows[name] = {
                    "CAGR": cagr,
                    "Annual volatility": vol,
                    "Sharpe (rf=0)": sharpe,
                    "Sortino (rf=0)": sortino,
                    "Max drawdown": max_drawdown(x),
                    "Growth of $1": total_growth,
                }

            table = pd.DataFrame(rows).T
            if turnover_by_strategy:
                table["Mean rebalance turnover"] = pd.Series({
                    k: float(v.mean()) for k, v in turnover_by_strategy.items()
                })
            return table


        turnover_series = {
            "Full model": full_model["weights"]["turnover"],
            "All-asset MVO": conventional_mvo["weights"]["turnover"],
            "Equal weight": equal_weight["weights"]["turnover"],
        }
        metrics = performance_table(strategy_returns, turnover_series)
        display(
            metrics.style.format({
                "CAGR": "{:.2%}",
                "Annual volatility": "{:.2%}",
                "Sharpe (rf=0)": "{:.2f}",
                "Sortino (rf=0)": "{:.2f}",
                "Max drawdown": "{:.2%}",
                "Growth of $1": "{:.2f}",
                "Mean rebalance turnover": "{:.2%}",
            })
        )
        """
    ),
    code(
        """
        wealth = np.exp(strategy_returns.cumsum())
        running_peaks = wealth.cummax().clip(lower=1.0)
        drawdowns = wealth.div(running_peaks).sub(1.0)

        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
        wealth.plot(ax=axes[0], lw=2)
        axes[0].set(title="Strict walk-forward growth of $1", ylabel="Portfolio value")
        drawdowns.plot(ax=axes[1], lw=1.5)
        axes[1].set(title="Drawdowns", ylabel="Drawdown", xlabel="")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        topology_history = full_model["topology"]
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        topology_history["total_persistence_l2"].plot(
            ax=axes[0], color="darkorange", lw=1.8
        )
        axes[0].set(title="Causal topological persistence signal", ylabel="$P_2$")
        topology_history["regime_z"].plot(ax=axes[1], color="crimson", lw=1.8)
        axes[1].axhline(1.0, color="black", ls="--", lw=1)
        axes[1].set(title="Expanding robust regime z-score", ylabel="$R_t$")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        asset_columns = [c for c in returns.columns if c in full_model["weights"].columns]
        fig, ax = plt.subplots(figsize=(13, 7))
        full_model["weights"][asset_columns].plot.area(
            ax=ax, stacked=True, alpha=0.85, linewidth=0
        )
        ax.set(title="Full-model weights at each rebalance", ylabel="Weight", ylim=(0, 1))
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 10. Save reproducible outputs

        The output folder contains the exact daily strategy returns, summary metrics,
        rebalance weights, and topological regime history used by the figures.
        """
    ),
    code(
        """
        output_dir = Path(CFG.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy_returns.to_csv(output_dir / "daily_log_returns.csv")
        metrics.to_csv(output_dir / "performance_metrics.csv")
        full_model["weights"].to_csv(output_dir / "full_model_weights.csv")
        full_model["topology"].to_csv(output_dir / "topology_history.csv")
        cluster_table.to_csv(output_dir / "latest_clusters.csv")

        print(f"Saved outputs to: {output_dir.resolve()}")
        """
    ),
    md(
        r"""
        ## 11. How to interpret the result

        The research hypothesis is supported only if the full model performs more
        reliably **out of sample**, after costs, than the simpler controls. Do not accept
        the model because one backtest has a higher final value.

        Recommended next experiments:

        1. Change only one component at a time: TDA, Wasserstein, MMD, or clustering.
        2. Test multiple non-overlapping periods and universes.
        3. Add a cash/risk-free series and realistic fund-specific costs.
        4. Replace the quick robust persistence cutoff with the bootstrap mode.
        5. Use nested tuning: parameters must be chosen inside each training period.
        6. Report uncertainty with block-bootstrap confidence intervals.

        The central question remains:

        \[
        \boxed{\text{Does topology-aware filtering plus distribution-aware selection
        improve stable out-of-sample portfolio decisions?}}
        \]
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUT)
