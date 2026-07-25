import json
from pathlib import Path


NOTEBOOK = Path("/workspace/scratch/686f951bbed7/topology_aware_portfolio_optimization.ipynb")


MARKDOWN = {
0: r"""# Topology-aware, distribution-aware portfolio optimization

## Mathematical objective

At each rebalance date \(t\), the project chooses portfolio weights

\[
\mathbf w_t=(w_{1,t},\ldots,w_{n,t})^\top\in\mathbb R^n
\]

using only information available before \(t\). The final goal is not merely to
fit historical returns, but to produce stable out-of-sample allocations:

\[
\boxed{
\text{market data}
\longrightarrow
\text{structural information}
\longrightarrow
\text{asset selection}
\longrightarrow
\mathbf w_t
\longrightarrow
\text{future portfolio performance}
}
\]

This notebook combines two complementary geometric descriptions of the market:

\[
\begin{aligned}
\text{time geometry}
&:\quad
\text{persistent homology}
\longrightarrow \text{regime score }R_t,\\[2mm]
\text{asset geometry}
&:\quad
\text{Wasserstein distance and MMD}
\longrightarrow \text{asset clusters}.
\end{aligned}
\]

The complete pipeline is

\[
\text{prices}
\rightarrow \text{log returns}
\rightarrow
\begin{cases}
\text{topological regime estimation},\\
\text{distributional asset distances}
\end{cases}
\rightarrow
\text{representative assets}
\rightarrow
\text{spanning diagnostic}
\rightarrow
\text{regime-aware optimization}
\rightarrow
\text{walk-forward evaluation}.
\]

All estimates are causal: weights used after date \(t\) are constructed only
from observations dated at or before \(t\). Transaction costs are deducted.

> This notebook is a research prototype, not financial advice.
""",
1: r"""## Why Python is the recommended language

Python is the best primary language for this project because the full
mathematical pipeline can be expressed and tested in one environment:

- `pandas` and `NumPy` for prices, returns, vectors, and matrices;
- `SciPy` for Wasserstein distance and constrained optimization;
- `ripser` for Vietoris--Rips persistent homology;
- scikit-learn tools for preprocessing and statistical utilities;
- Jupyter for combining derivations, code, and experimental results.

A sensible future architecture is:

- **Python:** research, topology, clustering, optimization, and backtesting;
- **SQL:** persistent market and experiment data when the project grows;
- **C++ or Rust:** only for a measured production bottleneck.

The mathematical correctness of the model depends primarily on causal
estimation and valid out-of-sample testing, not on using a lower-level language.
""",
2: r"""## 0. Install the nonstandard packages

Run the installation cell once in each new environment. If Jupyter or Colab asks
for a kernel restart, restart it before continuing.
""",
4: r"""## 1. Imports and reproducibility

Fixing a random seed makes stochastic steps reproducible. If two runs use the
same data, parameters, and seed, they should produce the same sampled bootstrap
windows and clustering initializations.
""",
6: r"""## 2. Experiment configuration and information timing

Begin with `FAST_MODE=True`. After the entire notebook runs successfully, set it
to `False` to use more assets, more bootstrap samples, and a longer experiment.

Let \(r_{i,s}\) denote the log return of asset \(i\) on day \(s\). At rebalance
index \(t\), every estimator uses only the trailing information set

\[
\mathcal F_t
=
\sigma\!\left(
\{r_{i,s}:s<t,\ i=1,\ldots,n\}
\right).
\]

In the code this is enforced by slices of the form

```python
returns.iloc[t - lookback:t]
```

Python excludes the right endpoint, so observation \(t\) is not used to create
weights applied beginning at \(t\). This prevents look-ahead bias.
""",
8: r"""## 3. Prices and log returns

For adjusted closing price \(P_{i,t}>0\), the one-period log return is

\[
r_{i,t}
=
\log\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
=
\log P_{i,t}-\log P_{i,t-1}.
\]

Log returns are used because consecutive returns add across time:

\[
\sum_{s=t_0+1}^{t_1} r_{i,s}
=
\log\!\left(\frac{P_{i,t_1}}{P_{i,t_0}}\right).
\]

The code first attempts to download adjusted prices. A local adjusted-price CSV
can be supplied when online data are unavailable.
""",
11: r"""## 4. Persistent-homology branch: market geometry through time

### 4.1 Rolling market-state point cloud

Choose \(m\) broad market indices. Their returns on day \(s\) form one market
state

\[
\mathbf x_s
=
\begin{bmatrix}
r_{1,s}\\
r_{2,s}\\
\vdots\\
r_{m,s}
\end{bmatrix}
\in\mathbb R^m.
\]

For a trailing window of \(L\) days ending before rebalance \(t\), define

\[
X_t
=
\left\{
\mathbf x_{t-L},\ldots,\mathbf x_{t-1}
\right\}.
\]

Thus, the **vertices are trading days**, while the coordinates describe the
simultaneous returns of the chosen indices.

### 4.2 Robust coordinate scaling

For coordinate \(q\), use its median and median absolute deviation:

\[
\operatorname{MAD}_q
=
\operatorname{median}_{s}
\left|
x_{s,q}-\operatorname{median}_{u}(x_{u,q})
\right|.
\]

The robustly standardized coordinate is approximately

\[
\widetilde x_{s,q}
=
\frac{x_{s,q}-\operatorname{median}_{u}(x_{u,q})}
{1.4826\,\operatorname{MAD}_q+\varepsilon_{\mathrm{num}}},
\]

where \(\varepsilon_{\mathrm{num}}>0\) prevents division by zero.

### 4.3 Vietoris--Rips filtration and first homology

For distance threshold \(\epsilon\), the Vietoris--Rips complex is

\[
\operatorname{VR}_{\epsilon}(X_t)
=
\left\{
\sigma\subseteq X_t:
d(\mathbf x,\mathbf y)\le \epsilon
\text{ for every }\mathbf x,\mathbf y\in\sigma
\right\}.
\]

Increasing \(\epsilon\) produces a filtration

\[
\operatorname{VR}_{\epsilon_1}(X_t)
\subseteq
\operatorname{VR}_{\epsilon_2}(X_t)
\subseteq\cdots,
\qquad
\epsilon_1\le\epsilon_2\le\cdots.
\]

The first homology group is

\[
H_1
=
\frac{\ker(\partial_1)}{\operatorname{im}(\partial_2)}.
\]

Here, \(\ker(\partial_1)\) contains closed edge cycles, while
\(\operatorname{im}(\partial_2)\) contains cycles that are merely boundaries of
filled triangles. Their quotient identifies genuine one-dimensional holes.

The persistence diagram is

\[
D_t^{(1)}
=
\left\{(b_{j,t},d_{j,t})\right\}_{j=1}^{N_t},
\]

where loop \(j\) appears at \(b_{j,t}\), disappears at \(d_{j,t}\), and has
lifetime

\[
\ell_{j,t}=d_{j,t}-b_{j,t}.
\]

### 4.4 Topological noise filter and regime score

Short-lived features are treated as topological noise:

\[
D_{t,\mathrm{signal}}^{(1)}
=
\left\{
(b_{j,t},d_{j,t})\in D_t^{(1)}:
\ell_{j,t}>\tau_t
\right\}.
\]

The threshold \(\tau_t\) is estimated robustly or by bootstrap. Importantly,
this step filters **features in the persistence diagram**; it does not delete
daily returns.

The surviving persistence is summarized into a causal scalar \(R_t\). A larger
\(R_t\) means that the recent market-state cloud contains stronger or more
unusual persistent loop structure. This is a regime indicator, not automatically
a crash prediction.
""",
14: r"""## 5. Distributional asset geometry: Wasserstein distance and MMD

Within a training window, asset \(i\) is represented by its empirical return
distribution

\[
\widehat P_{i,t}
=
\frac{1}{L}\sum_{s=t-L}^{t-1}\delta_{r_{i,s}},
\]

where \(\delta_x\) is a point mass at \(x\). This retains more information than
representing the asset only by its sample mean and variance.

### 5.1 First Wasserstein distance

For one-dimensional distributions \(P\) and \(Q\),

\[
W_1(P,Q)
=
\int_0^1
\left|
F_P^{-1}(u)-F_Q^{-1}(u)
\right|\,du.
\]

It measures the minimum transportation cost required to deform one return
distribution into the other. The pairwise matrix is

\[
\left[D_t^{(W)}\right]_{ij}
=
W_1(\widehat P_{i,t},\widehat P_{j,t}).
\]

### 5.2 Maximum mean discrepancy

Let \(k\) be a positive-definite kernel with feature map \(\phi\) into a
reproducing-kernel Hilbert space \(\mathcal H\). Then

\[
\operatorname{MMD}_k(P,Q)
=
\left\|
\mathbb E_{X\sim P}[\phi(X)]
-
\mathbb E_{Y\sim Q}[\phi(Y)]
\right\|_{\mathcal H}.
\]

Its squared kernel form is

\[
\operatorname{MMD}_k^2(P,Q)
=
\mathbb E[k(X,X')]
+\mathbb E[k(Y,Y')]
-2\mathbb E[k(X,Y)].
\]

The corresponding asset-distance matrix is

\[
\left[D_t^{(M)}\right]_{ij}
=
\operatorname{MMD}_k(\widehat P_{i,t},\widehat P_{j,t}).
\]

After robust normalization, the two geometries are combined:

\[
D_t
=
\omega_W\widetilde D_t^{(W)}
+\omega_M\widetilde D_t^{(M)},
\qquad
\omega_W,\omega_M\ge 0,
\quad
\omega_W+\omega_M=1.
\]

K-medoids is appropriate because \(D_t\) is a precomputed, potentially
non-Euclidean distance matrix. Unlike a K-means centroid, each medoid is an
actual traded asset.
""",
16: r"""### 5.3 K-medoids clustering and representative assets

Let \(c(i)\in\{1,\ldots,K\}\) be asset \(i\)'s cluster and let \(m_k\) be the
medoid index of cluster \(k\). K-medoids approximately solves

\[
\min_{\{c(i)\},\,\{m_k\}}
\sum_{i=1}^{n} D_{i,m_{c(i)}}.
\]

The medoid

\[
m_k
\in
\arg\min_{j\in C_k}
\sum_{i\in C_k}D_{ij}
\]

is the asset most centrally located within cluster \(C_k\). Selecting one
medoid per cluster reduces redundancy while retaining different regions of the
asset-distribution geometry.
""",
18: r"""## 6. Mean--variance spanning diagnostic

For a candidate asset set with estimated mean vector
\(\widehat{\boldsymbol\mu}\in\mathbb R^n\) and covariance matrix
\(\widehat\Sigma\in\mathbb R^{n\times n}\), the long-only minimum-variance
portfolio at target return \(r_\star\) solves

\[
\begin{aligned}
\min_{\mathbf w\in\mathbb R^n}
\quad&
\mathbf w^\top\widehat\Sigma\mathbf w\\
\text{subject to}\quad&
\mathbf w^\top\widehat{\boldsymbol\mu}\ge r_\star,\\
&
\mathbf 1^\top\mathbf w=1,\\
&
\mathbf w\ge\mathbf 0.
\end{aligned}
\]

Repeating this problem over a grid of \(r_\star\) values traces an estimated
efficient frontier.

Let \(\sigma_{\mathrm{base}}(r_\star)\) and
\(\sigma_{\mathrm{selected}}(r_\star)\) be the minimum volatilities obtained
from a baseline universe and the selected universe. A useful descriptive
improvement is

\[
\Delta\sigma(r_\star)
=
\sigma_{\mathrm{base}}(r_\star)
-
\sigma_{\mathrm{selected}}(r_\star).
\]

Positive \(\Delta\sigma\) means the selected set attains the target return with
less estimated volatility. This notebook computes an **economic frontier
diagnostic**, not the formal Huberman--Kandel statistical spanning test.
Out-of-sample performance remains the more important evidence.
""",
20: r"""## 7. Regime-aware portfolio optimizer

For the assets selected at rebalance \(t\), estimate

\[
\widehat{\boldsymbol\mu}_t
=
\frac{1}{L}\sum_{s=t-L}^{t-1}\mathbf r_s,
\]

and a regularized covariance matrix \(\widehat\Sigma_t\).

The new portfolio solves

\[
\begin{aligned}
\mathbf w_t^\star
\in
\arg\min_{\mathbf w}
\quad&
-\widehat{\boldsymbol\mu}_t^\top\mathbf w
+\gamma_t\,\mathbf w^\top\widehat\Sigma_t\mathbf w
+\eta\left\|\mathbf w-\mathbf w_{t-1}\right\|_1\\
\text{subject to}\quad&
\mathbf 1^\top\mathbf w=1,\\
&
0\le w_i\le u_t.
\end{aligned}
\]

The three objective terms represent:

\[
\underbrace{-\widehat{\boldsymbol\mu}_t^\top\mathbf w}_{\text{reward expected return}}
\;+\;
\underbrace{\gamma_t\mathbf w^\top\widehat\Sigma_t\mathbf w}_{\text{penalize risk}}
\;+\;
\underbrace{\eta\|\mathbf w-\mathbf w_{t-1}\|_1}_{\text{penalize turnover}}.
\]

Risk aversion responds to the topological regime score:

\[
\gamma_t
=
\gamma_0
\left(1+\alpha\max\{R_t,0\}\right).
\]

The maximum weight \(u_t\) is also reduced in high-regime states. Therefore,
larger \(R_t\) produces a more conservative and more diversified allocation.
""",
22: r"""## 8. Strict walk-forward backtest

At rebalance date \(t_k\), the algorithm performs the following causal sequence:

1. Form the training sample
   \(\mathcal D_{t_k}=\{\mathbf r_s:t_k-L\le s<t_k\}\).
2. Compute Wasserstein/MMD distances using only \(\mathcal D_{t_k}\).
3. Cluster assets and select one representative per cluster.
4. Compute the trailing topological score \(R_{t_k}\).
5. Estimate \(\widehat{\boldsymbol\mu}_{t_k}\) and
   \(\widehat\Sigma_{t_k}\), then solve for \(\mathbf w_{t_k}\).
6. Hold those weights over the next out-of-sample interval.
7. Deduct transaction costs at the rebalance.

For day \(s\) in the holding interval, the gross portfolio return is

\[
r_{p,s}^{\mathrm{gross}}
=
\mathbf w_{t_k}^\top\mathbf r_s.
\]

One-way turnover is measured by

\[
\operatorname{TO}_{t_k}
=
\frac12
\left\|
\mathbf w_{t_k}-\mathbf w_{t_{k-1}}
\right\|_1.
\]

If the proportional cost rate is \(c\), the rebalance cost is

\[
C_{t_k}=c\,\operatorname{TO}_{t_k}.
\]

The notebook compares four strategies:

- **Full model:** topology, distributional clustering, and regime-aware weights;
- **All-asset MVO:** conventional optimization without topology or selection;
- **Equal weight:** \(w_i=1/n\) for all available assets;
- **SPY:** a simple investable benchmark.
""",
24: r"""### 8.1 Run all strategies

The following cell executes the same evaluation dates for every strategy so
that differences are not caused by mismatched samples.
""",
26: r"""## 9. Performance, drawdown, turnover, and regime diagnostics

For daily portfolio returns \(r_{p,1},\ldots,r_{p,T}\), cumulative wealth is

\[
V_t
=
\prod_{s=1}^{t}(1+r_{p,s}),
\qquad V_0=1.
\]

Using \(A=252\) trading days per year, annualized return and volatility are
estimated as

\[
\widehat\mu_{\mathrm{ann}}
=
A\,\overline r_p,
\qquad
\widehat\sigma_{\mathrm{ann}}
=
\sqrt{A}\,s(r_p).
\]

With daily risk-free rate \(r_{f,s}\), the annualized Sharpe ratio is

\[
\widehat{\operatorname{SR}}
=
\sqrt{A}\,
\frac{\overline{(r_p-r_f)}}{s(r_p-r_f)}.
\]

Running peak wealth and drawdown are

\[
M_t=\max_{0\le u\le t}V_u,
\qquad
\operatorname{DD}_t=\frac{V_t}{M_t}-1.
\]

Maximum drawdown is

\[
\operatorname{MDD}
=
\min_{1\le t\le T}\operatorname{DD}_t.
\]

These metrics must be read together: a strategy with a high return but extreme
drawdown, unstable weights, or excessive turnover may not be practically
superior.
""",
31: r"""## 10. Save reproducible outputs

The output folder records the exact numerical objects behind the figures:

- daily out-of-sample strategy returns;
- summary performance metrics;
- weights at every rebalance;
- turnover and transaction costs;
- topological regime history.

Saving these objects makes later ablation tests directly comparable.
""",
33: r"""## 11. Statistical interpretation and next experiments

The full method is useful only if it improves decisions on data that were not
used to estimate or tune it. One favorable backtest is not sufficient evidence.

The core research hypothesis is

\[
\begin{aligned}
H_0:\;&
\text{topology-aware filtering and distribution-aware selection}\\
&\text{do not improve out-of-sample portfolio performance},\\[1mm]
H_1:\;&
\text{they improve risk-adjusted performance or stability after costs}.
\end{aligned}
\]

“Improvement” should be specified before testing, for example:

\[
\Delta\operatorname{Sharpe}>0,\qquad
\Delta|\operatorname{MDD}|<0,\qquad
\Delta\operatorname{Turnover}\le 0,
\]

subject to acceptable realized return.

Recommended next experiments:

1. Perform ablations by changing only one component at a time: persistent
   homology, Wasserstein distance, MMD, or clustering.
2. Test multiple non-overlapping periods and asset universes.
3. Add a cash or risk-free asset and realistic fund-specific trading costs.
4. Compare robust and bootstrap persistence thresholds.
5. Use nested walk-forward tuning so hyperparameters are selected only inside
   each training period.
6. Report uncertainty using moving-block bootstrap confidence intervals.

The central question is

\[
\boxed{
\begin{gathered}
\text{Does persistent market topology plus distribution-aware asset selection}\\
\text{produce more stable out-of-sample portfolios than conventional methods?}
\end{gathered}
}
\]
""",
}


def source_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
for index, text in MARKDOWN.items():
    cell = notebook["cells"][index]
    if cell["cell_type"] != "markdown":
        raise ValueError(f"Cell {index} is not Markdown")
    cell["source"] = source_lines(text)

NOTEBOOK.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
