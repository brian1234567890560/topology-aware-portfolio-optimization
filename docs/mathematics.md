# Mathematics and Model Structure

This page collects the mathematical structure implemented in the current notebook.

## 1. Information timing

Let $P_{i,t}>0$ be the adjusted closing price of asset $i$ at time $t$. Its log return is

```math
r_{i,t}
=
\log\left(\frac{P_{i,t}}{P_{i,t-1}}\right).
```

At rebalance date $t$, every estimator must be measurable with respect to the available information

```math
\mathcal F_t=\sigma(r_{i,s}\colon s<t,\ i=1,\ldots,n).
```

The weights selected at $t$ may be applied only to returns at $t$ and later. This is the core no-look-ahead condition.

## 2. Market-state point clouds

Choose $m$ broad market indices. One day is represented by

```math
\mathbf x_s
=
\begin{bmatrix}
r_{1,s}\\
\vdots\\
r_{m,s}
\end{bmatrix}
\in\mathbb R^m.
```

The trailing point cloud at rebalance $t$ is

```math
X_t
=
\{\mathbf x_{t-L},\ldots,\mathbf x_{t-1}\}.
```

The vertices are trading days; their coordinates are simultaneous market returns.

### Robust scaling

For coordinate $q$, define its sample median and median absolute deviation (MAD) by

```math
\mathrm{MAD}_q
=
\mathrm{median}_s
\left|
x_{s,q}-\mathrm{median}_u(x_{u,q})
\right|.
```

Let $s_q^{\mathrm{std}}$ be the sample standard deviation of coordinate $q$. The code chooses the scale

```math
a_q
=
\begin{cases}
1.4826\,\mathrm{MAD}_q,
&1.4826\,\mathrm{MAD}_q>\varepsilon_{\mathrm{num}},\\
s_q^{\mathrm{std}},
&s_q^{\mathrm{std}}>\varepsilon_{\mathrm{num}},\\
1,
&\text{otherwise}.
\end{cases}
```

The robustly standardized coordinate is

```math
\widetilde x_{s,q}
=
\frac{x_{s,q}-\mathrm{median}_u(x_{u,q})}
{a_q}.
```

The factor $1.4826$ makes MAD comparable to the standard deviation for Gaussian data.

## 3. Vietoris-Rips persistent homology

For a metric space $(X_t,d)$ and threshold $\epsilon$, the Vietoris-Rips complex is

```math
\mathrm{VR}_\epsilon(X_t)
=
\left\{
\sigma\subseteq X_t:
d(\mathbf x,\mathbf y)\le\epsilon
\text{ for every }\mathbf x,\mathbf y\in\sigma
\right\}.
```

As $\epsilon$ increases,

```math
\mathrm{VR}_{\epsilon_1}(X_t)
\subseteq
\mathrm{VR}_{\epsilon_2}(X_t)
\subseteq\cdots.
```

Using coefficients in the field $\mathbb F_2$, the first homology group is

```math
H_1
=
\frac{\ker(\partial_1)}{\mathrm{im}(\partial_2)}.
```

- $\ker(\partial_1)$ contains closed one-dimensional cycles.
- $\mathrm{im}(\partial_2)$ contains cycles that are boundaries of filled triangles.
- The quotient retains cycles not already filled by two-dimensional simplices.

The first-homology persistence diagram is

```math
D_t^{(1)}
=
\{(b_{j,t},d_{j,t})\}_{j=1}^{N_t},
```

with lifetimes

```math
\ell_{j,t}=d_{j,t}-b_{j,t}.
```

### Topological noise threshold

Short-lived loops are filtered:

```math
D_{t,\mathrm{signal}}^{(1)}
=
\left\{
(b_{j,t},d_{j,t})\in D_t^{(1)}:
\ell_{j,t}>\tau_t
\right\}.
```

The current notebook offers two thresholds.

For the robust threshold,

```math
\tau_t
=
\mathrm{median}(\ell_t)
+1.4826\,c\,\mathrm{MAD}(\ell_t).
```

Here $c$ is `tda_noise_mad_multiplier`. For the moving-block-bootstrap threshold, blocks of the market-state sequence are resampled to preserve short-range dependence. If $\ell_{\max}^{*(b)}$ is the largest finite $H_1$ lifetime in bootstrap sample $b$, the implemented threshold is

```math
\tau_t
=
Q_q\!\left(
\ell_{\max}^{*(1)},\ldots,\ell_{\max}^{*(B)}
\right),
```

where $q$ is `tda_bootstrap_quantile` and $B$ is `tda_bootstrap_samples`.

The code summarizes surviving loops with

```math
P_{2,t}
=
\left(
\sum_{j:\ell_{j,t}>\tau_t}\ell_{j,t}^2
\right)^{1/2}.
```

An expanding robust z-score converts this into the regime score $R_t$. First,
use only the earlier persistence summaries to define the historical center

```math
m_t
=
\mathrm{median}\!\left(P_{2,1},\ldots,P_{2,t-1}\right)
```

and the corresponding historical MAD

```math
a_t
=
\mathrm{median}_{1\le u\le t-1}
\left|P_{2,u}-m_t\right|.
```

Then the regime score is

```math
R_t
=
\frac{P_{2,t}-m_t}{1.4826\,a_t}.
```

This notation makes the causal timing explicit: neither $m_t$ nor $a_t$ uses
the current value $P_{2,t}$. The code returns $0$ until there are at least six
earlier observations and falls back to the sample standard deviation if
$a_t$ is nearly zero.

## 4. Wasserstein asset distance

Let $\mu_i$ and $\mu_j$ be empirical trailing return distributions for assets $i$ and $j$. The $p$-Wasserstein distance is

```math
W_p(\mu_i,\mu_j)
=
\left[
\inf_{\gamma\in\Pi(\mu_i,\mu_j)}
\int |x-y|^p\,d\gamma(x,y)
\right]^{1/p}.
```

$\Pi(\mu_i,\mu_j)$ is the set of joint probability measures with marginals $\mu_i$ and $\mu_j$.

For one-dimensional empirical distributions, this is equivalent to integrating the difference between empirical quantile functions. The current code applies it to raw trailing log returns; it does **not** standardize each asset first.

## 5. Maximum mean discrepancy

For a positive-definite kernel $k$ with feature map $\phi$, MMD is

```math
\mathrm{MMD}_k^2(P,Q)
=
\left\|
\mathbb E_{X\sim P}[\phi(X)]
-
\mathbb E_{Y\sim Q}[\phi(Y)]
\right\|_{\mathcal H}^2.
```

Equivalently,

```math
\mathrm{MMD}_k^2(P,Q)
=
\mathbb E[k(X,X')]
+\mathbb E[k(Y,Y')]
-2\mathbb E[k(X,Y)].
```

The notebook uses the Gaussian RBF kernel

```math
k(x,y)
=
\exp\!\left(
-\frac{(x-y)^2}{2h^2}
\right),
```

where $h$ is the median positive pairwise distance in the pooled samples. The implemented finite-sample estimate includes diagonal terms:

```math
\widehat{\mathrm{MMD}}_k^2
=
\frac{1}{n^2}\sum_{a=1}^{n}\sum_{b=1}^{n}k(x_a,x_b)
+\frac{1}{m^2}\sum_{a=1}^{m}\sum_{b=1}^{m}k(y_a,y_b)
-\frac{2}{nm}\sum_{a=1}^{n}\sum_{b=1}^{m}k(x_a,y_b).
```

The distance passed to clustering is $\sqrt{\max(\widehat{\mathrm{MMD}}_k^2,0)}$. This is the biased but nonnegative empirical MMD estimator used by the code.

## 6. Combined distributional geometry

Let $s_W$ and $s_M$ be the medians of the positive upper-triangular entries of the Wasserstein and MMD distance matrices. The code normalizes

```math
\widetilde W_{ij}=\frac{W_{ij}}{s_W},
\qquad
\widetilde M_{ij}=\frac{M_{ij}}{s_M}.
```

The combined distance is

```math
D_{ij}
=
\alpha\,\widetilde W_{ij}
+(1-\alpha)\,\widetilde M_{ij},
\qquad 0\le\alpha\le1.
```

The default weights are $\alpha=0.65$ for Wasserstein and $1-\alpha=0.35$ for MMD.

## 7. K-medoids clustering

For clusters $C_1,\ldots,C_K$ and observed medoids $m_1,\ldots,m_K$, K-medoids minimizes

```math
\min_{\{C_k,m_k\}}
\sum_{k=1}^{K}
\sum_{i\in C_k}
D_{i,m_k}.
```

Unlike Euclidean K-means, it needs only a distance matrix. Each medoid is an actual asset, which makes cluster representatives directly interpretable.

The assets finally selected are not necessarily the medoids. Within each cluster, the code ranks asset $i$ by

```math
S_i
=
-z\!\left(
\frac{1}{|C_k|}
\sum_{j\in C_k}D_{ij}
\right)
+0.25\,z(\widehat{\mathrm{SR}}_i),
\qquad i\in C_k,
```

so it primarily favors distributional centrality with a modest in-sample Sharpe tilt.

## 8. Efficient-frontier diagnostic

Let $\widehat{\boldsymbol\mu}$ and $\widehat\Sigma$ be annualized moment estimates. For target return $r_\star$, solve

```math
\begin{aligned}
\min_{\mathbf w}\quad&
\mathbf w^\top\widehat\Sigma\mathbf w\\
\text{subject to}\quad&
\mathbf w^\top\widehat{\boldsymbol\mu}\ge r_\star,\\
&\mathbf 1^\top\mathbf w=1,\\
&\mathbf w\ge\mathbf 0.
\end{aligned}
```

For baseline and expanded universes, define

```math
\Delta\sigma(r_\star)
=
\sigma_{\mathrm{base}}(r_\star)
-\sigma_{\mathrm{expanded}}(r_\star).
```

Positive $\Delta\sigma$ means the expanded universe reaches the target return with lower estimated volatility. This is descriptive, not a formal statistical spanning test.

The target grid is restricted to the overlap of the baseline and expanded universes' estimated-return ranges. Failed or infeasible numerical optimizations are recorded as missing values.

## 9. Regime-aware portfolio optimization

For selected assets, the optimizer solves

```math
\begin{aligned}
\mathbf w_t^\star\in\arg\min_{\mathbf w}\quad&
-\widehat{\boldsymbol\mu}_t^\top\mathbf w
+\gamma_t\mathbf w^\top\widehat\Sigma_t^{\mathrm{shrunk}}\mathbf w
+\eta\|\mathbf w-\mathbf w_{t-1}\|_1\\
\text{subject to}\quad&
\mathbf1^\top\mathbf w=1,\\
&0\le w_i\le u_t.
\end{aligned}
```

The terms respectively reward estimated return, penalize estimated risk, and penalize turnover.

Risk aversion is

```math
\gamma_t
=
\gamma_0\left(1+\beta\max\{R_t,0\}\right).
```

The covariance estimator is shrunk toward its diagonal:

```math
\widehat\Sigma_t^{\mathrm{shrunk}}
=
(1-\rho_t)\widehat\Sigma_t
+\rho_t\mathrm{diag}(\widehat\Sigma_t)
+\delta I.
```

In the current code,

```math
\rho_t
=
\min\!\left\{
0.60,\,
\max\!\left[
0.10,\,
0.10+0.10\max(R_t,0)
\right]
\right\}.
```

The normal weight cap is $0.40$ and the stressed cap is $0.25$ when $R_t\ge1$. To preserve feasibility, the implemented cap is never below $1/n_t$, where $n_t$ is the number of selected assets.

## 10. Walk-forward returns and costs

For holding-period day $s$,

```math
r_{p,s}^{\mathrm{gross}}
=
\mathbf w_t^\top\mathbf r_s.
```

The code records the full reallocation distance

```math
\mathrm{Q}_t
=
\|\mathbf w_t-\mathbf w_{t-1}\|_1.
```

If proportional cost is $c$, then

```math
C_t=c\,\mathrm{Q}_t.
```

The cost is deducted once, from the first log return of the new holding period:

```math
r_{p,t}^{\mathrm{net}}
\approx
r_{p,t}^{\mathrm{gross}}-C_t.
```

Many finance texts instead define one-way turnover as $\mathrm{TO}_t=\tfrac12\mathrm Q_t$. Therefore, the current cost convention charges twice the cost that would result from multiplying the same rate $c$ by conventional one-way turnover. This must be kept explicit when comparing results.

## 11. Evaluation metrics

Cumulative wealth from log returns is

```math
V_t
=
\exp\left(\sum_{s=1}^t r_{p,s}\right).
```

For $A=252$ trading days per year,

```math
\widehat\mu_{\mathrm{ann}}=A\overline r_p,
\qquad
\widehat\sigma_{\mathrm{ann}}=\sqrt A\,s(r_p).
```

The zero-risk-free-rate Sharpe estimate is

```math
\widehat{\mathrm{SR}}
=
\sqrt A\frac{\overline r_p}{s(r_p)}.
```

Drawdown is

```math
\mathrm{DD}_t
=
\frac{V_t}{\max_{0\le u\le t}V_u}-1,
```

and maximum drawdown is $\mathrm{MDD}=\min_t\mathrm{DD}_t$.

## 12. Research hypothesis

```math
\begin{aligned}
H_0:\quad&
\text{topology-aware filtering and distribution-aware selection}\\
&\text{do not improve out-of-sample portfolio performance},\\[1mm]
H_1:\quad&
\text{they improve risk-adjusted performance or stability after costs}.
\end{aligned}
```

The hypothesis must be evaluated with preregistered metrics, causal model selection, and uncertainty estimates rather than a single favorable backtest.
