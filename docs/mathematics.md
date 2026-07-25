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
\mathcal F_t
=
\sigma\left(\{r_{i,s}:s<t,\ i=1,\ldots,n\}\right).
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

For coordinate $q$,

```math
\operatorname{MAD}_q
=
\operatorname{median}_s
\left|
x_{s,q}-\operatorname{median}_u(x_{u,q})
\right|.
```

The robustly standardized coordinate is

```math
\widetilde x_{s,q}
=
\frac{x_{s,q}-\operatorname{median}_u(x_{u,q})}
{1.4826\,\operatorname{MAD}_q+\varepsilon_{\mathrm{num}}}.
```

## 3. Vietoris-Rips persistent homology

For a metric space $(X_t,d)$ and threshold $\epsilon$, the Vietoris-Rips complex is

```math
\operatorname{VR}_\epsilon(X_t)
=
\left\{
\sigma\subseteq X_t:
d(\mathbf x,\mathbf y)\le\epsilon
\text{ for every }\mathbf x,\mathbf y\in\sigma
\right\}.
```

As $\epsilon$ increases,

```math
\operatorname{VR}_{\epsilon_1}(X_t)
\subseteq
\operatorname{VR}_{\epsilon_2}(X_t)
\subseteq\cdots.
```

The first homology group is

```math
H_1
=
\frac{\ker(\partial_1)}{\operatorname{im}(\partial_2)}.
```

- $\ker(\partial_1)$ contains closed one-dimensional cycles.
- $\operatorname{im}(\partial_2)$ contains cycles that are boundaries of filled triangles.
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
\operatorname{median}(\ell_t)
+c\operatorname{MAD}(\ell_t).
```

For the moving-block-bootstrap threshold, blocks of the time series are resampled to preserve short-range dependence. The threshold is a selected upper quantile of bootstrap persistence.

The code summarizes surviving loops with

```math
P_{2,t}
=
\left(
\sum_{j:\ell_{j,t}>\tau_t}\ell_{j,t}^2
\right)^{1/2}.
```

An expanding robust z-score converts this into the regime score $R_t$.

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

The current code uses one-dimensional Wasserstein distance on standardized trailing returns.

## 5. Maximum mean discrepancy

For a positive-definite kernel $k$ with feature map $\phi$, MMD is

```math
\operatorname{MMD}_k^2(P,Q)
=
\left\|
\mathbb E_{X\sim P}[\phi(X)]
-
\mathbb E_{Y\sim Q}[\phi(Y)]
\right\|_{\mathcal H}^2.
```

Equivalently,

```math
\operatorname{MMD}_k^2(P,Q)
=
\mathbb E[k(X,X')]
+\mathbb E[k(Y,Y')]
-2\mathbb E[k(X,Y)].
```

The notebook uses a Gaussian RBF kernel and a median-distance bandwidth heuristic.

## 6. Combined distributional geometry

After scaling the Wasserstein and MMD matrices to comparable ranges, the combined distance is

```math
D_{ij}
=
\alpha\,\widetilde W_{ij}
+(1-\alpha)\,\widetilde M_{ij},
\qquad 0\le\alpha\le1.
```

The default weight is $\alpha=0.65$.

## 7. K-medoids clustering

For clusters $C_1,\ldots,C_K$ and observed medoids $m_1,\ldots,m_K$, K-medoids minimizes

```math
\min_{\{C_k,m_k\}}
\sum_{k=1}^{K}
\sum_{i\in C_k}
D_{i,m_k}.
```

Unlike Euclidean K-means, it needs only a distance matrix. Each medoid is an actual asset, which makes cluster representatives directly interpretable.

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

## 9. Regime-aware portfolio optimization

For selected assets, the optimizer solves

```math
\begin{aligned}
\mathbf w_t^\star\in\arg\min_{\mathbf w}\quad&
-\widehat{\boldsymbol\mu}_t^\top\mathbf w
+\gamma_t\mathbf w^\top\widehat\Sigma_t\mathbf w
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
+\rho_t\operatorname{diag}(\widehat\Sigma_t)
+\delta I.
```

Both $\rho_t$ and the weight cap $u_t$ become more conservative in stressed regimes.

## 10. Walk-forward returns and costs

For holding-period day $s$,

```math
r_{p,s}^{\mathrm{gross}}
=
\mathbf w_t^\top\mathbf r_s.
```

The standard one-way turnover definition is

```math
\operatorname{TO}_t
=
\frac12\|\mathbf w_t-\mathbf w_{t-1}\|_1.
```

If proportional cost is $c$, then

```math
C_t=c\,\operatorname{TO}_t.
```

The current implementation records the full $L^1$ weight change and applies the configured basis-point cost to that quantity. This convention should be made consistent with the one-way definition before final empirical reporting; see the limitations page.

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
\widehat{\operatorname{SR}}
=
\sqrt A\frac{\overline r_p}{s(r_p)}.
```

Drawdown is

```math
\operatorname{DD}_t
=
\frac{V_t}{\max_{0\le u\le t}V_u}-1,
```

and maximum drawdown is $\min_t\operatorname{DD}_t$.

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

