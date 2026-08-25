# M7 — Curvature approximations: GGN, Fisher, empirical Fisher, KFAC, diagonal

**Source: read** (2026-08-24, `paper/` in this repo plus the references below), with
two measured checks marked inline.

Equations use `$...$` / `$$...$$` (Obsidian and GitHub both render it) and are also
written to be readable as plain text. Every symbol is defined at the point it first
appears; the full list is in [Symbols](#symbols) at the end.

---

## 1. The one equation everything hangs on

A Laplace approximation replaces the posterior over weights by a Gaussian centred at
the MAP estimate:

$$p(\theta \mid \mathcal{D}) \;\approx\; \mathcal{N}\!\left(\theta_{\text{MAP}},\; \Sigma\right),
\qquad \Sigma = \left(C + \tau I\right)^{-1}$$

- $\theta$ — the vector of all weights being treated as Bayesian, length $P$.
- $\theta_{\text{MAP}}$ — the trained weights (the mode of the posterior).
- $\Sigma$ — the posterior covariance, a $P \times P$ matrix.
- $C$ — a **curvature matrix** of the negative log likelihood, $P \times P$.
- $\tau = 1/\mathrm{var}_0$ — the prior precision, a single number. With a Gaussian
  prior $\mathcal{N}(0, \mathrm{var}_0 I)$ the prior's own curvature is exactly
  $\tau I$, which is why it enters as a constant added to the diagonal.
- $I$ — the $P \times P$ identity matrix.

Everything that follows is about **$C$**. Every "flavour" of Laplace is a different
choice of $C$, along two independent axes:

| axis | question | options |
|---|---|---|
| **curvature** | which matrix approximates the true second derivative? | exact Hessian, GGN, Fisher, MC Fisher, empirical Fisher |
| **structure** | which entries of that matrix do we even store? | dense, per-layer blocks, KFAC, diagonal |

A third, separate choice is *scope*: which weights go into $\theta$ at all (just the
last layer, or the whole network).

**The short version: the structure axis is about affording big networks. The
curvature axis is about the matrix existing at all.**

---

## 2. The curvature axis, as one shared shape

For binary classification every candidate for $C$ has the **same sandwich form**.
Write the model and loss first:

- $x_n, y_n$ — the $n$-th input and its label, $y_n \in \{0, 1\}$, for $n = 1 \dots N$.
- $f_\theta(x_n)$ — the network's scalar output (the **logit**) at $x_n$.
- $p_n = \sigma(f_\theta(x_n))$ — the predicted probability, where
  $\sigma(z) = 1/(1+e^{-z})$ is the sigmoid.
- $\ell$ — the per-example loss (binary cross-entropy),
  $\ell = -y\log p - (1-y)\log(1-p)$.
- $j_n = \nabla_\theta f_\theta(x_n)$ — the gradient of the *logit* with respect to
  the weights: a vector of length $P$. (Stack all $N$ of them and you get the
  Jacobian $J$, an $N \times P$ matrix, with $j_n^\top$ as its $n$-th row.)

Then every curvature choice looks like this:

$$C \;=\; \sum_{n=1}^{N} c_n \, j_n j_n^\top$$

where $j_n j_n^\top$ is an outer product — a $P \times P$ rank-1 matrix — and $c_n$
is a **single scalar weight per data point**. The whole taxonomy is: *what is $c_n$?*

| $C$ | scalar weight $c_n$ | what that weight looks at |
|---|---|---|
| **GGN** | $p_n(1-p_n)$ | the **prediction** only — never the label |
| **Fisher** | $\mathbb{E}_{y \sim \mathrm{Bern}(p_n)}\big[(p_n - y)^2\big]$ | the prediction, averaged over labels the *model* would generate |
| **MC Fisher** | $(p_n - \tilde y_n)^2$ with $\tilde y_n \sim \mathrm{Bern}(p_n)$ | one random label drawn from the model |
| **empirical Fisher** | $(p_n - y_n)^2$ with the **observed** $y_n$ | the **error** the model made |
| **exact Hessian** | $p_n(1-p_n)$, *plus a second term* (below) | prediction, plus the network's own curvature |

Two facts fall straight out of this table.

**Fisher = GGN, exactly.** The variance of a Bernoulli variable with success
probability $p$ is $p(1-p)$, so

$$\mathbb{E}_{y \sim \mathrm{Bern}(p_n)}\big[(p_n - y)^2\big] = p_n(1-p_n).$$

Averaging the *error-weighted* form over labels drawn from the model gives the
*prediction-weighted* form. So "GGN" and "Fisher" name the same matrix here, reached
from two directions — one by differentiating, one by sampling labels. (This holds for
exponential-family likelihoods with their canonical link, i.e. Bernoulli+logit and
Categorical+softmax — every case in this repo.) MC Fisher is the same thing with the
average replaced by a single draw: noisy, but unbiased.

**GGN is positive semi-definite, always.** Each $c_n = p_n(1-p_n) \ge 0$ and each
$j_n j_n^\top$ is PSD, so the sum is PSD for *any* network, however non-convex. Add
$\tau I$ with $\tau > 0$ and $C + \tau I$ is strictly positive definite — hence
invertible, and a legitimate covariance. **That is the entire reason these
approximations exist.**

### 2.1 What the exact Hessian has that the GGN drops

The chain rule through the scalar logit gives the exact second derivative as two
terms:

$$\underbrace{\nabla^2_\theta L}_{\text{exact Hessian}} \;=\;
\underbrace{\sum_n p_n(1-p_n)\, j_n j_n^\top}_{\text{= the GGN}} \;+\;
\underbrace{\sum_n (p_n - y_n)\, \nabla^2_\theta f_\theta(x_n)}_{\text{dropped by the GGN}}$$

- $L = \sum_n \ell$ — the total loss over the training set.
- $\nabla^2_\theta f_\theta(x_n)$ — the curvature of the *network output* as a
  function of its own weights, a $P \times P$ matrix.
- $(p_n - y_n)$ — the residual, i.e. the error. **Its sign is unconstrained.**

The dropped term is a sum of matrices that are not PSD, scaled by numbers that can be
positive or negative. That is exactly why the exact Hessian can be indefinite — and
[I11](../gotchas/full-bnn-laplace-not-pd.md) measured it: **79 negative eigenvalues
of 500** over all Linear weights of our two-moons network.

Two consequences worth memorising:

- **The GGN is the exact Hessian of the *linearised* model.** If you freeze $J$ and
  pretend $f$ is linear in $\theta$ around $\theta_{\text{MAP}}$, then
  $\nabla^2_\theta f = 0$ and the dropped term vanishes by construction.
- **For a last layer with frozen features, nothing is dropped at all.** There
  $f = w^\top \phi(x)$ is genuinely linear in the weights $w$, so
  $\nabla^2_w f = 0$ exactly. **Measured** (2026-08-24, `full_bnn_ablation.py`): GGN
  and exact Hessian agree to $1.2 \times 10^{-10}$ maximum relative eigenvalue
  difference on the `ll` arm — see
  [decision 0012](../decisions/0012-ggn-laplace-for-deep-arms.md).

### 2.2 The empirical Fisher, and why it is the odd one out

From the table: the empirical Fisher keeps the sandwich but weights each point by
$(p_n - y_n)^2$ — **how wrong the model was** — instead of $p_n(1-p_n)$ — **how
uncertain the model is**. It is popular because it is free: $(p_n - y_n) j_n$ is just
the ordinary gradient, already computed by backprop, so no extra work is needed.

The two weights are not interchangeable. Take a confidently *correct* prediction,
$y_n = 1$ and $p_n = 0.99$:

| weight | value |
|---|---|
| GGN / Fisher: $p(1-p)$ | $0.99 \times 0.01 = 9.9\times10^{-3}$ |
| empirical Fisher: $(p-y)^2$ | $(0.01)^2 = 1.0\times10^{-4}$ |

**A factor of 100 too small.** In general the ratio is
$(1-p)^2 / \big(p(1-p)\big) = (1-p)/p$, which goes to zero as the model becomes
confident. So on well-fit data the empirical Fisher systematically *underestimates*
curvature, and since $\Sigma = (C + \tau I)^{-1}$, underestimating $C$ means
**overestimating the posterior variance**. On badly-fit points it errs the other way.
It agrees with the Fisher only on average, and only if the model is correctly
specified — which is precisely when you needed the uncertainty estimate least.

> **Correction to an earlier draft of this note.** I previously wrote that the
> empirical Fisher "goes to zero at a stationary point because the gradients vanish".
> That is wrong: at a stationary point the *summed* gradient $\sum_n g_n$ vanishes,
> but the empirical Fisher is a sum of *per-example outer products*
> $\sum_n g_n g_n^\top$, which does not. The real objection is the mis-weighting
> above. Kunstner, Balles & Hennig (2019) is the careful treatment.

---

## 3. The structure axis, drawn out

$C$ is $P \times P$. For our two-moons network $P = 501$, so $C$ has ~251 000
entries — fine. For a ResNet with $P = 10^7$ it would have $10^{14}$, which is why
the paper needed structure. Take a toy network with two layers,
$W_1$ of shape $3\times2$ (6 weights) and $W_2$ of shape $1\times3$ (3 weights), so
$P = 9$. Writing `X` for "stored" and `.` for "forced to zero":

```
   dense (all P²)          per-layer blocks         diagonal
   XXXXXX XXX              XXXXXX ...              X........
   XXXXXX XXX              XXXXXX ...              .X.......
   XXXXXX XXX              XXXXXX ...              ..X......
   XXXXXX XXX              XXXXXX ...              ...X.....
   XXXXXX XXX              XXXXXX ...              ....X....
   XXXXXX XXX              XXXXXX ...              .....X...
   XXXXXX XXX              ...... XXX              ......X..
   XXXXXX XXX              ...... XXX              .......X.
   XXXXXX XXX              ...... XXX              ........X

   45 free numbers         21 + 6 = 27             9
   (9×9 symmetric)         (cross-layer dropped)   (all correlations dropped)
```

**Diagonal** is the crudest, and its error has a known direction: for any positive
definite $H$, $(H^{-1})_{ii} \ge 1/H_{ii}$. Inverting only the diagonal therefore
**underestimates every marginal posterior variance**. It is not a neutral
simplification.

### 3.1 KFAC, structurally

KFAC (Kronecker-Factored Approximate Curvature) sits between per-layer blocks and
diagonal: it keeps a whole layer's block, but forces it into a **product form**.

Consider just the $W_1$ layer, shape $3 \times 2$: 2 inputs, 3 outputs, 6 weights,
so its block of $C$ is $6\times6$. Two new symbols, both *per layer*:

- $a$ — the layer's **input activations**, a vector of length $\mathrm{fan\_in} = 2$.
- $\delta$ — the gradient of the loss with respect to the layer's **pre-activation
  output**, a vector of length $\mathrm{fan\_out} = 3$.

For a linear layer the weight gradient is exactly an outer product,
$\nabla_{W} \ell = \delta a^\top$. So the exact block entry pairing weight
$W[i,j]$ with weight $W[k,l]$ is

$$C_{(i,j),(k,l)} \;=\; \mathbb{E}\big[\, \delta_i a_j \, \delta_k a_l \,\big].$$

KFAC assumes that expectation **factorises** — that the output-side and input-side
parts are uncorrelated:

$$\mathbb{E}\big[\delta_i a_j \delta_k a_l\big] \;\approx\;
\underbrace{\mathbb{E}[\delta_i \delta_k]}_{G_{ik}} \cdot
\underbrace{\mathbb{E}[a_j a_l]}_{A_{jl}}$$

- $A = \mathbb{E}[a a^\top]$ — input second moment.
  Shape $\mathrm{fan\_in} \times \mathrm{fan\_in}$, here $2 \times 2$.
- $G = \mathbb{E}[\delta \delta^\top]$ — output-gradient second moment,
  $\mathrm{fan\_out} \times \mathrm{fan\_out}$, here $3 \times 3$.

Compactly, $C_{\text{layer}} \approx G \otimes A$, where $\otimes$ is the **Kronecker
product**: it tiles a copy of $A$ for every entry of $G$, scaled by that entry. Drawn
out, the $6\times6$ block becomes a $3\times3$ grid of $2\times2$ tiles:

```
              input pair (j,l)
              ┌─────────┬─────────┬─────────┐
              │ G₁₁ · A │ G₁₂ · A │ G₁₃ · A │
  output      ├─────────┼─────────┼─────────┤
  pair (i,k)  │ G₂₁ · A │ G₂₂ · A │ G₂₃ · A │
              ├─────────┼─────────┼─────────┤
              │ G₃₁ · A │ G₃₂ · A │ G₃₃ · A │
              └─────────┴─────────┴─────────┘

  every tile is the SAME 2×2 matrix A, only rescaled
```

That is the whole idea, and the picture makes the trade explicit:

| | free numbers for this layer |
|---|---|
| exact $6\times6$ block | **21** (symmetric) |
| KFAC: $A$ (3) + $G$ (6) | **9**, and they must multiply out as above |

**What is lost:** in the exact block all 21 numbers are independent. KFAC forces them
onto a product grid, so it can only represent correlations that separate into
"output-side $\times$ input-side". Cross-layer correlations are dropped too, as in
the per-layer-block picture. **What is bought** is decisive at scale: storage falls
from $(\mathrm{fan\_in} \cdot \mathrm{fan\_out})^2$ to
$\mathrm{fan\_in}^2 + \mathrm{fan\_out}^2$; inversion factorises as
$(G \otimes A)^{-1} = G^{-1} \otimes A^{-1}$ on the small factors; and sampling
becomes a matrix-normal draw, $W = M + U\,E\,V$ with $E$ a matrix of standard normals
and $U, V$ Cholesky factors of the two small matrices — literally the line in
`paper/laplace/kfla.py`.

One subtlety that is easy to miss: $G \otimes A + \tau I$ is **not** a Kronecker
product, so the prior damping cannot be applied exactly in factored form. The
standard heuristic, and what `kfla.py` does, is
$(G + \sqrt{\tau} I) \otimes (A + \sqrt{\tau} I)$ — a *third* approximation stacked on
the factorisation and the block-diagonality.

---

## 4. What BackPack is

[BackPack](https://github.com/f-dangel/backpack) (Dangel, Kunstner & Hennig, ICLR
2020) is a PyTorch library that extends autograd so that one backward pass yields
**more than the averaged gradient** — per-example gradients, the diagonal GGN, KFAC's
$A$ and $G$ factors, and so on. Usage is the pattern in `paper/laplace/llla.py`:

```python
extend(lossfunc); extend(model.linear)      # mark what to instrument
with backpack(KFAC()):
    lossfunc(model(x), y).backward()        # ordinary backward pass
    U, V = W.kfac                           # ...but the factors are now attached
```

So "KFAC via BackPack" simply means *the two Kronecker factors were computed as a
by-product of backprop*. Which extension matters: BackPack's `KFAC` is the
**MC-sampled Fisher** variant (it draws labels from the model internally), while its
`KFLR` is the exact-GGN Kronecker counterpart. `llla.py` uses `KFAC`, so its curvature
is an MC Fisher even though the loop passes the true labels.

---

## 5. What the paper's code actually computes

| file | curvature ($c_n$) | structure | scope |
|---|---|---|---|
| `laplace/llla.py` | MC Fisher (BackPack `KFAC`) | Kronecker | last layer |
| `laplace/llla_binary.py` | **exact Hessian** of the likelihood | dense | last layer |
| `laplace/dla.py` | **MC Fisher** — labels drawn from the model | diagonal | all layers |
| `laplace/kfla.py` | MC Fisher (own `util/kfac.py`) | Kronecker | all layers |
| `notebooks/laplace/diag_laplace.py` | **GGN** (`ggn_vector_product`) | **dense**, despite the filename | all layers |

Quirks to know before comparing numbers against them:

- **`dla.py` is an MC Fisher, not an empirical Fisher.** It calls
  `y = distribution.sample()` — labels from the *model*. Given §2, that is an
  unbiased 1-sample estimate of $p(1-p)$, not the error-weighted
  $(p-y)^2$. The distinction is §2.2's whole point.
- **`dla.py` squares the batch-*mean* gradient.** Its criterion is
  `reduction='mean'`, so each accumulated term is
  $\big(\tfrac{1}{B}\sum_n g_n\big)^2$ rather than $\sum_n g_n^2$ — a batch-level
  statistic including cross-terms between examples, whose scale depends on the batch
  size $B$. `n_data` is computed in `get_hessian` and never used, so the result is
  not normalised to the dataset either. Both wash out in practice because
  `gridsearch_var0` tunes $\mathrm{var}_0$ against validation and OOD loss, absorbing
  any global rescaling of $C$.
- **`kfla.py` leaves biases deterministic** (`if name == 'bias': w = mean`), and
  `util/kfac.py` has its bias-augmentation of the input commented out to match.
- Both `llla.py` and `kfla.py` accumulate their factors as a **running average**
  across batches ($\rho = 0.95$), then rescale each factor by $\sqrt{N}$ so that
  $G \otimes A$ scales like a sum over the dataset rather than a mean.

---

## 6. Summary: what each variant gives up

Taking "full Laplace" to mean *dense exact Hessian over all weights*:

| variant | gives up | buys |
|---|---|---|
| **dense GGN** | the $(p-y)\nabla^2 f$ term | PSD, hence invertible at all; **exact** for a frozen-feature last layer |
| **KFAC** | that term, plus within-layer factorisation and cross-layer correlation | $\mathrm{fan\_in}^2 + \mathrm{fan\_out}^2$ per layer instead of $P^2$ |
| **diagonal** | all of the above, plus every correlation | $O(P)$; systematically **under**-estimates marginal variances |
| **empirical Fisher** | curvature itself — it weights by error, not uncertainty | nothing beyond the gradients already computed |
| **last-layer only** | uncertainty over the features | convexity, so the exact Hessian is PD and all of the above coincide |

Costs, with $P$ weights, $N$ data points, layers indexed by $l$:

| variant | memory | compute |
|---|---|---|
| exact Hessian | $P^2$ | $\sim P$ backward passes |
| dense GGN | $P^2$ | $N$ backward passes (one per Jacobian row) — cheaper than the Hessian when $N < P$ |
| KFAC | $\sum_l (\mathrm{fan\_in}_l^2 + \mathrm{fan\_out}_l^2)$ | one backward pass per batch |
| diagonal | $P$ | one backward pass per batch |

This repo needs only the curvature axis: $P = 501$ is small enough for a dense
matrix, but the exact Hessian is indefinite. The paper needed both axes, because it
ran on ResNets.

---

## 7. Mapping onto `laplace-torch`, for the Q6 cross-check

`laplace-torch` exposes exactly these axes as constructor arguments — a subset of
weights (last layer vs all) and a Hessian structure (diagonal, Kronecker, full,
low-rank), with a selectable GGN/Fisher backend. The configuration matching
`full_bnn_ablation.py`'s `Laplace (GGN)` is *all weights + full structure + GGN*; the
one matching the paper's headline method is *last layer + Kronecker*. Check the
current argument names against its docs before relying on them —
[Q6](../open-questions.md#q6).

---

## Symbols

| symbol | meaning |
|---|---|
| $\theta$, $P$ | the Bayesian weight vector, and its length |
| $\theta_{\text{MAP}}$ | trained weights = posterior mode |
| $\Sigma$ | posterior covariance, $P \times P$ |
| $C$ | curvature matrix of the negative log likelihood, $P \times P$ |
| $\tau = 1/\mathrm{var}_0$ | prior precision (a scalar); $\mathrm{var}_0$ is the prior variance |
| $I$ | identity matrix |
| $N$, $x_n$, $y_n$ | number of training points, the $n$-th input, its label $\in \{0,1\}$ |
| $f_\theta(x)$ | network output (logit) |
| $\sigma(z)$, $p_n$ | sigmoid; predicted probability $\sigma(f_\theta(x_n))$ |
| $\ell$, $L$ | per-example loss (binary cross-entropy); total loss $\sum_n \ell$ |
| $j_n$, $J$ | $\nabla_\theta f_\theta(x_n)$, length $P$; the $N \times P$ Jacobian stacking them |
| $c_n$ | the scalar weight in $C = \sum_n c_n j_n j_n^\top$ — the whole taxonomy |
| $\phi(x)$, $w$ | frozen feature map and last-layer weights, for the last-layer case |
| $a$, $\delta$ | *per layer*: input activations; gradient w.r.t. pre-activation output |
| $A$, $G$ | $\mathbb{E}[a a^\top]$ and $\mathbb{E}[\delta \delta^\top]$ — KFAC's two factors |
| $\otimes$ | Kronecker product |
| $\mathrm{fan\_in}$, $\mathrm{fan\_out}$ | a layer's input and output widths |
| $B$, $\rho$ | mini-batch size; running-average decay in the paper's factor accumulation |

## References
- **Martens & Grosse (2015)** — *Optimizing Neural Networks with Kronecker-factored
  Approximate Curvature*, ICML. KFAC itself. <https://arxiv.org/abs/1503.05671>
- **Botev, Ritter, Barber (2017)** — *Practical Gauss-Newton Optimisation for Deep
  Learning*, ICML. GGN and its Kronecker factorisation.
  <https://arxiv.org/abs/1706.03662>
- **Kunstner, Balles, Hennig (2019)** — *Limitations of the Empirical Fisher
  Approximation*, NeurIPS. <https://arxiv.org/abs/1905.12558>
- **Dangel, Kunstner, Hennig (2020)** — *BackPACK: Packing more into Backprop*, ICLR.
  <https://arxiv.org/abs/1912.10985>
- **Immer, Korzepa, Bauer (2021)** — *Improving predictions of Bayesian neural nets
  via local linearization*, AISTATS. Why a GGN posterior wants the *linearised*
  predictive — [decision 0012](../decisions/0012-ggn-laplace-for-deep-arms.md).
  <https://arxiv.org/abs/2008.08400>

**Transferable?** yes — this is the standard taxonomy for any Laplace or second-order
method, and none of it is specific to two moons or to this repo.
