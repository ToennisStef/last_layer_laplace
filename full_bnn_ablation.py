"""Ablation over *how much* of the network is Bayesian: last layer -> full BNN.

Answers Q2 (doc/open-questions.md): is the Laplace-vs-NUTS gap a property of the
Laplace approximation, or of the last-layer restriction? One architecture is fitted
at several "Bayesian depths" - only the output layer, the last two layers, then every
layer - with the same four inference methods each time. Any difference is then
attributable to the depth of the Bayesian treatment alone.

`two_moons_comparison` is imported, not copied: the data, the predictive estimator
(`predict_probs`), the uncertainty decomposition, the panel plot and `Config` all
come from there unchanged, so the numbers stay comparable to the last-layer results.
Nothing in that file - or in `prior_scale_sweep.py` - is modified.

What an "arm" is, the exact architecture and the per-arm parameter counts:
**doc/setup/bayesian-depth-arms.md**.

Design notes, all of them deliberate:

**BatchNorm stays in, but is never Bayesian.** The architecture matches
`two_moons_comparison` - `Linear -> BN -> ReLU` twice, then the output layer - so
that "how much is Bayesian" is the only thing this ablation varies. The BN affine
parameters are point-estimated in stage 1 along with any deterministic Linear
layers, then frozen; only Linear weights and biases ever become `pyro.sample`
sites, at every depth.

Frozen BN is a well-posed object where a Bayesian BN is not. In *train* mode BN
makes the likelihood depend on the whole batch, so `pyro.plate("data", ...)` stops
describing a product over observations, and the sampled posterior would belong to a
different model than the one used at prediction time. Its `gamma` is also
non-identifiable against the next layer's weights - ReLU is positively homogeneous,
so scaling one up and the other down leaves the function unchanged - which would
give NUTS a ridge to explore for no function-space gain. Freezing removes both
problems at once: in eval mode BN is an affine map with *fixed* constants, the
likelihood factorises again, and `gamma` is not a latent variable.

Two properties of that choice, neither of them bugs: the weight prior then lives in
post-BN coordinates, so a given `var0` does not mean the same thing with and without
BN (`--norm none` is available, but its arms are not comparable to BN arms at equal
`var0`); and BN's running statistics are calibrated at the MAP while posterior draws
sit far from it - which is equally true of the frozen feature map in
`two_moons_comparison`, so it is a shared property of the setup rather than a new
confound here.

**The last-layer arm is re-run here rather than reused.** Even at `--norm batchnorm`
this script's `ll` arm is not byte-identical to `two_moons_comparison` (its output
layer carries a bias, giving 21 latent dimensions against 20). Re-running it gives
the within-architecture baseline the ladder needs.

**Weight-space `r_hat` is not interpretable for the deeper arms.** Hidden-unit
permutations and ReLU sign flips make the posterior massively multimodal in weight
space, and equivalent chains can sit in different modes. `mcmc_diagnostics.json` is
still written, but read it as "did the chains mix over *this* parametrisation",
which for `full` they will not. Function-space diagnostics are the right tool and
are deliberately **not** implemented here - no metric is added by this script.

Cost warning: `full` has ~500 latent dimensions against 20 for `ll`. NUTS and the
full-rank guide both scale badly in that number. Use `--arms`, `--quick` and
`--skip-full-rank` rather than launching everything blind.

Layout (mirrors doc/decisions/0011, one timestamped folder, nothing overwritten):

    ablations/<stamp>_bayes_depth[_<tag>]/
        ablation_config.json   base Config, arm specs, flags, versions
        ablation_status.json   per-arm outcome: ok / skipped / the exception
        ablation_summary.json  collected scalars
        ablation_summary.csv   one row per (arm, method)
        figures/               cross-arm figures
        arms/arm<i>_<name>_d<latent_dim>/
            run.log            that arm's full stdout
            artifacts/         doc/decisions/0010 layout, plus deterministic_params.pt
            figures/           that arm's own panels

Usage (see README section "Running the ablation" for the uv form):
    uv run python full_bnn_ablation.py --quick --tag smoke     # plumbing check
    uv run python full_bnn_ablation.py --arms ll               # cheapest real arm
    uv run python full_bnn_ablation.py --arms ll last2 full    # the full ladder
    uv run python full_bnn_ablation.py --plot-only ablations/<dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before two_moons_comparison pulls in pyplot

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import (
    AutoLaplaceApproximation,
    AutoMultivariateNormal,
    AutoNormal,
    init_to_sample,
    init_to_value,
)
from pyro.optim import ClippedAdam, SGD
from torch import Tensor

from pyro import poutine

from prior_scale_sweep import METHOD_COLOR, METHOD_MARKER, REFERENCE, Tee
from prior_scale_sweep import region_masks
from two_moons_comparison import (
    Config,
    decompose_uncertainty,
    make_grid,
    plot_panels,
    predict_probs,
    set_seed,
    two_moons,
)

ABLATION_ROOT = "ablations"

# The two GGN rows are extra *methods*, not extra metrics: same confidence /
# epistemic / aleatoric quantities, a different posterior and (for the linearised
# row) a different way of turning that posterior into probabilities.
GGN = "Laplace (GGN)"
GGN_LIN = "Laplace (GGN, linearised)"
EXACT = "Laplace"

ABLATION_METHODS: tuple[str, ...] = (
    EXACT,
    GGN,
    GGN_LIN,
    "VI mean-field",
    "VI full-rank",
    REFERENCE,
)
COLORS = {**METHOD_COLOR, GGN: "#FB4F62", GGN_LIN: "#6E0019"}  # coral, bordeaux
MARKERS = {**METHOD_MARKER, GGN: "v", GGN_LIN: "P"}


# --- the architecture --------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """One rung of the ladder: how deep the Bayesian treatment goes.

    `bayes_depth` counts *trailing* Linear layers that are Bayesian, so the
    deterministic layers are always a prefix - which is what keeps the forward
    pass sane under vectorised sampling (only the tail carries sample dimensions).
    """

    name: str
    bayes_depth: int
    description: str


# The ladder. Parameter counts per arm, and why BatchNorm is never Bayesian:
# doc/setup/bayesian-depth-arms.md
ARMS: dict[str, ArmSpec] = {
    "ll": ArmSpec("ll", 1, "output layer only - the last-layer baseline"),
    "last2": ArmSpec("last2", 2, "output layer + second hidden layer"),
    "full": ArmSpec("full", 3, "every layer Bayesian - the full BNN"),
}


class PartiallyBayesianMLP(pyro.nn.PyroModule):
    """`n -> h -> h -> k` MLP whose *last* `bayes_depth` Linear layers are Bayesian.

    Deterministic layers stay as `nn.Linear` submodules, so their parameters land in
    the Pyro param store and stage 1 optimises them jointly with the MAP of the
    Bayesian sites - exactly the two-stage pattern `two_moons_comparison` uses for
    its feature map. Bayesian layers hold no parameters at all; their weights are
    `pyro.sample` sites applied functionally.

    Sites are named `w{i}` / `b{i}` by layer index, counting from the input, so an
    artifact can be read without knowing which arm produced it.
    """

    def __init__(
        self,
        n: int,
        h: int,
        k: int,
        *,
        bayes_depth: int,
        norm: str = "none",
    ) -> None:
        """Build the layer stack. `norm` is applied to deterministic layers only."""
        super().__init__()
        self.dims = [(n, h), (h, h), (h, k)]
        self.n_layers = len(self.dims)
        if not 1 <= bayes_depth <= self.n_layers:
            msg = f"bayes_depth must be in 1..{self.n_layers}, got {bayes_depth}"
            raise ValueError(msg)
        self.bayes_depth = bayes_depth
        self.norm = norm
        self.first_bayes = self.n_layers - bayes_depth
        self._prior_scale: float | None = None
        self._frozen = False

        # Deterministic Linear prefix. `nn.ModuleList` keeps the param names stable
        # across arms, which matters because the param store is what gets reloaded.
        self.det_layers = nn.ModuleList(
            [nn.Linear(i, o) for (i, o) in self.dims[: self.first_bayes]]
        )

        # BatchNorm after every hidden layer, independent of `bayes_depth`: it sits
        # *between* Bayesian layers in the deeper arms, so it cannot be tied to the
        # deterministic prefix. It is never Bayesian - see the module docstring.
        if norm == "batchnorm":
            self.norms = nn.ModuleList(
                [nn.BatchNorm1d(o) for (_, o) in self.dims[:-1]]
            )
        elif norm == "none":
            self.norms = None
        else:
            msg = f"unknown norm {norm!r}, expected 'none' or 'batchnorm'"
            raise ValueError(msg)

    @property
    def prior_scale(self) -> float | None:
        """Prior standard deviation shared by every Bayesian weight and bias."""
        return self._prior_scale

    @prior_scale.setter
    def prior_scale(self, value: float) -> None:
        self._prior_scale = value

    @property
    def bayes_sites(self) -> list[str]:
        """Names of every latent site, in forward order."""
        names = []
        for i in range(self.first_bayes, self.n_layers):
            names += [f"w{i}", f"b{i}"]
        return names

    @property
    def latent_dim(self) -> int:
        """Total number of latent dimensions - the number NUTS has to explore."""
        total = 0
        for i in range(self.first_bayes, self.n_layers):
            fan_in, fan_out = self.dims[i]
            total += fan_in * fan_out + fan_out
        return total

    def fan_in_values(self) -> dict[str, Tensor]:
        """A fan-in draw for every latent site, for use as a *guide init*.

        `U(-1/sqrt(fan_in), +1/sqrt(fan_in))`, i.e. what `nn.Linear` does. This is
        the guide-initialisation knob and explicitly not the prior scale: the
        defaults (`init_to_median`) would start every weight at 0 and never break
        the hidden-unit symmetry.
        """
        values = {}
        for i in range(self.first_bayes, self.n_layers):
            fan_in, fan_out = self.dims[i]
            bound = 1.0 / math.sqrt(fan_in)
            values[f"w{i}"] = (torch.rand(fan_out, fan_in) * 2 - 1) * bound
            values[f"b{i}"] = (torch.rand(fan_out) * 2 - 1) * bound
        return values

    def apply_norm(self, i: int, h_: Tensor) -> Tensor:
        """Apply BatchNorm layer *i*, by module while training, functionally once frozen.

        `nn.BatchNorm1d` accepts only ``(N, C)`` or ``(N, C, L)``, but under
        `Predictive(parallel=True)` or vectorised particles the activations arrive as
        ``(S, N, C)`` - so after freezing, BN is applied from its buffers instead:
        in eval mode it *is* just an affine map, and written out by hand it
        broadcasts over any number of leading sample dimensions.
        """
        if self.norms is None:
            return h_
        bn = self.norms[i]
        if not self._frozen:
            return bn(h_)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return (h_ - bn.running_mean) * scale + bn.bias

    def model(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Pyro model. Bayesian layers are applied functionally via `einsum`.

        `einsum('...oi,ni->...no')` rather than `F.linear` because sampled weights
        carry arbitrary leading sample dimensions under `Predictive(parallel=True)`,
        which `F.linear` will not broadcast over.
        """
        if self.prior_scale is None:
            msg = "No prior scale value set."
            raise ValueError(msg)

        h_ = x
        for i in range(self.first_bayes):
            h_ = self.det_layers[i](h_)
            h_ = torch.relu(self.apply_norm(i, h_))

        for i in range(self.first_bayes, self.n_layers):
            fan_in, fan_out = self.dims[i]
            w = pyro.sample(
                f"w{i}",
                dist.Normal(
                    loc=torch.zeros(fan_out, fan_in),
                    scale=self.prior_scale,
                ).to_event(2),
            )
            b = pyro.sample(
                f"b{i}",
                dist.Normal(
                    loc=torch.zeros(fan_out),
                    scale=self.prior_scale,
                ).to_event(1),
            )
            h_ = torch.einsum(
                "...oi,...ni->...no", w, h_.expand(*w.shape[:-2], *h_.shape[-2:])
            )
            h_ = h_ + b.unsqueeze(-2)
            if i < self.n_layers - 1:
                h_ = torch.relu(self.apply_norm(i, h_))

        logits = h_.squeeze(-1)
        pyro.deterministic("logits", logits)
        with pyro.plate("data", x.shape[0]):
            return pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    def deterministic_parameters(self) -> list[nn.Parameter]:
        """Every point-estimated parameter: the Linear prefix plus all BN affines."""
        params = list(self.det_layers.parameters())
        if self.norms is not None:
            params += list(self.norms.parameters())
        return params

    def freeze_deterministic(self) -> None:
        """Freeze the prefix after stage 1: eval mode plus `requires_grad_(False)`.

        Both are needed, as in `two_moons_comparison` - `eval()` alone leaves the
        parameters trainable, and `requires_grad_(False)` alone leaves BatchNorm
        updating its running statistics.
        """
        self.det_layers.eval()
        if self.norms is not None:
            self.norms.eval()
        for p in self.deterministic_parameters():
            p.requires_grad_(requires_grad=False)
        # Switches `apply_norm` to the functional path; also the point past which BN
        # stops updating its running statistics.
        self._frozen = True

    def deterministic_state(self) -> dict[str, Tensor]:
        """State dict of the frozen prefix, for reloading without refitting."""
        state = {f"det_layers.{k}": v for k, v in self.det_layers.state_dict().items()}
        if self.norms is not None:
            # state_dict, not parameters: the running statistics are buffers and are
            # part of what has to be reloaded to reproduce a prediction.
            state |= {f"norms.{k}": v for k, v in self.norms.state_dict().items()}
        return state


# --- one arm -----------------------------------------------------------------


def init_loc_fn_for(bnn: PartiallyBayesianMLP) -> Callable:
    """Guide init from a fan-in draw (see `fan_in_values` for why not the default)."""
    return init_to_value(values=bnn.fan_in_values())


def mcmc_init_strategy(
    bnn: PartiallyBayesianMLP, kind: str, lap_guide: AutoLaplaceApproximation
) -> Callable:
    """NUTS initialisation strategy.

    `fan_in` is the default here, unlike the last-layer script's `init_to_sample`.
    With a prior scale of ~45 over 500 dimensions, a prior draw starts the chain on
    a shell of radius `s*sqrt(d)` ~ 1000 where every logit is saturated; NUTS spends
    its whole warmup walking back. A fan-in draw is a well-conditioned start that is
    still *not* the mode, so it does not reintroduce the mode-seeding bias that
    Q5 retracted - unlike `map`, which does and is offered only for diagnosis.
    """
    if kind == "fan_in":
        return init_to_value(values=bnn.fan_in_values())
    if kind == "sample":
        return init_to_sample()
    if kind == "map":
        return init_to_value(values=map_point(lap_guide))
    msg = f"unknown mcmc init {kind!r}"
    raise ValueError(msg)


def map_point(lap_guide: AutoLaplaceApproximation) -> dict[str, Tensor]:
    """The stage-1 MAP, per latent site.

    `AutoLaplaceApproximation.median()` cannot be used: `AutoContinuous.median`
    goes through `_loc_scale`, which this guide only implements *after*
    `laplace_approximation()` has produced a covariance - and the whole point of the
    GGN route is to work on arms where that call fails. So take the guide's flat
    `loc` (its AutoDelta point) and let the guide itself split it into sites via
    `_unpack_latent`; that keeps pyro's packing order authoritative instead of
    assuming it matches ours. Every site here has real support, so the constraint
    transform `median()` would apply is the identity.
    """
    loc = dict(lap_guide.named_pyro_params())["loc"].detach().clone()
    return {
        site["name"]: value.detach().clone()
        for site, value in lap_guide._unpack_latent(loc)  # noqa: SLF001
    }


def site_shapes(bnn: PartiallyBayesianMLP) -> dict[str, torch.Size]:
    """Shape of every latent site, in forward order - defines the flat ordering."""
    shapes = {}
    for i in range(bnn.first_bayes, bnn.n_layers):
        fan_in, fan_out = bnn.dims[i]
        shapes[f"w{i}"] = torch.Size((fan_out, fan_in))
        shapes[f"b{i}"] = torch.Size((fan_out,))
    return shapes


def flatten_sites(values: dict[str, Tensor], shapes: dict[str, torch.Size]) -> Tensor:
    """Pack per-site values into one vector, in `shapes` order."""
    return torch.cat([values[k].reshape(-1) for k in shapes])


def unflatten_sites(
    theta: Tensor, shapes: dict[str, torch.Size], *, batch: bool = False
) -> dict[str, Tensor]:
    """Inverse of `flatten_sites`. With `batch`, `theta` is ``(S, P)``."""
    out, i = {}, 0
    for name, shape in shapes.items():
        n = int(np.prod(shape)) if len(shape) else 1
        chunk = theta[..., i : i + n]
        out[name] = chunk.reshape(*chunk.shape[:-1], *shape) if batch else chunk.reshape(shape)
        i += n
    return out


class GGNLaplace:
    """Laplace posterior built from the Generalized Gauss-Newton, plus `tau*I`.

    The exact Hessian of the negative log likelihood splits as

        H = J^T Lambda J  +  sum_c (grad_f l)_c * grad^2_theta f_c

    with `J = df/dtheta` and `Lambda = grad^2_f l` the curvature of the loss in the
    *output*. The GGN keeps the first term and drops the second. Since the binary
    cross-entropy is convex in the logit, `Lambda = p(1-p) >= 0`, so
    `J^T Lambda J` is a sum of rank-1 PSD terms and is positive semi-definite for
    *any* network. Adding the prior precision `tau = 1/var0` then makes it strictly
    positive definite, so unlike the exact Hessian it can always be inverted
    (doc/gotchas/full-bnn-laplace-not-pd.md).

    The dropped term is exactly the one that made the exact Hessian indefinite: it
    carries the network's own curvature in weight space, weighted by a residual
    whose sign is unconstrained.

    For the `ll` arm the drop costs nothing - with the features frozen the logit is
    linear in `w`, so `grad^2_theta f = 0` and GGN *is* the exact Hessian. `run_arm`
    checks that numerically.

    Two predictives are exposed, because the GGN is the exact Hessian of the
    *linearised* model and the two ways of using it are not equivalent:

    * `sample_sites` draws weights and pushes them through the real, nonlinear
      network - the estimator every other method here uses (decision 0009), so the
      numbers stay comparable.
    * `linearised_probs` propagates the Jacobian instead,
      `f ~ N(f_map, diag(J Sigma J^T))`, which is self-consistent with the GGN and
      is what Kristiadi et al. do (their `forward_linearized`). Its mean matches the
      MacKay probit formula `sigmoid(f_map / sqrt(1 + pi/8 v))`; it is drawn rather
      than evaluated in closed form only so that the same
      aleatoric/epistemic decomposition applies to it.
    """

    def __init__(
        self,
        bnn: PartiallyBayesianMLP,
        cfg: Config,
        theta_map: Tensor,
        precision: Tensor,
        shapes: dict[str, torch.Size],
    ) -> None:
        """Store the fitted posterior and the pieces needed to predict from it."""
        self.bnn = bnn
        self.cfg = cfg
        self.theta_map = theta_map
        self.precision = precision
        self.shapes = shapes
        self.posterior = torch.distributions.MultivariateNormal(
            theta_map, precision_matrix=precision
        )
        self.covariance = self.posterior.covariance_matrix

    def logits_at(self, theta: Tensor, x: Tensor) -> Tensor:
        """Model logits at a flat parameter vector - the *same* model, via poutine.

        Going through `poutine.condition` rather than a hand-written forward pass
        means there is only one definition of the network; a second copy would be
        free to drift away from the one every other method is using.
        """
        values = unflatten_sites(theta, self.shapes)
        conditioned = poutine.condition(self.bnn.model, data=values)
        trace = poutine.trace(conditioned).get_trace(x, None)
        return trace.nodes["logits"]["value"]

    def jacobian(self, x: Tensor) -> Tensor:
        """``d logits / d theta`` at the MAP, shape ``(len(x), P)``."""

        def f(theta: Tensor) -> Tensor:
            return self.logits_at(theta, x)

        try:
            return torch.autograd.functional.jacobian(f, self.theta_map, vectorize=True)
        except RuntimeError:
            # vectorize=True vmaps the backward pass; fall back if that trips over
            # anything in the trace.
            return torch.autograd.functional.jacobian(f, self.theta_map)

    def draw_weights(self, n_samples: int) -> dict[str, Tensor]:
        """Draw weights, shaped like MCMC samples so `Predictive` accepts them."""
        theta = self.posterior.sample((n_samples,))
        return unflatten_sites(theta, self.shapes, batch=True)

    def linearised_probs(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Linearised predictive: returns ``(probs (S, N), probit_mean (N,))``.

        `probs` are MC draws of `sigmoid(f)` with `f ~ N(f_map, v)` per point;
        `probit_mean` is MacKay's closed form for the same quantity, returned only
        as a check that the two agree.
        """
        chunks, probits = [], []
        for start in range(0, x.shape[0], self.cfg.pred_chunk):
            batch = x[start : start + self.cfg.pred_chunk]
            jac = self.jacobian(batch)
            with torch.no_grad():
                f_map = self.logits_at(self.theta_map, batch)
                var = ((jac @ self.covariance) * jac).sum(-1).clamp_min(0.0)
                eps = torch.randn(self.cfg.pred_samples, batch.shape[0])
                chunks.append(torch.sigmoid(f_map + var.sqrt() * eps))
                probits.append(
                    torch.sigmoid(f_map / torch.sqrt(1 + math.pi / 8 * var))
                )
        return torch.cat(chunks, dim=-1), torch.cat(probits, dim=-1)


def fit_ggn_laplace(
    bnn: PartiallyBayesianMLP,
    cfg: Config,
    x_train: Tensor,
    y_train: Tensor,
    theta_map_values: dict[str, Tensor],
) -> GGNLaplace:
    """Assemble ``GGN + tau*I`` at the MAP and return the posterior."""
    shapes = site_shapes(bnn)
    missing = set(shapes) - set(theta_map_values)
    if missing:
        msg = f"MAP is missing latent sites {sorted(missing)}"
        raise ValueError(msg)
    theta_map = flatten_sites(theta_map_values, shapes).detach().clone()
    expected = sum(int(np.prod(sh)) if len(sh) else 1 for sh in shapes.values())
    if theta_map.numel() != expected:
        msg = f"flattened MAP has {theta_map.numel()} entries, expected {expected}"
        raise ValueError(msg)
    tau = 1.0 / cfg.var0

    lap = GGNLaplace(bnn, cfg, theta_map, torch.eye(theta_map.numel()), shapes)
    jac = lap.jacobian(x_train)  # (N, P)
    with torch.no_grad():
        p = torch.sigmoid(lap.logits_at(theta_map, x_train))
        lam = (p * (1 - p)).clamp_min(0.0)  # curvature of BCE in the logit
        ggn = jac.T @ (lam.unsqueeze(-1) * jac)
        ggn = 0.5 * (ggn + ggn.T)  # symmetrise away autodiff round-off
        precision = ggn + tau * torch.eye(theta_map.numel())
    return GGNLaplace(bnn, cfg, theta_map, precision, shapes)


def posterior_spread(samples: dict[str, Tensor]) -> float:
    """Mean over coordinates of the posterior std, pooled across all sites.

    The same quantity `two_moons_comparison` reports as `std mean`, generalised to
    several sites by flattening and concatenating. Not a new metric.
    """
    flat = [v.reshape(v.shape[0], -1).std(0) for v in samples.values()]
    return float(torch.cat(flat).mean())


def run_arm(cfg: Config, spec: ArmSpec, arm_dir: Path, *, args) -> str:
    """Fit one arm: MAP, Laplace, two VI guides, NUTS; save artifacts and figures."""
    torch.set_default_dtype(torch.float64)
    pyro.clear_param_store()
    set_seed(cfg.seed)

    figure_dir = Path(cfg.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = two_moons(cfg.n_train, sigma=cfg.noise)
    x1_grid, x2_grid, x_test = make_grid(cfg)
    _, n = x_train.shape

    bnn = PartiallyBayesianMLP(
        n=n, h=cfg.h, k=cfg.k, bayes_depth=spec.bayes_depth, norm=args.norm
    )
    bnn.prior_scale = cfg.prior_scale
    print(
        f"arm {spec.name!r}: {spec.description}\n"
        f"  bayes_depth = {spec.bayes_depth}/{bnn.n_layers}, "
        f"latent dim = {bnn.latent_dim}, sites = {bnn.bayes_sites}\n"
        f"  prior_scale = {cfg.prior_scale:.3f} (var0 = {cfg.var0:.1f}), norm = {args.norm}"
    )

    # --- stage 1: MAP of the latent sites + the deterministic prefix ---------
    # AutoLaplaceApproximation acts as AutoDelta here, so the ELBO is -log p(w, D).
    # NOTE: this objective contains the prior, so the prefix that gets frozen below
    # depends on the prior scale - see doc/gotchas/prior-enters-map-fit.md.
    lap_guide = AutoLaplaceApproximation(bnn.model, init_loc_fn=init_loc_fn_for(bnn))
    pyro.module("_lap_guide", lap_guide)  # fixes the param names map_point() reads
    if bnn.norms is not None:
        bnn.norms.train()
    svi_map = SVI(
        model=bnn.model,
        guide=lap_guide,
        optim=SGD(
            {
                "lr": cfg.map_lr,
                "momentum": cfg.map_momentum,
                "weight_decay": cfg.map_weight_decay,
            }
        ),
        loss=Trace_ELBO(),
    )
    print(f"stage 1: MAP, {cfg.map_steps} steps")
    map_losses = [svi_map.step(x_train, y_train) for _ in range(cfg.map_steps)]
    print(f"  -log p(w, D) = {map_losses[-1]:.3f}")

    bnn.freeze_deterministic()

    # --- stage 2: the three Gaussian posteriors ------------------------------
    # Each posterior is fitted defensively: for the deeper arms a method can fail
    # for reasons that are themselves the finding (see below), and one failure must
    # not cost the arm its other three methods.
    method_status: dict[str, str] = {}

    print("stage 2: Laplace at the MAP")
    lap_posterior_guide = None
    try:
        lap_posterior_guide = lap_guide.laplace_approximation(x_train, y_train)
        method_status["Laplace"] = "ok"
    except Exception as exc:  # noqa: BLE001
        # Expected for `full`, and worth stating plainly rather than patching over:
        # with the features frozen, the last-layer posterior is a logistic
        # regression in `w` - log-concave, so its Hessian is positive definite and
        # the Laplace approximation is well posed. A full BNN has no such guarantee:
        # permutation/sign symmetries give exactly flat directions and the loss
        # surface has saddles, so the *exact* Hessian at the point SGD reached need
        # not be invertible. Pyro offers no GGN/Fisher fallback (which is what makes
        # `laplace-torch` PSD by construction), so the honest report is that this
        # method did not apply - not a silently damped substitute.
        method_status["Laplace"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {method_status['Laplace']}")
        print(
            "  -> exact-Hessian Laplace is not well posed here; continuing with the "
            "remaining methods. See doc/gotchas/full-bnn-laplace-not-pd.md"
        )

    # --- the GGN Laplace, which unlike the exact Hessian always exists ------
    print("stage 2: Laplace (GGN + tau*I)")
    ggn_lap = None
    try:
        ggn_lap = fit_ggn_laplace(bnn, cfg, x_train, y_train, map_point(lap_guide))
        method_status[GGN] = "ok"
        method_status[GGN_LIN] = "ok"
        print(f"  precision {tuple(ggn_lap.precision.shape)}, tau = {1 / cfg.var0:.3e}")
        if bnn.bayes_depth == 1 and lap_posterior_guide is not None:
            # With the features frozen the logit is linear in w, so the term the
            # GGN drops is identically zero and the two precisions must coincide.
            # Compared through eigenvalues, which are invariant to how each side
            # happens to order the flattened latent.
            exact_cov = lap_posterior_guide.get_posterior().covariance_matrix
            exact_prec = torch.linalg.inv(exact_cov)
            e_exact = torch.linalg.eigvalsh(0.5 * (exact_prec + exact_prec.T))
            e_ggn = torch.linalg.eigvalsh(ggn_lap.precision)
            rel = float(
                ((e_exact - e_ggn).abs() / e_exact.abs().clamp_min(1e-12)).max()
            )
            print(f"  check vs exact Hessian: max relative eigenvalue diff {rel:.3e}")
            method_status[GGN] = f"ok (matches exact Hessian, max rel eig diff {rel:.1e})"
    except Exception as exc:  # noqa: BLE001
        method_status[GGN] = f"{type(exc).__name__}: {exc}"
        method_status[GGN_LIN] = method_status[GGN]
        print(f"  FAILED: {method_status[GGN]}")

    def fit_vi(guide: Callable, name: str) -> list[float]:
        svi = SVI(
            model=bnn.model,
            guide=guide,
            optim=ClippedAdam({"lr": cfg.vi_lr, "lrd": cfg.vi_lrd, "clip_norm": 10.0}),
            loss=Trace_ELBO(num_particles=cfg.vi_particles, vectorize_particles=True),
        )
        print(f"stage 2: {name}, {cfg.vi_steps} steps")
        losses = [svi.step(x_train, y_train) for _ in range(cfg.vi_steps)]
        print(f"  final ELBO loss = {losses[-1]:.3f}")
        if not math.isfinite(losses[-1]):
            msg = f"final ELBO is {losses[-1]} - the guide diverged"
            raise RuntimeError(msg)
        return losses

    mf_guide = AutoNormal(bnn.model, init_loc_fn=init_loc_fn_for(bnn))
    mf_losses: list[float] = []
    try:
        mf_losses = fit_vi(mf_guide, "AutoNormal (mean-field)")
        method_status["VI mean-field"] = "ok"
    except Exception as exc:  # noqa: BLE001
        mf_guide = None
        method_status["VI mean-field"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {method_status['VI mean-field']}")

    fr_guide, fr_losses = None, []
    if args.skip_full_rank:
        # d(d+1)/2 covariance entries: 210 for `ll`, ~125k for `full`.
        print(f"stage 2: full-rank skipped (--skip-full-rank, d = {bnn.latent_dim})")
        method_status["VI full-rank"] = "skipped (--skip-full-rank)"
    else:
        fr_guide = AutoMultivariateNormal(bnn.model, init_loc_fn=init_loc_fn_for(bnn))
        try:
            fr_losses = fit_vi(fr_guide, "AutoMultivariateNormal (full-rank)")
            method_status["VI full-rank"] = "ok"
        except Exception as exc:  # noqa: BLE001
            fr_guide = None
            method_status["VI full-rank"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {method_status['VI full-rank']}")

    # --- stage 3: NUTS -------------------------------------------------------
    print(
        f"stage 3: NUTS, {cfg.mcmc_chains} chains x {cfg.mcmc_samples} samples "
        f"over {bnn.latent_dim} dims, init = {args.mcmc_init}"
    )
    kernel = NUTS(bnn.model, init_strategy=mcmc_init_strategy(bnn, args.mcmc_init, lap_guide))
    mcmc = MCMC(
        kernel,
        num_samples=cfg.mcmc_samples,
        warmup_steps=cfg.mcmc_warmup,
        num_chains=cfg.mcmc_chains,
    )
    try:
        mcmc.run(x_train, y_train)
    except RuntimeError as exc:
        # Same Windows multiprocessing trap as two_moons_comparison: num_chains > 1
        # spawns processes and needs the caller behind `if __name__ == "__main__"`.
        if "bootstrapping phase" not in str(exc):
            raise
        total = cfg.mcmc_samples * cfg.mcmc_chains
        print(
            '  WARNING: multi-chain sampling needs an `if __name__ == "__main__"` '
            f"guard in the entry script.\n  Falling back to 1 chain x {total} "
            "samples - no cross-chain r_hat available."
        )
        kernel = NUTS(bnn.model, init_strategy=mcmc_init_strategy(bnn, args.mcmc_init, lap_guide))
        mcmc = MCMC(kernel, num_samples=total, warmup_steps=cfg.mcmc_warmup, num_chains=1)
        mcmc.run(x_train, y_train)
    method_status[REFERENCE] = "ok"
    if bnn.latent_dim <= args.summary_max_dim:
        mcmc.summary()
    else:
        print(
            f"  (mcmc.summary() suppressed: {bnn.latent_dim} dims > "
            f"--summary-max-dim {args.summary_max_dim}; see mcmc_diagnostics.json)"
        )
    print(
        "  reminder: weight-space r_hat is not interpretable for the deeper arms - "
        "permutation and sign symmetries make equivalent chains disagree."
    )

    # --- predictive comparison ----------------------------------------------
    print("predictive:")
    method_kwargs: list[tuple[str, dict]] = []
    if lap_posterior_guide is not None:
        method_kwargs.append((EXACT, {"guide": lap_posterior_guide}))
    if ggn_lap is not None:
        # Same posterior as the linearised row below, pushed through the *real*
        # network rather than its linearisation.
        method_kwargs.append(
            (GGN, {"posterior_samples": ggn_lap.draw_weights(cfg.pred_samples)})
        )
    if mf_guide is not None:
        method_kwargs.append(("VI mean-field", {"guide": mf_guide}))
    if fr_guide is not None:
        method_kwargs.append(("VI full-rank", {"guide": fr_guide}))
    method_kwargs.append((REFERENCE, {"posterior_samples": mcmc.get_samples()}))
    dropped = {k: v for k, v in method_status.items() if v != "ok"}
    if dropped:
        print(f"  methods not available for this arm: {json.dumps(dropped, indent=2)}")

    results: dict[str, dict[str, Tensor]] = {}
    w_samples: dict[str, dict[str, Tensor]] = {}
    for name, kwargs in method_kwargs:
        probs = predict_probs(bnn.model, x_test, cfg, **kwargs)
        results[name] = decompose_uncertainty(probs)
        w_samples[name] = sample_sites(bnn, cfg, x_train, **kwargs)
        print(f"  {name:20s} done ({probs.shape[0]} posterior draws)")

    if ggn_lap is not None:
        # Not routed through `predict_probs`: this predictive propagates the
        # Jacobian instead of sampling weights, so it needs its own call.
        probs_lin, probit = ggn_lap.linearised_probs(x_test)
        results[GGN_LIN] = decompose_uncertainty(probs_lin)
        w_samples[GGN_LIN] = w_samples[GGN]  # identical posterior, other predictive
        diff = (probs_lin.mean(0) - probit).abs()
        # Two sources separate these: MC noise, which shrinks as 1/sqrt(S) and
        # dominates the *max* over a large grid, and the probit approximation's own
        # bias, which does not shrink and shows up in the *mean*. `mc_scale` is the
        # crude worst-case Bernoulli standard error for reference.
        mc_scale = 0.5 / math.sqrt(probs_lin.shape[0])
        print(f"  {GGN_LIN:20s} done ({probs_lin.shape[0]} draws)")
        print(
            f"    MC mean vs MacKay probit: mean {float(diff.mean()):.4f}, "
            f"max {float(diff.max()):.4f}  (MC noise scale ~{mc_scale:.4f})"
        )

    print(f"\n{'method':26s} {'std mean':>10s} {'conf':>8s} {'epistemic':>10s}")
    for name in ABLATION_METHODS:
        if name not in results:
            continue
        res = results[name]
        print(
            f"{name:26s} {posterior_spread(w_samples[name]):10.3f} "
            f"{res['confidence'].mean():8.3f} {res['epistemic'].mean():10.4f}"
        )

    # --- figures -------------------------------------------------------------
    print("figures:")
    plot_panels(
        results,
        "confidence",
        x1_grid,
        x2_grid,
        x_train,
        y_train,
        title=f"confidence  max(p, 1-p)   [arm: {spec.name}]",
        cmap="Blues",
        levels=np.arange(0.5, 1.001, 0.025),
        out_path=figure_dir / "confidence_comparison.svg",
    )
    plot_panels(
        results,
        "epistemic",
        x1_grid,
        x2_grid,
        x_train,
        y_train,
        title=f"epistemic uncertainty  I[y; w | x]  (nats)   [arm: {spec.name}]",
        cmap="magma",
        levels=None,
        out_path=figure_dir / "epistemic_comparison.svg",
    )

    save_arm_artifacts(
        cfg,
        bnn,
        spec,
        lap_posterior_guide,
        mcmc,
        tensors={
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "x1_grid": x1_grid,
            "x2_grid": x2_grid,
        },
        results=results,
        losses={"map": map_losses, "mean_field": mf_losses, "full_rank": fr_losses},
        w_samples=w_samples,
        norm=args.norm,
        method_status=method_status,
    )
    return "ok"


def sample_sites(
    bnn: PartiallyBayesianMLP,
    cfg: Config,
    x: Tensor,
    *,
    guide: Callable | None = None,
    posterior_samples: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Draw posterior samples of every latent site, shape ``(S, ...)`` each.

    Routed through `Predictive` for the same reason as in `two_moons_comparison`:
    `AutoNormal` has no `get_posterior()`, so no analytic route covers every guide
    family and mixing analytic with sampled spread would not be comparable.
    """
    if posterior_samples is not None:
        return {k: v.detach() for k, v in posterior_samples.items()}
    from pyro.infer import Predictive

    predictive = Predictive(
        bnn.model,
        guide=guide,
        num_samples=cfg.pred_samples,
        return_sites=tuple(bnn.bayes_sites),
        parallel=True,
    )
    with torch.no_grad():
        drawn = predictive(x[:2], None)
    return {k: v.squeeze(1).detach() if v.dim() > 1 else v.detach() for k, v in drawn.items()}


def save_arm_artifacts(
    cfg: Config,
    bnn: PartiallyBayesianMLP,
    spec: ArmSpec,
    lap_guide,
    mcmc: MCMC,
    tensors: dict[str, Tensor],
    results: dict[str, dict[str, Tensor]],
    losses: dict[str, list[float]],
    w_samples: dict[str, dict[str, Tensor]],
    *,
    norm: str,
    method_status: dict[str, str],
) -> Path:
    """Persist one arm in the doc/decisions/0010 layout, plus the frozen prefix."""
    out = Path(cfg.artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": asdict(cfg),
        "arm": {**asdict(spec), "latent_dim": bnn.latent_dim, "norm": norm,
                "sites": bnn.bayes_sites},
        "method_status": method_status,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyro": pyro.__version__,
            "numpy": np.__version__,
        },
        "default_dtype": str(torch.get_default_dtype()),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    torch.save(tensors, out / "data.pt")
    torch.save(bnn.deterministic_state(), out / "deterministic_params.pt")
    pyro.get_param_store().save(str(out / "param_store.pt"))

    if lap_guide is not None:
        posterior = lap_guide.get_posterior()
        torch.save(
            {
                "loc": posterior.mean.detach(),
                "covariance_matrix": posterior.covariance_matrix.detach(),
                "stddev": posterior.stddev.detach(),
            },
            out / "laplace_posterior.pt",
        )

    torch.save(
        {
            "samples": {k: v.detach() for k, v in mcmc.get_samples().items()},
            "samples_by_chain": {
                k: v.detach() for k, v in mcmc.get_samples(group_by_chain=True).items()
            },
        },
        out / "mcmc_samples.pt",
    )
    diagnostics = {
        site: {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in stats.items()}
        for site, stats in mcmc.diagnostics().items()
    }
    (out / "mcmc_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    torch.save(w_samples, out / "w_samples.pt")
    torch.save(results, out / "predictions.pt")
    torch.save(losses, out / "losses.pt")
    print(f"  wrote artifacts to {out.resolve()}")
    return out


# --- collection --------------------------------------------------------------


def is_complete(arm_dir: Path) -> bool:
    """An arm counts as finished once its predictive fields are on disk."""
    return (arm_dir / "artifacts" / "predictions.pt").exists()


def collect_arm(arm_dir: Path) -> dict:
    """Reduce one finished arm to scalars. Same quantities as the last-layer runs."""
    art = arm_dir / "artifacts"
    meta = json.loads((art / "config.json").read_text(encoding="utf-8"))
    preds = torch.load(art / "predictions.pt", weights_only=False)
    data = torch.load(art / "data.pt", weights_only=False)
    w_samples = torch.load(art / "w_samples.pt", weights_only=False)
    losses = torch.load(art / "losses.pt", weights_only=False)
    diagnostics = json.loads((art / "mcmc_diagnostics.json").read_text(encoding="utf-8"))
    masks = region_masks(data["x_test"])

    methods = {}
    for name, res in preds.items():
        row = {"w_std_mean": posterior_spread(w_samples[name])}
        for key in ("confidence", "epistemic", "aleatoric", "total"):
            for region, mask in masks.items():
                suffix = "" if region == "all" else f"_{region}"
                row[f"{key}{suffix}"] = float(res[key][mask].mean())
        methods[name] = row

    gaps = {}
    if REFERENCE in preds:
        ref = preds[REFERENCE]
        for name, res in preds.items():
            if name == REFERENCE:
                continue
            key = name.replace(" ", "_")
            gaps[f"conf_gap_{key}"] = float(
                res["confidence"].mean() - ref["confidence"].mean()
            )
            gaps[f"p_mean_l1_{key}"] = float((res["p_mean"] - ref["p_mean"]).abs().mean())

    r_hats = [
        v
        for site, stats in diagnostics.items()
        if site != "divergences"
        for v in (stats.get("r_hat") or [])
        if isinstance(v, (int, float))
    ]
    return {
        "arm_dir": arm_dir.name,
        "arm": meta["arm"]["name"],
        "bayes_depth": meta["arm"]["bayes_depth"],
        "latent_dim": meta["arm"]["latent_dim"],
        "norm": meta["arm"]["norm"],
        "var0": float(meta["config"]["var0"]),
        "prior_scale": math.sqrt(float(meta["config"]["var0"])),
        "method_status": meta.get("method_status", {}),
        "r_hat_max": max(r_hats) if r_hats else None,
        "map_loss_final": losses["map"][-1] if losses.get("map") else None,
        "methods": methods,
        "gaps": gaps,
    }


def collect_ablation(ablation_dir: Path) -> list[dict]:
    """Collect every finished arm, in ladder order."""
    rows = []
    for arm_dir in sorted(p for p in (ablation_dir / "arms").iterdir() if p.is_dir()):
        if not is_complete(arm_dir):
            print(f"  skipping incomplete arm {arm_dir.name}")
            continue
        rows.append(collect_arm(arm_dir))
    return sorted(rows, key=lambda r: r["bayes_depth"])


def write_tables(ablation_dir: Path, rows: list[dict]) -> None:
    """One row per (arm, method), plus the full JSON."""
    (ablation_dir / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    method_fields = sorted({k for r in rows for m in r["methods"].values() for k in m})
    gap_keys = sorted({k for r in rows for k in r["gaps"]})
    with (ablation_dir / "ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["arm", "bayes_depth", "latent_dim", "norm", "prior_scale", "method",
             *method_fields, *gap_keys]
        )
        for row in rows:
            for name in ABLATION_METHODS:
                if name not in row["methods"]:
                    continue
                m = row["methods"][name]
                key = name.replace(" ", "_")
                writer.writerow(
                    [
                        row["arm"], row["bayes_depth"], row["latent_dim"], row["norm"],
                        row["prior_scale"], name,
                        *(m.get(k, "") for k in method_fields),
                        *(row["gaps"].get(k, "") if k.endswith(key) else ""
                          for k in gap_keys),
                    ]
                )
    print(f"  wrote tables to {ablation_dir.resolve()}")


# --- cross-arm figures -------------------------------------------------------


def plot_by_arm(
    rows: list[dict],
    key: str,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    hline: float | None = None,
) -> None:
    """Metric against Bayesian depth, one line per method, one panel per region."""
    labels = [f"{r['arm']}\nd={r['latent_dim']}" for r in rows]
    xs = list(range(len(rows)))
    regions = [("all", "whole grid"), ("near", "near data"), ("far", "far field")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, (region, label) in zip(axes, regions, strict=True):
        suffix = "" if region == "all" else f"_{region}"
        for name in ABLATION_METHODS:
            ys = [r["methods"].get(name, {}).get(f"{key}{suffix}") for r in rows]
            pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pairs:
                continue
            ax.plot(
                [x for x, _ in pairs],
                [y for _, y in pairs],
                marker=MARKERS[name],
                color=COLORS[name],
                label=name,
                linewidth=1.6,
                markersize=5,
            )
        if hline is not None:
            ax.axhline(hline, color="grey", linestyle=":", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("how much of the net is Bayesian")
        ax.grid(visible=True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_confidence_grid(ablation_dir: Path, rows: list[dict], out_path: Path) -> None:
    """Rows = arm, columns = method."""
    loaded = []
    for row in rows:
        art = ablation_dir / "arms" / row["arm_dir"] / "artifacts"
        loaded.append(
            (
                row,
                torch.load(art / "data.pt", weights_only=False),
                torch.load(art / "predictions.pt", weights_only=False),
            )
        )
    if not loaded:
        return
    shape = loaded[0][1]["x1_grid"].shape
    levels = np.arange(0.5, 1.001, 0.025)
    fig, axes = plt.subplots(
        len(loaded), len(ABLATION_METHODS), figsize=(2.5 * len(ABLATION_METHODS), 2.5 * len(loaded)),
        squeeze=False,
    )
    im = None
    for i, (row, data, preds) in enumerate(loaded):
        for j, name in enumerate(ABLATION_METHODS):
            ax = axes[i][j]
            res = preds.get(name)
            if res is None:
                ax.axis("off")
                continue
            im = ax.contourf(
                data["x1_grid"], data["x2_grid"],
                res["confidence"].reshape(shape).numpy(),
                levels=levels, cmap="Blues", extend="neither",
            )
            ax.contour(
                data["x1_grid"], data["x2_grid"],
                res["p_mean"].reshape(shape).numpy(),
                levels=[0.5], colors="black", linewidths=1.2,
            )
            ax.scatter(
                data["x_train"][:, 0], data["x_train"][:, 1], c=data["y_train"],
                cmap="coolwarm", s=4, edgecolors="k", linewidths=0.15,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"mean {float(res['confidence'].mean()):.3f}", fontsize=8)
            if i == 0:
                ax.set_xlabel(name, fontsize=10)
                ax.xaxis.set_label_position("top")
            if j == 0:
                ax.set_ylabel(f"{row['arm']}  (d={row['latent_dim']})", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes, label="confidence  max(p, 1-p)", fraction=0.015)
    fig.suptitle("Confidence by Bayesian depth (rows) and inference method (columns)",
                 y=0.995)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_figures(ablation_dir: Path, rows: list[dict]) -> None:
    """All cross-arm figures. Reads artifacts only."""
    fig_dir = ablation_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("ablation figures:")
    plot_by_arm(
        rows, "confidence",
        title="Mean confidence by Bayesian depth",
        ylabel="mean  max(p, 1-p)",
        out_path=fig_dir / "confidence_by_arm.svg",
        hline=0.5,
    )
    plot_by_arm(
        rows, "epistemic",
        title="Mean epistemic uncertainty  $I[y; w \\mid x]$  by Bayesian depth",
        ylabel="mean epistemic (nats)",
        out_path=fig_dir / "epistemic_by_arm.svg",
    )
    plot_confidence_grid(ablation_dir, rows, fig_dir / "confidence_grid.svg")


def report(rows: list[dict]) -> None:
    """The table worth reading straight after the ablation."""
    print(
        f"\n{'arm':8s} {'d':>5s} {'method':26s} {'conf':>7s} {'conf_far':>9s} "
        f"{'epist':>8s} {'w_std':>9s} {'|dp| vs NUTS':>13s}"
    )
    for row in rows:
        for name in ABLATION_METHODS:
            m = row["methods"].get(name)
            if m is None:
                continue
            l1 = row["gaps"].get(f"p_mean_l1_{name.replace(' ', '_')}")
            l1_str = "-" if l1 is None else f"{l1:.4f}"
            print(
                f"{row['arm']:8s} {row['latent_dim']:5d} {name:26s} "
                f"{m['confidence']:7.3f} {m['confidence_far']:9.3f} "
                f"{m['epistemic']:8.4f} {m['w_std_mean']:9.3f} {l1_str:>13s}"
            )


# --- entry point -------------------------------------------------------------


QUICK = {
    "n_train": 100,
    "grid_n": 40,
    "map_steps": 200,
    "vi_steps": 300,
    "mcmc_samples": 50,
    "mcmc_warmup": 50,
    "mcmc_chains": 2,
    "pred_samples": 100,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(ARMS),
        default=["ll", "last2", "full"],
        help="which rungs of the ladder to run, cheapest first",
    )
    p.add_argument(
        "--norm",
        choices=("batchnorm", "none"),
        default="batchnorm",
        help="frozen, never-Bayesian BatchNorm after each hidden layer, matching "
        "two_moons_comparison (default); 'none' drops it, but arms are then not "
        "comparable to BN arms at equal var0",
    )
    p.add_argument(
        "--mcmc-init",
        choices=("fan_in", "sample", "map"),
        default="fan_in",
        help="NUTS initialisation; 'map' reintroduces the bias Q5 retracted",
    )
    p.add_argument(
        "--skip-full-rank",
        action="store_true",
        help="skip AutoMultivariateNormal - its covariance is d(d+1)/2 parameters",
    )
    p.add_argument(
        "--summary-max-dim",
        type=int,
        default=100,
        help="suppress mcmc.summary() above this latent dimension (it prints one "
        "row per coordinate)",
    )
    p.add_argument("--tag", default=None, help="suffix for the ablation folder name")
    p.add_argument("--quick", action="store_true", help="tiny settings: plumbing check")
    p.add_argument("--seed", type=int, default=None, help="override Config.seed")
    p.add_argument("--var0", type=float, default=None, help="override Config.var0")
    p.add_argument(
        "--pred-samples",
        type=int,
        default=None,
        help="override Config.pred_samples; raise it to separate MC noise from the "
        "probit approximation bias in the linearised row",
    )
    p.add_argument(
        "--resume", type=Path, default=None, help="continue an existing ablation folder"
    )
    p.add_argument("--force", action="store_true", help="re-run arms that already ran")
    p.add_argument(
        "--plot-only", type=Path, default=None, help="re-collect and re-plot, no fitting"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the requested arms, then collect, tabulate and plot."""
    args = parse_args(argv)

    if args.plot_only is not None:
        rows = collect_ablation(args.plot_only)
        if not rows:
            msg = f"no finished arms under {args.plot_only}"
            raise SystemExit(msg)
        write_tables(args.plot_only, rows)
        make_figures(args.plot_only, rows)
        report(rows)
        return

    base = Config()
    overrides: dict = dict(QUICK) if args.quick else {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.var0 is not None:
        overrides["var0"] = args.var0
    if args.pred_samples is not None:
        overrides["pred_samples"] = args.pred_samples
    if overrides:
        base = replace(base, **overrides)

    if args.resume is not None:
        ablation_dir = args.resume
        if not ablation_dir.exists():
            msg = f"--resume: {ablation_dir} does not exist"
            raise SystemExit(msg)
        stored = json.loads(
            (ablation_dir / "ablation_config.json").read_text(encoding="utf-8")
        )
        base = Config(**stored["base_config"])
        args.norm = stored["norm"]
        print(f"resuming {ablation_dir} with its stored config")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
        name = f"{stamp}_bayes_depth" + (f"_{args.tag}" if args.tag else "")
        ablation_dir = Path(ABLATION_ROOT) / name
        if ablation_dir.exists() and not args.force:
            msg = f"{ablation_dir} already exists; use --resume or --force"
            raise SystemExit(msg)
        (ablation_dir / "arms").mkdir(parents=True, exist_ok=True)

    specs = [ARMS[a] for a in args.arms]
    plan = []
    for i, spec in enumerate(specs):
        probe = PartiallyBayesianMLP(
            n=2, h=base.h, k=base.k, bayes_depth=spec.bayes_depth, norm=args.norm
        )
        arm_dir = ablation_dir / "arms" / f"arm{i}_{spec.name}_d{probe.latent_dim}"
        plan.append((spec, arm_dir, probe.latent_dim))

    (ablation_dir / "ablation_config.json").write_text(
        json.dumps(
            {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_config": asdict(base),
                "norm": args.norm,
                "mcmc_init": args.mcmc_init,
                "skip_full_rank": args.skip_full_rank,
                "quick": args.quick,
                "arms": [
                    {"name": s.name, "bayes_depth": s.bayes_depth, "latent_dim": d,
                     "dir": p.name, "description": s.description}
                    for s, p, d in plan
                ],
                "versions": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "pyro": pyro.__version__,
                    "numpy": np.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"ablation -> {ablation_dir.resolve()}")
    for spec, _, dim in plan:
        print(f"  {spec.name:6s} d = {dim:4d}   {spec.description}")
    print(
        f"NUTS: {base.mcmc_chains} chains x {base.mcmc_samples} draws "
        f"(warmup {base.mcmc_warmup}) per arm"
        + ("  [--quick: not a result]" if args.quick else "")
    )

    status: dict[str, str] = {}
    for spec, arm_dir, _ in plan:
        if is_complete(arm_dir) and not args.force:
            print(f"\n=== arm {spec.name}: already finished, skipping ===")
            status[arm_dir.name] = "skipped"
            continue
        arm_dir.mkdir(parents=True, exist_ok=True)
        cfg = replace(
            base,
            artifact_dir=str(arm_dir / "artifacts"),
            figure_dir=str(arm_dir / "figures"),
        )
        print(f"\n=== arm {spec.name} -> {arm_dir} ===")
        with (arm_dir / "run.log").open("w", encoding="utf-8") as handle:
            with redirect_stdout(Tee(sys.stdout, handle)):
                print(f"# arm {spec.name}: {spec.description}")
                print(f"# config: {json.dumps(asdict(cfg), indent=2)}")
                try:
                    status[arm_dir.name] = run_arm(cfg, spec, arm_dir, args=args)
                except Exception as exc:  # noqa: BLE001 - one arm must not kill the rest
                    print(f"# FAILED: {type(exc).__name__}: {exc}")
                    status[arm_dir.name] = f"{type(exc).__name__}: {exc}"

    (ablation_dir / "ablation_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    failed = {k: v for k, v in status.items() if v not in ("ok", "skipped")}
    if failed:
        print(f"\n{len(failed)} arm(s) failed: {json.dumps(failed, indent=2)}")

    rows = collect_ablation(ablation_dir)
    if not rows:
        msg = "every arm failed; nothing to summarise"
        raise SystemExit(msg)
    write_tables(ablation_dir, rows)
    make_figures(ablation_dir, rows)
    report(rows)
    print(f"\ndone: {ablation_dir.resolve()}")


if __name__ == "__main__":
    main()
