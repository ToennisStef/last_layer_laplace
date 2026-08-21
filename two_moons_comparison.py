import functools
import json
import math
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# from math import *
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from pyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import (
    AutoLaplaceApproximation,
    AutoMultivariateNormal,
    AutoNormal,
    init_to_value,
    init_to_sample,
)
from pyro.optim import ClippedAdam, SGD
from torch import Tensor

from hessian import exact_hessian
from pairplot import pair_plot


def two_moons(n: int, sigma: float = 1e-1) -> tuple[Tensor, Tensor]:
    """Two Moon synthetic function.

    Generates samples (location and lables) for the two moons synthetic classification
    problem.

    Args:
        n (int): number of samples.
        sigma (float): standard deviation of normal to sample from.

    Returns:
        tuple[Tensor, Tensor]: Locations and Lables

    """
    theta = 2 * torch.pi * torch.rand(n)
    label = (theta > torch.pi).double()

    x = torch.stack(
        (
            torch.cos(theta) + label - 1 / 2,
            torch.sin(theta) + label / 2 - 1 / 4,
        ),
        axis=-1,
    )

    return torch.normal(x, sigma), label


class Model(nn.Module):
    def __init__(self, n: int, h: int, k: int) -> None:
        """Initialize."""
        super().__init__()
        self.hidden_dim = h
        self.input_dim = n
        self.output_dim = k
        self.feature_map = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
        )

        self.clf = nn.Linear(h, k, bias=False)
        self._w_map: Tensor | None = None
        self._sigma: Tensor | None = None
        self._var0: Tensor | float | None = None
        self._x_train: Tensor | None = None
        self._y_train: Tensor | None = None

    @property
    def w_map(self) -> Tensor | None:
        """W_map."""
        return self._w_map

    @property
    def sigma(self) -> Tensor | None:
        """Sigma."""
        return self._sigma

    @property
    def var0(self) -> Tensor | None:
        """Var0."""
        return self._var0

    @var0.setter
    def var0(self, value: Tensor | float) -> None:
        self._var0 = value

    @property
    def x_train(self) -> Tensor | None:
        """X_train."""
        return self._x_train

    @x_train.setter
    def x_train(self, value: Tensor) -> None:
        self._x_train = value

    @property
    def y_train(self) -> Tensor | None:
        """Y_train."""
        return self._y_train

    @y_train.setter
    def y_train(self, value: Tensor) -> None:
        self._y_train = value

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): _description_

        Returns:
            _type_: Logit prediction

        """
        x = self.feature_map(x)
        return self.clf(x)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Predict class lables of inputs.

        Args:
            x (Tensor): input locations.

        Returns:
            Tensor: Logit prediction from Laplace Approximation at w_map

        """
        if self.w_map is None:
            self.set_w_map()
        if self.sigma is None:
            self.set_sigma()
        phi = self.feature_map(x)  # Feature vector of x
        m = phi @ self.w_map  # MAP prediction

        # "Moderate" the MAP prediction using the variance (see MacKay 1992 "Evidence Framework ...")
        # This is an approximation of the expected sigmoid (the so-called "probit approximation")
        v = torch.diag(phi @ self.sigma @ phi.T)
        return torch.sigmoid(m / np.sqrt(1 + pi / 8 * v))

    def neg_log_prior(self, var0: Tensor | float) -> Tensor:
        """Negative log prior.

        Args:
            var0 (Tensor): prior variance of the uncertain parameters.

        Returns:
            Tensor: Negative log prior values

        """
        if var0 is None:
            msg: str = "No prior variance set."
            raise ValueError(msg)

        w = list(self.parameters())[-1]

        return 1 / 2 * w.flatten() @ (1 / var0 * torch.eye(w.numel())) @ w.flatten()

    def neg_log_likelihood(self, x: Tensor, y: Tensor) -> Tensor:
        """Negative Log Likelihood.

        Args:
            x (Tensor): (Training) location
            y (Tensor): (Training) labels

        Returns:
            Tensor: Negative log likelihood values

        """
        if x is None:
            msg: str = "No x-values passed."
            raise ValueError(msg)
        if y is None:
            msg: str = "No y-values passed."
            raise ValueError(msg)

        return F.binary_cross_entropy_with_logits(
            self.forward(x).squeeze(),
            y,
            reduction="sum",
        )

    def neg_log_posterior(
        self,
        var0: Tensor | float,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """Negative log posterior.

        Args:
            var0 (Tensor | float): Prior variance values
            x (Tensor): (Training) locations
            y (Tesnor): (Training) labels

        Returns:
            Tensor: Negative log posterior values

        """
        # Negative-log-likelihood
        nll = self.neg_log_likelihood(x=x, y=y)
        # Negative-log-prior
        nlp = self.neg_log_prior(var0)

        return nll + nlp

    def get_covariance(self, var0: Tensor | float, x: Tensor, y: Tensor) -> Tensor:
        """Inverse-Hessian of the negative-log-posterior at the MAP estimate."""
        # This is the posterior covariance

        w = list(self.parameters())[-1]
        loss = self.neg_log_posterior(var0=var0, x=x, y=y)
        lambda_ = exact_hessian(loss, [w])  # The Hessian of the negative log-posterior
        return torch.inverse(lambda_).detach().numpy()

    def set_w_map(self) -> None:
        """W_map setter."""
        w = list(self.parameters())[-1]
        self._w_map = w.view(-1).data

    def set_sigma(
        self,
        var0: Tensor | float | None = None,
        x: Tensor | None = None,
        y: Tensor | None = None,
    ) -> None:
        """Sigma setter."""
        if var0 is None:
            var0 = self.var0
        if x is None:
            x = self.x_train
        if y is None:
            y = self.y_train
        self._sigma = self.get_covariance(var0=var0, x=x, y=y)

    def set_trainig_data(self, x_train: Tensor, y_train: Tensor) -> None:
        """Trainig data setter.

        Args:
            x_train (Tensor): Training locations
            y_train (Tensor): Training labels

        """
        self.x_train = x_train
        self.y_train = y_train


class BayesianMLP(pyro.nn.PyroModule):
    """Bayesian MLP."""

    def __init__(self, n: int, h: int, k: int) -> None:
        """Initialize model.

        Args:
            n (int): input dimension
            h (int): hidden dimension
            k (int): output dimension

        """
        super().__init__()
        self.h = h
        self.n = n
        self.k = k
        self._prior_scale = None
        self.feature_map = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
        )
        self._laplace_guide = None
        self._mvn_guide = None
        self._normal_guide = None
        self._w_map: Tensor | None = None
        self._sigma: Tensor | None = None

    @property
    def prior_scale(self) -> Tensor | float | None:
        """Prior_scale."""
        return self._prior_scale

    @prior_scale.setter
    def prior_scale(self, value: Tensor | float) -> None:
        """Prior_scale setter."""
        self._prior_scale = value

    def model(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Pyro model.

        Args:
            x (Tensor): Input locations
            y (Tensor | None, optional): Observation lables. Defaults to None.

        """
        if self.prior_scale is None:
            msg: str = "No prior scale value set."
            raise ValueError(msg)
        phi = self.feature_map(x)
        w = pyro.sample(
            "w",
            dist.Normal(
                loc=torch.zeros(self.h),
                scale=self.prior_scale,
            ).to_event(1),
        )
        logits = (phi @ w.unsqueeze(-1)).squeeze(-1)
        pyro.deterministic("logits", logits)
        with pyro.plate("data", x.shape[0]):
            return pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    @property
    def laplace_guide(self) -> AutoLaplaceApproximation:
        """Laplace Approximation guide."""
        return self._laplace_guide

    @property
    def mvn_guide(self) -> AutoMultivariateNormal:
        """Laplace Approximation guide."""
        return self._mvn_guide

    @property
    def normal_guide(self) -> AutoNormal:
        """Laplace Approximation guide."""
        return self._normal_guide

    @property
    def w_map(self) -> Tensor | None:
        """W_map."""
        return self._w_map

    @property
    def sigma(self) -> Tensor | None:
        """Sigma."""
        return self._sigma

    def init_laplace_guide(self) -> None:
        """Initialize the laplace approximation guide."""
        self._laplace_guide = AutoLaplaceApproximation(
            model=self.model,
            init_loc_fn=self.init_loc_fn_guides(),
        )

    def init_mvn_guide(self) -> None:
        """Initialize the multivariate normal guide."""
        self._mvn_guide = AutoMultivariateNormal(
            model=self.model,
            init_loc_fn=self.init_loc_fn_guides(),
        )

    def init_normal_guide(self) -> None:
        """Initialize the normal guide."""
        self._normal_guide = AutoNormal(
            model=self.model,
            init_loc_fn=self.init_loc_fn_guides(),
        )

    def init_loc_fn_guides(self) -> callable:
        """Init loc function for guides."""
        return init_to_value(
            values={"w": (torch.rand(self.h) * 2 - 1) * self.h**-0.5},
        )

    def init_to_w_map(self) -> callable:
        """Init to w_map."""
        return init_to_value(values={"w": self.w_map})

    def init_guides(self) -> None:
        """Initialize the guides."""
        self.init_laplace_guide()
        self.init_mvn_guide()
        self.init_normal_guide()

    def set_w_map(self) -> None:
        """Set w_map values."""
        self._w_map = dict(self.laplace_guide.named_pyro_params())["loc"].data.clone()


@dataclass(frozen=True)
class Config:
    """Everything that determines a run, serialised next to its artifacts.

    Anything that changes the numbers belongs here, so that `config.json` plus
    the seed is enough to reproduce a run exactly.
    """

    seed: int = 7777

    # data / grid
    n_train: int = 200
    noise: float = 1e-1
    grid_lim: float = 5.0
    grid_n: int = 100

    # architecture
    h: int = 20  # hidden units per layer
    k: int = 1  # output units

    # prior: var0 == 1 / weight_decay, and dist.Normal wants a STANDARD DEVIATION
    var0: float = 1 / 5e-4

    # stage 1: MAP of w + the feature map (AutoLaplaceApproximation is AutoDelta here).
    # SGD with momentum, as in the paper's training loop - it finds a better MAP here
    # than Adam/ClippedAdam do. The VI stage below cannot use it (see `vi_*`).
    map_steps: int = 5000
    map_lr: float = 1e-3
    map_momentum: float = 0.9
    map_weight_decay: float = 5e-4

    # stage 2: the two VI guides. ClippedAdam, not SGD: SGD+momentum diverges for
    # AutoMultivariateNormal (its scale_tril goes through an exp transform, so one
    # oversized step compounds into `nan`).
    vi_steps: int = 10000
    vi_lr: float = 1e-2
    vi_particles: int = 8
    # Per-step LR decay: lr is multiplied by vi_lrd each step, so the run ends at
    # vi_lr_final_frac * vi_lr. Without it the reported final ELBO is wherever the
    # last noisy step happened to land, which hurts run-to-run reproducibility.
    vi_lr_final_frac: float = 0.1

    # stage 3: NUTS
    mcmc_samples: int = 500
    mcmc_warmup: int = 1000
    mcmc_chains: int = 5

    # predictive
    pred_samples: int = 1000
    pred_chunk: int = 2500  # grid points per predictive batch, caps peak memory

    artifact_dir: str = "artifacts"
    figure_dir: str = "figures"

    @property
    def vi_lrd(self) -> float:
        """Per-step LR decay factor for the VI optimizer."""
        return self.vi_lr_final_frac ** (1 / self.vi_steps)

    @property
    def prior_scale(self) -> float:
        """Prior standard deviation (`sqrt(var0)`), NOT the variance."""
        return math.sqrt(self.var0)


def set_seed(seed: int) -> None:
    """Seed pyro, torch, numpy and python in one call."""
    pyro.set_rng_seed(seed)


def make_grid(cfg: Config) -> tuple[Tensor, Tensor, Tensor]:
    """Return `(x1_grid, x2_grid, x_test)` for the square evaluation grid."""
    ticks = torch.linspace(-cfg.grid_lim, cfg.grid_lim, cfg.grid_n)
    x1_grid, x2_grid = torch.meshgrid(ticks, ticks, indexing="ij")
    x_test = torch.stack((x1_grid.flatten(), x2_grid.flatten()), dim=-1)
    return x1_grid, x2_grid, x_test


def predict_probs(
    model: Callable,
    x_test: Tensor,
    cfg: Config,
    *,
    guide: Callable | None = None,
    posterior_samples: dict[str, Tensor] | None = None,
) -> Tensor:
    """Draw predictive class probabilities, shape ``(n_samples, n_grid)``.

    Every method is scored with the *same* estimator - sample weights, push them
    through the network, take `sigmoid` - rather than the probit approximation
    used for Laplace in the paper. That keeps the four posteriors comparable:
    any difference in the maps is a difference in the posterior, not in how the
    logits were turned into probabilities (see doc/open-questions.md Q4).

    The grid is evaluated in chunks because ``pred_samples x n_grid`` doubles
    would otherwise be hundreds of MB.
    """
    num_samples = None if posterior_samples is not None else cfg.pred_samples
    predictive = Predictive(
        model,
        guide=guide,
        posterior_samples=posterior_samples,
        num_samples=num_samples,
        return_sites=("logits",),
        parallel=True,
    )
    chunks = []
    for start in range(0, x_test.shape[0], cfg.pred_chunk):
        batch = x_test[start : start + cfg.pred_chunk]
        with torch.no_grad():
            logits = predictive(batch, None)["logits"]
        chunks.append(logits.squeeze().sigmoid())
    return torch.cat(chunks, dim=-1)


def sample_w(
    bnn: BayesianMLP,
    *,
    guide: Callable,
    cfg: Config,
    x: Tensor,
) -> Tensor:
    """Draw ``cfg.pred_samples`` weight vectors from *guide*, shape ``(S, h)``.

    Used only for reporting posterior spread. Going through `Predictive` keeps
    this identical across guide families - `AutoNormal` has no `get_posterior`,
    so no analytic route works for all of them.
    """
    predictive = Predictive(
        bnn.model,
        guide=guide,
        num_samples=cfg.pred_samples,
        return_sites=("w",),
        parallel=True,
    )
    with torch.no_grad():
        return predictive(x[:2], None)["w"].squeeze().detach()


def binary_entropy(p: Tensor, eps: float = 1e-12) -> Tensor:
    """Entropy of a Bernoulli with success probability *p*, in nats."""
    p = p.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def decompose_uncertainty(probs: Tensor) -> dict[str, Tensor]:
    """Split predictive uncertainty into its aleatoric and epistemic parts.

    *probs* is ``(n_samples, n_grid)``: one predictive probability per posterior
    sample per grid point. With ``p_bar`` the posterior-mean probability,

        total     = H[p_bar]                    (predictive entropy)
        aleatoric = E_q[ H[p] ]                 (expected entropy)
        epistemic = total - aleatoric           (mutual information I[y; w | x])

    The epistemic term is what shrinks as data arrives; it is the quantity that
    actually distinguishes the four inference methods. Confidence, by contrast,
    is a property of ``p_bar`` alone and says nothing about posterior spread.
    """
    p_bar = probs.mean(dim=0)
    total = binary_entropy(p_bar)
    aleatoric = binary_entropy(probs).mean(dim=0)
    return {
        "p_mean": p_bar,
        "confidence": torch.maximum(p_bar, 1 - p_bar),
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": (total - aleatoric).clamp_min(0.0),
    }


def plot_panels(
    results: dict[str, dict[str, Tensor]],
    key: str,
    x1_grid: Tensor,
    x2_grid: Tensor,
    x_train: Tensor,
    y_train: Tensor,
    *,
    title: str,
    cmap: str,
    levels: np.ndarray | None,
    out_path: Path,
) -> None:
    """One contour panel per method, shared colour scale, saved as SVG."""
    shape = x1_grid.shape
    fields = {name: res[key].reshape(shape).numpy() for name, res in results.items()}
    if levels is None:
        vmax = max(float(f.max()) for f in fields.values())
        levels = np.linspace(0.0, max(vmax, 1e-6), 21)

    fig, axes = plt.subplots(1, len(fields), figsize=(4.4 * len(fields), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (name, field_) in zip(axes, fields.items(), strict=True):
        im = ax.contourf(
            x1_grid,
            x2_grid,
            field_,
            levels=levels,
            cmap=cmap,
            extend="max" if key != "confidence" else "neither",
        )
        # decision boundary of the posterior-mean probability
        ax.contour(
            x1_grid,
            x2_grid,
            results[name]["p_mean"].reshape(shape).numpy(),
            levels=[0.5],
            colors="black",
            linewidths=1.5,
        )
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=y_train,
            cmap="coolwarm",
            s=14,
            edgecolors="k",
            linewidths=0.3,
        )
        ax.set_title(f"{name}\nmean {field_.mean():.3f}", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, label=title, fraction=0.02)
    fig.suptitle(title, y=0.99)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def save_artifacts(
    cfg: Config,
    bnn: BayesianMLP,
    lap_guide: AutoLaplaceApproximation,
    mcmc: MCMC,
    tensors: dict[str, Tensor],
    results: dict[str, dict[str, Tensor]],
    losses: dict[str, list[float]],
    w_samples: dict[str, Tensor],
) -> Path:
    """Persist everything needed to reload the fitted objects without refitting.

    Layout (see `load_artifacts` for the way back):
        config.json          the Config plus library versions and a timestamp
        data.pt              training data and evaluation grid
        feature_map.pt       frozen feature-map state_dict
        param_store.pt       pyro param store: all guide parameters
        laplace_posterior.pt loc / covariance of the Laplace guide
        mcmc_samples.pt      posterior samples, and per-chain samples
        w_samples.pt         posterior draws of w per method (reconstruction-free)
        mcmc_diagnostics.json r_hat and n_eff per parameter
        predictions.pt       per-method predictive fields, to re-plot cheaply
        losses.pt            SVI loss traces
    """
    out = Path(cfg.artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": asdict(cfg),
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
    torch.save(bnn.feature_map.state_dict(), out / "feature_map.pt")
    pyro.get_param_store().save(str(out / "param_store.pt"))

    posterior = lap_guide.get_posterior()
    torch.save(
        {
            "loc": posterior.mean.detach(),
            "covariance_matrix": posterior.covariance_matrix.detach(),
            "stddev": posterior.stddev.detach(),
            "w_map": bnn.w_map,
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
    diagnostics = {}
    for site, stats in mcmc.diagnostics().items():
        diagnostics[site] = {
            k: (v.tolist() if torch.is_tensor(v) else v) for k, v in stats.items()
        }
    (out / "mcmc_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    torch.save(w_samples, out / "w_samples.pt")
    torch.save(results, out / "predictions.pt")
    torch.save(losses, out / "losses.pt")
    print(f"  wrote artifacts to {out.resolve()}")
    return out


def load_param_store(path: str | Path) -> None:
    """Load a saved pyro param store, working around torch >= 2.6 defaults.

    `ParamStore.save` pickles constraint objects, which `torch.load`'s
    `weights_only=True` default refuses. Allow-listing the classes one by one is
    whack-a-mole (torch *and* pyro constraint variants), so this trusts the file -
    which is fine for artifacts this script wrote itself.

    **Do not rely on this to restore guides.** Reloading the store repopulates
    `AutoNormal` correctly but silently leaves `AutoMultivariateNormal.loc` at its
    initial value - see doc/gotchas/param-store-reload.md. Use `w_samples.pt`,
    which holds posterior draws for every method and needs no reconstruction.
    """
    original = torch.load
    torch.load = functools.partial(original, weights_only=False)
    try:
        pyro.get_param_store().load(str(path))
    finally:
        torch.load = original


def load_artifacts(artifact_dir: str = "artifacts") -> dict:
    """Reload a finished run without redoing any inference.

    Returns the raw tensors/metadata. To get *working guides* back, rebuild the
    same `BayesianMLP`, load `feature_map.pt` into `bnn.feature_map`, call
    `bnn.init_guides()`, then `pyro.get_param_store().load(param_store.pt)` -
    the parameter names line up because the guides are constructed identically.
    The Laplace posterior is stored directly as loc/covariance, so it needs no
    reconstruction at all.
    """
    out = Path(artifact_dir)
    return {
        "config": json.loads((out / "config.json").read_text(encoding="utf-8")),
        "data": torch.load(out / "data.pt", weights_only=False),
        "feature_map": torch.load(out / "feature_map.pt", weights_only=False),
        "laplace_posterior": torch.load(
            out / "laplace_posterior.pt", weights_only=False
        ),
        "mcmc": torch.load(out / "mcmc_samples.pt", weights_only=False),
        "diagnostics": json.loads(
            (out / "mcmc_diagnostics.json").read_text(encoding="utf-8")
        ),
        "w_samples": torch.load(out / "w_samples.pt", weights_only=False),
        "predictions": torch.load(out / "predictions.pt", weights_only=False),
        "losses": torch.load(out / "losses.pt", weights_only=False),
    }


def main(cfg: Config | None = None) -> None:
    """Four-way comparison over one frozen feature map: Laplace, VI x2, NUTS.

    Every posterior is fitted over the *same* deterministic feature map, so any
    difference between the maps is attributable to the inference method alone.
    """
    cfg = cfg or Config()
    torch.set_default_dtype(torch.float64)
    pyro.clear_param_store()
    set_seed(cfg.seed)

    figure_dir = Path(cfg.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = two_moons(cfg.n_train, sigma=cfg.noise)
    x1_grid, x2_grid, x_test = make_grid(cfg)
    _, n = x_train.shape

    print(f"prior_scale = {cfg.prior_scale:.3f}  (var0 = {cfg.var0:.1f})")

    bnn = BayesianMLP(n=n, h=cfg.h, k=cfg.k)
    bnn.prior_scale = cfg.prior_scale
    bnn.init_guides()

    # --- stage 1: MAP of w and the feature map ------------------------------
    # AutoLaplaceApproximation behaves as AutoDelta here, so the ELBO is just
    # -log p(w, D). SGD with momentum, matching the paper's training loop: it
    # reaches a better MAP here than Adam/ClippedAdam. Stage 2 has to switch
    # optimizers because SGD+momentum diverges for the VI guides.
    bnn.feature_map.train()
    svi_map = SVI(
        model=bnn.model,
        guide=bnn.laplace_guide,
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

    # --- freeze the features -------------------------------------------------
    # eval() puts BatchNorm on its running statistics; requires_grad_(False) stops
    # any later SVI from touching the feature map. Both are needed - see
    # doc/gotchas/batchnorm-frozen-features.md.
    bnn.feature_map.eval()
    for p in bnn.feature_map.parameters():
        p.requires_grad_(requires_grad=False)
    bnn.set_w_map()

    with torch.no_grad():
        acc = (
            (((bnn.feature_map(x_train) @ bnn.w_map) > 0).double() == y_train)
            .double()
            .mean()
        )
    print(f"  train accuracy {acc:.3f}  (1.000 => separable => collapsed Hessian)")

    # --- stage 2: the three Gaussian posteriors -----------------------------
    lap_guide = bnn.laplace_guide.laplace_approximation(x_train, y_train)

    def fit_vi(guide: Callable, name: str) -> list[float]:
        svi = SVI(
            model=bnn.model,
            guide=guide,
            optim=ClippedAdam(
                {"lr": cfg.vi_lr, "lrd": cfg.vi_lrd, "clip_norm": 10.0}
            ),
            loss=Trace_ELBO(
                num_particles=cfg.vi_particles, vectorize_particles=True
            ),
        )
        print(f"stage 2: {name}, {cfg.vi_steps} steps")
        losses = [svi.step(x_train, y_train) for _ in range(cfg.vi_steps)]
        print(f"  final ELBO loss = {losses[-1]:.3f}")
        return losses

    mf_losses = fit_vi(bnn.normal_guide, "AutoNormal (mean-field)")
    fr_losses = fit_vi(bnn.mvn_guide, "AutoMultivariateNormal (full-rank)")

    # --- stage 3: NUTS, the reference ---------------------------------------
    # init_to_sample() rather than the MAP: seeding at w_map biases the chains
    # towards reporting a narrow posterior simply because they never leave the
    # mode (doc/open-questions.md Q5).
    print(f"stage 3: NUTS, {cfg.mcmc_chains} chains x {cfg.mcmc_samples} samples")
    kernel = NUTS(bnn.model, init_strategy=init_to_sample())
    mcmc = MCMC(
        kernel,
        num_samples=cfg.mcmc_samples,
        warmup_steps=cfg.mcmc_warmup,
        num_chains=cfg.mcmc_chains,
    )
    try:
        mcmc.run(x_train, y_train)
    except RuntimeError as exc:
        # On Windows, num_chains > 1 spawns processes, which requires the caller
        # to sit behind `if __name__ == "__main__":`. Importing main() into a
        # notebook or an unguarded script trips this. Fall back to one long chain
        # rather than dying - but say so loudly, because r_hat across chains is
        # the whole point of running several.
        if "bootstrapping phase" not in str(exc):
            raise
        total = cfg.mcmc_samples * cfg.mcmc_chains
        print(
            "  WARNING: multi-chain sampling needs an `if __name__ == \"__main__\"` "
            f"guard in the entry script.\n  Falling back to 1 chain x {total} "
            "samples - no cross-chain r_hat available."
        )
        kernel = NUTS(bnn.model, init_strategy=init_to_sample())
        mcmc = MCMC(
            kernel,
            num_samples=total,
            warmup_steps=cfg.mcmc_warmup,
            num_chains=1,
        )
        mcmc.run(x_train, y_train)
    mcmc.summary()
    w_nuts = mcmc.get_samples()["w"]

    # --- predictive comparison ----------------------------------------------
    print("predictive:")
    results: dict[str, dict[str, Tensor]] = {}
    for name, kwargs in (
        ("Laplace", {"guide": lap_guide}),
        ("VI mean-field", {"guide": bnn.normal_guide}),
        ("VI full-rank", {"guide": bnn.mvn_guide}),
        ("NUTS (reference)", {"posterior_samples": mcmc.get_samples()}),
    ):
        probs = predict_probs(bnn.model, x_test, cfg, **kwargs)
        results[name] = decompose_uncertainty(probs)
        print(f"  {name:20s} done ({probs.shape[0]} posterior draws)")

    print(f"\n{'method':22s} {'std mean':>10s} {'conf':>8s} {'epistemic':>10s}")
    # Posterior spread is measured the same way for every method - the std of
    # sampled w. AutoNormal has no `get_posterior()`, and mixing an analytic
    # std with a sampled one would not be comparable anyway.
    stds = {
        "Laplace": sample_w(bnn, guide=lap_guide, cfg=cfg, x=x_train),
        "VI mean-field": sample_w(bnn, guide=bnn.normal_guide, cfg=cfg, x=x_train),
        "VI full-rank": sample_w(bnn, guide=bnn.mvn_guide, cfg=cfg, x=x_train),
        "NUTS (reference)": w_nuts,
    }
    w_samples = stds
    stds = {name: w.std(0).mean().item() for name, w in w_samples.items()}
    for name, res in results.items():
        print(
            f"{name:22s} {stds[name]:10.3f} {res['confidence'].mean():8.3f} "
            f"{res['epistemic'].mean():10.4f}"
        )
    print(f"(prior_scale for comparison: {cfg.prior_scale:.3f})")

    # --- figures -------------------------------------------------------------
    print("figures:")
    plot_panels(
        results,
        "confidence",
        x1_grid,
        x2_grid,
        x_train,
        y_train,
        title="confidence  max(p, 1-p)",
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
        title="epistemic uncertainty  I[y; w | x]  (nats)",
        cmap="magma",
        levels=None,
        out_path=figure_dir / "epistemic_comparison.svg",
    )

    fig, _ = pair_plot(
        w_nuts.numpy(),
        truth=bnn.w_map.numpy(),
        panel_size=0.6,
        point_alpha=0.08,
    )
    fig.suptitle(
        f"NUTS posterior over w  ({w_nuts.shape[0]} draws, red = MAP)", y=1.0
    )
    pair_path = figure_dir / "mcmc_pairplot.svg"
    fig.savefig(pair_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pair_path}")

    # --- persist -------------------------------------------------------------
    save_artifacts(
        cfg,
        bnn,
        lap_guide,
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
        w_samples={k: v.detach() for k, v in w_samples.items()},
    )


if __name__ == "__main__":
    main()
