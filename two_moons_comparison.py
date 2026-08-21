import math
from dataclasses import dataclass

# from math import *
from math import pi

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
)
from pyro.optim import ClippedAdam, SGD
from torch import Tensor

from hessian import exact_hessian


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

    def set_w_map(self) -> None:
        """Set w_map values."""
        self._w_map = dict(self.laplace_guide.named_pyro_params())["loc"].data.clone()


def main() -> None:
    """Comparison of Lapalce approx, MVN-VI, MCMC."""
    pyro.clear_param_store()
    torch.set_default_dtype(torch.float64)

    x_train, y_train = two_moons(200)
    x1 = torch.linspace(-5, 5, 100)
    x2 = torch.linspace(-5, 5, 100)
    x1_grid, x2_grid = torch.meshgrid(x1, x2, indexing="ij")
    x_test = torch.stack((x1_grid.flatten(), x2_grid.flatten()), dim=-1)

    _, n = x_train.shape
    h = 20  # num. hidden units per layer
    k = 1  # num. of output unit

    # prior scale/variance
    var0 = 1 / 5e-4
    std0 = math.sqrt(var0)

    # Optimizers (SGD for Laplace)
    n_steps: int = 5000
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lrd = 0.1 ** (1 / 5000)
    sgd: SGD = SGD({"lr": lr, "momentum": momentum, "weight_decay": weight_decay})
    clipped_adam = ClippedAdam({"lr": lr, "lrd": lrd, "eps": 1e-12, "clip_norm": 10.0})

    elbo_loss = Trace_ELBO(
        num_particles=1,
    )

    # Lapalce Approx. from paper.
    model = Model(n=n, h=h, k=k)
    model.set_trainig_data(x_train=x_train, y_train=y_train)
    opt = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    losses = []
    for _ in range(n_steps):
        y_pred = model(x_train).squeeze()
        loss = F.binary_cross_entropy_with_logits(y_pred, y_train)
        losses.append(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    model.set_w_map()
    model.var0 = var0
    model.set_sigma()

    with torch.no_grad():
        model.eval()
        py_map = torch.sigmoid(model(x_test).squeeze())

    conf_map = np.maximum(py_map, 1 - py_map)

    py_exp_laplace = model.predict(x_test)
    conf_lapalce = np.maximum(py_exp_laplace, 1 - py_exp_laplace)

    # 1. Pyro Lapalce Approx Guide
    bnn = BayesianMLP(n=n, h=h, k=k)
    bnn.prior_scale = std0
    bnn.init_guides()

    bnn.feature_map.train()

    svi = SVI(
        model=bnn.model,
        guide=bnn.laplace_guide,
        optim=sgd,
        loss=elbo_loss,
    )
    losses = [svi.step(x_train, y_train) for _ in range(n_steps)]
    bnn.feature_map.eval()
    for p in bnn.feature_map.parameters():
        p.requires_grad_(requires_grad=False)

    bnn.set_w_map()
    bnn.laplace_guide.laplace_approximation(x_train, y_train)

    # 2. MCMC
    kernel = NUTS(
        bnn.model,
        init_strategy=bnn.init_to_w_map(),
    )
    mcmc = MCMC(
        kernel,
        num_samples=500,
        warmup_steps=1000,
        num_chains=5,
    )
    mcmc.run(x_train, y_train)
    print(mcmc.summary())
    samples = mcmc.get_samples()
    pred = Predictive(
        bnn.model,
        posterior_samples=samples,
        return_sites=("logits",),
    )
    logits_mcmc = pred(x_test)["logits"]
    logits_mcmc_grid = logits_mcmc.detach().clone().reshape(x1_grid.shape)


if __name__ == "__main__":
    main()
