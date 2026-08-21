"""Corner-style pair plot for MCMC posterior samples."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def pair_plot(
    W,
    labels=None,
    indices=None,
    panel_size=1.1,
    bins=30,
    point_alpha=0.15,
    hist_color="#4C72B0",
    truth=None,
):
    """Corner plot: scatter below diagonal, marginals on it, correlations above.

    Parameters
    ----------
    W : array (n_samples, n_params)
    labels : list of str, optional
    indices : list of int, optional
        Subset of parameters to plot. Defaults to all.
    panel_size : float
        Inches per panel. Total figure is panel_size * k on a side.
    truth : array (n_params,), optional
        Reference values (e.g. MAP estimate) marked in red.

    Returns
    -------
    fig, axes
    """
    W = np.asarray(W)
    if indices is None:
        indices = range(W.shape[1])
    indices = list(indices)
    X = W[:, indices]
    k = X.shape[1]

    if labels is None:
        labels = [f"$w_{{{i}}}$" for i in indices]
    else:
        labels = [labels[i] for i in indices]

    C = np.corrcoef(X, rowvar=False)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap("RdBu_r")

    fig, axes = plt.subplots(
        k, k, figsize=(panel_size * k, panel_size * k), sharex="col"
    )
    axes = np.atleast_2d(axes)

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]

            if i == j:
                ax.hist(
                    X[:, i], bins=bins, color=hist_color,
                    histtype="stepfilled", alpha=0.85, density=True,
                )
                if truth is not None:
                    ax.axvline(truth[indices[i]], color="crimson", lw=1.2)
                ax.set_yticks([])

            elif i > j:
                ax.scatter(
                    X[:, j], X[:, i], s=4, alpha=point_alpha,
                    color="#333333", edgecolors="none", rasterized=True,
                )
                if truth is not None:
                    ax.plot(truth[indices[j]], truth[indices[i]],
                            "x", color="crimson", ms=6, mew=1.5)

            else:
                r = C[i, j]
                ax.set_facecolor(cmap(norm(r)))
                ax.text(
                    0.5, 0.5, f"{r:+.2f}", ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=max(5, 9 - 0.15 * k),
                    color="white" if abs(r) > 0.55 else "black",
                )
                ax.set_xticks([])
                ax.set_yticks([])

            if i < k - 1 and i != j:
                ax.tick_params(labelbottom=False)
            if j > 0 or i == 0:
                ax.tick_params(labelleft=False)

            ax.tick_params(labelsize=max(5, 8 - 0.1 * k))
            for s in ax.spines.values():
                s.set_linewidth(0.5)

    for j in range(k):
        axes[-1, j].set_xlabel(labels[j], fontsize=max(6, 10 - 0.15 * k))
    for i in range(1, k):
        axes[i, 0].set_ylabel(labels[i], fontsize=max(6, 10 - 0.15 * k))

    fig.subplots_adjust(wspace=0.08, hspace=0.08, left=0.07,
                        bottom=0.07, right=0.98, top=0.98)
    return fig, axes
