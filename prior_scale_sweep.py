"""Prior-scale sensitivity sweep over the four-way comparison.

Answers Q8 (doc/open-questions.md): the typical set of an isotropic prior sits at
radius ``prior_scale * sqrt(d)``, far outside the mode (M6). Shrinking the prior
should pull that shell toward ``w_map`` and close the Laplace-vs-NUTS gap. This
script runs `two_moons_comparison.main` once per prior scale and reports how every
summary quantity moves.

`two_moons_comparison` is *imported*, never copied: the sweep must measure exactly
the pipeline that script defines, and a forked copy would drift from it. What varies
between runs is `Config.var0` and the output directories. `--couple-weight-decay`
additionally derives the SGD weight decay from the prior as decision 0003 does
(`wd = 1/var0`).

Note what the prior touches either way: stage 1's ELBO *is* `-log p(w, D)`, so the
prior enters the MAP fit and hence the feature map regardless of `map_weight_decay`.
No setting of this sweep isolates the prior's effect on the posterior from its
effect on the fit - `train_accuracy` is reported per run so the two can be told
apart after the fact. Only the training data itself is invariant across runs (fixed
seed, drawn before any fitting); `data_fingerprint` checks that.

What this sweep is, what it collects and what it costs:
**doc/setup/prior-scale-sweep.md**. The experiment it repeats:
**doc/setup/two-moons-comparison.md**.

Layout, one timestamped folder per sweep so nothing is ever overwritten
(doc/decisions/0011-sweep-layout.md):

    sweeps/<stamp>_prior_scale[_<tag>]/
        sweep_config.json     base Config, var0 grid, flags, library versions
        sweep_status.json     per-run outcome: ok / skipped / the exception
        sweep_summary.json    everything below, machine-readable
        sweep_summary.csv     one row per (run, method)
        sweep_runs.csv        one row per run: diagnostics + gaps vs NUTS
        figures/              cross-run figures
        runs/run<i>_var0_<v>_ps_<s>/
            run.log           that run's full stdout
            artifacts/        exactly the doc/decisions/0010 layout
            figures/          that run's own three figures

Usage:
    python prior_scale_sweep.py                        # default grid, full settings
    python prior_scale_sweep.py --var0 1 100 10000     # explicit grid
    python prior_scale_sweep.py --couple-weight-decay  # wd = 1/var0 (decision 0003)
    python prior_scale_sweep.py --quick --tag smoke    # fast plumbing check
    python prior_scale_sweep.py --resume sweeps/<dir>  # continue, skip finished runs
    python prior_scale_sweep.py --plot-only sweeps/<dir>   # re-plot, no inference
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before two_moons_comparison pulls in pyplot

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch

from two_moons_comparison import BayesianMLP, Config, load_artifacts
from two_moons_comparison import main as run_comparison

# --- constants ---------------------------------------------------------------

METHODS: tuple[str, ...] = (
    "Laplace",
    "VI mean-field",
    "VI full-rank",
    "NUTS (reference)",
)
REFERENCE = "NUTS (reference)"

# BAM corporate palette, categorical order (AI-OS references/bam-color-scheme.md)
METHOD_COLOR = {
    "Laplace": "#D4002A",  # BAM red
    "VI mean-field": "#009FE3",  # azure
    "VI full-rank": "#0083A4",  # teal
    "NUTS (reference)": "#000000",  # black - it is the reference
}
METHOD_MARKER = {
    "Laplace": "o",
    "VI mean-field": "s",
    "VI full-rank": "^",
    "NUTS (reference)": "D",
}

# var0 is the prior *variance*; prior_scale = sqrt(var0) is the std (I1).
# Four decades of scale around the current setting of var0 = 2000. This range is
# only safe because the weight decay is decoupled by default: under
# `--couple-weight-decay`, var0 = 1e-2 would mean wd = 100 and a dead feature map,
# so narrow the grid to var0 >= 1 in that mode and watch `train_accuracy`.
DEFAULT_VAR0: tuple[float, ...] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 2e3, 1e4)

# Radii splitting the evaluation grid into "where the data is" and "far field".
# The two moons live inside ||x|| ~ 1.5; the grid corner is at ||x|| = 5*sqrt(2).
NEAR_R = 1.5
FAR_R = 4.0

SWEEP_ROOT = "sweeps"


# --- run bookkeeping ---------------------------------------------------------


def fmt_sci(x: float) -> str:
    """Filesystem-safe scientific notation, e.g. ``2.0e03`` / ``1.0e-02``."""
    return f"{x:.1e}".replace("+", "")


def run_dir_name(index: int, var0: float) -> str:
    """Directory name carrying both the variance and the scale actually used."""
    return f"run{index:02d}_var0_{fmt_sci(var0)}_ps_{fmt_sci(math.sqrt(var0))}"


def sweep_dir_name(tag: str | None, stamp: str) -> str:
    """Timestamped sweep folder name; the timestamp is what prevents collisions."""
    return f"{stamp}_prior_scale" + (f"_{tag}" if tag else "")


def build_configs(
    base: Config,
    var0_grid: tuple[float, ...],
    sweep_dir: Path,
    *,
    couple_weight_decay: bool,
) -> list[tuple[int, float, Path, Config]]:
    """One `Config` per prior scale, each pointed at its own output folders.

    Only `var0` changes by default; `map_weight_decay` stays at its base value.
    That does **not** hold the network fixed: stage 1 maximises the ELBO of
    `bnn.model`, which contains the prior, so a different `var0` trains a
    different feature map and `w_map` even with the weight decay pinned. What
    decoupling buys is that the prior is applied *once*, through the model, rather
    than twice - for `w`, an L2 weight decay is a second copy of the same Gaussian
    prior.

    `--couple-weight-decay` sets `map_weight_decay = 1/var0`, the consistency
    argument of decision 0003: the duplicated prior at least agrees with itself.
    Its cost is that stage 1 applies that decay to the *whole* network, so a tight
    prior degrades the feature map outright (`var0 = 1e-2` means wd = 100).

    Either way `train_accuracy` is recorded, because in neither mode is the fit
    quality constant across the sweep.
    """
    out = []
    for i, var0 in enumerate(var0_grid):
        run_dir = sweep_dir / "runs" / run_dir_name(i, var0)
        overrides = {
            "var0": float(var0),
            "artifact_dir": str(run_dir / "artifacts"),
            "figure_dir": str(run_dir / "figures"),
        }
        if couple_weight_decay:
            overrides["map_weight_decay"] = 1.0 / float(var0)
        out.append((i, float(var0), run_dir, replace(base, **overrides)))
    return out


class Tee:
    """Duplicate a stream to a file, so each run gets its own `run.log`."""

    def __init__(self, stream, handle) -> None:
        """Wrap *stream*, mirroring everything written into *handle*."""
        self._stream = stream
        self._handle = handle

    def write(self, data: str) -> int:
        """Write to both targets."""
        self._handle.write(data)
        self._stream.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush both targets."""
        self._handle.flush()
        self._stream.flush()


def is_complete(run_dir: Path) -> bool:
    """A run counts as finished once its predictive fields are on disk."""
    return (run_dir / "artifacts" / "predictions.pt").exists()


def execute_run(cfg: Config, run_dir: Path, *, var0: float, label: str) -> str:
    """Run the comparison for one prior scale. Returns ``"ok"`` or the error."""
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label}  var0 = {var0:.4g}  prior_scale = {cfg.prior_scale:.4g} ===")
    with (run_dir / "run.log").open("w", encoding="utf-8") as handle:
        tee = Tee(sys.stdout, handle)
        with redirect_stdout(tee):
            print(f"# {label}: var0 = {var0}, prior_scale = {cfg.prior_scale}")
            print(f"# config: {json.dumps(asdict(cfg), indent=2)}")
            try:
                pyro.clear_param_store()
                run_comparison(cfg)
            except Exception as exc:  # noqa: BLE001 - keep the sweep alive
                print(f"# FAILED: {type(exc).__name__}: {exc}")
                return f"{type(exc).__name__}: {exc}"
    return "ok"


# --- collection --------------------------------------------------------------


def region_masks(x_test: torch.Tensor) -> dict[str, torch.Tensor]:
    """Near-data / far-field / whole-grid masks over the evaluation grid."""
    r = x_test.norm(dim=-1)
    return {
        "all": torch.ones_like(r, dtype=torch.bool),
        "near": r <= NEAR_R,
        "far": r >= FAR_R,
    }


def train_accuracy(art: dict) -> float:
    """Rebuild the frozen feature map and score `w_map` on the training data.

    The prior is part of stage 1's objective, so every prior scale trains a
    different network and the fit quality has to be reported next to the
    calibration numbers. `save_artifacts` stores the accuracy nowhere, but
    everything needed to recompute it is there.
    """
    cfg = art["config"]["config"]
    x_train, y_train = art["data"]["x_train"], art["data"]["y_train"]
    bnn = BayesianMLP(n=x_train.shape[-1], h=int(cfg["h"]), k=int(cfg["k"]))
    bnn.feature_map.double()
    bnn.feature_map.load_state_dict(art["feature_map"])
    bnn.feature_map.eval()
    w_map = torch.as_tensor(art["laplace_posterior"]["w_map"]).double()
    with torch.no_grad():
        logits = bnn.feature_map(x_train.double()) @ w_map
    return float(((logits > 0).double() == y_train.double()).double().mean())


def collect_run(run_dir: Path) -> dict:
    """Reduce one finished run's artifacts to scalars. No inference, no refit."""
    art = load_artifacts(str(run_dir / "artifacts"))
    cfg = art["config"]["config"]
    preds: dict[str, dict[str, torch.Tensor]] = art["predictions"]
    masks = region_masks(art["data"]["x_test"])

    methods: dict[str, dict] = {}
    for name, res in preds.items():
        w = art["w_samples"][name]
        row = {
            "w_std_mean": float(w.std(0).mean()),
            "w_norm_median": float(w.norm(dim=-1).median()),
        }
        for key in ("confidence", "epistemic", "aleatoric", "total"):
            for region, mask in masks.items():
                suffix = "" if region == "all" else f"_{region}"
                row[f"{key}{suffix}"] = float(res[key][mask].mean())
        methods[name] = row

    # Q8's actual claim: does the Laplace/NUTS disagreement shrink with the prior?
    gaps: dict[str, float] = {}
    if REFERENCE in preds:
        ref = preds[REFERENCE]
        for name, res in preds.items():
            if name == REFERENCE:
                continue
            key = name.replace(" ", "_")
            gaps[f"conf_gap_{key}"] = float(
                res["confidence"].mean() - ref["confidence"].mean()
            )
            gaps[f"conf_gap_far_{key}"] = float(
                res["confidence"][masks["far"]].mean()
                - ref["confidence"][masks["far"]].mean()
            )
            gaps[f"p_mean_l1_{key}"] = float((res["p_mean"] - ref["p_mean"]).abs().mean())

    r_hat = art["diagnostics"].get("w", {}).get("r_hat", [])
    n_eff = art["diagnostics"].get("w", {}).get("n_eff", [])
    losses = art["losses"]

    return {
        "run_dir": run_dir.name,
        "var0": float(cfg["var0"]),
        "prior_scale": math.sqrt(float(cfg["var0"])),
        "map_weight_decay": float(cfg["map_weight_decay"]),
        "seed": int(cfg["seed"]),
        # the same seed must mean the same data; a changed fingerprint means it did not
        "data_fingerprint": round(float(art["data"]["x_train"].sum()), 6),
        "w_map_norm": float(torch.as_tensor(art["laplace_posterior"]["w_map"]).norm()),
        "train_accuracy": train_accuracy(art),
        "typical_prior_radius": math.sqrt(float(cfg["var0"]) * int(cfg["h"])),
        "r_hat_max": max(r_hat) if r_hat else None,
        "n_eff_min": min(n_eff) if n_eff else None,
        "map_loss_final": losses["map"][-1] if losses.get("map") else None,
        "elbo_mean_field_final": (
            losses["mean_field"][-1] if losses.get("mean_field") else None
        ),
        "elbo_full_rank_final": (
            losses["full_rank"][-1] if losses.get("full_rank") else None
        ),
        "methods": methods,
        "gaps": gaps,
    }


def collect_sweep(sweep_dir: Path) -> list[dict]:
    """Collect every finished run in a sweep folder, ordered by run index."""
    runs_root = sweep_dir / "runs"
    rows = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if not is_complete(run_dir):
            print(f"  skipping incomplete run {run_dir.name}")
            continue
        rows.append(collect_run(run_dir))
    return rows


def write_tables(sweep_dir: Path, rows: list[dict]) -> None:
    """Long-format per-method CSV, wide per-run CSV, and the full JSON."""
    (sweep_dir / "sweep_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    method_fields = sorted({k for r in rows for m in r["methods"].values() for k in m})
    with (sweep_dir / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "var0", "prior_scale", "method", *method_fields])
        for row in rows:
            for name in METHODS:
                if name not in row["methods"]:
                    continue
                metrics = row["methods"][name]
                writer.writerow(
                    [
                        row["run_dir"],
                        row["var0"],
                        row["prior_scale"],
                        name,
                        *(metrics.get(k, "") for k in method_fields),
                    ]
                )

    scalar_keys = [k for k, v in rows[0].items() if not isinstance(v, dict)]
    gap_keys = sorted({k for r in rows for k in r["gaps"]})
    with (sweep_dir / "sweep_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*scalar_keys, *gap_keys])
        for row in rows:
            writer.writerow(
                [
                    *(row[k] for k in scalar_keys),
                    *(row["gaps"].get(k, "") for k in gap_keys),
                ]
            )
    print(f"  wrote tables to {sweep_dir.resolve()}")


# --- cross-run figures -------------------------------------------------------


def _style_log_x(ax, xs: list[float]) -> None:
    """Log x-axis labelled with the actual swept prior scales."""
    ax.set_xscale("log")
    ax.set_xlabel("prior scale  $s=\\sqrt{\\mathrm{var}_0}$")
    ax.grid(visible=True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.3g}" for x in xs], fontsize=8, rotation=45)
    ax.minorticks_off()


def plot_metric_vs_prior(
    rows: list[dict],
    key: str,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    hline: float | None = None,
) -> None:
    """One panel per region (whole grid / near data / far field)."""
    xs = [r["prior_scale"] for r in rows]
    regions = [
        ("all", "whole grid"),
        ("near", f"near data  $\\|x\\| \\leq {NEAR_R}$"),
        ("far", f"far field  $\\|x\\| \\geq {FAR_R}$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, (region, label) in zip(axes, regions, strict=True):
        suffix = "" if region == "all" else f"_{region}"
        for name in METHODS:
            ys = [r["methods"].get(name, {}).get(f"{key}{suffix}") for r in rows]
            if any(y is None for y in ys):
                continue
            ax.plot(
                xs,
                ys,
                marker=METHOD_MARKER[name],
                color=METHOD_COLOR[name],
                label=name,
                linewidth=1.6,
                markersize=5,
            )
        if hline is not None:
            ax.axhline(hline, color="grey", linestyle=":", linewidth=1.0)
        _style_log_x(ax, xs)
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_posterior_width(rows: list[dict], out_path: Path) -> None:
    """Posterior width and ``||w||`` against the prior, with prior references.

    The diagonal ``y = s`` is the prior's own std: a method sitting on it has a
    posterior no narrower than the prior, i.e. the data is not constraining the
    weight magnitude at all (M6). ``||w_map||`` is *not* constant along the sweep -
    the prior is part of the stage-1 objective, so the mode moves with it.
    """
    xs = [r["prior_scale"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    for name in METHODS:
        axes[0].plot(
            xs,
            [r["methods"][name]["w_std_mean"] for r in rows],
            marker=METHOD_MARKER[name],
            color=METHOD_COLOR[name],
            label=name,
            linewidth=1.6,
            markersize=5,
        )
        axes[1].plot(
            xs,
            [r["methods"][name]["w_norm_median"] for r in rows],
            marker=METHOD_MARKER[name],
            color=METHOD_COLOR[name],
            label=name,
            linewidth=1.6,
            markersize=5,
        )
    axes[0].plot(xs, xs, color="grey", linestyle="--", linewidth=1.0, label="prior std  $s$")
    axes[1].plot(
        xs,
        [r["typical_prior_radius"] for r in rows],
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label="prior typical  $s\\sqrt{d}$",
    )
    axes[1].plot(
        xs,
        [r["w_map_norm"] for r in rows],
        color="#6E0019",
        linestyle="-.",
        linewidth=1.2,
        label="$\\|w_{MAP}\\|$",
    )

    for ax, ylab, title in (
        (axes[0], "mean over coordinates of  std$(w)$", "posterior width"),
        (axes[1], "median  $\\|w\\|$", "where the mass sits"),
    ):
        _style_log_x(ax, xs)
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Posterior spread vs. prior scale")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_gaps(rows: list[dict], out_path: Path) -> None:
    """The Q8 verdict: does disagreement with NUTS shrink as the prior tightens?"""
    xs = [r["prior_scale"] for r in rows]
    others = [m for m in METHODS if m != REFERENCE]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for name in others:
        key = name.replace(" ", "_")
        axes[0].plot(
            xs,
            [r["gaps"].get(f"conf_gap_{key}") for r in rows],
            marker=METHOD_MARKER[name],
            color=METHOD_COLOR[name],
            label=name,
            linewidth=1.6,
            markersize=5,
        )
        axes[1].plot(
            xs,
            [r["gaps"].get(f"p_mean_l1_{key}") for r in rows],
            marker=METHOD_MARKER[name],
            color=METHOD_COLOR[name],
            label=name,
            linewidth=1.6,
            markersize=5,
        )
    axes[0].axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("mean confidence $-$ NUTS mean confidence")
    axes[0].set_title("signed confidence gap", fontsize=10)
    axes[1].set_ylabel("grid mean  $|\\bar{p} - \\bar{p}_{\\mathrm{NUTS}}|$")
    axes[1].set_title("predictive disagreement", fontsize=10)
    for ax in axes:
        _style_log_x(ax, xs)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Disagreement with the NUTS reference vs. prior scale  (Q8)")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_diagnostics(rows: list[dict], out_path: Path) -> None:
    """NUTS convergence per prior scale - a tight prior may sample differently."""
    xs = [r["prior_scale"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    have_rhat = [(x, r["r_hat_max"]) for x, r in zip(xs, rows) if r["r_hat_max"]]
    have_neff = [(x, r["n_eff_min"]) for x, r in zip(xs, rows) if r["n_eff_min"]]
    if have_rhat:
        axes[0].plot(
            [x for x, _ in have_rhat],
            [v for _, v in have_rhat],
            marker="o",
            color="#D4002A",
            linewidth=1.6,
            markersize=5,
        )
    axes[0].axhline(1.01, color="grey", linestyle=":", linewidth=1.0, label="$\\hat{r}=1.01$")
    axes[0].set_ylabel("max $\\hat{r}$ over the 20 coordinates")
    axes[0].set_title("NUTS convergence", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    if have_neff:
        axes[1].plot(
            [x for x, _ in have_neff],
            [v for _, v in have_neff],
            marker="s",
            color="#009FE3",
            linewidth=1.6,
            markersize=5,
        )
    axes[1].set_ylabel("min $n_{\\mathrm{eff}}$")
    axes[1].set_title("effective sample size", fontsize=10)
    # Not an invariant: the prior sits in the stage-1 ELBO, so every point of the
    # sweep trains a different feature map. Where this line drops, part of the
    # calibration change is a worse fit rather than a wider posterior.
    axes[2].plot(
        xs,
        [r["train_accuracy"] for r in rows],
        marker="^",
        color="#00485C",
        linewidth=1.6,
        markersize=5,
    )
    axes[2].axhline(1.0, color="grey", linestyle=":", linewidth=1.0, label="separable")
    axes[2].set_ylabel("train accuracy at $w_{MAP}$")
    axes[2].set_title("did the network still fit?", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
        _style_log_x(ax, xs)
    fig.suptitle("Diagnostics vs. prior scale")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_confidence_grid(sweep_dir: Path, rows: list[dict], out_path: Path) -> None:
    """Rows = prior scale, columns = method. The poster-candidate figure."""
    runs_root = sweep_dir / "runs"
    loaded = [
        (row, load_artifacts(str(runs_root / row["run_dir"] / "artifacts")))
        for row in rows
    ]
    if not loaded:
        return

    shape = loaded[0][1]["data"]["x1_grid"].shape
    levels = np.arange(0.5, 1.001, 0.025)
    n_rows, n_cols = len(loaded), len(METHODS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), squeeze=False
    )
    im = None
    for i, (row, art) in enumerate(loaded):
        x1, x2 = art["data"]["x1_grid"], art["data"]["x2_grid"]
        x_train, y_train = art["data"]["x_train"], art["data"]["y_train"]
        for j, name in enumerate(METHODS):
            ax = axes[i][j]
            res = art["predictions"].get(name)
            if res is None:
                ax.axis("off")
                continue
            im = ax.contourf(
                x1,
                x2,
                res["confidence"].reshape(shape).numpy(),
                levels=levels,
                cmap="Blues",
                extend="neither",
            )
            ax.contour(
                x1,
                x2,
                res["p_mean"].reshape(shape).numpy(),
                levels=[0.5],
                colors="black",
                linewidths=1.2,
            )
            ax.scatter(
                x_train[:, 0],
                x_train[:, 1],
                c=y_train,
                cmap="coolwarm",
                s=4,
                edgecolors="k",
                linewidths=0.15,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"mean {float(res['confidence'].mean()):.3f}", fontsize=8)
            if i == 0:
                ax.set_xlabel(name, fontsize=10)
                ax.xaxis.set_label_position("top")
            if j == 0:
                ax.set_ylabel(f"$s$ = {row['prior_scale']:.3g}", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes, label="confidence  max(p, 1-p)", fraction=0.015)
    fig.suptitle(
        "Confidence vs. prior scale (rows) and inference method (columns)", y=0.995
    )
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_figures(sweep_dir: Path, rows: list[dict]) -> None:
    """All cross-run figures. Reads artifacts only - never refits anything."""
    fig_dir = sweep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("sweep figures:")
    plot_metric_vs_prior(
        rows,
        "confidence",
        title="Mean confidence vs. prior scale",
        ylabel="mean  max(p, 1-p)",
        out_path=fig_dir / "confidence_vs_prior_scale.svg",
        hline=0.5,
    )
    plot_metric_vs_prior(
        rows,
        "epistemic",
        title="Mean epistemic uncertainty  $I[y; w \\mid x]$  vs. prior scale",
        ylabel="mean epistemic (nats)",
        out_path=fig_dir / "epistemic_vs_prior_scale.svg",
    )
    plot_posterior_width(rows, fig_dir / "posterior_width_vs_prior_scale.svg")
    plot_gaps(rows, fig_dir / "gap_vs_nuts_vs_prior_scale.svg")
    plot_diagnostics(rows, fig_dir / "mcmc_diagnostics_vs_prior_scale.svg")
    plot_confidence_grid(sweep_dir, rows, fig_dir / "confidence_grid.svg")


def report(rows: list[dict]) -> None:
    """Print the one table worth reading straight after a sweep."""
    print(
        f"\n{'s':>8s} {'wd':>9s} {'acc':>5s} {'method':22s} {'conf':>7s} "
        f"{'conf_far':>9s} {'epist':>8s} {'w_std':>8s} {'|dp| vs NUTS':>13s}"
    )
    for row in rows:
        for name in METHODS:
            m = row["methods"].get(name)
            if m is None:
                continue
            key = name.replace(" ", "_")
            l1 = row["gaps"].get(f"p_mean_l1_{key}")
            l1_str = "-" if l1 is None else f"{l1:.4f}"
            print(
                f"{row['prior_scale']:8.3g} {row['map_weight_decay']:9.2g} "
                f"{row['train_accuracy']:5.3f} {name:22s} {m['confidence']:7.3f} "
                f"{m['confidence_far']:9.3f} {m['epistemic']:8.4f} "
                f"{m['w_std_mean']:8.3f} {l1_str:>13s}"
            )
    fingerprints = {r["data_fingerprint"] for r in rows}
    if len(fingerprints) > 1:
        print(
            "\nWARNING: training data differs between runs "
            f"(fingerprints {sorted(fingerprints)}) - the comparison is unpaired."
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
        "--var0",
        type=float,
        nargs="+",
        default=None,
        help=f"prior variances to sweep (default: {DEFAULT_VAR0})",
    )
    p.add_argument("--tag", default=None, help="suffix for the sweep folder name")
    p.add_argument(
        "--quick",
        action="store_true",
        help="tiny steps/grid - a plumbing check, not a result",
    )
    p.add_argument(
        "--couple-weight-decay",
        action="store_true",
        help="derive the weight decay from the prior, map_weight_decay = 1/var0 "
        "(decision 0003); self-consistent, but retrains the network per run",
    )
    p.add_argument("--seed", type=int, default=None, help="override Config.seed")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue an existing sweep folder, skipping finished runs",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="allow re-running into a run folder that already holds results",
    )
    p.add_argument(
        "--plot-only",
        type=Path,
        default=None,
        help="re-collect and re-plot an existing sweep, no inference",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Sweep `Config.var0`, one full four-way comparison per value."""
    args = parse_args(argv)

    if args.plot_only is not None:
        sweep_dir = args.plot_only
        rows = collect_sweep(sweep_dir)
        if not rows:
            msg = f"no finished runs under {sweep_dir}"
            raise SystemExit(msg)
        write_tables(sweep_dir, rows)
        make_figures(sweep_dir, rows)
        report(rows)
        return

    base = Config()
    overrides: dict = dict(QUICK) if args.quick else {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        base = replace(base, **overrides)

    var0_grid = tuple(args.var0) if args.var0 else DEFAULT_VAR0

    if args.resume is not None:
        sweep_dir = args.resume
        if not sweep_dir.exists():
            msg = f"--resume: {sweep_dir} does not exist"
            raise SystemExit(msg)
        stored = json.loads((sweep_dir / "sweep_config.json").read_text(encoding="utf-8"))
        base = Config(**stored["base_config"])
        var0_grid = tuple(stored["var0_grid"])
        args.couple_weight_decay = stored["couple_weight_decay"]
        print(f"resuming {sweep_dir} with its stored config and grid")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
        sweep_dir = Path(SWEEP_ROOT) / sweep_dir_name(args.tag, stamp)
        if sweep_dir.exists() and not args.force:
            msg = f"{sweep_dir} already exists; use --resume or --force"
            raise SystemExit(msg)
        (sweep_dir / "runs").mkdir(parents=True, exist_ok=True)

    configs = build_configs(
        base, var0_grid, sweep_dir, couple_weight_decay=args.couple_weight_decay
    )

    (sweep_dir / "sweep_config.json").write_text(
        json.dumps(
            {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_config": asdict(base),
                "var0_grid": list(var0_grid),
                "prior_scale_grid": [math.sqrt(v) for v in var0_grid],
                "couple_weight_decay": args.couple_weight_decay,
                "quick": args.quick,
                "runs": [run_dir_name(i, v) for i, v, _, _ in configs],
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

    print(f"sweep -> {sweep_dir.resolve()}")
    print(
        f"{len(configs)} runs, prior scales: "
        f"{[f'{math.sqrt(v):.3g}' for _, v, _, _ in configs]}"
    )

    status: dict[str, str] = {}
    for i, var0, run_dir, cfg in configs:
        label = run_dir_name(i, var0)
        if is_complete(run_dir) and not args.force:
            print(f"\n=== {label}: already finished, skipping ===")
            status[label] = "skipped"
            continue
        status[label] = execute_run(cfg, run_dir, var0=var0, label=label)

    (sweep_dir / "sweep_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    failed = {k: v for k, v in status.items() if v not in ("ok", "skipped")}
    if failed:
        print(f"\n{len(failed)} run(s) failed: {json.dumps(failed, indent=2)}")

    rows = collect_sweep(sweep_dir)
    if not rows:
        msg = "every run failed; nothing to summarise"
        raise SystemExit(msg)
    write_tables(sweep_dir, rows)
    make_figures(sweep_dir, rows)
    report(rows)
    print(f"\ndone: {sweep_dir.resolve()}")


if __name__ == "__main__":
    main()
