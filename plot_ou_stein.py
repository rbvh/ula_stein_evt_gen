"""
Plot results from run_ou_stein.py.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from ula_stein_evt_gen.plot_style import (
    COLORS,
    TEXTWIDTH,
    configure_matplotlib,
    style_axes,
    style_symlog_axis,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "results",
    type=str,
    nargs="*",
    help=(
        "Optional path(s) to results.json from run_ou_stein.py. "
        "By default, the d=4 and d=16 OU result files are auto-discovered."
    ),
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/ou_stein",
    help="Directory searched for d=4 and d=16 result files when no paths are provided.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output plot path (default: plots/ou_stein_<dims>[_mass_cut_<fraction>].pdf)",
)
parser.add_argument("--no_tex", action="store_true")
args = parser.parse_args()

configure_matplotlib(use_tex=not args.no_tex)


def safe_label_value(value):
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def find_default_results(results_dir, dims=(4, 16)):
    results_dir = Path(results_dir)
    paths_by_dim = {dim: [] for dim in dims}
    for path in sorted(results_dir.glob("*/results.json")):
        with open(path) as f:
            dim = int(json.load(f)["dim"])
        if dim in paths_by_dim:
            paths_by_dim[dim].append(path)

    result_paths = []
    for dim in dims:
        candidates = paths_by_dim[dim]
        if not candidates:
            raise FileNotFoundError(
                f"No d={dim} OU result file found under {results_dir}"
            )
        if len(candidates) > 1:
            candidate_list = "\n".join(f"  {path}" for path in candidates)
            raise ValueError(
                f"Multiple d={dim} OU result files found under {results_dir}:\n"
                f"{candidate_list}\nPass explicit result paths to select one."
            )
        result_paths.append(str(candidates[0]))
    return result_paths


def load_result(path):
    with open(path) as f:
        raw = json.load(f)

    data = {
        "path": path,
        "ou_checkpoints": np.array(raw["ou_checkpoints"]),
        "mcmc_checkpoints": np.array(raw["mcmc_checkpoints"]),
        "analytical_lsd": np.array(raw["analytical_lsd"]),
        "ou_learned_lsd": np.array(raw["ou_learned_lsd"]),
        "ou_learned_lsd_std": np.array(raw["ou_learned_lsd_std"]),
        "mcmc_learned_lsd": np.array(raw["mcmc_learned_lsd"]),
        "mcmc_learned_lsd_std": np.array(raw["mcmc_learned_lsd_std"]),
        "dim": int(raw["dim"]),
        "boundary_mass_cut": raw.get("boundary_mass_cut"),
        "boundary_r2_cut": raw.get("boundary_r2_cut"),
    }
    data["analytical_lsd_std"] = np.array(
        raw.get("analytical_lsd_std", np.zeros_like(data["analytical_lsd"]))
    )

    no_boundary_keys = (
        "ou_learned_lsd_no_boundary",
        "ou_learned_lsd_no_boundary_std",
        "mcmc_learned_lsd_no_boundary",
        "mcmc_learned_lsd_no_boundary_std",
    )
    data["has_no_boundary_ablation"] = all(raw.get(key) is not None for key in no_boundary_keys)
    if data["has_no_boundary_ablation"]:
        for key in no_boundary_keys:
            data[key] = np.array(raw[key])

    return data


result_paths = args.results or find_default_results(args.results_dir)
all_data = sorted((load_result(path) for path in result_paths), key=lambda item: item["dim"])

if args.output is None:
    boundary_mass_cuts = {item["boundary_mass_cut"] for item in all_data}
    boundary_r2_cuts = {item["boundary_r2_cut"] for item in all_data}
    suffix = ""
    if len(boundary_mass_cuts) == 1 and next(iter(boundary_mass_cuts)) is not None:
        suffix = f"_mass_cut_{safe_label_value(next(iter(boundary_mass_cuts)))}"
    elif len(boundary_r2_cuts) == 1 and next(iter(boundary_r2_cuts)) is not None:
        suffix = f"_r2_cut_{safe_label_value(next(iter(boundary_r2_cuts)))}"

    dims = "_".join(f"d{item['dim']}" for item in all_data)
    args.output = f"plots/ou_stein_{dims}{suffix}.pdf"

out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

linthresh = 1e-2
fig_height = 3.35 if len(all_data) > 1 else 4.6
fig, axes = plt.subplots(
    1,
    len(all_data),
    figsize=(TEXTWIDTH, fig_height),
    sharey=True,
    squeeze=False,
    gridspec_kw={"wspace": 0.06},
)
axes = axes[0]


def plot_panel(ax, data, show_legend=False):
    ou_checkpoints = data["ou_checkpoints"]
    mcmc_checkpoints = data["mcmc_checkpoints"]
    analytical_lsd = data["analytical_lsd"]
    analytical_lsd_std = data["analytical_lsd_std"]
    ou_learned_lsd = data["ou_learned_lsd"]
    ou_learned_lsd_std = data["ou_learned_lsd_std"]
    mcmc_learned_lsd = data["mcmc_learned_lsd"]
    mcmc_learned_lsd_std = data["mcmc_learned_lsd_std"]

    ax.plot(
        ou_checkpoints,
        analytical_lsd,
        color="0.1",
        linestyle="--",
        dashes=(5.0, 2.5),
        linewidth=1.35,
        label=r"OU analytical $S_{\mathrm{opt}}$",
        zorder=8,
    )
    if np.any(analytical_lsd_std > 0):
        ax.fill_between(
            ou_checkpoints,
            analytical_lsd - analytical_lsd_std,
            analytical_lsd + analytical_lsd_std,
            alpha=0.18,
            color="k",
        )
        ax.errorbar(
            ou_checkpoints,
            analytical_lsd,
            yerr=analytical_lsd_std,
            fmt="none",
            ecolor="0.15",
            elinewidth=0.8,
            capsize=2.0,
            alpha=0.65,
        )

    ax.plot(
        ou_checkpoints,
        ou_learned_lsd,
        marker="o",
        color=COLORS[0],
        linewidth=1.8,
        markersize=3.2,
        label="OU LSD",
        zorder=6,
    )
    ax.fill_between(
        ou_checkpoints,
        ou_learned_lsd - ou_learned_lsd_std,
        ou_learned_lsd + ou_learned_lsd_std,
        alpha=0.22,
        color=COLORS[0],
        linewidth=0,
        zorder=2,
    )

    if data["has_no_boundary_ablation"]:
        ax.plot(
            ou_checkpoints,
            data["ou_learned_lsd_no_boundary"],
            marker="o",
            linestyle="--",
            color=COLORS[0],
            markerfacecolor="white",
            label=r"OU LSD (no $h$)",
        )
        ax.fill_between(
            ou_checkpoints,
            data["ou_learned_lsd_no_boundary"] - data["ou_learned_lsd_no_boundary_std"],
            data["ou_learned_lsd_no_boundary"] + data["ou_learned_lsd_no_boundary_std"],
            alpha=0.12,
            color=COLORS[0],
            linewidth=0,
        )

    ax.plot(
        mcmc_checkpoints,
        mcmc_learned_lsd,
        marker="s",
        color=COLORS[1],
        linewidth=1.25,
        label="ULA LSD",
        zorder=5,
    )
    ax.fill_between(
        mcmc_checkpoints,
        mcmc_learned_lsd - mcmc_learned_lsd_std,
        mcmc_learned_lsd + mcmc_learned_lsd_std,
        alpha=0.22,
        color=COLORS[1],
        linewidth=0,
    )

    if data["has_no_boundary_ablation"]:
        ax.plot(
            mcmc_checkpoints,
            data["mcmc_learned_lsd_no_boundary"],
            marker="s",
            linestyle="--",
            color=COLORS[1],
            markerfacecolor="white",
            label=r"ULA LSD (no $h$)",
        )
        ax.fill_between(
            mcmc_checkpoints,
            data["mcmc_learned_lsd_no_boundary"] - data["mcmc_learned_lsd_no_boundary_std"],
            data["mcmc_learned_lsd_no_boundary"] + data["mcmc_learned_lsd_no_boundary_std"],
            alpha=0.12,
            color=COLORS[1],
            linewidth=0,
        )

    ax.axhline(0, color="0.25", linewidth=0.6, linestyle=":")
    ax.set_xlabel("OU / ULA steps $t$")
    ax.text(
        0.03,
        0.035,
        rf"$d={data['dim']}$",
        transform=ax.transAxes,
        ha="left",
        va="center",
    )

    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.995, 0.78),
            fontsize=8.0,
        )


for i, (ax, data) in enumerate(zip(axes, all_data)):
    plot_panel(ax, data, show_legend=(i == 0))

all_y_values = []
for data in all_data:
    for key in (
        "analytical_lsd",
        "ou_learned_lsd",
        "mcmc_learned_lsd",
        "ou_learned_lsd_no_boundary",
        "mcmc_learned_lsd_no_boundary",
    ):
        if key in data:
            all_y_values.extend(np.asarray(data[key]).ravel())

y_values = np.array(all_y_values)
y_values = y_values[np.isfinite(y_values)]
ymin = min(-1e-1, float(np.min(y_values)) if y_values.size else -1e-1)

for i, ax in enumerate(axes):
    ax.axhspan(-linthresh, linthresh, color="0.92", alpha=0.5, zorder=0)
    ax.axhline(linthresh, color="0.55", linewidth=0.8, linestyle=":", zorder=0.5)
    ax.axhline(-linthresh, color="0.55", linewidth=0.8, linestyle=":", zorder=0.5)
    style_symlog_axis(ax, linthresh, ymin=max(ymin, -1e-1), ymax=300.0)
    style_axes(ax)
    ax.tick_params(top=False)
    ax.tick_params(which="minor", top=False)
    ax.tick_params(right=False)
    ax.tick_params(which="minor", right=False)
    if i > 0:
        ax.tick_params(left=False, labelleft=False)

axes[0].set_ylabel("Stein discrepancy")

fig.savefig(args.output, bbox_inches="tight")
print(f"Plot saved to {args.output}")
