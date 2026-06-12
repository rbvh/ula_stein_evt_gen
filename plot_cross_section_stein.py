"""
Plot cross-section Stein diagnostics from run_cross_section_mcmc.py.
"""

import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

from ula_stein_evt_gen.plot_style import (
    TEXTWIDTH,
    configure_matplotlib,
    multiplicity_color,
    style_axes,
    style_symlog_axis,
)


def result_paths(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            direct = os.path.join(item, "stein_results.json")
            if os.path.exists(direct):
                paths.append(direct)
                continue
            for root, _, files in os.walk(item):
                if "stein_results.json" in files:
                    paths.append(os.path.join(root, "stein_results.json"))
        else:
            if os.path.basename(item) != "stein_results.json":
                raise ValueError(
                    "Cross-section Stein plotting only supports split-format "
                    "stein_results.json files."
                )
            paths.append(item)
    return sorted(set(paths))


def load_result(path):
    with open(path) as f:
        data = json.load(f)
    data["_path"] = path
    return data


def vb_label(vb):
    return "Z" if vb == "z" else "W"


def multiplicity_label(data):
    return rf"${vb_label(data['vb'])}+{int(data['n_jets'])}g$"


def is_surrogate_result(data):
    return bool(data.get("surrogate_warm_start", False))


def fade_color(color, amount=0.38):
    rgb = np.array(to_rgb(color))
    return tuple((1.0 - amount) * rgb + amount * np.ones(3))


def method_style(data):
    if is_surrogate_result(data):
        return {
            "linestyle": "-",
            "linewidth": 1.55,
            "marker": None,
            "markersize": 0.0,
            "alpha": 0.98,
            "band_alpha": 0.20,
            "fade": False,
            "label": "ULA (Surrogate)",
        }
    return {
        "linestyle": "--",
        "linewidth": 1.15,
        "marker": None,
        "markersize": 0.0,
        "alpha": 0.9,
        "band_alpha": 0.08,
        "fade": True,
        "label": "ULA",
    }


def nice_upper_limit(ymax):
    if not np.isfinite(ymax) or ymax <= 0.0:
        return 100.0
    power = 10.0 ** np.floor(np.log10(ymax))
    scaled = ymax / power
    if scaled <= 1.0:
        nice = 1.0
    elif scaled <= 3.0:
        nice = 3.0
    else:
        nice = 10.0
    return nice * power


def default_output(results):
    if len(results) == 1:
        process = results[0].get(
            "process", f"{results[0]['vb']}_{results[0]['n_jets']}j"
        )
        return f"plots/cross_section_stein_{process}.pdf"
    vbs = sorted({data["vb"] for data in results})
    suffix = vbs[0] if len(vbs) == 1 else "combined"
    return f"plots/cross_section_stein_{suffix}.pdf"


parser = argparse.ArgumentParser()
parser.add_argument(
    "results",
    type=str,
    nargs="*",
    default=["results/cross_section_mcmc"],
    help=(
        "Path(s) to Stein result files or directories from run_cross_section_mcmc.py "
        "(default: results/cross_section_mcmc)"
    ),
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output plot path (default: plots/cross_section_stein_<process-or-vb>.pdf)",
)
parser.add_argument("--no_tex", action="store_true")
args = parser.parse_args()

configure_matplotlib(use_tex=not args.no_tex)

paths = result_paths(args.results)
if not paths:
    raise RuntimeError("No cross-section Stein result files found.")

results = [
    data
    for data in (load_result(path) for path in paths)
    if "mcmc_learned_lsd" in data and "mcmc_learned_lsd_std" in data
]
if not results:
    raise RuntimeError("No cross-section Stein diagnostics found in supplied paths.")
results.sort(
    key=lambda data: (
        data.get("vb", ""),
        int(data.get("n_jets", 0)),
        is_surrogate_result(data),
    )
)

if args.output is None:
    args.output = default_output(results)

out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(0.5 * TEXTWIDTH, 3.35))
linthresh = 1e-3
all_y_values = []
all_y_upper_values = []
multiplicity_handles = {}
method_keys = set()

for data in results:
    checkpoints = np.array(data["mcmc_checkpoints"], dtype=float)
    lsd = np.array(data["mcmc_learned_lsd"], dtype=float)
    lsd_std = np.array(data["mcmc_learned_lsd_std"], dtype=float)
    n_jets = int(data["n_jets"])
    color = multiplicity_color(n_jets)
    style = method_style(data)
    plot_color = fade_color(color) if style["fade"] else color
    method_keys.add("surrogate" if is_surrogate_result(data) else "mcmc")
    multiplicity_handles.setdefault(
        (data["vb"], n_jets),
        Line2D(
            [],
            [],
            color=color,
            linestyle="-",
            linewidth=1.4,
            label=multiplicity_label(data),
        ),
    )
    all_y_values.extend(lsd[np.isfinite(lsd)])
    upper = lsd + lsd_std
    all_y_upper_values.extend(upper[np.isfinite(upper)])

    ax.plot(
        checkpoints,
        lsd,
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=plot_color,
        linewidth=style["linewidth"],
        markersize=style["markersize"],
        alpha=style["alpha"],
    )
    ax.fill_between(
        checkpoints,
        lsd - lsd_std,
        lsd + lsd_std,
        alpha=style["band_alpha"],
        color=plot_color,
        linewidth=0,
    )

ax.axhline(0, color="0.25", linewidth=0.6, linestyle=":")
ax.set_xlabel(r"ULA steps $t$")
ax.set_ylabel("Stein discrepancy")
upper_values = np.array(all_y_upper_values)
ymax = nice_upper_limit(float(np.nanmax(upper_values)) if upper_values.size else 100.0)
ax.axhspan(-linthresh, linthresh, color="0.92", alpha=0.5, zorder=0)
ax.axhline(linthresh, color="0.55", linewidth=0.8, linestyle=":", zorder=0.5)
ax.axhline(-linthresh, color="0.55", linewidth=0.8, linestyle=":", zorder=0.5)
style_symlog_axis(ax, linthresh, ymin=-1e-2, ymax=ymax)
ax.set_xlim(left=0.0)

vbs = sorted({data["vb"] for data in results})
if len(vbs) == 1:
    process_text = rf"${vb_label(vbs[0])}$+gluons"
else:
    process_text = "vector-boson+gluons"
ax.text(
    0.03,
    0.065,
    process_text,
    transform=ax.transAxes,
    ha="left",
    va="center",
)

method_handles = []
if "mcmc" in method_keys:
    method_handles.append(
        Line2D(
            [],
            [],
            color="0.55",
            linestyle="--",
            linewidth=1.15,
            label="ULA",
        )
    )
if "surrogate" in method_keys:
    method_handles.append(
        Line2D(
            [],
            [],
            color="0.1",
            linestyle="-",
            linewidth=1.55,
            label="ULA (Surrogate)",
        )
    )
method_legend = ax.legend(
    handles=method_handles,
    loc="upper right",
    bbox_to_anchor=(0.995, 0.78),
    fontsize=8.0,
    handlelength=1.6,
)
ax.add_artist(method_legend)
ax.legend(
    handles=list(multiplicity_handles.values()),
    loc="upper right",
    fontsize=8.0,
    handlelength=1.6,
)
style_axes(ax)
ax.tick_params(top=False)
ax.tick_params(which="minor", top=False)
ax.tick_params(right=False)
ax.tick_params(which="minor", right=False)

fig.savefig(args.output, bbox_inches="tight")
print(f"Plot saved to {args.output}")
