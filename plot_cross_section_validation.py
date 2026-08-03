import argparse
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib import ticker
import numpy as np

from ula_stein_evt_gen.plot_style import (
    TEXTWIDTH,
    configure_matplotlib,
    multiplicity_color,
    style_axes,
)


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    data["_path"] = path
    return data


def load_split_result(directory, vb, n_jets, seed):
    metadata_path = os.path.join(directory, "metadata.json")
    histogram_dir = os.path.join(directory, "histograms")
    if not os.path.exists(metadata_path) or not os.path.isdir(histogram_dir):
        return None

    metadata = load_json(metadata_path)
    try:
        data_n_jets = int(metadata.get("n_jets", -1))
        data_seed = int(metadata.get("seed", -1))
    except (TypeError, ValueError):
        return None
    if metadata.get("vb") != vb or data_n_jets != n_jets or data_seed != seed:
        return None

    histograms = {}
    for filename in sorted(os.listdir(histogram_dir)):
        if not filename.endswith(".json"):
            continue
        payload = load_json(os.path.join(histogram_dir, filename))
        observable = payload.get("observable") or os.path.splitext(filename)[0]
        histogram = payload.get("histogram")
        if isinstance(histogram, dict):
            histograms[observable] = histogram
    if not histograms:
        return None

    data = dict(metadata)
    data["_path"] = metadata_path
    data["histograms"] = histograms
    return data


def load_split_path(path_or_directory, vb, n_jets, seed):
    if os.path.isfile(path_or_directory):
        if os.path.basename(path_or_directory) != "metadata.json":
            raise ValueError(
                "Observable plotting only supports split-format metadata.json "
                "files or directories containing metadata.json and histograms/."
            )
        path_or_directory = os.path.dirname(path_or_directory)
    return load_split_result(path_or_directory, vb, n_jets, seed)


def discover_split_results(path_or_directory, vb, n_jets, seed):
    if os.path.isfile(path_or_directory):
        result = load_split_path(path_or_directory, vb, n_jets, seed)
        return [result] if result is not None else []

    candidates = []
    if not os.path.isdir(path_or_directory):
        return candidates

    exact = load_split_result(
        os.path.join(path_or_directory, f"{vb}_{n_jets}j_seed{seed}"),
        vb,
        n_jets,
        seed,
    )
    if exact is not None:
        candidates.append(exact)

    for root, _, files in os.walk(path_or_directory):
        if "metadata.json" not in files:
            continue
        split = load_split_result(root, vb, n_jets, seed)
        if split is not None:
            candidates.append(split)

    unique = {data["_path"]: data for data in candidates}
    return list(unique.values())


def load_mcmc_variants(path_or_directory, vb, n_jets, seed):
    variants = []
    seen_paths = set()

    def add_variant(data):
        if data is None:
            return
        path = data.get("_path")
        if path in seen_paths:
            return
        seen_paths.add(path)
        label = "Surrogate" if data.get("surrogate_warm_start") else "MCMC"
        key = "surrogate" if data.get("surrogate_warm_start") else "mcmc"
        variants.append({"key": key, "label": label, "data": data})

    if os.path.isfile(path_or_directory):
        add_variant(load_split_path(path_or_directory, vb, n_jets, seed))
        return variants

    for data in discover_split_results(path_or_directory, vb, n_jets, seed):
        add_variant(data)

    variants.sort(key=lambda variant: 1 if variant["key"] == "surrogate" else 0)
    return variants


def get_variant_hist_pair(variant, vegas, observable):
    mcmc = variant["data"]
    if observable not in mcmc["histograms"] or observable not in vegas["histograms"]:
        return None
    return mcmc["histograms"][observable], vegas["histograms"][observable]


def item_has_observable(item, observable):
    _, variants, vegas = item
    if observable not in vegas["histograms"]:
        return False
    return any(observable in variant["data"]["histograms"] for variant in variants)


def variant_plot_style(key):
    if key == "surrogate":
        return {
            "kind": "step",
            "linestyle": ":",
            "linewidth": 0.9,
            "alpha": 0.95,
            "elinewidth": 0.65,
            "capsize": 0.0,
        }
    return {
        "kind": "step",
        "linestyle": "-",
        "linewidth": 0.9,
        "alpha": 0.95,
        "elinewidth": 0.65,
        "capsize": 0.0,
    }


def ratio_and_error(num, num_err, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0.0, num / den, np.nan)
        ratio_err = np.where(den > 0.0, num_err / den, np.nan)
    return ratio, ratio_err


def denominator_ratio_error(den, den_err):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.0, den_err / den, np.nan)


def vb_label(vb):
    return "Z" if vb == "z" else "W"


def multiplicity_label(vb, n_jets):
    return rf"${vb_label(vb)}+{n_jets}g$"


def observable_labels(name, vb):
    if name == "pt_v":
        boson = vb_label(vb)
        return (
            rf"$p_{{\mathrm{{T}}}}^{{{boson}}}$ [GeV]",
            rf"p_{{\mathrm{{T}}}}^{{{boson}}}",
            r"$\mathrm{[GeV]}^{-1}$",
        )
    if name == "y_v":
        boson = vb_label(vb)
        return rf"$y^{{{boson}}}$", rf"y^{{{boson}}}", ""
    if name == "h_t":
        return r"$H_{\mathrm{T}}$ [GeV]", r"H_{\mathrm{T}}", r"$\mathrm{[GeV]}^{-1}$"
    match = re.fullmatch(r"pt_j(\d)", name)
    if match:
        i_jet = match.group(1)
        return (
            rf"$p_{{\mathrm{{T}}}}^{{j_{i_jet}}}$ [GeV]",
            rf"p_{{\mathrm{{T}}}}^{{j_{i_jet}}}",
            r"$\mathrm{[GeV]}^{-1}$",
        )
    if name == "y_j1":
        return r"$y^{j_1}$", r"y^{j_1}", ""
    if name == "max_abs_y_j":
        return r"$\max_i |y^{j_i}|$", r"\max_i |y^{j_i}|", ""
    if name == "min_deltaR_Zj":
        boson = vb_label(vb)
        return (
            rf"$\min_i \Delta R({boson},j_i)$",
            rf"\min_i \Delta R({boson},j_i)",
            "",
        )
    if name == "min_deltaR_jj":
        return r"$\min_{i<j} \Delta R(j_i,j_j)$", r"\min_{i<j} \Delta R(j_i,j_j)", ""
    if name == "x_a":
        return r"$x_a$", r"x_a", ""
    if name == "x_b":
        return r"$x_b$", r"x_b", ""
    if name == "log_xaxb":
        return r"$\log(x_a x_b)$", r"\log(x_a x_b)", ""
    if name == "y_x":
        return r"$\frac{1}{2}\log(x_a/x_b)$", r"\frac{1}{2}\log(x_a/x_b)", ""
    if name == "m_jj":
        return r"$m_{j_1j_2}$ [GeV]", r"m_{j_1j_2}", r"$\mathrm{[GeV]}^{-1}$"
    if name == "m_all_jets":
        return (
            r"$m_{\mathrm{jets}}$ [GeV]",
            r"m_{\mathrm{jets}}",
            r"$\mathrm{[GeV]}^{-1}$",
        )

    match = re.fullmatch(r"([dy])(\d)(\d)", name)
    if match:
        family, i_scale, j_scale = match.groups()
        if family == "d":
            obs = rf"d_{{{i_scale}{j_scale}}}"
            return (
                rf"$\sqrt{{{obs}}}$ [GeV]",
                rf"\sqrt{{{obs}}}",
                r"$\mathrm{[GeV]}^{-1}$",
            )
        obs = rf"y_{{{i_scale}{j_scale}}}"
        return rf"${obs}$", obs, ""

    return name.replace("_", " "), name.replace("_", r"\_"), ""


def y_axis_label(vb, derivative_label, unit_label):
    label = rf"$\sigma^{{-1}}\,d\sigma/d{derivative_label}$"
    if unit_label:
        label = f"{label} {unit_label}"
    return label


def set_x_ticks(ax, observable):
    if observable == "pt_v" or re.fullmatch(r"pt_j\d", observable):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50.0))
    elif observable in {"h_t", "m_jj", "m_all_jets"}:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(250.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(125.0))
    elif re.fullmatch(r"d\d\d", observable):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50.0))
    elif observable in {"min_deltaR_Zj", "min_deltaR_jj"} or re.fullmatch(
        r"y\d\d", observable
    ):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    else:
        ax.xaxis.set_major_locator(
            ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        )
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def set_log_decade_ticks(ax):
    ymin, ymax = ax.get_ylim()
    if ymin <= 0.0 or ymax <= 0.0:
        return
    lo_power = int(np.floor(np.log10(ymin)))
    hi_power = int(np.ceil(np.log10(ymax)))
    ticks = 10.0 ** np.arange(lo_power, hi_power + 1)
    ticks = ticks[(ticks >= ymin) & (ticks <= ymax)]
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100)
    )


def discover_observables(loaded):
    observables = []
    preferred = [
        "pt_v",
        "y_v",
        "h_t",
        "pt_j1",
        "pt_j2",
        "pt_j3",
        "pt_j4",
        "pt_j5",
        "y_j1",
        "max_abs_y_j",
        "min_deltaR_Zj",
        "min_deltaR_jj",
        "x_a",
        "x_b",
        "log_xaxb",
        "y_x",
        "m_jj",
        "m_all_jets",
    ]
    for base in preferred:
        if any(
            item_has_observable(item, base) and is_observable_applicable(item[0], base)
            for item in loaded
        ):
            observables.append(base)

    scale_names = set()
    pattern = re.compile(r"[dy]\d\d")
    for _, variants, vegas in loaded:
        for variant in variants:
            common = set(variant["data"]["histograms"]) & set(vegas["histograms"])
            scale_names.update(name for name in common if pattern.fullmatch(name))

    def scale_key(name):
        match = re.fullmatch(r"([dy])(\d)(\d)", name)
        if match is None:
            return (99, 99, name)
        family, i_scale, j_scale = match.groups()
        return (0 if family == "d" else 1, int(i_scale), int(j_scale))

    observables.extend(sorted(scale_names, key=scale_key))
    return observables


def required_n_jets(observable):
    if observable == "pt_v":
        return 1
    if observable in {"h_t", "y_j1", "max_abs_y_j", "min_deltaR_Zj"}:
        return 1
    if observable in {"min_deltaR_jj", "m_jj", "m_all_jets"}:
        return 2
    match = re.fullmatch(r"pt_j(\d)", observable)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"[dy]\d(\d)", observable)
    if match:
        return int(match.group(1))
    return 0


def is_observable_applicable(n_jets, observable):
    return n_jets >= required_n_jets(observable)


def is_rapidity_like(observable):
    return observable in {"y_v", "y_j1", "max_abs_y_j", "y_x"}


def fixed_ratio_ylim(observable):
    if is_rapidity_like(observable):
        return 0.9, 1.1
    if observable in {
        "pt_v",
        "h_t",
        "m_jj",
        "m_all_jets",
    } or re.fullmatch(r"(pt_j\d|d\d\d)", observable):
        return 0.75, 1.25
    return None


def ratio_ylim(ratios, errors):
    vals = []
    for ratio, err in zip(ratios, errors):
        mask = np.isfinite(ratio) & np.isfinite(err)
        if np.any(mask):
            vals.append(ratio[mask])
            vals.append((ratio - err)[mask])
            vals.append((ratio + err)[mask])
    if not vals:
        return 0.5, 1.5
    finite = np.concatenate(vals)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.5, 1.5
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    span = max(hi - lo, 0.2)
    lo = min(0.95, lo - 0.25 * span)
    hi = max(1.05, hi + 0.25 * span)
    return max(0.0, lo), hi


def positive_limits(values, errors):
    finite = []
    for value, error in zip(values, errors):
        value = np.asarray(value)
        error = np.asarray(error)
        mask = np.isfinite(value) & (value > 0.0)
        if np.any(mask):
            finite.append(value[mask])
            low = value[mask] - error[mask]
            finite.append(low[low > 0.0])
            finite.append(value[mask] + error[mask])
    if not finite:
        return None
    finite = np.concatenate(finite)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return None
    lo = np.nanpercentile(finite, 1.0)
    hi = np.nanpercentile(finite, 99.5)
    return max(lo / 3.0, 1e-16), hi * 3.0


def linear_limits(values, errors):
    finite = []
    for value, error in zip(values, errors):
        value = np.asarray(value)
        error = np.asarray(error)
        mask = np.isfinite(value)
        if np.any(mask):
            finite.append(value[mask])
            finite.append((value - error)[mask])
            finite.append((value + error)[mask])
    if not finite:
        return None
    finite = np.concatenate(finite)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    lo = np.nanmin(finite)
    hi = np.nanmax(finite)
    span = max(hi - lo, 1e-12)
    return min(0.0, lo - 0.08 * span), hi + 0.45 * span


def plot_observable(
    loaded,
    observable,
    x_label,
    y_label,
    out_path,
    vb,
    log_y=False,
    compact=False,
):
    results = [
        item
        for item in loaded
        if is_observable_applicable(item[0], observable)
        and item_has_observable(item, observable)
    ]
    if not results:
        return False

    n_ratios = len(results)
    if compact:
        # Preserve the font and legend sizes while reducing only the vertical
        # extent of the data panels.  The ratio axes remain tall enough for
        # their tick labels and in-panel multiplicity labels.
        height_ratios = [3.7] + [0.9] * n_ratios
        fig_height = max(3.15, 1.65 + 0.525 * n_ratios)
    else:
        height_ratios = [4.4] + [1.15] * n_ratios
        fig_height = max(3.35, 2.25 + 0.6 * n_ratios)
    fig, axes = plt.subplots(
        n_ratios + 1,
        1,
        figsize=(0.5 * TEXTWIDTH, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.0},
    )
    ax = axes[0]
    ratio_axes = axes[1:]

    all_ratio_bands = []
    all_values = []
    all_errors = []
    multiplicity_handles = []
    plotted_variant_keys = set()
    for idx, item in enumerate(results):
        n_jets, variants, vegas = item
        if observable not in vegas["histograms"]:
            continue
        hist_v = vegas["histograms"][observable]
        edges = np.asarray(hist_v["bin_edges"], dtype=float)
        centers = 0.5 * (edges[1:] + edges[:-1])
        half_widths = 0.5 * (edges[1:] - edges[:-1])

        v_val = np.asarray(hist_v["density"], dtype=float)
        v_err = np.asarray(hist_v["density_error"], dtype=float)

        color = multiplicity_color(n_jets)
        label = multiplicity_label(vb, n_jets)
        multiplicity_handles.append(
            Line2D([], [], color=color, linestyle="-", linewidth=1.2, label=label)
        )

        ax.errorbar(
            centers,
            v_val,
            xerr=half_widths,
            yerr=v_err,
            fmt="o",
            color=color,
            markersize=2.4,
            elinewidth=0.65,
            capsize=0.0,
            markerfacecolor=color,
            markeredgewidth=0.0,
        )
        den_ratio_err = denominator_ratio_error(v_val, v_err)
        all_ratio_bands.append((np.ones_like(den_ratio_err), den_ratio_err))
        all_values.append(v_val)
        all_errors.append(v_err)

        axr = ratio_axes[idx]
        axr.fill_between(
            edges,
            np.r_[1.0 - den_ratio_err, 1.0 - den_ratio_err[-1]],
            np.r_[1.0 + den_ratio_err, 1.0 + den_ratio_err[-1]],
            step="post",
            color="0.55",
            alpha=0.22,
            linewidth=0,
        )
        for variant in variants:
            hist_pair = get_variant_hist_pair(variant, vegas, observable)
            if hist_pair is None:
                continue
            hist_m, _ = hist_pair
            m_val = np.asarray(hist_m["density"], dtype=float)
            m_err = np.asarray(hist_m["density_error"], dtype=float)
            style = variant_plot_style(variant["key"])
            plotted_variant_keys.add(variant["key"])
            if style["kind"] == "step":
                ax.stairs(
                    m_val,
                    edges,
                    color=color,
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                )
                ax.errorbar(
                    centers,
                    m_val,
                    xerr=half_widths,
                    yerr=m_err,
                    fmt="none",
                    ecolor=color,
                    elinewidth=style["elinewidth"],
                    capsize=style["capsize"],
                    alpha=style["alpha"],
                )
            else:
                ax.errorbar(
                    centers,
                    m_val,
                    yerr=m_err,
                    fmt=style["marker"],
                    color=color,
                    markersize=style["markersize"],
                    elinewidth=style["elinewidth"],
                    capsize=style["capsize"],
                    markerfacecolor=style["markerfacecolor"],
                    markeredgewidth=style["markeredgewidth"],
                )
            all_values.append(m_val)
            all_errors.append(m_err)

            ratio, ratio_err = ratio_and_error(m_val, m_err, v_val)
            all_ratio_bands.append((ratio, ratio_err))
            if style["kind"] == "step":
                axr.stairs(
                    ratio,
                    edges,
                    color=color,
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                )
                axr.errorbar(
                    centers,
                    ratio,
                    xerr=half_widths,
                    yerr=ratio_err,
                    fmt="none",
                    ecolor=color,
                    elinewidth=style["elinewidth"],
                    capsize=style["capsize"],
                    alpha=style["alpha"],
                )
            else:
                axr.errorbar(
                    centers,
                    ratio,
                    yerr=ratio_err,
                    fmt=style["marker"],
                    color=color,
                    markersize=max(style["markersize"] - 0.2, 1.8),
                    elinewidth=style["elinewidth"],
                    capsize=style["capsize"],
                    markerfacecolor=style["markerfacecolor"],
                    markeredgewidth=style["markeredgewidth"],
                )
        axr.axhline(1.0, color="0.05", linewidth=0.75)
        label_x = 0.5 if is_rapidity_like(observable) else 0.035
        label_ha = "center" if is_rapidity_like(observable) else "left"
        axr.text(
            label_x,
            0.90 if compact else 0.82,
            label,
            transform=axr.transAxes,
            ha=label_ha,
            va="top",
            fontsize=8.0,
        )
        style_axes(axr, labelsize=8.5)
        axr.xaxis.label.set_size(9.0)
        set_x_ticks(axr, observable)
        axr.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="both"))
        axr.tick_params(top=False)
        axr.tick_params(which="minor", top=False)
        axr.tick_params(right=False)
        axr.tick_params(which="minor", right=False)

    fixed_y_limits = fixed_ratio_ylim(observable)
    use_compact_ratio_ticks = compact and fixed_y_limits == (0.75, 1.25)
    if use_compact_ratio_ticks:
        # Keep the boundary tick labels of adjacent compact ratio panels from
        # crowding one another without changing the central comparison range.
        fixed_y_limits = (0.70, 1.30)
    if fixed_y_limits is None:
        ylo, yhi = ratio_ylim(
            [ratio for ratio, _ in all_ratio_bands],
            [err for _, err in all_ratio_bands],
        )
    else:
        ylo, yhi = fixed_y_limits
    for idx, axr in enumerate(ratio_axes):
        axr.set_ylim(ylo, yhi)
        axr.set_xlim(edges[0], edges[-1])
        if use_compact_ratio_ticks:
            axr.yaxis.set_major_locator(ticker.FixedLocator([0.8, 1.0, 1.2]))
        if idx != len(ratio_axes) - 1:
            axr.tick_params(labelbottom=False)
    ratio_axes[-1].set_xlabel(x_label, labelpad=2)

    ax.set_ylabel(y_label, labelpad=1, fontsize=9.0)
    set_x_ticks(ax, observable)
    style_axes(ax, labelsize=8.5)
    ax.tick_params(top=False)
    ax.tick_params(which="minor", top=False)
    ax.tick_params(right=False)
    ax.tick_params(which="minor", right=False)
    if log_y and any(
        np.any(np.asarray(item[2]["histograms"][observable]["density"]) > 0.0)
        for item in results
        if observable in item[2]["histograms"]
    ):
        ax.set_yscale("log")
        limits = positive_limits(all_values, all_errors)
        if limits is not None:
            ax.set_ylim(*limits)
    else:
        limits = linear_limits(all_values, all_errors)
        if limits is not None:
            ax.set_ylim(*limits)
    if ax.get_yscale() == "log":
        set_log_decade_ticks(ax)
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    method_handles = [
        Line2D(
            [],
            [],
            color="0.1",
            marker="o",
            linestyle="None",
            markerfacecolor="0.1",
            markeredgewidth=0.0,
            markersize=3.0,
            label="VEGAS",
        )
    ]
    if "mcmc" in plotted_variant_keys:
        method_handles.append(
            Line2D(
                [],
                [],
                color="0.1",
                linestyle="-",
                linewidth=1.0,
                label="ULA",
            )
        )
    if "surrogate" in plotted_variant_keys:
        method_handles.append(
            Line2D(
                [],
                [],
                color="0.1",
                linestyle=":",
                linewidth=1.0,
                label="ULA (Surrogate)",
            )
        )
    method_legend_kwargs = {
        "loc": "lower left",
        "bbox_to_anchor": (0.04, 0.05),
        "ncol": 1,
    }
    if is_rapidity_like(observable):
        method_legend_kwargs = {
            "loc": "lower center",
            "bbox_to_anchor": (0.5, 0.0 if compact else 0.05),
            "ncol": 1,
        }
    first_legend = ax.legend(
        handles=method_handles,
        fontsize=8.0,
        handlelength=1.5,
        **method_legend_kwargs,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=multiplicity_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=8.0,
        handlelength=1.5,
        ncol=1,
    )
    fig.align_ylabels()
    ratio_top = ratio_axes[0].get_position().y1
    ratio_bottom = ratio_axes[-1].get_position().y0
    fig.text(
        0.02,
        0.5 * (ratio_top + ratio_bottom),
        "ULA / VEGAS",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8.5,
    )
    # A tight bounding box varies with the width of each observable's labels,
    # which makes equally sized LaTeX panels appear to have different widths.
    # Compact workshop plots instead use a common fixed-width canvas, with a
    # small left allowance for the differential and shared ratio labels.
    if compact:
        fig_width, fig_height = fig.get_size_inches()
        compact_bbox = Bbox.from_bounds(
            -0.10, -0.08, fig_width + 0.20, fig_height - 0.17
        )
        fig.savefig(out_path, bbox_inches=compact_bbox)
    else:
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--vb", type=str, default="z", choices=["w", "z"])
parser.add_argument("--n_jets", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--mcmc_dir", type=str, default="results/cross_section_mcmc")
parser.add_argument(
    "--vegas_dir", type=str, default="results/cross_section_validation/vegas"
)
parser.add_argument("--out_dir", type=str, default="plots/cross_section_validation")
parser.add_argument(
    "--compact",
    action="store_true",
    help="Use a shorter workshop layout without changing text or legend sizes.",
)
parser.add_argument("--no_tex", action="store_true")
args = parser.parse_args()

configure_matplotlib(use_tex=not args.no_tex)
os.makedirs(args.out_dir, exist_ok=True)

loaded = []
for n_jets in args.n_jets:
    mcmc_variants = load_mcmc_variants(args.mcmc_dir, args.vb, n_jets, args.seed)
    vegas_candidates = discover_split_results(args.vegas_dir, args.vb, n_jets, args.seed)
    if len(vegas_candidates) > 1:
        paths = "\n  ".join(data["_path"] for data in vegas_candidates)
        raise RuntimeError(
            f"Found multiple VEGAS results for {args.vb}_{n_jets}j seed {args.seed}:\n"
            f"  {paths}\nPass a more specific VEGAS directory or metadata file."
        )
    vegas = vegas_candidates[0] if vegas_candidates else None
    if not mcmc_variants or vegas is None:
        print(f"Skipping {args.vb}_{n_jets}j: missing MCMC or VEGAS JSON.")
        continue
    loaded.append((n_jets, mcmc_variants, vegas))

if not loaded:
    raise RuntimeError("No matching MCMC/VEGAS validation files found.")

for observable in discover_observables(loaded):
    x_label, derivative_label, unit_label = observable_labels(observable, args.vb)
    out_path = os.path.join(args.out_dir, f"{args.vb}_{observable}_seed{args.seed}.pdf")
    log_y = not is_rapidity_like(observable)
    if plot_observable(
        loaded,
        observable,
        x_label,
        y_axis_label(args.vb, derivative_label, unit_label),
        out_path,
        args.vb,
        log_y=log_y,
        compact=args.compact,
    ):
        print(f"Saved {out_path}")
