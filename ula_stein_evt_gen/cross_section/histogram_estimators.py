import json

import numpy as np


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(val) for val in value]
    return value


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(to_jsonable(data), f, indent=2)


def _as_histogram_inputs(values, bins, weights=None):
    values = np.asarray(values)
    bins = np.asarray(bins, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(values, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(~np.isfinite(weights)):
            raise ValueError("Histogram weights must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("Normalized histogram weights must be non-negative.")

    if values.shape != weights.shape:
        raise ValueError(
            f"values and weights must have the same shape, got "
            f"{values.shape} and {weights.shape}."
        )
    return values, bins, weights


def init_histogram_accumulator(bins):
    bins = np.asarray(bins, dtype=np.float64)
    n_bins = bins.size - 1
    return {
        "bin_edges": bins,
        "sumw": np.zeros(n_bins, dtype=np.float64),
        "sumw2": np.zeros(n_bins, dtype=np.float64),
        "total_all": 0.0,
        "n": 0,
    }


def accumulate_histogram(accumulator, values, weights=None):
    values, bins, weights = _as_histogram_inputs(
        values, accumulator["bin_edges"], weights=weights
    )
    sumw, _ = np.histogram(values, bins=bins, weights=weights)
    sumw2, _ = np.histogram(values, bins=bins, weights=weights**2)
    accumulator["sumw"] += sumw
    accumulator["sumw2"] += sumw2
    accumulator["total_all"] += np.sum(weights)
    accumulator["n"] += values.size
    return accumulator


def finalize_histogram(accumulator):
    bins = np.asarray(accumulator["bin_edges"], dtype=np.float64)
    sumw = np.asarray(accumulator["sumw"], dtype=np.float64)
    sumw2 = np.asarray(accumulator["sumw2"], dtype=np.float64)
    total = np.sum(sumw)
    total_all = float(accumulator["total_all"])
    if total <= 0.0:
        raise ValueError("Histogram bin range contains no positive total weight.")

    integral = sumw / total
    integral_err = np.zeros_like(integral, dtype=np.float64)
    n = int(accumulator["n"])
    if n > 1:
        total_w2 = np.sum(sumw2)
        sum_terms2 = (1.0 - integral) ** 2 * sumw2
        sum_terms2 += integral**2 * (total_w2 - sumw2)
        integral_err = np.sqrt(np.maximum(n * sum_terms2 / ((n - 1) * total**2), 0.0))

    widths = np.diff(bins)
    return {
        "bin_edges": bins,
        "integral": integral,
        "integral_error": integral_err,
        "density": integral / widths,
        "density_error": integral_err / widths,
        "normalization": "unit_area_in_range",
        "in_range_weight_fraction": total / total_all if total_all > 0.0 else np.nan,
    }


def _normalized_histogram(values, bins, weights=None):
    """Unit-area histogram over the supplied bin range."""
    accumulator = init_histogram_accumulator(bins)
    accumulate_histogram(accumulator, values, weights=weights)
    return finalize_histogram(accumulator)


def mcmc_histogram(values, bins):
    """Normalized differential histogram for MCMC target samples."""
    return _normalized_histogram(values, bins)


def vegas_histogram(values, weights, bins):
    """Normalized differential weighted-MC histogram for VEGAS samples."""
    return _normalized_histogram(values, bins, weights=weights)


def default_observable_bins(
    pt_max=500.0,
    n_pt_bins=50,
    ht_max=1000.0,
    n_ht_bins=50,
    mass_max=1500.0,
    n_mass_bins=50,
    delta_r_max=6.0,
    n_delta_r_bins=50,
    log_xaxb_min=-12.0,
    log_xaxb_max=0.0,
    n_x_bins=50,
    kt_clustering_scale_max=300.0,
    ca_clustering_scale_max=6.0,
    n_clustering_scale_bins=50,
    y_max=5.0,
    n_y_bins=50,
):
    bins = {
        "pt_v": np.linspace(0.0, pt_max, n_pt_bins + 1),
        "y_v": np.linspace(-y_max, y_max, n_y_bins + 1),
        "h_t": np.linspace(0.0, ht_max, n_ht_bins + 1),
        "y_j1": np.linspace(-y_max, y_max, n_y_bins + 1),
        "max_abs_y_j": np.linspace(0.0, y_max, n_y_bins + 1),
        "min_deltaR_Zj": np.linspace(0.0, delta_r_max, n_delta_r_bins + 1),
        "min_deltaR_jj": np.linspace(0.0, delta_r_max, n_delta_r_bins + 1),
        "x_a": np.linspace(0.0, 1.0, n_x_bins + 1),
        "x_b": np.linspace(0.0, 1.0, n_x_bins + 1),
        "log_xaxb": np.linspace(log_xaxb_min, log_xaxb_max, n_x_bins + 1),
        "y_x": np.linspace(-y_max, y_max, n_y_bins + 1),
        "m_jj": np.linspace(0.0, mass_max, n_mass_bins + 1),
        "m_all_jets": np.linspace(0.0, mass_max, n_mass_bins + 1),
    }
    for i_jet in range(5):
        bins[f"pt_j{i_jet + 1}"] = np.linspace(0.0, pt_max, n_pt_bins + 1)
    for i_scale in range(5):
        bins[f"d{i_scale}{i_scale + 1}"] = np.linspace(
            0.0, kt_clustering_scale_max, n_clustering_scale_bins + 1
        )
        bins[f"y{i_scale}{i_scale + 1}"] = np.linspace(
            0.0, ca_clustering_scale_max, n_clustering_scale_bins + 1
        )
    return bins
