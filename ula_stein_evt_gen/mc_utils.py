"""Weight-tail statistics for positive cross-section importance samples."""
import numpy as np


def compute_unweighting_statistics(weights, quantiles=(0.01, 0.001)):
    """Measure efficiencies for nonnegative importance weights f(x)/q(x).

    Samples must come from q, including zero-weight trials that fail cuts.
    For each tail fraction a, choose the smallest observed threshold M such
    that sum(w[w > M]) / sum(w) <= a. This is a weight contribution quantile,
    not an event-count quantile or a bound on sum(max(w - M, 0)). Ties at M
    are not overweights. a=0 uses the observed maximum, not a proven bound.

    Following arXiv:2604.03511, epsilon_a = mean(w) / M. With overweights,
    actual acceptance is mean(min(1, w/M)); accepted events carry residual
    weight max(1, w/M). Both conventions and their reciprocals are returned.
    Sorting and accumulation use host float64 even when JAX uses float32.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("Weights must be a nonempty one-dimensional array.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Weights must be finite and nonnegative.")
    fractions = list(dict.fromkeys([*quantiles, 0.0]))
    if any(not np.isfinite(a) or not 0.0 <= a < 1.0 for a in fractions):
        raise ValueError("Allowed overweight integral fractions must lie in [0, 1).")
    maximum = float(np.max(weights))
    if maximum == 0.0:
        raise ValueError("At least one weight must be positive.")

    # Normalize first to avoid overflowing a sum of large finite weights.
    sorted_weights = np.sort(weights / maximum)[::-1]
    tail_cumsum = np.cumsum(sorted_weights, dtype=np.float64)
    total = tail_cumsum[-1]
    mean = total / weights.size
    result = {}
    for a in fractions:
        idx = min(int(np.searchsorted(tail_cumsum, a * total, side="right")),
                  weights.size - 1)
        threshold = sorted_weights[idx]
        # Count strict overweights, including ties at the selected threshold.
        n_over = int(np.searchsorted(-sorted_weights, -threshold, side="left"))
        over_sum = tail_cumsum[n_over - 1] if n_over else 0.0
        epsilon = mean / threshold
        acceptance = (n_over + (total - over_sum) / threshold) / weights.size
        result[float(a)] = {
            "max_weight_eff": float(threshold * maximum),
            "epsilon": float(epsilon),
            "inverse_epsilon": float(1.0 / epsilon),
            "acceptance": float(acceptance),
            "trials_per_accepted_event": float(1.0 / acceptance),
            "overweight_event_fraction": float(n_over / weights.size),
            "overweight_integral_fraction": float(over_sum / total),
        }
    return result


def compute_unweighting_efficiency(weights, quantiles=(0.01, 0.001)):
    """Return the paper convention epsilon_a = mean(w)/max_weight_eff."""
    return {a: stats["epsilon"] for a, stats in
            compute_unweighting_statistics(weights, quantiles).items()}
