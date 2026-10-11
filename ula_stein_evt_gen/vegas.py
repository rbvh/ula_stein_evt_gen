import jax
import jax.numpy as jnp
import numpy as np

from ula_stein_evt_gen.mc_utils import compute_unweighting_statistics


class VegasIntegrator:
    """
    Minimal VEGAS-style product-grid sampler for positive cross sections.

    ``integrand`` is expected to accept an ``(n, dim)`` array and return
    pointwise cross-section weights.
    """

    def __init__(self, integrand, dim, n_bins=100, alpha=1.5, do_smearing=False):
        self.integrand = integrand
        self.dim = dim
        self.n_bins = n_bins
        self.alpha = alpha
        self.do_smearing = do_smearing
        self.subdivisions = jnp.broadcast_to(
            jnp.linspace(0.0, 1.0, n_bins + 1)[None, :],
            (dim, n_bins + 1),
        )

    def sample(self, rng, n_points):
        r = jax.random.uniform(rng, shape=(n_points, self.dim))
        bin_nums = jnp.floor(r * self.n_bins).astype(jnp.int32)
        r_aux = r * self.n_bins - bin_nums

        x_min = jnp.take_along_axis(self.subdivisions, bin_nums.T, axis=1).T
        x_max = jnp.take_along_axis(self.subdivisions, (bin_nums + 1).T, axis=1).T
        delta_x = x_max - x_min
        x = x_min + delta_x * r_aux
        proposal_density = jnp.prod(1.0 / (self.n_bins * delta_x), axis=1)
        return x, proposal_density, bin_nums

    def weighted_sample(self, rng, n_points):
        x, proposal_density, bin_nums = self.sample(rng, n_points)
        f_vals = self.integrand(x)
        weights = f_vals / proposal_density
        return x, f_vals, proposal_density, weights, bin_nums

    def _refine_grid(self, bin_nums, weights_sq, eps=1e-30):
        new_subdivisions = []
        for i_dim, dim_bins in enumerate(bin_nums.T):
            bin_weights = jnp.bincount(
                dim_bins, weights=weights_sq, minlength=self.n_bins
            )
            bin_weights = jnp.maximum(bin_weights, eps)

            if self.do_smearing:
                padded = jnp.pad(bin_weights, (1, 1), mode="edge")
                bin_weights = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0

            total = jnp.sum(bin_weights)
            scaled = (
                (1.0 - bin_weights / total)
                / (jnp.log(total) - jnp.log(bin_weights))
            ) ** self.alpha
            target = jnp.mean(scaled)

            edges = [0.0]
            accumulated = 0.0
            old_bin = -1
            current = 0.0
            previous = 0.0
            for _ in range(self.n_bins - 1):
                while accumulated <= target:
                    old_bin += 1
                    accumulated += scaled[old_bin]
                    previous = current
                    current = self.subdivisions[i_dim, old_bin + 1]
                accumulated -= target
                delta = (current - previous) * accumulated / scaled[old_bin]
                edges.append(float(current - delta))
            edges.append(1.0)
            new_subdivisions.append(jnp.asarray(edges))

        self.subdivisions = jnp.stack(new_subdivisions, axis=0)

    def train(self, rng, n_points=100_000, n_epochs=10, status_fn=print):
        for epoch in range(n_epochs):
            rng, rng_sample = jax.random.split(rng)
            _, _, _, weights, bin_nums = self.weighted_sample(rng_sample, n_points)
            mean = jnp.mean(weights)
            err = jnp.std(weights, ddof=1) / jnp.sqrt(n_points)
            if status_fn is not None:
                status_fn(
                    f"VEGAS epoch {epoch + 1}/{n_epochs}: "
                    f"sigma={float(mean):.6g} +/- {float(err):.3g}"
                )
            self._refine_grid(bin_nums, weights**2)

    def integrate(self, rng, n_points):
        x, f_vals, proposal_density, weights, _ = self.weighted_sample(rng, n_points)
        mean = jnp.mean(weights)
        err = jnp.std(weights, ddof=1) / jnp.sqrt(n_points)
        return {
            "x": x,
            "f_vals": f_vals,
            "proposal_density": proposal_density,
            "weights": weights,
            "value": mean,
            "error": err,
        }

    def measure_unweighting(self, rng, n_points=10_000_000, batch_size=100_000,
                           quantiles=(0.01, 0.001), status_fn=None):
        """Survey the frozen proposal; keep only weights on the host.

        n_points counts all proposed events, including those failing cuts.
        No grid adaptation or random accept/reject step is performed here.
        """
        if n_points < 2 or batch_size < 1:
            raise ValueError("n_points must be at least 2 and batch_size positive.")
        weights = np.empty(n_points, dtype=np.float64)
        for start in range(0, n_points, batch_size):
            rng, rng_batch = jax.random.split(rng)
            stop = min(start + batch_size, n_points)
            batch_weights = self.weighted_sample(rng_batch, stop - start)[3]
            weights[start:stop] = np.asarray(jax.device_get(batch_weights), dtype=np.float64)
            if status_fn is not None:
                status_fn(f"Surveyed {stop}/{n_points} VEGAS trials.")
        efficiencies = compute_unweighting_statistics(weights, quantiles)
        return {
            "n_trials": n_points,
            "zero_weight_fraction": float(np.mean(weights == 0.0)),
            "cross_section": float(np.mean(weights)),
            "cross_section_error": float(np.std(weights, ddof=1) / np.sqrt(n_points)),
            "efficiencies": efficiencies,
        }

    def get_unweighting_efficiency(self, rng, n_points=10_000_000,
                                  quantiles=(0.01, 0.001), batch_size=100_000):
        """Return epsilon_a for a fresh, batched survey of the fitted grid."""
        survey = self.measure_unweighting(rng, n_points, batch_size, quantiles)
        return {a: stats["epsilon"] for a, stats in survey["efficiencies"].items()}
