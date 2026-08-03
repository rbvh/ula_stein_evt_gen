import jax.numpy as jnp

from ula_stein_evt_gen.mirror import from_mirror


def harmonic_boundary(factors):
    inside = jnp.all(factors > 0.0)
    safe_factors = jnp.maximum(factors, 1e-12)
    h = factors.shape[0] / jnp.sum(1.0 / safe_factors)
    return jnp.where(inside, h, 0.0)


def make_mirror_boundary_fn(cut_factors):
    def boundary_fn(z):
        return harmonic_boundary(cut_factors(from_mirror(z)))

    return boundary_fn
