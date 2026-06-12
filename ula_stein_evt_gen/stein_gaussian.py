import jax.numpy as jnp


def gaussian_score(x, mu, sigma):
    return -(x - mu) / sigma**2


def fisher_divergence_gaussian(mu_p, mu_q, sigma_p, sigma_q):
    """
    Fisher divergence D_F(p || q) for diagonal Gaussians p and q.

    All arguments can be scalars or arrays of shape (dim,). The result
    is a scalar (summed over dimensions).
    """

    return jnp.sum(
        (mu_p - mu_q)**2 / sigma_q**4 +
        (sigma_p**2 - sigma_q**2)**2 / (sigma_q**4 * sigma_p**2)
    )


def empirical_gaussian_stein_lsd(x, mu_p, sigma_p, mu_q, sigma_q, lamb):
    """
    Estimate E_q[||score_p(x) - score_q(x)||^2] / (2 lambda) on samples x.

    If x is sampled from a truncated q, this estimates the corresponding
    conditional expectation on the truncated support.
    """

    delta = gaussian_score(x, mu_p, sigma_p) - gaussian_score(x, mu_q, sigma_q)
    vals = jnp.sum(delta**2, axis=-1) / (2.0 * lamb)
    return jnp.mean(vals), jnp.std(vals) / jnp.sqrt(vals.shape[0])


def optimal_gaussian_stein_discrepancy(mu_p, mu_q, sigma_p, sigma_q, lamb):
    """
    Optimal regularized Stein discrepancy S_lambda^* = D_F / (4 * lambda)
    for diagonal Gaussians p and q. See 1810.03545 and 2002.05616.

    The unregularized evaluation of the optimal critic gives D_F / (2 * lambda),
    which is twice this value.
    """

    return fisher_divergence_gaussian(mu_p, mu_q, sigma_p, sigma_q) / (4.0 * lamb)
