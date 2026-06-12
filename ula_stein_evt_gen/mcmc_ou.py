import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import field
from jax import random
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class GaussianOUChainState:
    # Random state
    rng: jax.Array = field(default_factory=lambda: random.PRNGKey(0))

    # Number of samples
    n_samples: jdc.Static[int] = 1000

    # Dimension of the samples
    dim: jdc.Static[int] = 4

    # Samples
    x: jnp.ndarray = None

    # Number of completed MCMC steps
    n_steps_completed: int = field(default_factory=lambda: 0)


class GaussianOUChain:
    def __init__(self, mu_target, sigma_target, eta, mu_init, sigma_init):
        """
        Exact OU chain for a diagonal Gaussian target.

        Unlike the ULA chain, the stationary distribution is exactly the target
        with no discretization bias. Supports both Gaussian initialization
        (sigma_init > 0) and fixed-point initialization (sigma_init = 0, all
        chains start from mu_init).

        mu_init, sigma_init, mu_target, sigma_target must be one-dimensional
        arrays of shape (dim,). eta must be a scalar.
        """

        self.mu_target = jnp.asarray(mu_target)
        self.sigma_target = jnp.asarray(sigma_target)
        self.eta = jnp.asarray(eta)
        self.mu_init = jnp.asarray(mu_init)
        self.sigma_init = jnp.asarray(sigma_init)

        self._validate_params()
        self.dim = self.mu_target.shape[0]

        # Precompute drift and noise coefficients
        self._drift = jnp.exp(-self.eta / self.sigma_target**2)
        self._noise = self.sigma_target * jnp.sqrt(
            1.0 - jnp.exp(-2.0 * self.eta / self.sigma_target**2)
        )

    def _validate_params(self):
        for name, value in (
            ("mu_target", self.mu_target),
            ("sigma_target", self.sigma_target),
            ("mu_init", self.mu_init),
            ("sigma_init", self.sigma_init),
        ):
            if value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional array.")

        dim_ref = self.mu_target.shape[0]
        for name, value in (
            ("sigma_target", self.sigma_target),
            ("mu_init", self.mu_init),
            ("sigma_init", self.sigma_init),
        ):
            if value.shape[0] != dim_ref:
                raise ValueError(f"{name} has dimension {value.shape[0]}, expected {dim_ref}.")

        if self.eta.ndim != 0:
            raise ValueError("eta must be a scalar.")

        if bool(jnp.any(self.sigma_target <= 0.0)):
            raise ValueError("sigma_target must be strictly positive in every dimension.")

        if bool(jnp.any(self.sigma_init < 0.0)):
            raise ValueError("sigma_init must be non-negative in every dimension.")

        if bool(self.eta <= 0.0):
            raise ValueError("eta must be strictly positive.")

    def _target_score(self, x):
        return -(x - self.mu_target) / self.sigma_target**2

    def _gaussian_log_prob_factory(self, mu, sigma):
        log_norm = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * sigma**2))

        def log_prob(x):
            return -0.5 * jnp.sum(((x - mu) / sigma) ** 2) - log_norm

        return log_prob

    def _gaussian_density_factory(self, mu, sigma):
        log_prob = self._gaussian_log_prob_factory(mu, sigma)

        def density(x):
            return jnp.exp(log_prob(x))

        return density

    def _gaussian_score_factory(self, mu, sigma):
        variance = sigma**2

        def score(x):
            return -(x - mu) / variance

        return score

    def _ou_kernel(self, x_now, rng):
        rng, rng_normal = jax.random.split(rng)
        eps = jax.random.normal(rng_normal, shape=x_now.shape)
        x_new = (
            self.mu_target
            + self._drift * (x_now - self.mu_target)
            + self._noise * eps
        )

        return x_new, rng

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, state):
        """
        Initialize state.x by sampling from the initial Gaussian distribution.

        When sigma_init = 0, all chains start from the fixed point mu_init.
        """

        n_samples = state.n_samples
        if state.dim != self.dim:
            raise ValueError(f"state.dim is {state.dim}, expected {self.dim}.")

        rng, rng_normal = jax.random.split(state.rng)
        eps = jax.random.normal(rng_normal, shape=(n_samples, self.dim))
        x = self.mu_init[None, :] + self.sigma_init[None, :] * eps

        new_state = jdc.replace(state, x=x, rng=rng, n_steps_completed=0)

        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def run_chain(self, state, n_steps):
        """
        Perform n_steps of the exact OU transition kernel.
        """

        def ou_step(i, args):
            x_now, rng = args
            x_new, rng = self._ou_kernel(x_now, rng)

            return x_new, rng

        x, rng = jax.lax.fori_loop(
            0, n_steps, ou_step,
            (state.x, state.rng)
        )

        new_state = jdc.replace(
            state,
            rng=rng,
            x=x,
            n_steps_completed=state.n_steps_completed + n_steps
        )

        return new_state

    def exact_marginal(self, n_steps):
        """
        Exact mean and standard deviation of the OU chain after n_steps.
        """

        drift_pow = self._drift**n_steps
        drift_pow_sq = drift_pow**2

        mu = self.mu_target + drift_pow * (self.mu_init - self.mu_target)
        sigma = jnp.sqrt(
            drift_pow_sq * self.sigma_init**2
            + self.sigma_target**2 * (1.0 - drift_pow_sq)
        )

        return mu, sigma

    def stationary_marginal(self):
        """
        Mean and standard deviation of the stationary distribution.

        For the exact OU chain, this is identical to the target.
        """

        return self.mu_target, self.sigma_target

    def get_target_log_prob(self):
        return self._gaussian_log_prob_factory(self.mu_target, self.sigma_target)

    def get_target_density(self):
        return self._gaussian_density_factory(self.mu_target, self.sigma_target)

    def get_target_score(self):
        return self._gaussian_score_factory(self.mu_target, self.sigma_target)

    def get_exact_log_prob(self, n_steps):
        mu, sigma = self.exact_marginal(n_steps)

        return self._gaussian_log_prob_factory(mu, sigma)

    def get_exact_density(self, n_steps):
        mu, sigma = self.exact_marginal(n_steps)

        return self._gaussian_density_factory(mu, sigma)

    def get_exact_score(self, n_steps):
        mu, sigma = self.exact_marginal(n_steps)

        return self._gaussian_score_factory(mu, sigma)
