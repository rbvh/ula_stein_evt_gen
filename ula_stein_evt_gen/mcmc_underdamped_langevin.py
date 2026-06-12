import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial
from dataclasses import field
from jax import random
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class UnderdampedLangevinState:
    # Random state
    rng: jax.Array = field(default_factory=lambda: random.PRNGKey(0))

    # Number of samples
    n_samples: jdc.Static[int] = 1000

    # Dimension of the samples
    dim: jdc.Static[int] = 4

    # Positions
    x: jnp.ndarray = None

    # Velocities
    v: jnp.ndarray = None

    # Log-density and score at current positions
    log_prob: jnp.ndarray = None
    score: jnp.ndarray = None

    # Number of completed MCMC steps
    n_steps_completed: int = field(default_factory=lambda: 0)

    # Average acceptance probability
    avg_log_accept: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))

    # Coordinate-wise proposal scales, represented as
    # log_eta = log_epsilon + log_direction with mean(log_direction)=0.
    log_eta: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))

    # Expected squared jump distance (ESJD) from last step
    esjd: jnp.ndarray = field(default_factory=lambda: jnp.array(0.0))


class UnderdampedLangevin:
    def __init__(self, log_prob_and_score, beta=0.8):
        """
        Metropolis-adjusted underdamped Langevin sampler with partial momentum
        refreshment, scalar ESJD scale adaptation, and determinant-one
        score-based diagonal preconditioning.

        beta: momentum retention coefficient (0 = MALA, 1 = HMC).

        log_prob_and_score: callable, maps x (n, d) to
                            (log_prob: (n,), score: (n, d)).
        """
        self.log_prob_and_score = log_prob_and_score
        self.beta = beta

    def _eta_vector(self, log_eta, dim):
        eta = jnp.exp(log_eta)
        if eta.ndim == 0:
            eta = jnp.full((dim,), eta)
        return eta

    def _log_eta_vector(self, log_eta, dim):
        log_eta = jnp.asarray(log_eta)
        if log_eta.ndim == 0:
            log_eta = jnp.full((dim,), log_eta)
        return log_eta

    def _split_scale_direction(self, log_eta, dim):
        log_eta = self._log_eta_vector(log_eta, dim)
        log_scale = jnp.mean(log_eta)
        log_direction = log_eta - log_scale
        return log_scale, log_direction

    def _score_log_direction(self, score):
        score_second_moment = jnp.mean(score**2, axis=0)
        mean_second_moment = jnp.mean(score_second_moment)
        floor = jnp.maximum(
            1e-6 * mean_second_moment,
            jnp.finfo(score_second_moment.dtype).tiny,
        )
        raw_log_direction = -0.5 * jnp.log(jnp.maximum(score_second_moment, floor))
        return raw_log_direction - jnp.mean(raw_log_direction)

    def _tame_score(self, score, eta):
        """Tame target scores to avoid large jumps (Brosse et al. 1710.05559)."""
        return score / (1.0 + eta * jnp.abs(score))

    def _underdamped_kernel(self, x_now, v_now, log_prob_now, score_now, eta, rng):
        """
        One step of Metropolis-adjusted underdamped Langevin:
          1. Half-step velocity update (kick)
          2. Full-step position update (drift)
          3. Half-step velocity update (kick)
          4. Metropolis accept/reject (before refreshment)
          5. Partial momentum refreshment

        eta: (d,) array of coordinate-wise proposal scales.
        """
        beta = self.beta

        score_now_tamed = self._tame_score(score_now, eta[None, :])

        # Leapfrog: half-step kick
        v_half = v_now + 0.5 * eta[None, :] * score_now_tamed

        # Leapfrog: full-step drift
        x_prop = x_now + eta[None, :] * v_half

        # New score
        log_prob_prop, score_prop = self.log_prob_and_score(x_prop)
        prop_finite = jnp.isfinite(log_prob_prop) & jnp.all(jnp.isfinite(score_prop), axis=-1)
        log_prob_prop_safe = jnp.where(prop_finite, log_prob_prop, -jnp.inf)
        score_prop_safe = jnp.where(prop_finite[:, None], score_prop, 0.0)
        score_prop_tamed = self._tame_score(score_prop_safe, eta[None, :])

        # Leapfrog: half-step kick
        v_prop = v_half + 0.5 * eta[None, :] * score_prop_tamed

        # --- Metropolis step (before refreshment) ---
        # K(v) = (1/2) v^T v
        kinetic_now = 0.5 * jnp.sum(v_now**2, axis=-1)
        kinetic_prop = 0.5 * jnp.sum(v_prop**2, axis=-1)

        # H = -log_prob + kinetic
        delta_H = (kinetic_prop - log_prob_prop_safe) - (kinetic_now - log_prob_now)
        log_accept = jnp.minimum(-delta_H, jnp.zeros_like(log_prob_now))
        log_accept = jnp.where(prop_finite, log_accept, -jnp.inf)

        # Accept/reject
        rng, rng_accept = jax.random.split(rng)
        log_uniform = jnp.log(jax.random.uniform(rng_accept, shape=(x_now.shape[0],)))
        acc_mask = log_uniform < log_accept

        x_out = jnp.where(acc_mask[:, None], x_prop, x_now)
        log_prob_out = jnp.where(acc_mask, log_prob_prop_safe, log_prob_now)
        score_out = jnp.where(acc_mask[:, None], score_prop_safe, score_now)
        # On reject, negate velocity (reversal)
        v_pre_refresh = jnp.where(acc_mask[:, None], v_prop, -v_now)

        # --- Partial momentum refreshment ---
        # Noise ~ N(0, I)
        rng, rng_refresh = jax.random.split(rng)
        noise = jax.random.normal(rng_refresh, shape=v_now.shape)
        v_out = beta * v_pre_refresh + jnp.sqrt(1 - beta**2) * noise

        info = {
            "x_prop": x_prop,
            "v_prop": v_prop,
            "log_prob_prop": log_prob_prop,
            "score_now_tamed": score_now_tamed,
            "score_now": score_now,
            "score_prop_tamed": score_prop_tamed,
            "score_prop": score_prop_safe,
            "delta_H": delta_H,
        }
        return x_out, v_out, log_prob_out, score_out, log_accept, rng, info

    def _esjd_log_eta_gradient(
        self,
        x_now,
        v_now,
        x_prop,
        v_prop,
        score_now_tamed,
        score_prop_tamed,
        score_prop,
        delta_H,
        accept_prob,
        eta,
    ):
        """
        Analytical frozen-score gradient d(expected_ESJD)/d(log_eta).

        Uses the frozen-score approximation: treats the target scores as
        constants w.r.t. eta. The taming function T(s, eta) introduces
        additional eta-dependence, which is differentiated exactly.

        All position/velocity inputs are (n, d) arrays.
        delta_H, accept_prob are (n,) arrays. eta is a (d,) array.
        """
        eta = eta[None, :]

        step_now = eta * score_now_tamed
        step_prop = eta * score_prop_tamed

        # d[eta_i T_i(score, eta_i)] / d(log eta_i)
        step_now_dot = step_now * (1.0 - eta * jnp.abs(score_now_tamed))
        step_prop_dot = step_prop * (1.0 - eta * jnp.abs(score_prop_tamed))

        # Frozen-score derivatives w.r.t. each log eta_i.
        x_dot = eta * v_now + 0.5 * eta * (step_now + step_now_dot)
        v_dot = 0.5 * (step_now_dot + step_prop_dot)

        # Displacement and its derivative
        displacement = x_prop - x_now
        sq_jump = jnp.sum(displacement**2, axis=-1)
        sq_jump_dot = 2.0 * displacement * x_dot

        # Hamiltonian error derivative
        # d(delta_H)/d(log eta_i):
        #   d(log p(x_prop))/d(log eta_i) = score_prop_i * x_dot_i
        #   d(K(v_prop))/d(log eta_i) = v_prop_i * v_dot_i
        hamiltonian_dot = -score_prop * x_dot + v_prop * v_dot

        # Per-chain derivative: d(accept_prob_i * sq_jump_i)/d_eta.
        correction = jnp.where(
            delta_H[:, None] > 0,
            hamiltonian_dot * sq_jump[:, None],
            0.0,
        )
        d_esjd_d_log_eta = accept_prob[:, None] * (sq_jump_dot - correction)

        return jnp.mean(d_esjd_d_log_eta, axis=0)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _initialize_x(self, state, log_thresh=-20.0, max_iteration=1000):
        """
        Initialize state.x as standard normal, resampling points with
        log-prob below threshold.
        """
        n_samples = state.n_samples
        dim = state.dim

        def initialize_x_condition(args):
            iteration, x_now, _ = args
            log_prob, _ = self.log_prob_and_score(x_now)
            max_log_prob = jnp.max(log_prob)
            mask = log_prob < (max_log_prob + log_thresh)
            return jnp.logical_and(jnp.any(mask), iteration < max_iteration)

        def initialize_x_body(args):
            iteration, x_now, rng = args
            log_prob, _ = self.log_prob_and_score(x_now)
            max_log_prob = jnp.max(log_prob)
            mask = log_prob < (max_log_prob + log_thresh)
            rng, rng_gauss = jax.random.split(rng)
            x_replace = jax.random.normal(rng_gauss, shape=x_now.shape)
            x_next = jnp.where(mask[:, None], x_replace, x_now)
            return (iteration + 1, x_next, rng)

        rng, rng_gauss = jax.random.split(state.rng)
        x_init = jax.random.normal(rng_gauss, shape=(n_samples, dim))

        _, x, rng = jax.lax.while_loop(
            initialize_x_condition, initialize_x_body, (0, x_init, rng)
        )

        # Initialize velocities ~ N(0, I)
        rng, rng_v = jax.random.split(rng)
        v = jax.random.normal(rng_v, shape=(n_samples, dim))
        log_prob, score = self.log_prob_and_score(x)

        new_state = jdc.replace(
            state, x=x, v=v, log_prob=log_prob, score=score, rng=rng
        )
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def _initialize_eta(self, state):
        """
        Find an initial shared coordinate scale by doubling/halving until
        acceptance is ~50%, then initialize all eta_i to that value.
        Algorithm 4 in Hoffman & Gelman (2014).
        """
        rng = state.rng
        x_init = state.x
        v_init = state.v

        eta = 0.5
        eta_vec = jnp.full((state.dim,), eta)

        _, _, _, _, log_accept, rng, _ = self._underdamped_kernel(
            x_init, v_init, state.log_prob, state.score, eta_vec, rng
        )
        avg_log_accept = logsumexp(log_accept) - jnp.log(x_init.shape[0])

        a = jax.lax.cond(avg_log_accept > -jnp.log(2), lambda _: 1, lambda _: -1, None)

        def condition(args):
            _, _, _, _, log_acc_val, a_val, _ = args
            return a_val * log_acc_val > -a_val * jnp.log(2)

        def body(args):
            x, log_prob, score, eta_val, _, a_val, rng_val = args
            eta_val = jax.lax.cond(
                a_val == 1, lambda e: e * 2.0, lambda e: e / 2.0, eta_val
            )
            eta_vec = jnp.full((state.dim,), eta_val)
            rng_val, rng_v = jax.random.split(rng_val)
            v_test = jax.random.normal(rng_v, shape=x.shape)

            _, _, _, _, log_acc, rng_new, _ = self._underdamped_kernel(
                x, v_test, log_prob, score, eta_vec, rng_val
            )
            avg = logsumexp(log_acc) - jnp.log(x.shape[0])
            return x, log_prob, score, eta_val, avg, a_val, rng_new

        x_init, _, _, eta_final, avg_log_accept, a, rng = jax.lax.while_loop(
            condition,
            body,
            (x_init, state.log_prob, state.score, eta, avg_log_accept, a, rng),
        )

        new_state = jdc.replace(
            state,
            rng=rng,
            avg_log_accept=avg_log_accept,
            log_eta=jnp.log(eta_final) + self._score_log_direction(state.score),
        )
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, state):
        """Initialize the state of the MCMC chain."""
        state = self._initialize_x(state)
        state = self._initialize_eta(state)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def initialize_eta(self, state):
        """Initialize only the proposal scales for an existing chain state."""
        return self._initialize_eta(state)

    @partial(jax.jit, static_argnums=(0,))
    def run_chain(self, state, n_steps):
        """
        Run n_steps of Metropolis-adjusted underdamped Langevin with continuous
        scale-shape adaptation.

        The geometric-mean scale is updated by the frozen-score ESJD gradient.
        The determinant-one direction is updated toward the diagonal
        score-covariance preconditioner with the same diminishing
        Robbins-Monro schedule.
        """
        # Robbins-Monro schedule: beta_t = beta_0 / (t_0 + t)^gamma
        beta_0 = 1.0
        t_0 = 4.0
        gamma = 0.75

        def step(i, args):
            x_now, v_now, log_prob_now, score_now, log_eta, _, _, rng = args

            log_scale, log_direction = self._split_scale_direction(log_eta, state.dim)
            log_eta = log_scale + log_direction
            eta = self._eta_vector(log_eta, state.dim)

            # --- Single MCMC step ---
            rng, rng_step = jax.random.split(rng)
            (
                x_out,
                v_out,
                log_prob_out,
                score_out,
                log_accept,
                _,
                info,
            ) = self._underdamped_kernel(
                x_now,
                v_now,
                log_prob_now,
                score_now,
                eta,
                rng_step,
            )

            # --- Analytical ESJD derivative (essentially free) ---
            accept_prob = jnp.exp(log_accept)
            esjd = jnp.mean(
                accept_prob * jnp.sum((info["x_prop"] - x_now) ** 2, axis=-1)
            )
            esjd_log_eta_gradient = self._esjd_log_eta_gradient(
                x_now,
                v_now,
                info["x_prop"],
                info["v_prop"],
                info["score_now_tamed"],
                info["score_prop_tamed"],
                info["score_prop"],
                info["delta_H"],
                accept_prob,
                eta,
            )
            # Normalise to d log(ESJD) / d log(epsilon), where changing
            # log(epsilon) shifts all log(eta_i) together at fixed direction.
            esjd_log_scale_gradient = jnp.sum(esjd_log_eta_gradient)
            esjd_log_scale_gradient = esjd_log_scale_gradient / jnp.maximum(esjd, 1e-8)
            esjd_log_scale_gradient = jnp.clip(esjd_log_scale_gradient, -5.0, 5.0)

            # --- Robbins-Monro schedule ---
            t = state.n_steps_completed + i + 1
            beta_t = beta_0 / jnp.power(t_0 + t, gamma)

            # --- Update scale and determinant-one direction separately ---
            log_scale_new = log_scale + beta_t * esjd_log_scale_gradient
            target_log_direction = self._score_log_direction(score_out)
            log_direction_new = log_direction + beta_t * (
                target_log_direction - log_direction
            )
            log_direction_new = log_direction_new - jnp.mean(log_direction_new)
            log_eta_new = log_scale_new + log_direction_new

            avg_log_accept = logsumexp(log_accept) - jnp.log(x_now.shape[0])

            return (
                x_out,
                v_out,
                log_prob_out,
                score_out,
                log_eta_new,
                avg_log_accept,
                esjd,
                rng,
            )

        log_eta_init = self._log_eta_vector(state.log_eta, state.dim)

        x, v, log_prob, score, log_eta, avg_log_accept, esjd, rng = jax.lax.fori_loop(
            0,
            n_steps,
            step,
            (
                state.x,
                state.v,
                state.log_prob,
                state.score,
                log_eta_init,
                jnp.array(0.0),
                jnp.array(0.0),
                state.rng,
            ),
        )

        new_state = jdc.replace(
            state,
            rng=rng,
            x=x,
            v=v,
            log_prob=log_prob,
            score=score,
            avg_log_accept=avg_log_accept,
            log_eta=log_eta,
            esjd=esjd,
            n_steps_completed=state.n_steps_completed + n_steps,
        )
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def run_chain_fixed(self, state, n_steps):
        """
        Run n_steps with fixed step size (no adaptation).
        """

        def step(i, args):
            x_now, v_now, log_prob_now, score_now, _, rng = args
            rng, rng_step = jax.random.split(rng)
            eta = self._eta_vector(state.log_eta, state.dim)

            x_out, v_out, log_prob_out, score_out, log_accept, _, _ = (
                self._underdamped_kernel(
                    x_now, v_now, log_prob_now, score_now, eta, rng_step
                )
            )

            avg_log_accept = logsumexp(log_accept) - jnp.log(x_now.shape[0])
            return (x_out, v_out, log_prob_out, score_out, avg_log_accept, rng)

        x, v, log_prob, score, avg_log_accept, rng = jax.lax.fori_loop(
            0,
            n_steps,
            step,
            (state.x, state.v, state.log_prob, state.score, jnp.array(0.0), state.rng),
        )

        new_state = jdc.replace(
            state,
            rng=rng,
            x=x,
            v=v,
            log_prob=log_prob,
            score=score,
            avg_log_accept=avg_log_accept,
            n_steps_completed=state.n_steps_completed + n_steps,
        )
        return new_state
