"""
Run the learned Stein discrepancy test for the exact OU chain
and an underdamped Langevin sampler on a diagonal Gaussian target.
Results are saved to results/ou_stein/.
"""

import os
import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2

from ula_stein_evt_gen.mcmc_ou import GaussianOUChain
from ula_stein_evt_gen.mcmc_underdamped_langevin import (
    UnderdampedLangevin,
    UnderdampedLangevinState,
)
from ula_stein_evt_gen.stein_gaussian import (
    empirical_gaussian_stein_lsd,
    fisher_divergence_gaussian,
)
from ula_stein_evt_gen.stein_nn import train_stein_discrepancy, train_stein_discrepancy_kfac

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dim", type=int, default=4)
parser.add_argument("--n_samples", type=lambda x: int(float(x)), default=100_000)
parser.add_argument("--eta", type=float, default=0.5)
parser.add_argument("--lamb", type=float, default=0.1)
parser.add_argument("--num_hidden_layers", type=int, default=5)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "kfac"])
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--ou_steps", type=int, default=50)
parser.add_argument("--ula_steps", type=int, default=100)
parser.add_argument("--stein_checkpoint_patience", type=int, default=5)
parser.add_argument("--beta", type=float, default=0.8)
parser.add_argument("--cross_val", action="store_true")
parser.add_argument("--boundary_mass_cut", type=float, default=None)
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()


if args.n_samples < 1:
    raise ValueError("--n_samples must be positive.")
if args.batch_size < 1:
    raise ValueError("--batch_size must be positive.")
if args.learning_rate <= 0.0:
    raise ValueError("--learning_rate must be positive.")
if args.ou_steps < 0:
    raise ValueError("--ou_steps must be non-negative.")
if args.ula_steps < 0:
    raise ValueError("--ula_steps must be non-negative.")
if args.stein_checkpoint_patience < 1:
    raise ValueError("--stein_checkpoint_patience must be positive.")
if not 0.0 <= args.beta < 1.0:
    raise ValueError("--beta must satisfy 0 <= beta < 1.")

requested_n_samples = args.n_samples
stein_sample_multiple = 10 * args.batch_size
rounded_n_samples = (
    (args.n_samples + stein_sample_multiple - 1) // stein_sample_multiple
) * stein_sample_multiple
args.n_samples = max(rounded_n_samples, stein_sample_multiple)

if args.n_samples != requested_n_samples:
    print(
        f"Rounded n_samples from {requested_n_samples} to {args.n_samples} "
        f"to satisfy 10 * batch_size={stein_sample_multiple}."
    )

if args.boundary_mass_cut is not None:
    if not 0.0 < args.boundary_mass_cut < 1.0:
        raise ValueError("--boundary_mass_cut must be strictly between 0 and 1.")
    boundary_mass_cut = float(args.boundary_mass_cut)
    r2_cut = float(chi2.ppf(boundary_mass_cut, df=args.dim))
else:
    r2_cut = None
    boundary_mass_cut = None

has_r2_cut = r2_cut is not None

# Output directory
cut_suffix = ""
if has_r2_cut:
    cut_label = f"{boundary_mass_cut:g}".replace(".", "p").replace("-", "m")
    cut_suffix = f"_mass_cut_{cut_label}"
out_dir = args.out_dir or f"results/ou_stein/d{args.dim}_seed{args.seed}{cut_suffix}"
os.makedirs(out_dir, exist_ok=True)

rng = jax.random.PRNGKey(args.seed)

dim = args.dim
lamb = args.lamb

# Target: diagonal Gaussian with varying means and widths
mu_target = jnp.linspace(-2.0, 2.0, dim)
sigma_target = jnp.linspace(0.5, 2.0, dim)

# Initialization: fixed point far from the mean
x_star = mu_target + 3.0
sigma_init = jnp.zeros(dim)


def target_r2(x):
    z = (x - mu_target) / sigma_target
    return jnp.sum(z**2, axis=-1)


def target_score(x):
    return -(x - mu_target) / sigma_target**2


def gaussian_target_log_prob(x):
    return -0.5 * target_r2(x)


if r2_cut is not None:

    def boundary_fn(x):
        r2 = target_r2(x)
        return jnp.maximum(1.0 - r2_cut / jnp.maximum(r2, 1e-12), 0.0)

    def target_log_prob_and_score(x):
        inside = target_r2(x) > r2_cut
        log_prob = jnp.where(inside, gaussian_target_log_prob(x), -jnp.inf)
        scores = jnp.where(inside[:, None], target_score(x), 0.0)
        return log_prob, scores

else:
    boundary_fn = None

    def target_log_prob_and_score(x):
        return gaussian_target_log_prob(x), target_score(x)


def sample_ou_marginal(rng, n_samples, mu, sigma, r2_cut):
    samples = []
    n_accepted = 0
    n_proposed = 0

    while n_accepted < n_samples:
        rng, rng_proposal = jax.random.split(rng)
        eps = jax.random.normal(rng_proposal, shape=(n_samples, dim))
        x = mu[None, :] + sigma[None, :] * eps
        accepted = x if r2_cut is None else x[target_r2(x) > r2_cut]
        samples.append(accepted)
        n_accepted += int(accepted.shape[0])
        n_proposed += n_samples

    x = jnp.concatenate(samples, axis=0)[:n_samples]
    return rng, x, n_accepted / n_proposed


def eta_summary(log_eta):
    eta = np.asarray(jax.device_get(jnp.exp(log_eta)))
    if eta.ndim == 0:
        return eta.tolist(), f"{float(eta):.6f}"
    if eta.size <= 8:
        text = np.array2string(eta, precision=4, separator=", ")
    else:
        text = (
            f"min={float(np.min(eta)):.4g}, "
            f"median={float(np.median(eta)):.4g}, "
            f"max={float(np.max(eta)):.4g}"
        )
    return eta.tolist(), text


def train_stein_critic(x, scores, rng_train, boundary):
    kwargs = {
        "x": x,
        "scores": scores,
        "rng": rng_train,
        "num_hidden_layers": args.num_hidden_layers,
        "hidden_size": args.hidden_size,
        "lamb": lamb,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "cross_val": args.cross_val,
        "boundary_fn": boundary,
    }
    if args.optimizer == "adam":
        return train_stein_discrepancy(
            **kwargs,
            learning_rate=args.learning_rate,
        )
    return train_stein_discrepancy_kfac(**kwargs)


def make_stein_checkpoints(max_steps):
    checkpoints = list(range(0, max_steps + 1, args.stein_checkpoint_patience))
    if checkpoints[-1] != max_steps:
        checkpoints.append(max_steps)
    return checkpoints


# Checkpoints
ou_checkpoints = make_stein_checkpoints(args.ou_steps)
mcmc_checkpoints = make_stein_checkpoints(args.ula_steps)

# =====================================================================
# Underdamped Langevin sampler
# =====================================================================
print("=" * 60)
print("UNDERDAMPED LANGEVIN")
print("=" * 60)
optimizer_text = args.optimizer
if args.optimizer == "adam":
    optimizer_text += f"(lr={args.learning_rate:g})"
print(f"Stein optimizer: {optimizer_text}")
if has_r2_cut:
    print(
        f"Boundary cut: target r^2 > {r2_cut:.6g} "
        f"(removes {boundary_mass_cut:.3%} of target mass)"
    )

mcmc = UnderdampedLangevin(
    log_prob_and_score=target_log_prob_and_score,
    beta=args.beta,
)

mcmc_state = UnderdampedLangevinState(
    rng=jax.random.PRNGKey(args.seed + 2),
    n_samples=args.n_samples,
    dim=dim,
)

mcmc_state = mcmc.initialize(mcmc_state)
_, initial_eta_text = eta_summary(mcmc_state.log_eta)
print(f"  Initial eta: {initial_eta_text}")

mcmc_learned_lsd = []
mcmc_learned_lsd_std = []
mcmc_learned_lsd_no_boundary = []
mcmc_learned_lsd_no_boundary_std = []

prev_mcmc_steps = 0
for t in mcmc_checkpoints:
    steps_to_run = t - prev_mcmc_steps
    if steps_to_run > 0:
        mcmc_state = mcmc.run_chain(mcmc_state, steps_to_run)
    prev_mcmc_steps = t

    eta_t, eta_text = eta_summary(mcmc_state.log_eta)
    acc_t = float(jnp.exp(mcmc_state.avg_log_accept))
    esjd_t = float(mcmc_state.esjd)
    print(f"  step {t:4d}  eta={eta_text}  acc={acc_t:.4f}  " f"ESJD={esjd_t:.6f}")

    print(f"\n----- ULA: t = {t} -----")
    rng, rng_train = jax.random.split(rng)
    test_lsd, test_lsd_se = train_stein_critic(
        mcmc_state.x,
        mcmc_state.score,
        rng_train,
        boundary_fn,
    )
    mcmc_learned_lsd.append(float(test_lsd))
    mcmc_learned_lsd_std.append(float(test_lsd_se))
    print(f"  Learned LSD:           {float(test_lsd):.6f} ± {float(test_lsd_se):.6f}")

    if has_r2_cut:
        rng, rng_train_no_boundary = jax.random.split(rng)
        no_boundary_lsd, no_boundary_lsd_se = train_stein_critic(
            mcmc_state.x,
            mcmc_state.score,
            rng_train_no_boundary,
            None,
        )
        mcmc_learned_lsd_no_boundary.append(float(no_boundary_lsd))
        mcmc_learned_lsd_no_boundary_std.append(float(no_boundary_lsd_se))
        print(
            "  Learned LSD without h: "
            f"{float(no_boundary_lsd):.6f} ± {float(no_boundary_lsd_se):.6f}"
        )

# =====================================================================
# OU chain
# =====================================================================
print("\n" + "=" * 60)
print("OU CHAIN")
print("=" * 60)

chain = GaussianOUChain(
    mu_target=mu_target,
    sigma_target=sigma_target,
    eta=args.eta,
    mu_init=x_star,
    sigma_init=sigma_init,
)

analytical_lsd = []
analytical_lsd_std = []
ou_learned_lsd = []
ou_learned_lsd_std = []
ou_learned_lsd_no_boundary = []
ou_learned_lsd_no_boundary_std = []
ou_rejection_acceptance = []

for t in ou_checkpoints:
    print(f"\n----- OU: t = {t} -----")

    mu_t, sigma_t = chain.exact_marginal(t)

    rng, rng_ou = jax.random.split(rng)
    rng_ou, x_ou, accept = sample_ou_marginal(
        rng_ou, args.n_samples, mu_t, sigma_t, r2_cut
    )
    rng = rng_ou
    _, scores_ou = target_log_prob_and_score(x_ou)

    if bool(jnp.any(sigma_t <= 0.0)):
        analytical_value = np.nan
        analytical_se = np.nan
        if has_r2_cut:
            ou_rejection_acceptance.append(float(accept))
            print(f"  OU rejection acceptance: {accept:.4f}")
        print("  Analytical: singular point-mass marginal at t=0.")
    elif has_r2_cut:
        analytical_value, analytical_se = empirical_gaussian_stein_lsd(
            x_ou, mu_target, sigma_target, mu_t, sigma_t, lamb
        )
        analytical_value = float(analytical_value)
        analytical_se = float(analytical_se)
        ou_rejection_acceptance.append(float(accept))
        print(f"  OU rejection acceptance: {accept:.4f}")
        print(
            "  Analytical MC (conditional D_F / 2λ): "
            f"{analytical_value:.6f} ± {analytical_se:.6f}"
        )
    else:
        df = fisher_divergence_gaussian(mu_target, mu_t, sigma_target, sigma_t)
        analytical_value = float(df / (2.0 * lamb))
        analytical_se = 0.0
        print(f"  Analytical (D_F / 2λ): {analytical_value:.6f}")

    analytical_lsd.append(analytical_value)
    analytical_lsd_std.append(analytical_se)

    # Learned
    rng, rng_train = jax.random.split(rng)
    test_lsd, test_lsd_se = train_stein_critic(
        x_ou,
        scores_ou,
        rng_train,
        boundary_fn,
    )
    ou_learned_lsd.append(float(test_lsd))
    ou_learned_lsd_std.append(float(test_lsd_se))
    print(f"  Learned LSD:           {float(test_lsd):.6f} ± {float(test_lsd_se):.6f}")

    if has_r2_cut:
        rng, rng_train_no_boundary = jax.random.split(rng)
        no_boundary_lsd, no_boundary_lsd_se = train_stein_critic(
            x_ou,
            scores_ou,
            rng_train_no_boundary,
            None,
        )
        ou_learned_lsd_no_boundary.append(float(no_boundary_lsd))
        ou_learned_lsd_no_boundary_std.append(float(no_boundary_lsd_se))
        print(
            "  Learned LSD without h: "
            f"{float(no_boundary_lsd):.6f} ± {float(no_boundary_lsd_se):.6f}"
        )

# =====================================================================
# Save results
# =====================================================================
results = {
    "ou_checkpoints": ou_checkpoints,
    "mcmc_checkpoints": mcmc_checkpoints,
    "analytical_lsd": analytical_lsd,
    "ou_learned_lsd": ou_learned_lsd,
    "ou_learned_lsd_std": ou_learned_lsd_std,
    "mcmc_learned_lsd": mcmc_learned_lsd,
    "mcmc_learned_lsd_std": mcmc_learned_lsd_std,
    "analytical_lsd_std": analytical_lsd_std,
    "dim": dim,
    "lamb": lamb,
    "requested_n_samples": requested_n_samples,
    "n_samples": args.n_samples,
    "ou_steps": args.ou_steps,
    "ula_steps": args.ula_steps,
    "stein_checkpoint_patience": args.stein_checkpoint_patience,
    "mu_target": np.array(mu_target).tolist(),
    "sigma_target": np.array(sigma_target).tolist(),
    "mu_init": np.array(x_star).tolist(),
    "boundary_r2_cut": r2_cut,
    "boundary_mass_cut": boundary_mass_cut,
    "beta": args.beta,
    "eta_adaptation": "score_direction_scalar_esjd",
    "stein_optimizer": args.optimizer,
    "stein_learning_rate": args.learning_rate if args.optimizer == "adam" else None,
}
if has_r2_cut:
    results["ou_rejection_acceptance"] = ou_rejection_acceptance
    results["ou_learned_lsd_no_boundary"] = ou_learned_lsd_no_boundary
    results["ou_learned_lsd_no_boundary_std"] = ou_learned_lsd_no_boundary_std
    results["mcmc_learned_lsd_no_boundary"] = mcmc_learned_lsd_no_boundary
    results["mcmc_learned_lsd_no_boundary_std"] = mcmc_learned_lsd_no_boundary_std

with open(f"{out_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_dir}/results.json")
