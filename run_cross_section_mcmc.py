import argparse
import math
import os
import time
import warnings

import ula_stein_evt_gen.jax_bootstrap  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from ula_stein_evt_gen.cross_section.boundaries import make_mirror_boundary_fn
from ula_stein_evt_gen.cross_section.cross_section import build_cross_section_log_prob_and_score
from ula_stein_evt_gen.cross_section.histogram_estimators import (
    accumulate_histogram,
    default_observable_bins,
    finalize_histogram,
    init_histogram_accumulator,
    write_json,
)
from ula_stein_evt_gen.cross_section.observables import compute_observables
from ula_stein_evt_gen.cross_section.surrogate import (
    build_surrogate_log_prob_and_score,
    load_surrogate,
)
from ula_stein_evt_gen.mcmc_underdamped_langevin import (
    UnderdampedLangevin,
    UnderdampedLangevinState,
)
from ula_stein_evt_gen.mirror import from_mirror
from ula_stein_evt_gen.stein_nn import train_stein_discrepancy, train_stein_discrepancy_kfac

warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

RUN_START = time.perf_counter()
DEFAULT_BETA = 0.8


def status(message):
    elapsed = time.perf_counter() - RUN_START
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def safe_float_label(value):
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def beta_output_suffix(beta):
    parts = []
    if not math.isclose(beta, DEFAULT_BETA):
        parts.append(f"beta_{safe_float_label(beta)}")
    return f"_{'_'.join(parts)}" if parts else ""


def surrogate_output_suffix(surrogate_path):
    if surrogate_path is None:
        return ""
    return "_surrogate"


def eta_summary(log_eta):
    eta = np.asarray(jax.device_get(jnp.exp(log_eta)))
    if eta.ndim == 0:
        value = float(eta)
        return value, f"{value:.6g}"
    values = eta.tolist()
    if eta.size <= 8:
        text = np.array2string(eta, precision=4, separator=", ")
    else:
        text = (
            f"min={float(np.min(eta)):.4g}, "
            f"median={float(np.median(eta)):.4g}, "
            f"max={float(np.max(eta)):.4g}"
        )
    return values, text


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)

# Physics parameters
parser.add_argument("--vb", type=str, default="z", choices=["w", "z"])
parser.add_argument("--n_jets", type=int, default=1)
parser.add_argument("--sqrt_s", type=float, default=13000.0)
parser.add_argument("--pt_cut", type=float, default=20.0)
parser.add_argument("--y_cut", type=float, default=5.0)
parser.add_argument("--deltar_cut", type=float, default=0.4)

# MCMC parameters
parser.add_argument("--n_samples", type=lambda x: int(float(x)), default=1_000_000)
parser.add_argument(
    "--mcmc_steps",
    type=int,
    default=300,
    help=(
        "Maximum checkpointed ULA steps when Stein is enabled, or fixed ULA steps when Stein is disabled."
    ),
)
parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
parser.add_argument("--cross_section_chunk_size", type=int, default=1024)

# Optional surrogate warm start
parser.add_argument(
    "--surrogate_path",
    type=str,
    default=None,
    help="Optional trained surrogate pickle used for a cheap ULA warm start.",
)
parser.add_argument(
    "--surrogate_steps",
    type=int,
    default=500,
    help="Number of ULA steps to run on the surrogate before switching to the exact target.",
)
parser.add_argument(
    "--surrogate_chunk_size",
    type=int,
    default=8192,
    help="Chunk size for surrogate log-probability and score evaluations.",
)

# Stein parameters
parser.add_argument(
    "--stein",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable learned Stein discrepancy evaluations.",
)
parser.add_argument("--stein_checkpoint_patience", type=int, default=5)
parser.add_argument("--stein_lamb", type=float, default=0.1)
parser.add_argument(
    "--stein_optimizer",
    type=str,
    default="adam",
    choices=["kfac", "adam"],
)
parser.add_argument("--stein_learning_rate", type=float, default=1e-3)
parser.add_argument("--stein_num_hidden_layers", type=int, default=5)
parser.add_argument("--stein_hidden_size", type=int, default=128)
parser.add_argument("--stein_batch_size", type=int, default=8192)
parser.add_argument("--stein_training_patience", type=int, default=3)
parser.add_argument("--stein_cross_val", action="store_true")

# Observable parameters
parser.add_argument("--post_convergence_fraction", type=float, default=0.5)
parser.add_argument("--post_convergence_min_steps", type=int, default=50)
parser.add_argument(
    "--observable_batch_size", type=lambda x: int(float(x)), default=100_000
)
parser.add_argument("--pt_max", type=float, default=500.0)
parser.add_argument("--n_pt_bins", type=int, default=50)
parser.add_argument("--ht_max", type=float, default=1000.0)
parser.add_argument("--n_ht_bins", type=int, default=50)
parser.add_argument("--mass_max", type=float, default=1500.0)
parser.add_argument("--n_mass_bins", type=int, default=50)
parser.add_argument("--delta_r_max", type=float, default=6.0)
parser.add_argument("--n_delta_r_bins", type=int, default=50)
parser.add_argument("--log_xaxb_min", type=float, default=-12.0)
parser.add_argument("--log_xaxb_max", type=float, default=0.0)
parser.add_argument("--n_x_bins", type=int, default=50)
parser.add_argument("--kt_clustering_scale_max", type=float, default=300.0)
parser.add_argument("--ca_clustering_scale_max", type=float, default=6.0)
parser.add_argument("--n_clustering_scale_bins", type=int, default=50)
parser.add_argument("--y_max", type=float, default=None)
parser.add_argument("--n_y_bins", type=int, default=50)
parser.add_argument("--jet_radius", type=float, default=1.0)
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()

if args.n_samples < 1:
    raise ValueError("--n_samples must be positive.")
if args.mcmc_steps < 1:
    raise ValueError("--mcmc_steps must be positive.")
if not 0.0 <= args.beta < 1.0:
    raise ValueError("--beta must satisfy 0 <= beta < 1.")
if args.cross_section_chunk_size is None or args.cross_section_chunk_size < 1:
    raise ValueError("--cross_section_chunk_size must be positive.")
if args.surrogate_steps < 0:
    raise ValueError("--surrogate_steps must be non-negative.")
if args.surrogate_chunk_size < 1:
    raise ValueError("--surrogate_chunk_size must be positive.")
if args.stein:
    if args.stein_checkpoint_patience < 1:
        raise ValueError("--stein_checkpoint_patience must be positive.")
    if args.stein_lamb <= 0.0:
        raise ValueError("--stein_lamb must be positive.")
    if args.stein_learning_rate <= 0.0:
        raise ValueError("--stein_learning_rate must be positive.")
    if args.stein_batch_size < 1:
        raise ValueError("--stein_batch_size must be positive.")
    if args.stein_training_patience < 1:
        raise ValueError("--stein_training_patience must be positive.")
if args.observable_batch_size < 1:
    raise ValueError("--observable_batch_size must be positive.")
if args.post_convergence_fraction < 0.0:
    raise ValueError("--post_convergence_fraction must be non-negative.")
if args.post_convergence_min_steps < 0:
    raise ValueError("--post_convergence_min_steps must be non-negative.")

requested_n_samples = args.n_samples
if args.stein:
    stein_sample_multiple = 10 * args.stein_batch_size
    rounded_n_samples = (
        (args.n_samples + stein_sample_multiple - 1) // stein_sample_multiple
    ) * stein_sample_multiple
    args.n_samples = max(rounded_n_samples, stein_sample_multiple)
    if args.n_samples != requested_n_samples:
        status(
            f"Rounded n_samples from {requested_n_samples} to {args.n_samples} "
            f"to satisfy 10 * stein_batch_size={stein_sample_multiple}."
        )

process_label = f"{args.vb}_{args.n_jets}j"
output_label = (
    f"{process_label}{beta_output_suffix(args.beta)}"
    f"{surrogate_output_suffix(args.surrogate_path)}"
)
out_dir = args.out_dir or f"results/cross_section_mcmc/{output_label}_seed{args.seed}"
os.makedirs(out_dir, exist_ok=True)

rng = jax.random.PRNGKey(args.seed)

status("Building cross-section log-probability and score.")
log_prob_and_score_mirror, cut_factors_base, dim = (
    build_cross_section_log_prob_and_score(
        sqrt_s=args.sqrt_s,
        vb=args.vb,
        n_jets=args.n_jets,
        pt_cut=args.pt_cut,
        y_cut=args.y_cut,
        deltar_cut=args.deltar_cut,
        mirror=True,
        chunk_size=args.cross_section_chunk_size,
    )
)
status(
    f"Built cut cross section for {args.vb.upper()}+{args.n_jets}j "
    f"(dim={dim}, chunk_size={args.cross_section_chunk_size})."
)
status(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

boundary_fn_mirror = make_mirror_boundary_fn(cut_factors_base)


@jax.jit
def summarize_batch(z, log_prob):
    finite = jnp.isfinite(log_prob)
    safe_log_prob = jnp.where(finite, log_prob, 0.0)
    h_vals = jax.vmap(boundary_fn_mirror)(z)
    return (
        jnp.sum(finite),
        jnp.sum(safe_log_prob),
        jnp.max(jnp.where(finite, log_prob, -jnp.inf)),
        jnp.sum(h_vals),
        jnp.min(h_vals),
    )


def summarize_samples(z, log_prob, batch_size):
    n_total = int(z.shape[0])
    n_finite = 0
    sum_log_prob = 0.0
    max_log_prob = -np.inf
    sum_h = 0.0
    min_h = np.inf

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch_stats = jax.device_get(
            summarize_batch(z[start:stop], log_prob[start:stop])
        )
        (
            batch_n_finite,
            batch_sum_log_prob,
            batch_max_log_prob,
            batch_sum_h,
            batch_min_h,
        ) = batch_stats
        n_finite += int(batch_n_finite)
        sum_log_prob += float(batch_sum_log_prob)
        max_log_prob = max(max_log_prob, float(batch_max_log_prob))
        sum_h += float(batch_sum_h)
        min_h = min(min_h, float(batch_min_h))

    return {
        "finite_fraction": n_finite / n_total,
        "mean_log_prob": sum_log_prob / max(n_finite, 1),
        "max_log_prob": max_log_prob,
        "mean_boundary_h": sum_h / n_total,
        "min_boundary_h": min_h,
    }


def compute_histograms(state):
    bins = default_observable_bins(
        pt_max=args.pt_max,
        n_pt_bins=args.n_pt_bins,
        ht_max=args.ht_max,
        n_ht_bins=args.n_ht_bins,
        mass_max=args.mass_max,
        n_mass_bins=args.n_mass_bins,
        delta_r_max=args.delta_r_max,
        n_delta_r_bins=args.n_delta_r_bins,
        log_xaxb_min=args.log_xaxb_min,
        log_xaxb_max=args.log_xaxb_max,
        n_x_bins=args.n_x_bins,
        kt_clustering_scale_max=args.kt_clustering_scale_max,
        ca_clustering_scale_max=args.ca_clustering_scale_max,
        n_clustering_scale_bins=args.n_clustering_scale_bins,
        y_max=args.y_cut if args.y_max is None else args.y_max,
        n_y_bins=args.n_y_bins,
    )

    status("Mapping terminal MCMC samples out of mirror space.")
    x = from_mirror(state.x)
    status(f"Computing observables in batches of {args.observable_batch_size}.")
    observables = compute_observables(
        x,
        vb=args.vb,
        n_jets=args.n_jets,
        sqrt_s=args.sqrt_s,
        pt_cut=args.pt_cut,
        y_cut=args.y_cut,
        jet_radius=args.jet_radius,
        batch_size=args.observable_batch_size,
    )

    histogram_accumulators = {}
    for name, values in observables.items():
        if name not in bins:
            continue
        if name not in histogram_accumulators:
            histogram_accumulators[name] = init_histogram_accumulator(bins[name])
        accumulate_histogram(histogram_accumulators[name], values)

    return {
        name: finalize_histogram(accumulator)
        for name, accumulator in histogram_accumulators.items()
    }


chunks_per_score_call = (
    args.n_samples + args.cross_section_chunk_size - 1
) // args.cross_section_chunk_size
diagnostic_batch_size = min(
    args.stein_batch_size if args.stein else args.observable_batch_size,
    args.n_samples,
)

print("=" * 60)
print("DIRECT CROSS-SECTION UNDERDAMPED LANGEVIN")
print("=" * 60)
print(f"process={args.vb.upper()}+{args.n_jets}j  dim={dim}")
print(
    f"sqrt_s={args.sqrt_s:g} GeV  pt_cut={args.pt_cut:g} GeV  "
    f"|y|<{args.y_cut:g}  deltaR>{args.deltar_cut:g}"
)
print(
    f"n_samples={args.n_samples}  beta={args.beta:g}  "
    f"stein={args.stein}  observables=True"
)
if args.stein:
    stein_optimizer_text = args.stein_optimizer
    if args.stein_optimizer == "adam":
        stein_optimizer_text += f"(lr={args.stein_learning_rate:g})"
    print(f"stein_optimizer={stein_optimizer_text}")
if args.surrogate_path is not None:
    print(
        f"surrogate warm start={args.surrogate_path}  "
        f"surrogate_steps={args.surrogate_steps}  "
        f"surrogate_chunk_size={args.surrogate_chunk_size}"
    )
print(
    f"cross-section score chunk_size={args.cross_section_chunk_size}  "
    f"chunks/score_call={chunks_per_score_call}"
)

mcmc = UnderdampedLangevin(
    log_prob_and_score=log_prob_and_score_mirror,
    beta=args.beta,
)
surrogate_summary = None
surrogate_eta = None
surrogate_acceptance = None
surrogate_esjd = None

if args.surrogate_path is None:
    status("Initializing ULA chains.")
    mcmc_state = UnderdampedLangevinState(
        rng=jax.random.PRNGKey(args.seed + 2),
        n_samples=args.n_samples,
        dim=dim,
    )
    mcmc_state = mcmc.initialize(mcmc_state)
    mcmc_state.x.block_until_ready()
    status("ULA initialization complete.")
else:
    status(f"Loading surrogate warm-start model from {args.surrogate_path}.")
    surrogate_payload = load_surrogate(args.surrogate_path)
    surrogate_dim = int(surrogate_payload["dim"])
    if surrogate_dim != dim:
        raise ValueError(
            f"Surrogate dimension ({surrogate_dim}) does not match exact "
            f"cross-section dimension ({dim})."
        )
    if "vb" in surrogate_payload and surrogate_payload["vb"] != args.vb:
        raise ValueError(
            f"Surrogate vb={surrogate_payload['vb']} does not match --vb={args.vb}."
        )
    if (
        "n_jets" in surrogate_payload
        and int(surrogate_payload["n_jets"]) != args.n_jets
    ):
        raise ValueError(
            f"Surrogate n_jets={surrogate_payload['n_jets']} does not match "
            f"--n_jets={args.n_jets}."
        )

    surrogate_log_prob_and_score = build_surrogate_log_prob_and_score(
        surrogate_payload,
        cut_factors_base=cut_factors_base,
        chunk_size=args.surrogate_chunk_size,
    )
    surrogate_mcmc = UnderdampedLangevin(
        log_prob_and_score=surrogate_log_prob_and_score,
        beta=args.beta,
    )
    status("Initializing surrogate ULA chains.")
    surrogate_state = UnderdampedLangevinState(
        rng=jax.random.PRNGKey(args.seed + 2),
        n_samples=args.n_samples,
        dim=dim,
    )
    surrogate_state = surrogate_mcmc.initialize(surrogate_state)
    surrogate_state.x.block_until_ready()
    status("Surrogate ULA initialization complete.")

    if args.surrogate_steps > 0:
        status(f"Running surrogate ULA warm start for {args.surrogate_steps} steps.")
        surrogate_state = surrogate_mcmc.run_chain(
            surrogate_state,
            args.surrogate_steps,
        )
        surrogate_state.x.block_until_ready()
        status("Surrogate ULA warm start complete.")

    surrogate_eta, surrogate_eta_text = eta_summary(surrogate_state.log_eta)
    surrogate_acceptance = float(jnp.exp(surrogate_state.avg_log_accept))
    surrogate_esjd = float(surrogate_state.esjd)
    surrogate_summary = summarize_samples(
        surrogate_state.x,
        surrogate_state.log_prob,
        diagnostic_batch_size,
    )
    print(
        "  Surrogate terminal state: "
        f"eta={surrogate_eta_text}  acc={surrogate_acceptance:.4f}  "
        f"ESJD={surrogate_esjd:.6g}  "
        f"finite={surrogate_summary['finite_fraction']:.3f}  "
        f"mean h={surrogate_summary['mean_boundary_h']:.3g}  "
        f"min h={surrogate_summary['min_boundary_h']:.3g}"
    )

    status("Switching warm-start samples to the exact cross-section target.")
    exact_log_prob, exact_score = log_prob_and_score_mirror(surrogate_state.x)
    exact_log_prob.block_until_ready()
    mcmc_state = UnderdampedLangevinState(
        rng=surrogate_state.rng,
        n_samples=args.n_samples,
        dim=dim,
        x=surrogate_state.x,
        v=surrogate_state.v,
        log_prob=exact_log_prob,
        score=exact_score,
    )
    status("Initializing exact-target proposal scales on warm-start samples.")
    mcmc_state = mcmc.initialize_eta(mcmc_state)
    mcmc_state.x.block_until_ready()
    status("Exact-target warm-start initialization complete.")

initial_summary = summarize_samples(
    mcmc_state.x, mcmc_state.log_prob, diagnostic_batch_size
)
if initial_summary["finite_fraction"] < 1.0:
    raise RuntimeError(
        "Underdamped Langevin initialization returned samples outside the "
        f"cross-section support: finite fraction "
        f"{initial_summary['finite_fraction']:.3f}."
    )

_, initial_eta_text = eta_summary(mcmc_state.log_eta)
print(f"  Initial eta: {initial_eta_text}")
print(
    "  Initial samples: "
    f"finite={initial_summary['finite_fraction']:.3f}  "
    f"mean log p={initial_summary['mean_log_prob']:.3f}  "
    f"mean h={initial_summary['mean_boundary_h']:.3g}  "
    f"min h={initial_summary['min_boundary_h']:.3g}"
)

mcmc_checkpoints = []
mcmc_learned_lsd = []
mcmc_learned_lsd_std = []
mcmc_eta = []
mcmc_acceptance = []
mcmc_esjd = []
mcmc_summary = []
mcmc_learned_lsd_folds = []
stein_converged = False
stein_convergence_step = None
post_convergence_steps = 0
terminal_mcmc_steps = 0
histograms = None
stein_out_path = os.path.join(out_dir, "stein_results.json")
metadata_out_path = os.path.join(out_dir, "metadata.json")
histogram_dir = os.path.join(out_dir, "histograms")


def build_common_metadata(
    terminal_eta,
    terminal_acceptance,
    terminal_esjd,
    terminal_summary,
    partial=False,
):
    return {
        "method": "mcmc",
        "process": process_label,
        "vb": args.vb,
        "n_jets": args.n_jets,
        "dim": dim,
        "seed": args.seed,
        "sqrt_s": args.sqrt_s,
        "pt_cut": args.pt_cut,
        "y_cut": args.y_cut,
        "deltar_cut": args.deltar_cut,
        "requested_n_samples": requested_n_samples,
        "n_samples": args.n_samples,
        "mcmc_steps": args.mcmc_steps,
        "terminal_mcmc_steps": terminal_mcmc_steps,
        "beta": args.beta,
        "sampler": "underdamped_langevin",
        "eta_adaptation": "score_direction_scalar_esjd",
        "surrogate_warm_start": args.surrogate_path is not None,
        "surrogate_path": args.surrogate_path,
        "surrogate_steps": args.surrogate_steps if args.surrogate_path else 0,
        "surrogate_chunk_size": (
            args.surrogate_chunk_size if args.surrogate_path else None
        ),
        "surrogate_eta": surrogate_eta,
        "surrogate_acceptance": surrogate_acceptance,
        "surrogate_esjd": surrogate_esjd,
        "surrogate_summary": surrogate_summary,
        "terminal_eta": terminal_eta,
        "terminal_acceptance": terminal_acceptance,
        "terminal_esjd": terminal_esjd,
        "initial_summary": initial_summary,
        "terminal_summary": terminal_summary,
        "cross_section_chunk_size": args.cross_section_chunk_size,
        "score_method": "autodiff_cut_cross_section",
        "score_vectorization": "chunked_vmap_value_and_grad",
        "partial": partial,
        "tasks": {
            "stein": bool(args.stein),
            "observables": True,
        },
    }


def build_stein_results(
    terminal_eta,
    terminal_acceptance,
    terminal_esjd,
    terminal_summary,
    partial=False,
):
    results = build_common_metadata(
        terminal_eta,
        terminal_acceptance,
        terminal_esjd,
        terminal_summary,
        partial=partial,
    )
    if args.stein:
        results.update(
            {
                "stein_lamb": args.stein_lamb,
                "stein_optimizer": args.stein_optimizer,
                "stein_learning_rate": (
                    args.stein_learning_rate if args.stein_optimizer == "adam" else None
                ),
                "stein_num_hidden_layers": args.stein_num_hidden_layers,
                "stein_hidden_size": args.stein_hidden_size,
                "stein_batch_size": args.stein_batch_size,
                "stein_training_patience": args.stein_training_patience,
                "stein_cross_val": bool(args.stein_cross_val),
                "stein_checkpoint_patience": args.stein_checkpoint_patience,
                "stein_converged": bool(stein_converged),
                "stein_convergence_step": stein_convergence_step,
                "stein_zero_sigma_rule": "lsd <= lsd_se",
                "post_convergence_fraction": args.post_convergence_fraction,
                "post_convergence_min_steps": args.post_convergence_min_steps,
                "post_convergence_steps": post_convergence_steps,
                "mcmc_checkpoints": mcmc_checkpoints,
                "mcmc_eta": mcmc_eta,
                "mcmc_acceptance": mcmc_acceptance,
                "mcmc_esjd": mcmc_esjd,
                "mcmc_summary": mcmc_summary,
                "mcmc_learned_lsd": mcmc_learned_lsd,
                "mcmc_learned_lsd_std": mcmc_learned_lsd_std,
                "mcmc_learned_lsd_folds": mcmc_learned_lsd_folds,
            }
        )
    return results


def build_observable_metadata(
    terminal_eta,
    terminal_acceptance,
    terminal_esjd,
    terminal_summary,
    histogram_names,
):
    metadata = build_common_metadata(
        terminal_eta,
        terminal_acceptance,
        terminal_esjd,
        terminal_summary,
        partial=False,
    )
    metadata.update(
        {
            "observable_mcmc_steps": terminal_mcmc_steps,
            "observable_batch_size": args.observable_batch_size,
            "pt_max": args.pt_max,
            "n_pt_bins": args.n_pt_bins,
            "ht_max": args.ht_max,
            "n_ht_bins": args.n_ht_bins,
            "mass_max": args.mass_max,
            "n_mass_bins": args.n_mass_bins,
            "delta_r_max": args.delta_r_max,
            "n_delta_r_bins": args.n_delta_r_bins,
            "log_xaxb_min": args.log_xaxb_min,
            "log_xaxb_max": args.log_xaxb_max,
            "n_x_bins": args.n_x_bins,
            "kt_clustering_scale_max": args.kt_clustering_scale_max,
            "ca_clustering_scale_max": args.ca_clustering_scale_max,
            "n_clustering_scale_bins": args.n_clustering_scale_bins,
            "clustering_observables": ["d", "y"],
            "jet_radius": args.jet_radius,
            "y_max": args.y_cut if args.y_max is None else args.y_max,
            "n_y_bins": args.n_y_bins,
            "histograms": sorted(histogram_names),
        }
    )
    if args.stein:
        metadata.update(
            {
                "stein_converged": bool(stein_converged),
                "stein_convergence_step": stein_convergence_step,
                "stein_results_path": "stein_results.json",
            }
        )
    return metadata


def write_stein_results(
    terminal_eta,
    terminal_acceptance,
    terminal_esjd,
    terminal_summary,
    partial=False,
):
    write_json(
        stein_out_path,
        build_stein_results(
            terminal_eta,
            terminal_acceptance,
            terminal_esjd,
            terminal_summary,
            partial=partial,
        ),
    )


def write_observable_results(
    terminal_eta,
    terminal_acceptance,
    terminal_esjd,
    terminal_summary,
    histograms,
):
    metadata = build_observable_metadata(
        terminal_eta,
        terminal_acceptance,
        terminal_esjd,
        terminal_summary,
        histograms.keys(),
    )
    write_json(metadata_out_path, metadata)
    os.makedirs(histogram_dir, exist_ok=True)
    for name, histogram in sorted(histograms.items()):
        payload = dict(metadata)
        payload["observable"] = name
        payload["histogram"] = histogram
        write_json(os.path.join(histogram_dir, f"{name}.json"), payload)


def evaluate_stein_checkpoint(t, state, rng):
    eta_t, eta_text = eta_summary(state.log_eta)
    acc_t = float(jnp.exp(state.avg_log_accept))
    esjd_t = float(state.esjd)
    summary_t = summarize_samples(state.x, state.log_prob, diagnostic_batch_size)

    mcmc_checkpoints.append(t)
    mcmc_eta.append(eta_t)
    mcmc_acceptance.append(acc_t)
    mcmc_esjd.append(esjd_t)
    mcmc_summary.append(summary_t)

    print(
        f"\n----- ULA: t = {t} -----\n"
        f"  eta={eta_text}  acc={acc_t:.4f}  ESJD={esjd_t:.6g}\n"
        f"  finite={summary_t['finite_fraction']:.3f}  "
        f"mean log p={summary_t['mean_log_prob']:.3f}  "
        f"mean h={summary_t['mean_boundary_h']:.3g}  "
        f"min h={summary_t['min_boundary_h']:.3g}"
    )

    rng, rng_train = jax.random.split(rng)
    status(f"Training/evaluating {args.stein_optimizer.upper()} Stein critic at t={t}.")
    stein_kwargs = {
        "x": state.x,
        "scores": state.score,
        "rng": rng_train,
        "num_hidden_layers": args.stein_num_hidden_layers,
        "hidden_size": args.stein_hidden_size,
        "lamb": args.stein_lamb,
        "batch_size": args.stein_batch_size,
        "patience": args.stein_training_patience,
        "cross_val": args.stein_cross_val,
        "boundary_fn": boundary_fn_mirror,
        "status_fn": status,
        "return_fold_results": True,
    }
    if args.stein_optimizer == "adam":
        stein_result = train_stein_discrepancy(
            **stein_kwargs,
            learning_rate=args.stein_learning_rate,
        )
    else:
        stein_result = train_stein_discrepancy_kfac(**stein_kwargs)
    test_lsd, test_lsd_se, fold_results = stein_result
    test_lsd = float(test_lsd)
    test_lsd_se = float(test_lsd_se)
    mcmc_learned_lsd.append(test_lsd)
    mcmc_learned_lsd_std.append(test_lsd_se)
    mcmc_learned_lsd_folds.append(fold_results)
    print(f"  Learned LSD: {test_lsd:.6f} +/- {test_lsd_se:.6f}")

    write_stein_results(
        eta_t,
        acc_t,
        esjd_t,
        summary_t,
        partial=True,
    )
    status(f"Wrote partial Stein results to {stein_out_path}.")

    return rng, test_lsd, test_lsd_se


if args.stein:
    target_mcmc_steps = args.mcmc_steps
    rng, test_lsd, test_lsd_se = evaluate_stein_checkpoint(
        terminal_mcmc_steps,
        mcmc_state,
        rng,
    )
    if test_lsd <= test_lsd_se:
        stein_converged = True
        stein_convergence_step = terminal_mcmc_steps
        post_convergence_steps = max(
            math.ceil(args.post_convergence_fraction * stein_convergence_step),
            args.post_convergence_min_steps,
        )
        target_mcmc_steps = terminal_mcmc_steps + post_convergence_steps
        status(
            f"Stein criterion reached at t={terminal_mcmc_steps}: "
            f"LSD={test_lsd:.6g} <= SE={test_lsd_se:.6g}. "
            f"Continuing Stein checks to observable point "
            f"t={target_mcmc_steps}."
        )

    while terminal_mcmc_steps < target_mcmc_steps:
        steps_to_run = min(
            args.stein_checkpoint_patience,
            target_mcmc_steps - terminal_mcmc_steps,
        )
        t = terminal_mcmc_steps + steps_to_run
        status(f"Running ULA from step {terminal_mcmc_steps} to {t}.")
        mcmc_state = mcmc.run_chain(mcmc_state, steps_to_run)
        mcmc_state.x.block_until_ready()
        status(f"ULA reached step {t}.")
        terminal_mcmc_steps = t

        rng, test_lsd, test_lsd_se = evaluate_stein_checkpoint(
            t,
            mcmc_state,
            rng,
        )

        if not stein_converged and test_lsd <= test_lsd_se:
            stein_converged = True
            stein_convergence_step = t
            post_convergence_steps = max(
                math.ceil(args.post_convergence_fraction * stein_convergence_step),
                args.post_convergence_min_steps,
            )
            target_mcmc_steps = terminal_mcmc_steps + post_convergence_steps
            status(
                f"Stein criterion reached at t={t}: "
                f"LSD={test_lsd:.6g} <= SE={test_lsd_se:.6g}. "
                f"Continuing Stein checks to observable point "
                f"t={target_mcmc_steps}."
            )
else:
    status(f"Running fixed-length ULA chain for {args.mcmc_steps} steps.")
    mcmc_state = mcmc.run_chain(mcmc_state, args.mcmc_steps)
    mcmc_state.x.block_until_ready()
    terminal_mcmc_steps = args.mcmc_steps

status(f"Computing terminal histograms at t={terminal_mcmc_steps}.")
histograms = compute_histograms(mcmc_state)
status("Observable histogram computation complete.")

terminal_eta, terminal_eta_text = eta_summary(mcmc_state.log_eta)
terminal_summary = summarize_samples(
    mcmc_state.x, mcmc_state.log_prob, diagnostic_batch_size
)
print(
    "\n----- Terminal ULA state -----\n"
    f"  t={terminal_mcmc_steps}  eta={terminal_eta_text}  "
    f"acc={float(jnp.exp(mcmc_state.avg_log_accept)):.4f}  "
    f"ESJD={float(mcmc_state.esjd):.6g}\n"
    f"  finite={terminal_summary['finite_fraction']:.3f}  "
    f"mean log p={terminal_summary['mean_log_prob']:.3f}  "
    f"mean h={terminal_summary['mean_boundary_h']:.3g}  "
    f"min h={terminal_summary['min_boundary_h']:.3g}"
)

if args.stein:
    write_stein_results(
        terminal_eta,
        float(jnp.exp(mcmc_state.avg_log_accept)),
        float(mcmc_state.esjd),
        terminal_summary,
        partial=False,
    )

write_observable_results(
    terminal_eta,
    float(jnp.exp(mcmc_state.avg_log_accept)),
    float(mcmc_state.esjd),
    terminal_summary,
    histograms,
)
status(f"Stein results saved to {stein_out_path}" if args.stein else "Stein disabled.")
status(f"Run metadata saved to {metadata_out_path}")
status(f"Histogram files saved to {histogram_dir}")
