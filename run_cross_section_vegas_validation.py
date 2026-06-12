import argparse
import os
import time

import ula_stein_evt_gen.jax_bootstrap  # noqa: F401
import jax
import numpy as np

from ula_stein_evt_gen.cross_section.cross_section import build_cross_section
from ula_stein_evt_gen.cross_section.histogram_estimators import (
    accumulate_histogram,
    default_observable_bins,
    finalize_histogram,
    init_histogram_accumulator,
    write_json,
)
from ula_stein_evt_gen.cross_section.observables import compute_observables
from ula_stein_evt_gen.vegas import VegasIntegrator

RUN_START = time.perf_counter()


def status(message):
    elapsed = time.perf_counter() - RUN_START
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--vb", type=str, default="z", choices=["w", "z"])
parser.add_argument("--n_jets", type=int, default=1)
parser.add_argument("--sqrt_s", type=float, default=13000.0)
parser.add_argument("--pt_cut", type=float, default=20.0)
parser.add_argument("--y_cut", type=float, default=5.0)
parser.add_argument("--deltar_cut", type=float, default=0.4)
parser.add_argument("--n_train", type=lambda x: int(float(x)), default=1_000_000)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_samples", type=lambda x: int(float(x)), default=1_000_000)
parser.add_argument("--n_bins_vegas", type=int, default=100)
parser.add_argument("--alpha", type=float, default=1.5)
parser.add_argument("--batch_size", type=lambda x: int(float(x)), default=1_000_000)
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
parser.add_argument(
    "--out_dir", type=str, default="results/cross_section_validation/vegas"
)
args = parser.parse_args()

if args.n_train < 2:
    raise ValueError("--n_train must be at least 2.")
if args.n_samples < 1:
    raise ValueError("--n_samples must be positive.")
if args.n_epochs < 1:
    raise ValueError("--n_epochs must be positive.")
if args.batch_size < 1:
    raise ValueError("--batch_size must be positive.")

process_label = f"{args.vb}_{args.n_jets}j"
run_dir = os.path.join(args.out_dir, f"{process_label}_seed{args.seed}")
histogram_dir = os.path.join(run_dir, "histograms")
os.makedirs(run_dir, exist_ok=True)
rng = jax.random.PRNGKey(args.seed)
rng, rng_train, rng_sample = jax.random.split(rng, 3)

status("Building grad-free cross-section evaluator.")
cross_section, dim = build_cross_section(
    sqrt_s=args.sqrt_s,
    vb=args.vb,
    n_jets=args.n_jets,
    pt_cut=args.pt_cut,
    y_cut=args.y_cut,
    deltar_cut=args.deltar_cut,
)

vegas = VegasIntegrator(
    integrand=cross_section,
    dim=dim,
    n_bins=args.n_bins_vegas,
    alpha=args.alpha,
)
status(
    f"Training VEGAS for {process_label}: "
    f"n_train={args.n_train}, epochs={args.n_epochs}, dim={dim}."
)
vegas.train(
    rng_train,
    n_points=args.n_train,
    n_epochs=args.n_epochs,
    status_fn=status,
)

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

n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
status(
    f"Sampling {args.n_samples} VEGAS events in {n_batches} batches "
    f"of up to {args.batch_size}."
)
histogram_accumulators = {}
n_seen = 0
for batch_idx in range(n_batches):
    n_batch = min(args.batch_size, args.n_samples - n_seen)
    rng_sample, rng_batch = jax.random.split(rng_sample)
    sample = vegas.integrate(rng_batch, n_batch)
    x = sample["x"]
    weights = np.asarray(jax.device_get(sample["weights"]), dtype=np.float64)

    n_seen += n_batch

    observables = compute_observables(
        x,
        vb=args.vb,
        n_jets=args.n_jets,
        sqrt_s=args.sqrt_s,
        pt_cut=args.pt_cut,
        y_cut=args.y_cut,
        jet_radius=args.jet_radius,
        batch_size=n_batch,
    )
    for name, values in observables.items():
        if name not in bins:
            continue
        if name not in histogram_accumulators:
            histogram_accumulators[name] = init_histogram_accumulator(bins[name])
        accumulate_histogram(histogram_accumulators[name], values, weights=weights)

    status(f"Completed VEGAS batch {batch_idx + 1}/{n_batches}.")

histograms = {
    name: finalize_histogram(accumulator)
    for name, accumulator in histogram_accumulators.items()
}

result = {
    "method": "vegas",
    "process": process_label,
    "vb": args.vb,
    "n_jets": args.n_jets,
    "dim": dim,
    "seed": args.seed,
    "sqrt_s": args.sqrt_s,
    "pt_cut": args.pt_cut,
    "y_cut": args.y_cut,
    "deltar_cut": args.deltar_cut,
    "n_train": args.n_train,
    "n_epochs": args.n_epochs,
    "n_samples": args.n_samples,
    "batch_size": args.batch_size,
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
    "n_bins_vegas": args.n_bins_vegas,
    "alpha": args.alpha,
    "kt_clustering_scale_max": args.kt_clustering_scale_max,
    "ca_clustering_scale_max": args.ca_clustering_scale_max,
    "n_clustering_scale_bins": args.n_clustering_scale_bins,
    "clustering_observables": ["d", "y"],
    "jet_radius": args.jet_radius,
    "y_max": args.y_cut if args.y_max is None else args.y_max,
    "n_y_bins": args.n_y_bins,
    "histograms": sorted(histograms),
}

metadata_path = os.path.join(run_dir, "metadata.json")
write_json(metadata_path, result)
os.makedirs(histogram_dir, exist_ok=True)
for name, histogram in sorted(histograms.items()):
    payload = dict(result)
    payload["observable"] = name
    payload["histogram"] = histogram
    write_json(os.path.join(histogram_dir, f"{name}.json"), payload)
status(f"Saved VEGAS validation metadata to {metadata_path}")
status(f"Saved VEGAS validation histograms to {histogram_dir}")
