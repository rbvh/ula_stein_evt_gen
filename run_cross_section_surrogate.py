import argparse
import os
import time
import warnings

import ula_stein_evt_gen.jax_bootstrap  # noqa: F401
import jax
import jax.numpy as jnp

from ula_stein_evt_gen.cross_section.cross_section import build_cross_section
from ula_stein_evt_gen.cross_section.surrogate import (
    round_up_to_multiple,
    save_surrogate,
    train_surrogate_adam,
    train_surrogate_kfac,
)
from ula_stein_evt_gen.mirror import to_mirror

warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

RUN_START = time.perf_counter()


def status(message):
    elapsed = time.perf_counter() - RUN_START
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def collect_uniform_accepted_dataset(
    rng,
    cross_section,
    dim,
    n_target,
    sample_batch_size,
):
    z_chunks = []
    y_chunks = []
    n_seen = 0
    n_kept = 0

    while n_kept < n_target:
        rng, rng_batch = jax.random.split(rng)
        x = jax.random.uniform(
            rng_batch,
            shape=(sample_batch_size, dim),
            minval=1e-7,
            maxval=1.0 - 1e-7,
        )
        sigma = cross_section(x)
        mask = jnp.isfinite(sigma) & (sigma > 0.0)
        n_batch_kept = int(jnp.sum(mask))
        n_seen += sample_batch_size

        if n_batch_kept:
            x_kept = x[mask]
            sigma_kept = sigma[mask]
            z_chunks.append(to_mirror(x_kept))
            y_chunks.append(jnp.log(sigma_kept))
            n_kept += n_batch_kept

        status(
            f"Collected {min(n_kept, n_target)}/{n_target} accepted points "
            f"from {n_seen} uniform proposals "
            f"(running acceptance {n_kept / max(n_seen, 1):.4f})."
        )

    z = jnp.concatenate(z_chunks, axis=0)[:n_target]
    y = jnp.concatenate(y_chunks, axis=0)[:n_target]
    return rng, z, y, n_seen, n_kept


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)

# Physics parameters
parser.add_argument("--vb", type=str, default="z", choices=["w", "z"])
parser.add_argument("--n_jets", type=int, default=1)
parser.add_argument("--sqrt_s", type=float, default=13000.0)
parser.add_argument("--pt_cut", type=float, default=20.0)
parser.add_argument("--y_cut", type=float, default=5.0)
parser.add_argument("--deltar_cut", type=float, default=0.4)

# Surrogate architecture: same residual MLP form as the Stein critic.
parser.add_argument("--num_hidden_layers", type=int, default=5)
parser.add_argument("--hidden_size", type=int, default=128)

# Dataset and training parameters. Counts refer to accepted, non-cut points.
parser.add_argument("--n_train", type=lambda x: int(float(x)), default=1_000_000)
parser.add_argument("--n_val", type=lambda x: int(float(x)), default=100_000)
parser.add_argument("--n_test", type=lambda x: int(float(x)), default=100_000)
parser.add_argument(
    "--sample_batch_size",
    type=lambda x: int(float(x)),
    default=1_000_000,
)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "kfac"])
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--rtol", type=float, default=1e-3)
parser.add_argument("--out_path", type=str, default=None)
args = parser.parse_args()

if args.n_train < 1 or args.n_val < 1 or args.n_test < 1:
    raise ValueError("--n_train, --n_val and --n_test must be positive.")
if args.sample_batch_size < 1:
    raise ValueError("--sample_batch_size must be positive.")
if args.batch_size < 1:
    raise ValueError("--batch_size must be positive.")
if args.learning_rate <= 0.0:
    raise ValueError("--learning_rate must be positive.")
if args.max_epochs < 1:
    raise ValueError("--max_epochs must be positive.")
if args.patience < 1:
    raise ValueError("--patience must be positive.")

requested_counts = {
    "n_train": args.n_train,
    "n_val": args.n_val,
    "n_test": args.n_test,
}
args.n_train = round_up_to_multiple(args.n_train, args.batch_size)
args.n_val = round_up_to_multiple(args.n_val, args.batch_size)
args.n_test = round_up_to_multiple(args.n_test, args.batch_size)
for name, requested in requested_counts.items():
    rounded = getattr(args, name)
    if rounded != requested:
        status(
            f"Rounded {name} from {requested} to {rounded} "
            f"to be a multiple of batch_size={args.batch_size}."
        )

process_label = f"{args.vb}_{args.n_jets}j"
optimizer_suffix = "" if args.optimizer == "adam" else f"_{args.optimizer}"
out_path = (
    args.out_path
    or f"models/cross_section_surrogate/{process_label}_seed{args.seed}{optimizer_suffix}.pkl"
)
out_dir = os.path.dirname(out_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

rng = jax.random.PRNGKey(args.seed)

status("Building grad-free cross-section evaluator.")
cross_section, dim = build_cross_section(
    sqrt_s=args.sqrt_s,
    vb=args.vb,
    n_jets=args.n_jets,
    pt_cut=args.pt_cut,
    y_cut=args.y_cut,
    deltar_cut=args.deltar_cut,
)
status(f"Built cross section for {args.vb.upper()}+{args.n_jets}j (dim={dim}).")
status(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

n_total = args.n_train + args.n_val + args.n_test
rng, z, y, n_seen, n_kept = collect_uniform_accepted_dataset(
    rng,
    cross_section,
    dim,
    n_total,
    args.sample_batch_size,
)

rng, rng_perm = jax.random.split(rng)
perm = jax.random.permutation(rng_perm, n_total)
z = z[perm]
y = y[perm]

z_train = z[: args.n_train]
y_train = y[: args.n_train]
val_start = args.n_train
val_stop = val_start + args.n_val
z_val = z[val_start:val_stop]
y_val = y[val_start:val_stop]
z_test = z[val_stop:]
y_test = y[val_stop:]

status(
    "Training surrogate on standardized log cross section: "
    f"train={args.n_train}, val={args.n_val}, test={args.n_test}, "
    f"optimizer={args.optimizer}."
)
rng, rng_train = jax.random.split(rng)
train_kwargs = {
    "z_train": z_train,
    "y_train": y_train,
    "z_val": z_val,
    "y_val": y_val,
    "z_test": z_test,
    "y_test": y_test,
    "rng": rng_train,
    "num_hidden_layers": args.num_hidden_layers,
    "hidden_size": args.hidden_size,
    "batch_size": args.batch_size,
    "max_epochs": args.max_epochs,
    "patience": args.patience,
    "rtol": args.rtol,
    "status_fn": status,
}
if args.optimizer == "adam":
    payload = train_surrogate_adam(
        **train_kwargs,
        learning_rate=args.learning_rate,
    )
else:
    payload = train_surrogate_kfac(**train_kwargs)

payload.update(
    {
        "method": "cross_section_surrogate",
        "input_space": "mirror",
        "target": "log_cross_section_on_accepted_uniform_points",
        "architecture": "stein_critic_residual_mlp_scalar_output",
        "optimizer": args.optimizer,
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
        "n_val": args.n_val,
        "n_test": args.n_test,
        "sample_batch_size": args.sample_batch_size,
        "uniform_proposals_seen": n_seen,
        "uniform_proposals_accepted": n_kept,
        "uniform_acceptance": n_kept / max(n_seen, 1),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate if args.optimizer == "adam" else None,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "rtol": args.rtol,
    }
)

save_surrogate(out_path, payload)
status(f"Saved surrogate to {out_path}")
status(f"Metrics: {payload['metrics']}")
