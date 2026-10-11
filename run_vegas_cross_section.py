"""Fit VEGAS to the exact cross section and measure unweighting costs."""
import argparse
import os
from pathlib import Path
import time

import ula_stein_evt_gen.jax_bootstrap  # noqa: F401
import jax
import numpy as np

from ula_stein_evt_gen.cross_section.cross_section import build_cross_section
from ula_stein_evt_gen.cross_section.histogram_estimators import write_json
from ula_stein_evt_gen.vegas import VegasIntegrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vb", choices=["w", "z"], default="z")
    parser.add_argument("--n_jets", type=int, choices=range(6), default=1)
    parser.add_argument("--sqrt_s", type=float, default=13000.0)
    parser.add_argument("--pt_cut", type=float, default=20.0)
    parser.add_argument("--y_cut", type=float, default=5.0)
    parser.add_argument("--deltar_cut", type=float, default=0.4)
    parser.add_argument("--n_train", type=lambda x: int(float(x)), default=1_000_000,
                        help="Trials per grid-training epoch.")
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--n_bins_vegas", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--n_eval", "--n_samples", type=lambda x: int(float(x)),
                        default=10_000_000, help="Survey trials per repeat.")
    parser.add_argument("--batch_size", type=lambda x: int(float(x)), default=100_000,
                        help="Survey batch size; training uses n_train per epoch.")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="Independent surveys of one frozen fitted grid.")
    parser.add_argument("--tail_fractions", type=float, nargs="+", default=[0.001, 0.01],
                        help="Allowed fractions of the integral from overweight events.")
    parser.add_argument("--out_dir", type=Path, default=Path("results/cross_section_unweighting/vegas"))
    args = parser.parse_args()
    if args.n_train < 2 or args.n_eval < 2 or args.n_bins_vegas < 2:
        parser.error("n_train, n_eval and n_bins_vegas must be at least 2.")
    if min(args.n_epochs, args.n_repeats, args.batch_size) < 1:
        parser.error("n_epochs, n_repeats and batch_size must be positive.")
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("alpha must be finite and positive.")
    if any(not np.isfinite(a) or not 0 <= a < 1 for a in args.tail_fractions):
        parser.error("tail_fractions must lie in [0, 1).")

    started = time.perf_counter()

    def status(message):
        print(f"[{time.perf_counter() - started:8.1f}s] {message}", flush=True)

    process = f"{args.vb}_{args.n_jets}j"
    run_dir = args.out_dir / f"{process}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rng_train, rng_survey = jax.random.split(jax.random.PRNGKey(args.seed))
    status("Building the exact grad-free cross-section evaluator.")
    cross_section, dim = build_cross_section(
        sqrt_s=args.sqrt_s, vb=args.vb, n_jets=args.n_jets,
        pt_cut=args.pt_cut, y_cut=args.y_cut, deltar_cut=args.deltar_cut,
    )
    vegas = VegasIntegrator(cross_section, dim, args.n_bins_vegas, args.alpha)
    status(f"Fitting {process}: dim={dim}, {args.n_epochs} epochs of {args.n_train} trials.")
    vegas.train(rng_train, n_points=args.n_train, n_epochs=args.n_epochs, status_fn=status)
    np.save(run_dir / "grid.npy", np.asarray(jax.device_get(vegas.subdivisions)))
    repeats = []
    for i in range(args.n_repeats):
        rng_survey, rng_repeat = jax.random.split(rng_survey)
        status(f"Survey {i + 1}/{args.n_repeats}.")
        repeats.append(vegas.measure_unweighting(
            rng_repeat, args.n_eval, args.batch_size, args.tail_fractions, status,
        ))
        status(f"Survey {i + 1}: sigma={repeats[-1]['cross_section']:.6g} pb.")

    summary = {}
    if args.n_repeats > 1:
        print(f"Means +/- standard errors across {args.n_repeats} independent surveys.",
              flush=True)
    print(f"{'tail fraction':>13}  {'epsilon':>26}  {'1/epsilon':>26}  "
          f"{'trials/accepted':>26}", flush=True)
    for a in repeats[0]["efficiencies"]:
        metrics = {}
        for metric in ("epsilon", "inverse_epsilon", "acceptance", "trials_per_accepted_event"):
            values = np.array([repeat["efficiencies"][a][metric] for repeat in repeats])
            metrics[metric] = float(values.mean())
            metrics[metric + "_standard_error"] = (
                float(values.std(ddof=1) / np.sqrt(args.n_repeats))
                if args.n_repeats > 1 else None
            )
        summary[a] = metrics
        columns = []
        for metric in ("epsilon", "inverse_epsilon", "trials_per_accepted_event"):
            value = f"{metrics[metric]:.6g}"
            error = metrics[metric + "_standard_error"]
            if error is not None:
                value += f" +/- {error:.3g}"
            columns.append(f"{value:>26}")
        print(f"{a:13g}  " + "  ".join(columns), flush=True)

    result = dict(vars(args))
    result.update(
        out_dir=str(args.out_dir), method="vegas", process=process, dim=dim,
        madjax_precision=os.environ.get("MADJAX_PRECISION", "float32"),
        jax_enable_x64=bool(jax.config.jax_enable_x64),
        training_trials=args.n_train * args.n_epochs,
        survey_trials=args.n_eval * args.n_repeats,
        definition="epsilon_a = mean(w)/M_a; sum(w[w > M_a])/sum(w) <= a",
        reference="https://arxiv.org/abs/2604.03511",
        summary=summary, repeats=repeats,
    )
    write_json(run_dir / "results.json", result)
    status(f"Saved efficiencies and grid to {run_dir}.")


if __name__ == "__main__":
    main()
