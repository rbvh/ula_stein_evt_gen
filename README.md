# Event Generation with Parallel Langevin Sampling and Learned Stein Diagnostics

This repository contains the code, trained surrogate models, numerical results
and plotting scripts for the reported event-generation study.

## Scripts

- `run_cross_section_mcmc.py`: direct ULA event-generation runner, with optional surrogate warm start and learned Stein diagnostics.
- `run_cross_section_vegas_validation.py`: VEGAS reference histogram runner.
- `run_vegas_cross_section.py`: fit VEGAS and measure unweighting efficiencies and proposal costs.
- `run_cross_section_surrogate.py`: cross-section surrogate trainer.
- `run_ou_stein.py`: Ornstein-Uhlenbeck validation runner.
- `plot_cross_section_stein.py`, `plot_cross_section_validation.py` and
  `plot_ou_stein.py`: figure-generation scripts.

## Installation

Python 3.10 or newer is required. For a CPU-only environment:

```bash
pip install -e .
```

For a CUDA 12 JAX environment:

```bash
pip install -e ".[gpu]"
```

The reported experiments were run with CUDA 12 on a single NVIDIA GeForce RTX
5090 GPU (32 GB). The generated matrix elements and NNPDF4.0 interpolation
tables are included in `ula_stein_evt_gen/cross_section/`; no external dataset
is required.

## Reproduction

Run the commands below from the repository root. All options not shown use the
defaults corresponding to the reported setup. The non-default sample counts,
cross-validation options, multiplicities and OU boundary mass are explicit.
All runs use the default seed, 0.

### Generate the numerical results

Train the surrogates:

```bash
for n_jets in 0 1 2 3; do
  python run_cross_section_surrogate.py --n_jets "$n_jets"
done
```

Run ULA with and without surrogate initialization:

```bash
for n_jets in 0 1 2 3; do
  python run_cross_section_mcmc.py \
    --n_jets "$n_jets" \
    --n_samples 10000000 \
    --stein_cross_val

  python run_cross_section_mcmc.py \
    --n_jets "$n_jets" \
    --n_samples 10000000 \
    --stein_cross_val \
    --surrogate_path "models/cross_section_surrogate/z_${n_jets}j_seed0.pkl"
done
```

Generate the independent VEGAS references:

```bash
for n_jets in 0 1 2 3; do
  python run_cross_section_vegas_validation.py \
    --n_jets "$n_jets" \
    --n_samples 10000000
done
```

Run the Ornstein--Uhlenbeck validation:

```bash
python run_ou_stein.py \
  --dim 4 \
  --n_samples 10000000 \
  --cross_val \
  --boundary_mass_cut 0.2

python run_ou_stein.py \
  --dim 16 \
  --n_samples 10000000 \
  --cross_val \
  --boundary_mass_cut 0.2
```

Outputs are written under `models/` and `results/`. Each cross-section run also
records its effective configuration in JSON.

### Measure VEGAS unweighting efficiencies

```bash
for n_jets in 0 1 2 3; do
  python run_vegas_cross_section.py \
    --n_jets "$n_jets" \
    --n_train 1e6 --n_epochs 15 --n_eval 1e7 \
    --batch_size 100000 --n_repeats 10
done
```

The runner fits the exact cross section with the same energy and cuts as the
MCMC runs. The transverse-momentum cut applies only for positive gluon
multiplicity. `n_train` is the number of trials per training epoch, while
`batch_size` controls only the survey. Each repeat independently surveys the
same frozen fitted grid; grid fitting is performed once.

It reports epsilon and its reciprocal for integral-tail allowances of zero,
0.001, and 0.01, including zero-weight proposals that fail cuts. Following
[arXiv:2604.03511](https://arxiv.org/abs/2604.03511), the threshold for each
allowance bounds the integral contribution from overweight events, rather
than their event-count fraction. The zero-allowance threshold is a finite
survey maximum, not a guaranteed global bound.

With overweights, the conventional epsilon = mean(w)/threshold differs from
the actual acceptance mean(min(1, w/threshold)); both are saved, together
with their reciprocals and overweight diagnostics. For multiple repeats,
the terminal table displays means +/- standard errors, calculated separately
for each quantity as the sample standard deviation divided by the square
root of the repeat count. These errors describe survey variation for one
fitted grid. With one repeat, error fields are null.

Grids and JSON results are saved under
`results/cross_section_unweighting/vegas/z_<n_jets>j_seed0/`.
`vegas_results.txt` records the table measurements, in order Z+0g through
Z+3g. The table uses means of the reciprocals, rather than reciprocals of
the mean efficiencies. Grid training is recorded separately; these numbers
are not complete wall-clock costs for comparison with MCMC relaxation steps.

Tail statistics use host float64. To also sample the grid and evaluate matrix
elements in float64, prefix the runner with
`MADJAX_PRECISION=float64 JAX_ENABLE_X64=true`.

Check the calculation with `python -m unittest test_unweighting`.

### Generate the figures

```bash
python plot_cross_section_stein.py \
  results/cross_section_mcmc \
  --compact \
  --output plots/cross_section_stein_z_workshop.pdf

python plot_cross_section_validation.py \
  --n_jets 0 1 2 3 \
  --compact \
  --out_dir plots/cross_section_validation_workshop

python plot_ou_stein.py \
  --output plots/ou_stein_d4_d16_mass_cut_0p2.pdf
```

The archive already includes the trained models and numerical outputs, so these
plotting commands can be run without regenerating the samples. The plotting
scripts use LaTeX for labels unless called with `--no_tex`.

## Scope

The reported artifacts cover the fixed tree-level
`u ubar -> Z + n gluons` channel for `n = 0, 1, 2, 3` with the included
phase-space map. This is a research prototype rather than a full event
generator implementation.

## License

The original code in this repository is released under the MIT License; see
`LICENSE`. Bundled third-party assets remain subject to their respective terms,
which are documented in `THIRD_PARTY_LICENSES.md`.
