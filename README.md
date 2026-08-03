# Event Generation with Parallel Langevin Sampling and Learned Stein Diagnostics

This repository contains the code, trained surrogate models, numerical results
and plotting scripts for the reported event-generation study.

## Scripts

- `run_cross_section_mcmc.py`: direct ULA event-generation runner, with optional surrogate warm start and learned Stein diagnostics.
- `run_cross_section_vegas_validation.py`: VEGAS reference histogram runner.
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
