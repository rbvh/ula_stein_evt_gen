# Langevin Event Generation

This repository contains the code, trained surrogate models, numerical results,
plotting scripts, and LaTeX source for the accompanying event-generation paper.

## Contents

- `run_cross_section_mcmc.py`: direct ULA event-generation runner, with optional
  surrogate warm start and learned Stein diagnostics.
- `run_cross_section_vegas_validation.py`: VEGAS reference histogram runner.
- `run_cross_section_surrogate.py`: cross-section surrogate trainer.
- `run_ou_stein.py`: Ornstein-Uhlenbeck validation runner.
- `plot_cross_section_stein.py`, `plot_cross_section_validation.py`,
  `plot_ou_stein.py`: scripts used to produce the paper figures.
- `ula_stein_evt_gen/`: minimal installable package used by the runners above.
- `results/`: JSON result files used by the plotters.
- `models/cross_section_surrogate/`: trained surrogate checkpoints used in the
  paper runs.
- `plots/`: generated PDF figures included by the LaTeX paper.
- `paper/`: LaTeX source, bibliography, and compiled PDF.

## Installation

For a CPU-only environment:

```bash
pip install -e .
```

For a CUDA 12 JAX environment:

```bash
pip install -e ".[cuda12]"
```

The runners are designed for large JAX jobs. For small smoke checks, reduce
`--n_samples`, disable cross validation, and keep `--n_jets` at 0 or 1.

## Reproducing the Paper Plots

The checked-in JSON results are sufficient to regenerate the plotted figures:

```bash
python plot_ou_stein.py

python plot_cross_section_stein.py

python plot_cross_section_validation.py
```

The LaTeX source expects these figures in `plots/` relative to the repository
root:

```bash
cd paper
latexmk -pdf main.tex
```
