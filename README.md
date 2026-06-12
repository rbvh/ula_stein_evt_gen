# Event Generation with Parallel Langevin Sampling and Learned Stein Diagnostics

This repository contains the code, trained surrogate models, numerical results and plotting scripts for the accompanying event-generation paper.

## Scripts

- `run_cross_section_mcmc.py`: direct ULA event-generation runner, with optional surrogate warm start and learned Stein diagnostics.
- `run_cross_section_vegas_validation.py`: VEGAS reference histogram runner.
- `run_cross_section_surrogate.py`: cross-section surrogate trainer.
- `run_ou_stein.py`: Ornstein-Uhlenbeck validation runner.
- `plot_cross_section_stein.py`, `plot_cross_section_validation.py`,
  `plot_ou_stein.py`: scripts used to produce the paper figures.

## Installation

For a CPU-only environment:

```bash
pip install -e .
```

For a CUDA 12 JAX environment:

```bash
pip install -e ".[gpu]"
```
