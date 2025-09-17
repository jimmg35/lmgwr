# lmgwr

## Introduction

`lmgwr` is a Python project for geographically weighted regression (GWR) and local GWR (LGWR) analysis, including clustering, simulation, optimization, and visualization tools. It provides scripts and modules for spatial data analysis, benchmarking, and hypothesis testing, with support for simulated and real datasets.

## Usage


You can run GWR/LGWR demos and experiments using the provided Makefile commands. This simplifies running scripts and setting up the environment.

Example usage:
```bash
# Run the vanilla GWR demo (Georgia dataset)
make demo-gwr

# Run the GWR demo with reinforcement learning
make demo-gwr-rl

# Run the GWR demo with reinforcement learning on simulated data
make demo-gwr-rl-simulated

# Run the LGWR demo with reinforcement learning
make demo-lgwr-rl

# Visualize the LGWR reinforcement learning results
make demo-lgwr-rl-visual
```

Notebooks for interactive demos and visualization are in `scripts/notebook/`.
Data files are in the `data/` and `src/dataset/data/` directories.

## Reproducing the Python Environment

To set up the environment using Conda:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you don't have it.
2. Create a new environment named `lmgwr` and install dependencies:
   ```bash
   conda create -n lmgwr python=3.11
   conda activate lmgwr
   pip install -r requirements.txt
   ```
3. (Optional) Install additional packages for Jupyter notebook support:
   ```bash
   pip install notebook jupyterlab
   ```

## Project Structure

- `src/` - Source code modules (clustering, dataset, dgp, distance, kernel, log, model, optimizer, utility, visualize)
- `scripts/` - Experiment and analysis scripts
- `data/` - Example datasets
- `docs/` - Documentation (see `docs/index.md`)
- `requirements.txt` - Python dependencies
- `Makefile` - Build and automation commands

## Documentation

See `docs/` for detailed documentation and API reference.
