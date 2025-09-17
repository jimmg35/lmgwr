# ==========================================================
# Project: LMGWR
# Author : Jim Chang
# ==========================================================

.PHONY: demo env test setup

# ╔════════════════════════════════════════╗
# 🌍 Demo Commands
# ╚════════════════════════════════════════╝

## Run the vanilla GWR demo (Georgia dataset).
demo-gwr:
	python script-gwr-vanilla-georgia.py

## Run the GWR demo with reinforcement learning.
demo-gwr-rl:
	python script-gwr-rl-demo.py

## Run the LGWR demo with reinforcement learning.
demo-lgwr-rl:
	python script-lgwr-rl-demo.py

## Visualize the LGWR reinforcement learning results.
demo-lgwr-rl-visual:
	python script-lgwr-rl-visualize.py

# ╔════════════════════════════════════════╗
# 📦 Conda Environment
# ╚════════════════════════════════════════╝

## Export the current conda environment (name: lmgwr).
env-export:
	conda env export -n lmgwr -f environment-linux.yml

## Import conda environment from file.
env-import:
	conda env create -f environment-linux.yml

## Install the source directory for absolute imports.
setup:
	pip install -e .

# ╔════════════════════════════════════════╗
# 🧪 Unit Testing
# ╚════════════════════════════════════════╝

## Run all unit tests with pytest.
test:
	export PYTHONPATH=$(PWD) && echo $(PYTHONPATH) && pytest
