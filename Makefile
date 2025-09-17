# ==========================================================
# Project: LMGWR
# Author : Jim Chang
# ==========================================================

.PHONY: demo env test setup

# ╔════════════════════════════════════════╗
# 🌍 Setting up development mode
# ╚════════════════════════════════════════╝

## Install the package in editable mode.
setup: 
	pip install -e .


# ╔════════════════════════════════════════╗
# 🌍 Demo Commands
# ╚════════════════════════════════════════╝

## Run the vanilla GWR demo (Georgia dataset).
demo-gwr:
	python ./scripts/script-gwr-vanilla-georgia.py

## Run the GWR demo with reinforcement learning.
demo-gwr-rl:
	python ./scripts/script-gwr-rl-demo.py

## Run the GWR demo with reinforcement learning on simulated data.
demo-gwr-rl-simulated:
	python ./scripts/script-gwr-rl-simulated.py

## Run the LGWR demo with reinforcement learning.
demo-lgwr-rl:
	python ./scripts/script-lgwr-rl-demo.py

## Visualize the LGWR reinforcement learning results.
demo-lgwr-rl-visual:
	python ./scripts/script-lgwr-rl-visualize.py



# ╔════════════════════════════════════════╗
# 🧪 Unit Testing
# ╚════════════════════════════════════════╝

## Run all unit tests with pytest. 
test: 
	export PYTHONPATH=$(PWD) && echo $(PYTHONPATH) && pytest 


