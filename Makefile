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

demo-test:
	python ./scripts/test_gwr_equivalence.py

## Run the vanilla GWR demo (Georgia dataset).
demo-gwr-vanilla-georgia:
	python ./scripts/gwr/vanilla/script-gwr-vanilla-georgia.py

demo-gwr-vanilla-simulated:
	python ./scripts/gwr/vanilla/script-gwr-vanilla-simulated.py

## Run the GWR demo with reinforcement learning on simulated data.
demo-gwr-rl-simulated:
	python ./scripts/gwr/rl/script-gwr-rl-simulated.py

## Run the LGWR demo with reinforcement learning on simulated data.
demo-lgwr-rl-simulated:
	python ./scripts/lgwr/rl/script-lgwr-rl-simulated.py

## Run the GWR demo with reinforcement learning.
demo-gwr-rl:
	python ./scripts/gwr/rl/script-gwr-rl-georgia.py


## Run the LGWR demo with reinforcement learning.
demo-lgwr-rl:
	python ./scripts/lgwr/rl/script-lgwr-rl-georgia.py

## Visualize the LGWR reinforcement learning results.
demo-lgwr-rl-visual:
	python ./scripts/lgwr/visualize/script-lgwr-rl-visualize.py

demo-test-dgp:
	python ./scripts/.unclassified/script-test-dgp.py

# ╔════════════════════════════════════════╗
# 🧪 Unit Testing
# ╚════════════════════════════════════════╝

## Run all unit tests with pytest. 
test: 
	export PYTHONPATH=$(PWD) && echo $(PYTHONPATH) && pytest 


