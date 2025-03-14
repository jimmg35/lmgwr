.PHONY: unittest run exportCondaEnv importCondaEnv setup

# Author: Jim Chang

# ================ Demo related commands ================

# The vanilla GWR demo.
gwrVanillaDemo:
	python script-gwr-demo.py

# The GWR demo with reinforcement learning.
gwrRlDemo:
	python script-gwr-rl-demo.py

# The LGWR demo with reinforcement learning.
lgwrRlDemo:
	python script-lgwr-rl-demo.py

# ========== Conda Environment related commands ==========

# Export the conda environment to a file. 
# (includes all dependencies and python version)
exportCondaEnv:
	conda env export -n lmgwr -f environment-linux.yml
# Create a new conda environment from the file.
importCondaEnv:
	conda env create -f environment-linux.yml
# Initialize the src directory (for absolute imports).
setup:
	pip install -e .

# ================ Unit test related commands ================

# Run unit test for all test scripts.
unittest:
	export PYTHONPATH=$(PWD) && echo $(PYTHONPATH) && pytest