.PHONY: unittest run exportCondaEnv importCondaEnv setup


# Run unit test for all test scripts.
unittest:
	export PYTHONPATH=$(PWD) && echo $(PYTHONPATH) && pytest


# Run the main script.
gwrDemo:
	python script-gwr-demo.py


lgwrDemo:
	python script-lgwr-demo.py
lgwrTorchDemo:
	python script-lgwr-torch-demo.py


lmgwrDemo:
	python script-lmgwr-demo.py


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