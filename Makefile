.PHONY: unittest run exportCondaEnv importCondaEnv setup


# Run unit test for all test scripts.
unittest:
	PYTHONPATH=./src pytest

# Run the main script.
run:
	python index.py


# Export the conda environment to a file. 
# (includes all dependencies and python version)
exportCondaEnv:
	conda env export -n gwr -f environment-linux.yml
# Create a new conda environment from the file.
importCondaEnv:
	conda env create -f environment-linux.yml
# Initialize the src directory (for absolute imports).
setup:
	pip install -e .