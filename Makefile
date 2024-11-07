.PHONY: unittest run exportCondaEnv importCondaEnv

# Run unit test for all test scripts.
unittest:
	pytest

# Run the main script.
run:
	python index.py


# Export the conda environment to a file. 
# (includes all dependencies and python version)
exportCondaEnv:
	conda env export > environment.yml
# Create a new conda environment from the file.
importCondaEnv:
	conda env create -f environment.yml