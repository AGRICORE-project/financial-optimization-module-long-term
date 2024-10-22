#!/bin/bash

# Install requirements
python3.10 -m pip install -r ./requirements.txt

# Run cProfile and output to program.prof
# python3.10 -m cProfile -o program.prof ./model/algorithm_ABM.py
python3.10 ./model/algorithm_ABM.py --skipParallel --profile

# Start snakeviz server to visualize the profile
python3.10 -m snakeviz stats.prof --server