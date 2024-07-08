#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
echo "run limited grid search CNNs"
N_setups=14
N_runs=10

for ((i = 1; i <= $N_setups; i++))
do
    for ((run = 1; run <= $N_runs; run++))
    do
        echo "running setup $i run $run"
        python LF_simulation.py inputs/gridsearch/TO/in_TO_$i.json $run
        
    done
done