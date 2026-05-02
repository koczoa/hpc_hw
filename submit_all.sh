#!/bin/bash
#BSUB -J all_floorplans
#BSUB -q gpua100
#BSUB -o all_results%J.out
#BSUB -e all_results%J.err
#BSUB -R "rusage[mem=8192]"
#BSUB -W 20
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

#python simulate_cupy_fixed.py 4571 > results.csv
#python simulate_cupy_fixed.py 10 jacobi > jacobi.csv
#python simulate_cupy_fixed.py 10 sor    > sor.csv

#for w in 1.85 1.90 1.93 1.95 1.97; do
#    python simulate_cupy_fixed.py 10 sor $w 2>&1 | grep "time="
#done

python simulate_cupy_fixed.py 4571 sor 1.93 > results.csv
