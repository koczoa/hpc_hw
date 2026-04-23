#!/bin/bash
#BSUB -J dynamic_parallel
#BSUB -q hpc
#BSUB -o dynamic%J.out
#BSUB -e dynamic%J.err
#BSUB -R "rusage[mem=4096]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 60
#BSUB -n 16
#BSUB -R "span[hosts=1]"

source /zhome/4a/9/168632/.bashrc
conda activate 02613_2026

N=100
for workers in 1 2 4 8 16; do
    python simulate_parallel_dynamic.py $N $workers
done