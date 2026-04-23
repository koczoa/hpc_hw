#!/bin/bash
#BSUB -J static_parallel
#BSUB -q hpc
#BSUB -o static%J.out
#BSUB -e static%J.err
#BSUB -R "rusage[mem=4096]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 60
#BSUB -n 16
#BSUB -R "span[hosts=1]"

conda activate 02613_2026

N=100
for workers in 1 2 4 8 16; do
    python simulate_parallel_static.py $N $workers
done