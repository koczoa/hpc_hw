#!/bin/bash
#BSUB -J numba_cpu
#BSUB -q hpc
#BSUB -o numba%J.out
#BSUB -e numba%J.err
#BSUB -R "rusage[mem=4096]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 30
#BSUB -n 1

conda activate 02613_2026

python simulate_numba.py 10