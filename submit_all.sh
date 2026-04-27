#!/bin/bash
#BSUB -J full_run
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o full_run_%J.out
#BSUB -e full_run_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python simulate_all.py 32 results_all.csv