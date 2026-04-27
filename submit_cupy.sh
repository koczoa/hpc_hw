#!/bin/bash
#BSUB -J cupy_test
#BSUB -q c02613
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o cupy_%J.out
#BSUB -e cupy_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python simulate_cupy.py 20 > /dev/null