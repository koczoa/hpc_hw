#!/bin/bash
#BSUB -J static_sweep
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o static_sweep_%J.out
#BSUB -e static_sweep_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

for W in 1 2 4 8 16 32; do
    echo "=== Workers: $W ===" >&2
    python simulate_static.py 100 $W > /dev/null
done
