#!/bin/bash
#BSUB -J dynamic_sweep
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o dynamic_sweep_%J.out
#BSUB -e dynamic_sweep_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

for P in 1 2 4 8 16 32; do
    echo "=== Workers: $P ===" >&2
    python simulate_dynamic.py 100 $P > /dev/null
done
