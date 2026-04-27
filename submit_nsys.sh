#!/bin/bash
#BSUB -J cupy_fixed
#BSUB -q c02613
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o cupy_fixed_%J.out
#BSUB -e cupy_fixed_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

echo "=== ORIGINAL CuPy (sync every iteration), N=20 ===" >&2
python simulate_cupy.py 20 > /dev/null

echo "=== FIXED CuPy (sync every 1000 iterations), N=20 ===" >&2
python simulate_cupy_fixed.py 20 > /dev/null
