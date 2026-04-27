#!/bin/bash
#BSUB -J jit_test
#BSUB -q hpc
#BSUB -W 00:30
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -o jit_%J.out
#BSUB -e jit_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Run on 20 floorplans for direct comparison with reference timing
python simulate_jit.py 20 > /dev/null
