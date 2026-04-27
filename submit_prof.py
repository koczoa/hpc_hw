#!/bin/bash
#BSUB -J prof_jacobi
#BSUB -q hpc
#BSUB -W 00:30
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -o prof_%J.out
#BSUB -e prof_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Run with kernprof — produces simulate_prof.py.lprof
kernprof -l simulate_prof.py 5 > /dev/null

# Format and print the report
python -m line_profiler -rmt simulate_prof.py.lprof
