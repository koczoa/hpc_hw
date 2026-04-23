#!/bin/bash
#BSUB -J cupy_fixed
#BSUB -q gpuv100
#BSUB -o cupy_fixed%J.out
#BSUB -e cupy_fixed%J.err
#BSUB -R "rusage[mem=4096]"
#BSUB -W 30
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"

source /zhome/4a/9/168632/.bashrc
conda activate 02613_2026

python simulate_cupy_fixed.py 10