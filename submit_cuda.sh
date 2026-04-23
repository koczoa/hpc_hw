#!/bin/bash
#BSUB -J cuda_jacobi
#BSUB -q gpuv100
#BSUB -o cuda%J.out
#BSUB -e cuda%J.err
#BSUB -R "rusage[mem=4096]"
#BSUB -W 30
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"

conda activate 02613_2026

python simulate_cuda.py 10