#!/bin/bash
#BSUB -J all_floorplans
#BSUB -q gpuv100
#BSUB -o all_results%J.out
#BSUB -e all_results%J.err
#BSUB -R "rusage[mem=8192]"
#BSUB -W 120
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"

source /zhome/4a/9/168632/.bashrc
conda activate 02613_2026

python simulate_cupy_fixed.py 4571 > results.csv