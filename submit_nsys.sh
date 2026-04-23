#!/bin/bash
#BSUB -J nsys_profile
#BSUB -q gpuv100
#BSUB -o nsys%J.out
#BSUB -e nsys%J.err
#BSUB -R "rusage[mem=4096]"
#BSUB -W 30
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"

source /zhome/4a/9/168632/.bashrc
conda activate 02613_2026

nsys profile --stats=true -o cupy_profile python profile_cupy.py