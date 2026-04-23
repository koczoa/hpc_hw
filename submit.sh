#!/bin/bash
#BSUB -J hw
#BSUB -q hpc
#BSUB -o hw%J.out
#BSUB -e hw%J.err
#BSUB -R "rusage[mem=2048]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 15
#BSUB -n 2
#BSUB -R "span[hosts=1]"


conda activate 02613_2026

python simulate.py 20