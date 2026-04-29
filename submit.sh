#!/bin/bash
#BSUB -J timing_ref
#BSUB -q hpc
#BSUB -o timing_ref_%J.out
#BSUB -e timing_ref_%J.err
#BSUB -R "rusage[mem=2048]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 15
#BSUB -n 1
#BSUB -R "span[hosts=1]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

N=20
START=$(date +%s%N)
python simulate.py $N > /dev/null
END=$(date +%s%N)

ELAPSED=$(echo "scale=3; ($END - $START) / 1000000000" | bc)
PER_FLOOR=$(echo "scale=3; $ELAPSED / $N" | bc)
TOTAL_H=$(echo "scale=2; $ELAPSED / $N * 4571 / 3600" | bc)

echo "# elapsed: ${ELAPSED} s, N: ${N}"
echo "# per floorplan: ${PER_FLOOR} s"
echo "# estimated total (4571 floorplans): ${TOTAL_H} h"