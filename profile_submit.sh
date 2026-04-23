#BSUB -J profile
#BSUB -q hpc
#BSUB -o profile%J.out
#BSUB -e profile%J.err
#BSUB -R "rusage[mem=2048]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 15
#BSUB -n 1

conda activate 02613_2026
kernprof -l -v profile_jacobi.py