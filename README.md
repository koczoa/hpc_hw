# Initial remarks
```shell
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
```

with these setups, the time needed for the 20 floor plans is: `402.26 sec`

Initial notes on the code:
```python
u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
```
this line is basically a 2d matrix eval, run a lot repetitively `MAX_ITER`, this is what we need to optimise.

Ideas:
- parallelism
  - loading the buildings
  - calculating the buildings
- better submit.sh
- using GPU
- Jakobi is slow:
