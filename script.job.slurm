#!/bin/bash
#SBATCH --job-name=cs633_assignment2
#SBATCH --output=cs633_assignment2_%j.out
#SBATCH --error=cs633_assignment2_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

module load mpi 2>/dev/null || true

mpicc src.c -lm -o a.out

d=7
T=5
seed=1000
F=2
isovalue=500

process_counts=(32 48 64 96)
px_values=(4 6 4 6)
py_values=(4 4 4 4)
pz_values=(2 2 4 4)
ppn_values=(32 48 32 48)
grid_sizes=(120 240)

for execution in 1 2 3 4 5; do
    for i in "${!process_counts[@]}"; do
        P=${process_counts[$i]}
        px=${px_values[$i]}
        py=${py_values[$i]}
        pz=${pz_values[$i]}
        ppn=${ppn_values[$i]}

        for n in "${grid_sizes[@]}"; do
            echo "execution=${execution} P=${P} ppn=${ppn} px=${px} py=${py} pz=${pz} nx=${n} ny=${n} nz=${n}"
            srun -n "${P}" --ntasks-per-node="${ppn}" ./a.out \
                "${d}" "${ppn}" "${px}" "${py}" "${pz}" \
                "${n}" "${n}" "${n}" \
                "${T}" "${seed}" "${F}" "${isovalue}"
        done
    done
done
