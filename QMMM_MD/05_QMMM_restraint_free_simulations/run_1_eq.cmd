#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --job-name=__JOBNAME__
#SBATCH --ntasks=__NTASK__
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

hostname
srun numactl -s

module load PrgEnv-gnu/8.5.0

source ~/.local/amber.sh



export MPICH_NO_BUFFER_ALIAS_CHECK=1

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun  --cpu-bind=cores sander.MPI -O -i in -o 0.out -c ../amber-nvt-workflow/04_NVT/structure_NVT_5.rst7 -r 0.rst7 -x 0.nc -inf 0.mdinfo -p ../amber-nvt-workflow/00_prep/structure.parm7


