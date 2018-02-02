#!/bin/bash

#SBATCH --qos=heavy
#SBATCH -p serial
#SBATCH -N 3 # node count
#SBATCH -t 100:00:00
#SBATCH -c 4
#SBATCH --ntasks-per-node=7
#SBATCH --mem=100000
export SLURM_MPI_TYPE=pmi2

export TMPDIR="/scratch/global/zhendong"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
srun python -u test_mpi.py > test_mpi.out
