#!/bin/bash -l
#
#SBATCH --job-name=clam_patching_01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --partition=work
#SBATCH --hint=nomultithread
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Set number of threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python
conda activate clam

bash /home/woody/iwi5/iwi5204h/HistGen/CLAM/patching_scripts/1.sh