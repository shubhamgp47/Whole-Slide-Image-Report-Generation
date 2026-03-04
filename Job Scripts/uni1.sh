#!/bin/bash -l
#SBATCH --job-name=uni1_training_1_seed46
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
source ~/.bashrc
module load python
conda activate histgen
bash /home/woody/iwi5/iwi5204h/HistGen/train_wsi_report_uni1_seed4x.sh