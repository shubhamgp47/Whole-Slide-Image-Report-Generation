#!/bin/bash -l
#SBATCH --job-name=clam_extract_CONCH_1_5
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --export=NONE

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

unset SLURM_EXPORT_ENV
source ~/.bashrc
module load python
conda activate clam_conch
bash /home/woody/iwi5/iwi5204h/CLAM/extract_features_calling_script.sh