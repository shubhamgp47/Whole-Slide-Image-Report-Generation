#!/bin/bash -l
#SBATCH --job-name=TITAN_Slide_Embeddings
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
conda activate histgen_titan
python /home/woody/iwi5/iwi5204h/HistGen4TITAN/extractSlideEmbeddings.py