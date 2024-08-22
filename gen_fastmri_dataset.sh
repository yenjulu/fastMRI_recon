#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=6:00:00
#SBATCH --output=log.txt

python fastMRI_to_dataset.py