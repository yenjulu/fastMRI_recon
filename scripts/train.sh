#!/bin/bash
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=20:50:00

GPU_NUM=0
TRAIN_CONFIG_YAML="configs/fastmri.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py \
    --config=$TRAIN_CONFIG_YAML \
    --write_image=5 \
