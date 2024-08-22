#!/bin/bash
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=23:00:00

GPU_NUM=0
TEST_CONFIG_YAML="configs/fastmri_varnet.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test.py \
    --config=$TEST_CONFIG_YAML \
    --write_image=5 \
