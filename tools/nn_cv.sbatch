#!/bin/bash
#SBATCH --job-name=hta_nn
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nn_%x_%j.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u nn_cv.py "$@" 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
