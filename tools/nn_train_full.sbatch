#!/bin/bash
#SBATCH --job-name=tcn_full
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tcn_full_%j.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
python3 -u nn_train_full.py --name v2_tcn --steps 1500 --seeds 0 2>&1 | grep --line-buffered -v -i warn
python3 -u nn_train_full.py --name v2_tcn3000 --steps 3000 --seeds 0 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
