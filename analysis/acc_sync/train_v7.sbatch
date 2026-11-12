#!/bin/bash
#SBATCH --job-name=v7_train
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/train_v7_%j.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync
export OMP_NUM_THREADS=4
python3 -u train_full_u4so.py --name v7_u4so_model --seed 0 --steps 3000 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
