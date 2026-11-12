#!/bin/bash
#SBATCH --job-name=tcn_v3
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tcn_v3_%j.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
G=local,rel,sess,eda2,hrv,spec,pos
python3 -u nn_train_full.py --name v3_pos3000 --groups $G --steps 3000 --seeds 0 2>&1 | grep --line-buffered -v -i warn
python3 -u nn_train_full.py --name v3_pos3000_ens3 --groups $G --steps 3000 --seeds 0,1,2 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
