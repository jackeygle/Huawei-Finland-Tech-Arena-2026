#!/bin/bash
#SBATCH --job-name=acc_u4
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/u4_%A_%a.log
# 0-2: 4-class (U as class), seeds 0/1/2;  3-5: 4-class + boundary head, seeds 0/1/2.  Same setup as TCN3000.
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync
export OMP_NUM_THREADS=4
i=$SLURM_ARRAY_TASK_ID; s=$((i % 3))
if [ $i -lt 3 ]; then args="--u_class 1 --tag u4_s$s"; else args="--u_class 1 --bnd 1 --tag u4b_s$s"; fi
echo "variant: $args seed $s"
python3 -u nn_cv_acc.py --model tcn --steps 3000 --seed $s $args 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
