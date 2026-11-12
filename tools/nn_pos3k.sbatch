#!/bin/bash
#SBATCH --job-name=tcn_pos3k
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/tcn_pos3k_%A_%a.log
# TCN anchor features + position-in-recording group (protocol timeline), seeds 0/1/2.
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
s=$SLURM_ARRAY_TASK_ID
echo "variant: +pos 3000 steps seed $s"
python3 -u nn_cv.py --model tcn --groups local,rel,sess,eda2,hrv,spec,pos --seed $s --steps 3000 --tag pos3k_s$s 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
