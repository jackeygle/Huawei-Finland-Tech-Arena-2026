#!/bin/bash
#SBATCH --job-name=tcn_physio
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/tcn_physio_%A_%a.log
# TCN anchor feature set G + one physiology group (or all three), seeds 0/1/2 each.
# Baseline = tcn_sweep1 tasks 0-2 (anchor seeds 0/1/2, same folds).
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
G=local,rel,sess,eda2,hrv,spec
ADD=(resp pulse recov resp,pulse,recov)
i=$SLURM_ARRAY_TASK_ID; v=${ADD[$((i / 3))]}; seed=$((i % 3)); tag=phys_$(echo $v | tr ',' '+')_s$seed
echo "variant: +$v seed $seed"
python3 -u nn_cv.py --model tcn --groups $G,$v --seed $seed --tag $tag 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
