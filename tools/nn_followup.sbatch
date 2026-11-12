#!/bin/bash
#SBATCH --job-name=tcn_follow
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-8
#SBATCH --output=logs/tcn_follow_%A_%a.log
# Follow-ups from the diagnostics, seeds 0/1/2 each:
#  a) anchor features, 3000 steps (learning curve still rising at 1500)
#  b) no session-level groups (drop rel, sess): local,eda2,hrv,spec
#  c) minimal: local,hrv,spec
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
V=("--steps 3000 --tag fu_steps3000"
   "--groups local,eda2,hrv,spec --tag fu_nosess"
   "--groups local,hrv,spec --tag fu_min")
i=$SLURM_ARRAY_TASK_ID; seed=$((i % 3)); v="${V[$((i / 3))]}"
echo "variant: $v seed $seed"
python3 -u nn_cv.py --model tcn ${v}_s$seed --seed $seed 2>&1 | sed -u "s/^fu_/fu_s${seed}_/" | grep --line-buffered -v -i warn
echo ALLDONE
