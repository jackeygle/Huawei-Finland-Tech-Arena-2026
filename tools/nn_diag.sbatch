#!/bin/bash
#SBATCH --job-name=tcn_diag
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/tcn_diag_%A_%a.log
# Diagnostics around the TCN anchor. 0: learning curve. 1-6: drop one feature group. 7-9: data changes.
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
V=("--split subject --steps 3000 --eval_every 250 --tag diag_curve3000"
   "--groups rel,sess,eda2,hrv,spec --tag diag_no_local"
   "--groups local,sess,eda2,hrv,spec --tag diag_no_rel"
   "--groups local,rel,eda2,hrv,spec --tag diag_no_sess"
   "--groups local,rel,sess,hrv,spec --tag diag_no_eda2"
   "--groups local,rel,sess,eda2,spec --tag diag_no_hrv"
   "--groups local,rel,sess,eda2,hrv --tag diag_no_spec"
   "--exclude_train AEROBIC/S15,AEROBIC/S16 --tag diag_excl_S15S16"
   "--boundary_ignore 15 --tag diag_bound15"
   "--boundary_ignore 30 --tag diag_bound30")
echo "variant: ${V[$SLURM_ARRAY_TASK_ID]}"
python3 -u nn_cv.py --model tcn ${V[$SLURM_ARRAY_TASK_ID]} 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
