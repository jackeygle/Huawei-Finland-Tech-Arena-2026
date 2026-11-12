#!/bin/bash
#SBATCH --job-name=tcn_stress
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/tcn_stress_%A_%a.log
# Stress specialist: same TCN and features, trained on STRESS recordings only (Rest vs Stress), 3000 steps.
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
s=$SLURM_ARRAY_TASK_ID
echo "variant: stress-only training seed $s"
python3 -u nn_cv.py --model tcn --train_prot STRESS --steps 3000 --seed $s --tag stressonly_s$s 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
