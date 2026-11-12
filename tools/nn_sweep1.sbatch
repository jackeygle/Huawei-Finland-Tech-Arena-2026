#!/bin/bash
#SBATCH --job-name=tcn_sweep1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/tcn_sweep1_%A_%a.log
# Anchor = TCN (ch 48, drop 0.2, featdrop 0.1, wd 1e-2, 1500 steps). Each task changes ONE setting.
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
V=("--tag tcnA_anchor"
   "--seed 1 --tag tcnA_seed1"
   "--seed 2 --tag tcnA_seed2"
   "--steps 750 --tag tcnA_steps750"
   "--drop 0.4 --tag tcnA_drop0.4"
   "--featdrop 0.3 --tag tcnA_featdrop0.3"
   "--wd 0.05 --tag tcnA_wd0.05"
   "--ch 32 --tag tcnA_ch32")
echo "variant: ${V[$SLURM_ARRAY_TASK_ID]}"
python3 -u nn_cv.py --model tcn ${V[$SLURM_ARRAY_TASK_ID]} 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
