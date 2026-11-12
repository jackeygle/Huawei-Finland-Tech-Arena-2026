#!/bin/bash
# Normal V100 queue, one small task per array index. Usage:
#   sbatch --array=0-N --job-name=<name> --export=ALL,RUNFILE=<file with one args line per task> arr.sbatch
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/%x_%A_%a.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync
export OMP_NUM_THREADS=4
# RUNFILE: one line of nn_cv_acc.py arguments per array index (sbatch --export splits values at commas)
args=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNFILE")
echo "args: $args"
[ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ -n "$POOLCHECK" ] && python3 -u onnx_pool_check.py 2>&1 | grep -v -i warn
python3 -u nn_cv_acc.py --model tcn --steps 3000 $args 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
