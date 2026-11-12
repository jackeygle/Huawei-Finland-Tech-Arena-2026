#!/bin/bash
# gpu-debug runner for nn_cv_phase.py: sbatch --job-name=<tag> run_dbg.sbatch <args>
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/phase_head/logs/%x_%j.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/analysis/phase_head
export OMP_NUM_THREADS=4
echo "args: $@"
python3 -u nn_cv_phase.py --model tcn --steps 3000 "$@" 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
