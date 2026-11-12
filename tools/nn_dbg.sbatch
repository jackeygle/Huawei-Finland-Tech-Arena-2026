#!/bin/bash
#SBATCH --job-name=hta_dbg
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dbg_%x_%j.log
# Generic gpu-debug CV runner: sbatch --job-name=<tag> tools/nn_dbg.sbatch <nn_cv.py args>
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
echo "args: $@"
python3 -u nn_cv.py "$@" 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
