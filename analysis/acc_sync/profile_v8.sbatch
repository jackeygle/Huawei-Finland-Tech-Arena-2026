#!/bin/bash
#SBATCH --job-name=v7_prof
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/prof_v8_%j.log
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
lscpu | grep "Model name"
../env313/bin/python -u ../analysis/acc_sync/profile_v7.py ../runs/v8_p10 "$@" 2>&1 | grep -v -i warn
echo ALLDONE
