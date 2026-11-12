#!/bin/bash
#SBATCH --job-name=hta_time
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/time_%j.log
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
lscpu | grep "Model name"
for z in "$@"; do echo "== $z"; ../env313/bin/python -u local_score.py $z --runs 20 2>&1 | grep --line-buffered -v -i warn; done
rm -rf ../runs/tmp*
echo ALLDONE
