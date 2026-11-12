#!/bin/bash
#SBATCH --job-name=hta_cv
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cv_groups_%j.log
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=8
for g in "local,rel,sess" "local,rel,sess,sess2" "local,rel,sess,sess2,pos"; do
  tag=$(echo $g | tr ',' '_'); echo "##### $g"
  ../env313/bin/python -u cv.py --groups $g --smooth 0 --save ../runs/cv_$tag 2>&1 | grep -v -i warn
done
echo ALLDONE
