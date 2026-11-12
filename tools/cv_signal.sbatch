#!/bin/bash
#SBATCH --job-name=hta_cvsig
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-4
#SBATCH --output=logs/cv_signal_%A_%a.log
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=8
V=("local,rel,sess" "local,rel,sess,eda2" "local,rel,sess,hrv" "local,rel,sess,spec" "local,rel,sess,eda2,hrv,spec")
g=${V[$SLURM_ARRAY_TASK_ID]}; tag=$(echo $g | tr ',' '_')
echo "##### $g"
../env313/bin/python -u cv.py --groups $g --smooth 0 --save ../runs/cvsig_$tag 2>&1 | grep -v -i warn
echo ALLDONE
