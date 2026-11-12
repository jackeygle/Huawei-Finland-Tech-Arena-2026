#!/bin/bash
#SBATCH --job-name=tcn_speedvar
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/tcn_speedvar_%A_%a.log
# Speed variants vs the current model (TCN ch48, 3000 steps, groups local,rel,sess,eda2,hrv,spec), seeds 0/1/2:
#  0-2: drop the signal-processing groups (local,rel,sess only)
#  3-5: narrower network (ch 32)
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=4
i=$SLURM_ARRAY_TASK_ID; s=$((i % 3))
if [ $i -lt 3 ]; then args="--groups local,rel,sess --tag spd_nosig_s$s"; else args="--ch 32 --tag spd_ch32_s$s"; fi
echo "variant: $args seed $s"
python3 -u nn_cv.py --model tcn --steps 3000 --seed $s $args 2>&1 | grep --line-buffered -v -i warn
echo ALLDONE
