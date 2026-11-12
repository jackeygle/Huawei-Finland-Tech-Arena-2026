#!/bin/bash
# One V100, several CV runs in parallel (each TCN run uses ~0.5 GB and ~20% of the GPU; CPU is the limit).
# Usage: sbatch --export=ALL,RUNS="args1;args2;..." packed.sbatch     (each item = extra args for nn_cv_acc.py)
#SBATCH --job-name=acc_pack
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/pack_%j.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync
export OMP_NUM_THREADS=2
IFS=';' read -ra R <<< "$RUNS"
for args in "${R[@]}"; do
  tag=$(echo "$args" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')
  python3 -u nn_cv_acc.py --model tcn --steps 3000 $args 2>&1 | grep --line-buffered -v -i warn > logs/run_${tag}.log &
done
wait
echo ALLDONE
