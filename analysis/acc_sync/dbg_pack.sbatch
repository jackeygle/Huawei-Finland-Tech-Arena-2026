#!/bin/bash
# gpu-debug (30 min, 1 job/user): smoke test, then several CV runs in parallel on ONE GPU.
# Usage: sbatch --job-name=<name> --export=ALL,RUNS="args1;args2;..." dbg_pack.sbatch
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/%x_%j.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync
export OMP_NUM_THREADS=2
IFS=';' read -ra R <<< "$RUNS"
first=$(echo "${R[0]}" | sed 's/--tag [^ ]*/--tag smoke_tmp/')
python3 -u nn_cv_acc.py --model tcn --steps 20 --split subject --folds 6 $first 2>&1 | grep -v -i warn | tail -3 || exit 1
rm -f oof/nn_smoke_tmp_*
for args in "${R[@]}"; do
  tag=$(echo "$args" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')
  python3 -u nn_cv_acc.py --model tcn --steps 3000 $args 2>&1 | grep --line-buffered -v -i warn > logs/run_${tag}.log &
done
wait
grep -h "macro-F1=" logs/run_*.log
echo ALLDONE
