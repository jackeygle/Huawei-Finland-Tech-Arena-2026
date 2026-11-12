#!/bin/bash
#SBATCH --job-name=acc_smoke
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/smoke_%j.log
source /etc/profile
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync
export OMP_NUM_THREADS=4
for cfg in "--tag smoke3" "--u_class 1 --tag smoke4" "--u_class 1 --bnd 1 --tag smoke4b"; do
  python3 -u nn_cv_acc.py --model tcn --steps 30 --split subject --folds 6 $cfg 2>&1 | grep --line-buffered -v -i warn
done
python3 -c "
import numpy as np
for t in ('smoke3','smoke4','smoke4b'):
    z=np.load(f'oof/nn_{t}_subject.npz'); print(t, z['prob'].shape, z['prob'][:2].round(3))"
echo ALLDONE
