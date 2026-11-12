#!/bin/bash
#SBATCH --job-name=hta_build
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/build_%j.log
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
OMP_NUM_THREADS=2 ../env313/bin/python -u -c "import common as C; d=C.build_dataset(rebuild=True); print(d['X'].shape)"
