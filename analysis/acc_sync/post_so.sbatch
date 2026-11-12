#!/bin/bash
#SBATCH --job-name=acc_post_so
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/post_so_%j.log
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=1
for t in u4so_s0 u4so_s1 u4so_s2; do
  (../env313/bin/python -u ../analysis/acc_sync/decode4.py nn_$t 4 3 2>&1 | grep -v -i warn | sed "s/^/$t /" > ../analysis/acc_sync/logs/dec_$t.txt) &
done
wait
cd ../analysis/acc_sync/logs && python3 - <<'PY' | tee so_summary.txt
import re, glob, collections
acc = collections.defaultdict(list)
for f in sorted(glob.glob("dec_u4so_s*.txt")) + sorted(glob.glob("dec_u4b_s*.txt")):
    for l in open(f):
        m = re.match(r"(u4\w+?)_s\d \[(\w+)\] (\{.*?\})\s+argmax ([\d.]+) \| decoded macro-F1 ([\d.]+) F1 \[([\d. ]+)\]", l)
        if m and m[2] == "subject": acc[(m[1], m[3].strip())].append((float(m[5]), float(m[6].split()[1])))
print("subject split, 3-seed means")
for (mdl, cfg), v in sorted(acc.items()):
    if mdl == "u4b" and cfg != "{'beta': 0.05, 'u_emit': True}": continue
    print(f"{mdl:6s} {cfg:52s} macro-F1 {sum(x for x,_ in v)/len(v):.4f} stressF1 {sum(y for _,y in v)/len(v):.3f} n={len(v)}")
PY
echo ALLDONE
