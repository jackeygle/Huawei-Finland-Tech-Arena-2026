#!/bin/bash
#SBATCH --job-name=acc_post
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/work/zhangx29/huawei_tech_arena_2026/analysis/acc_sync/logs/post_tune_%j.log
cd /scratch/work/zhangx29/huawei_tech_arena_2026/tools
export OMP_NUM_THREADS=1
for t in u4b_k1_s0 u4b_k1_s1 u4b_k6_s0 u4b_k6_s1 u4b_w3_s0 u4b_w3_s1; do
  (../env313/bin/python -u ../analysis/acc_sync/decode4.py nn_$t 4 1 2>&1 | grep -v -i warn | sed "s/^/$t /" > ../analysis/acc_sync/logs/dec_$t.txt) &
done
wait
cd ../analysis/acc_sync/logs && python3 - <<'PY' > tune_summary.txt
import re, glob, collections
acc = collections.defaultdict(list)
for f in glob.glob("dec_u4b*_s*.txt"):
    for l in open(f):
        m = re.match(r"(u4b\w*?)_s\d \[(\w+)\] (\{.*?\})\s+argmax ([\d.]+) \| decoded macro-F1 ([\d.]+) F1 \[([\d. ]+)\]", l)
        if m: acc[(m[1], m[3].strip(), m[2])].append((float(m[5]), float(m[6].split()[1])))
for (mdl, cfg) in sorted({(a, b) for a, b, _ in acc}):
    s = acc[(mdl, cfg, "subject")]; t = acc[(mdl, cfg, "session")]
    print(f"{mdl:8s} {cfg:34s} subject {sum(x for x,_ in s)/len(s):.4f} session {sum(x for x,_ in t)/len(t):.4f} stressF1 {sum(y for _,y in s)/len(s):.3f} n={len(s)}")
PY
cat tune_summary.txt
echo ALLDONE
