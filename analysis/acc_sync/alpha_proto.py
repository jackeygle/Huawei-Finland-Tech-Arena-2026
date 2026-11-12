import sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "../analysis/acc_sync")
import cens_test as CT
from align2 import align2
orig = CT.align2
def run_ap(tag, kind, a_st, a_ex):
    def patched(lp, segs, alpha, res, skip_pen, cap):
        a = a_st if any(g["c"] == 1 for g in segs) else a_ex
        return align2(lp, segs, a, res, skip_pen, cap)
    CT.align2 = patched
    try: return CT.run(tag, kind, 4.0)[0][0]
    finally: CT.align2 = orig
tags = ["nn_fu_steps3000_s0", "nn_fu_steps3000_s1", "nn_fu_steps3000_s2"]
def score(a_st, a_ex):
    v = [[run_ap(t, k, a_st, a_ex) for t in tags] for k in ("subject", "session")]
    return np.mean(v[0]), np.mean(v[1])
for a_st in (0.001, 0.0015, 0.002, 0.003, 0.004):
    s = score(a_st, 0.002); print(f"alpha STRESS {a_st:<6} exercise 0.002  mean over 3 seeds: subject {s[0]:.4f} session {s[1]:.4f}", flush=True)
for a_ex in (0.001, 0.004, 0.008, 0.02):
    s = score(0.002, a_ex); print(f"alpha STRESS 0.002   exercise {a_ex:<6} mean over 3 seeds: subject {s[0]:.4f} session {s[1]:.4f}", flush=True)
