"""3-seed mean of the v7 decoder config (beta 0.05, U emission, beta_so 0.1) for tag prefixes. From tools/:
   ../env313/bin/python ../analysis/acc_sync/eval_best.py u4so u4so_p5d5 u4so_p10d4"""
import os, sys, numpy as np
ACC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, "."); sys.path.insert(0, ACC)
import decode4 as D
from scoring import fold_of_session, window_score
from template_decode import build_templates
for pre in sys.argv[1:]:
    for kind in ("subject", "session"):
        r = []
        for s in (0, 1, 2):
            f = os.path.join(ACC, "oof", f"nn_{pre}_s{s}_{kind}.npz")
            if not os.path.exists(f): continue
            Z = np.load(f); prob, y, sess, win = Z["prob"], Z["y"], Z["sess"], Z["win"]
            fos = fold_of_session(kind, D.sid); pred = prob[:, :3].argmax(1).copy()
            for k in range(6):
                tr = np.flatnonzero(fos != k); T = build_templates([y[sess == i] for i in tr], [D.keys[i] for i in tr])
                for i in np.flatnonzero(fos == k):
                    m = sess == i; pred[m] = D.decode4(prob[m], T, D.keys[i][1], 4, 3, beta=0.05, u_emit=True, beta_so=0.1)
            r.append(window_score(y, pred, sess, win))
        if r:
            print(f"{pre:12s} {kind:8s} n={len(r)} macro-F1 {np.mean([x[0] for x in r]):.4f} (seeds {[round(x[0], 4) for x in r]})  "
                  f"F1 R/S/A {np.mean([x[1] for x in r], 0).round(3)}", flush=True)
