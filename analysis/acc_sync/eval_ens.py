"""v7/v8 decoder config: per-seed mean and 3-seed ensemble for tag prefixes. From tools/:
   ../env313/bin/python ../analysis/acc_sync/eval_ens.py u4so_p10d4 p10k6 ..."""
import os, sys, numpy as np
ACC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, "."); sys.path.insert(0, ACC)
import decode4 as D
from scoring import fold_of_session, window_score
from template_decode import build_templates
def run(tags, kind):
    Z = [np.load(os.path.join(ACC, "oof", f"nn_{t}_{kind}.npz")) for t in tags]
    prob = np.mean([z["prob"][:, :7] for z in Z], 0); y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
    fos = fold_of_session(kind, D.sid); pred = prob[:, :3].argmax(1).copy()
    for k in range(6):
        tr = np.flatnonzero(fos != k); T = build_templates([y[sess == i] for i in tr], [D.keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i; pred[m] = D.decode4(prob[m], T, D.keys[i][1], 4, 3, beta=0.05, u_emit=True, beta_so=0.1)
    return window_score(y, pred, sess, win)
for pre in sys.argv[1:]:
    out = []
    for kind in ("subject", "session"):
        tags = [f"{pre}_s{s}" for s in range(3)]
        if not all(os.path.exists(os.path.join(ACC, "oof", f"nn_{t}_{kind}.npz")) for t in tags):
            out.append("   (missing)   "); continue
        single = np.mean([run([t], kind)[0] for t in tags]); e = run(tags, kind)
        out.append(f"single {single:.4f} ens {e[0]:.4f} (stress {e[1][1]:.3f})")
    print(f"{pre:12s} subject: {out[0]} | session: {out[1]}", flush=True)
