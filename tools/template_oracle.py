"""Upper bounds: decoded score if the protocol template (or only its version / type) were known."""
import sys
import numpy as np
from scoring import fold_of_session, window_score
from template_decode import build_templates, decode

tag = sys.argv[1]
alpha, res = 0.002, 10
d = np.load("../runs/dataset_v4.npz", allow_pickle=True); sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
for kind in ("subject", "session"):
    z = np.load(f"../runs/{tag}_{kind}.npz"); prob, y, sess, win = z["prob"], z["y"], z["sess"], z["win"]
    fos = fold_of_session(kind, sid)
    preds = {m: prob.argmax(1).copy() for m in ("free", "oracle_version", "oracle_template")}
    for k in range(6):
        tr = np.flatnonzero(fos != k)
        T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i
            preds["free"][m] = decode(prob[m], T, alpha, res)
            preds["oracle_version"][m] = decode(prob[m], {kk: v for kk, v in T.items() if kk[1] == keys[i][1]}, alpha, res)
            preds["oracle_template"][m] = decode(prob[m], {keys[i]: T[keys[i]]}, alpha, res)
    for name, p in preds.items():
        r = window_score(y, p, sess, win)
        print(f"[{kind}] {name:16s} macro-F1 {r[0]:.4f}  F1 {r[1].round(3)}", flush=True)
