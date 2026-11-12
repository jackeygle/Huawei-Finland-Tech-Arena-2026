"""Compare emission models (and mixtures) under the full v7 pipeline: template + version from date."""
import sys, datetime as dt
import numpy as np
from scoring import fold_of_session, window_score
from template_decode import build_templates, decode

SETS = {
    "TCN (current, 3000 steps)": ["nn_fu_steps3000_s0"],
    "TCN 1500 steps": ["nn_tcnA_anchor"],
    "tree (GBDT, signal features)": ["cvsig_local_rel_sess_eda2_hrv_spec"],
    "transformer": ["nn_trans"],
    "raw-waveform CNN": ["nn_raw"],
    "TCN + tree": ["nn_fu_steps3000_s0", "cvsig_local_rel_sess_eda2_hrv_spec"],
    "TCN + transformer": ["nn_fu_steps3000_s0", "nn_trans"],
    "TCN + raw CNN": ["nn_fu_steps3000_s0", "nn_raw"],
    "TCN + tree + transformer + raw": ["nn_fu_steps3000_s0", "cvsig_local_rel_sess_eda2_hrv_spec", "nn_trans", "nn_raw"],
}
d = np.load("../runs/dataset_v4.npz", allow_pickle=True); sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
start = np.array([d["win"][d["sess"] == i][0] * 10 for i in range(len(sid))])
ver = ["V1" if dt.datetime.fromtimestamp(int(t), dt.UTC).month <= 4 else "V2" for t in start]
for kind in ("subject", "session"):
    y0 = None
    fos = fold_of_session(kind, sid)
    for name, tags in SETS.items():
        Z = [np.load(f"../runs/{t}_{kind}.npz") for t in tags]
        prob = np.mean([z["prob"] for z in Z], 0); y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
        assert all((z["y"] == y).all() for z in Z)
        raw = window_score(y, prob.argmax(1), sess, win)[0]
        p = prob.argmax(1).copy()
        for k in range(6):
            tr = np.flatnonzero(fos != k)
            T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
            for i in np.flatnonzero(fos == k):
                m = sess == i
                p[m] = decode(prob[m], T, 0.002, 10, version=ver[i])
        r = window_score(y, p, sess, win)
        print(f"[{kind}] {name:32s} model alone {raw:.4f} | + template + date {r[0]:.4f}  F1 {r[1].round(3)}", flush=True)
