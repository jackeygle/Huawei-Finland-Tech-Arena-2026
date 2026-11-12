"""Post-processing experiments with the template decoder on saved out-of-fold probabilities.
Each variant changes one thing relative to the current setting (alpha 0.002, res 10).
Usage: python template_exp.py <oof-prefix> <variant> [<variant> ...]
variants: base | date | sel:<a> | shift:<s> | alpha:<stress>,<exercise>"""
import sys
import datetime as dt
import numpy as np
from scoring import fold_of_session, window_score
from template_decode import build_templates, decode

tag, variants = sys.argv[1], sys.argv[2:]
d = np.load("../runs/dataset_v4.npz", allow_pickle=True); sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
# recording date -> version rule (Feb-Mar = V1, Jun-Jul = V2), checked against the truth
start = np.array([d["win"][d["sess"] == i][0] * 10 for i in range(len(sid))])
month = np.array([dt.datetime.fromtimestamp(int(t), dt.UTC).month for t in start])
date_ver = np.where(month <= 4, "V1", "V2")
print("date rule matches version:", int((date_ver == np.array([k[1] for k in keys])).sum()), "/", len(sid),
      "| months:", sorted(set(month.tolist())))
for kind in ("subject", "session"):
    z = np.load(f"../runs/{tag}_{kind}.npz"); prob, y, sess, win = z["prob"], z["y"], z["sess"], z["win"]
    fos = fold_of_session(kind, sid)
    tmpl = {}
    for k in range(6):
        tr = np.flatnonzero(fos != k)
        tmpl[k] = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
    for v in variants:
        kw = dict(alpha=0.002, res=10)
        name, _, arg = v.partition(":")
        pred = prob.argmax(1).copy(); hit = 0
        for i in range(len(sid)):
            m = sess == i
            kk = dict(kw)
            if name == "date": kk["version"] = date_ver[i]
            if name == "sel": kk["alpha_sel"] = float(arg)
            if name == "shift": kk["shift"] = int(arg)
            if name == "alpha":
                a_s, a_e = map(float, arg.split(","))
                kk["alpha"] = {"STRESS": a_s, "AEROBIC": a_e, "ANAEROBIC": a_e}; kk["alpha_sel"] = 0.002
            pred[m], key = decode(prob[m], tmpl[fos[i]], return_key=True, **kk)
            hit += key == keys[i]
        r = window_score(y, pred, sess, win)
        print(f"[{kind}] {v:20s} macro-F1 {r[0]:.4f}  F1 {r[1].round(3)}  template exact {hit}/85", flush=True)
