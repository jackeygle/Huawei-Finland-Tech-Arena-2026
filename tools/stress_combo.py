"""C2 evaluation: general model chooses the template (version from the date); inside STRESS templates the
alignment uses a stress-specialist model's probabilities instead. Usage: stress_combo.py <general> <specialist> [kinds]"""
import sys
import datetime as dt
import numpy as np
from scoring import fold_of_session, window_score
from template_decode import build_templates, decode

gen_tag, spe_tag = sys.argv[1], sys.argv[2]
kinds = sys.argv[3].split(",") if len(sys.argv) > 3 else ["subject", "session"]
d = np.load("../runs/dataset_v4.npz", allow_pickle=True); sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
start = np.array([d["win"][d["sess"] == i][0] * 10 for i in range(len(sid))])
ver = ["V1" if dt.datetime.fromtimestamp(int(t), dt.UTC).month <= 4 else "V2" for t in start]
for kind in kinds:
    G = np.load(f"../runs/{gen_tag}_{kind}.npz"); Sp = [np.load(f"../runs/{t}_{kind}.npz") for t in spe_tag.split("+")]
    pg, y, sess, win = G["prob"], G["y"], G["sess"], G["win"]
    ps = np.mean([z["prob"] for z in Sp], axis=0)
    fos = fold_of_session(kind, sid)
    base = pg.argmax(1).copy(); combo = base.copy()
    for k in range(6):
        tr = np.flatnonzero(fos != k)
        T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i
            base[m], key = decode(pg[m], T, 0.002, 10, return_key=True, version=ver[i])
            combo[m] = base[m]
            if key[0] == "STRESS":
                combo[m] = decode(ps[m], {key: T[key]}, 0.002, 10)
    st = np.isin(sess, [i for i in range(len(sid)) if prot[i] == "STRESS"])
    for name, p in (("general only", base), ("general + stress specialist", combo)):
        r = window_score(y, p, sess, win)
        rs = window_score(y[st], p[st], sess[st], win[st])
        print(f"[{kind}] {name:28s} overall macro-F1 {r[0]:.4f} F1 {r[1].round(3)} | stress sessions: Rest F1 {rs[1][0]:.3f} Stress F1 {rs[1][1]:.3f}", flush=True)
