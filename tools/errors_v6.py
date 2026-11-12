"""Remaining errors of the v6 pipeline (template + version from date) on out-of-fold predictions."""
import sys, datetime as dt, collections
import numpy as np
from scoring import fold_of_session, window_score
from template_decode import build_templates, decode

tag, kind = sys.argv[1], sys.argv[2]
d = np.load("../runs/dataset_v4.npz", allow_pickle=True); sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
start = np.array([d["win"][d["sess"] == i][0] * 10 for i in range(len(sid))])
ver = ["V1" if dt.datetime.fromtimestamp(int(t), dt.UTC).month <= 4 else "V2" for t in start]
z = np.load(f"../runs/{tag}_{kind}.npz"); prob, y, sess, win = z["prob"], z["y"], z["sess"], z["win"]
fos = fold_of_session(kind, sid)
pred = prob.argmax(1).copy(); chosen = {}
for k in range(6):
    tr = np.flatnonzero(fos != k)
    T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
    for i in np.flatnonzero(fos == k):
        m = sess == i
        pred[m], chosen[i] = decode(prob[m], T, 0.002, 10, return_key=True, version=ver[i])
r = window_score(y, pred, sess, win)
print(f"{tag} {kind}: macro-F1 {r[0]:.4f} F1 {r[1].round(3)}\nwindow confusion (rows true R/S/A):\n{r[2]}")
ok = y >= 0; err = ok & (pred != y)
st = np.isin(sess, [i for i in range(len(sid)) if prot[i] == "STRESS"])
print(f"\nerror seconds {err.sum()}: stress recordings {(err & st).sum()} ({(err & st).sum() / err.sum():.0%}), exercise {(err & ~st).sum()}")
names = {0: "R", 1: "S", 2: "A"}
c = collections.Counter((names[a], names[b], "stress-rec" if s else "exercise-rec") for a, b, s in zip(y[err], pred[err], st[err]))
for k_, v in c.most_common(): print(f"   true {k_[0]} -> pred {k_[1]} in {k_[2]}: {v}")
# errors by true phase segment index within stress recordings (which task is hard?)
print("\nstress recordings: per true segment (in order), seconds wrong / length")
for vv in ("V1", "V2"):
    rows = collections.defaultdict(lambda: [0, 0])
    for i in range(len(sid)):
        if prot[i] != "STRESS" or keys[i][1] != vv: continue
        m = np.flatnonzero(sess == i); yy, pp = y[m], pred[m]
        seg = np.r_[0, np.cumsum(yy[1:] != yy[:-1])]
        for j in np.unique(seg):
            mm = seg == j
            if yy[mm][0] < 0 or mm.sum() < 5: continue
            rows[(j, names[int(yy[mm][0])])][0] += int((pp[mm] != yy[mm]).sum()); rows[(j, names[int(yy[mm][0])])][1] += int(mm.sum())
    print(f"  {vv}: " + "  ".join(f"#{j}{c_}:{a}/{b}" for (j, c_), (a, b) in sorted(rows.items())))
d_ = np.full(len(y), 10**9)
for i in range(len(sid)):
    idx = np.flatnonzero(sess == i); yy = y[idx]; ch = np.flatnonzero(yy[1:] != yy[:-1]) + 1
    if len(ch): d_[idx] = np.min(np.abs(np.arange(len(yy))[:, None] - ch[None, :]), axis=1)
print("\nerror distance to nearest true boundary: " + "  ".join(f"<{h}s:{(err & (d_ < h)).sum() / err.sum():.0%}" for h in (10, 30, 60, 120)))
