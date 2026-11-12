"""Where do errors remain after template decoding? Per-session breakdown on out-of-fold predictions."""
import sys
import numpy as np
from scoring import fold_of_session, window_score
from template_decode import build_templates, decode

tag, kind = sys.argv[1], sys.argv[2]
alpha, res = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002, 10
d = np.load("../runs/dataset_v4.npz", allow_pickle=True); sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
z = np.load(f"../runs/{tag}_{kind}.npz"); prob, y, sess, win = z["prob"], z["y"], z["sess"], z["win"]
fos = fold_of_session(kind, sid)
pred = prob.argmax(1).copy(); chosen = {}
for k in range(6):
    tr = np.flatnonzero(fos != k)
    T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
    for i in np.flatnonzero(fos == k):
        m = sess == i
        pred[m], chosen[i] = decode(prob[m], T, alpha, res, return_key=True)
mf, f, cm = window_score(y, pred, sess, win)
print(f"{tag} {kind}: decoded macro-F1 {mf:.4f} F1 {f.round(3)}")
ok = y >= 0
err = ok & (pred != y)
# distance of each error to the nearest true label change (within the same session)
dist = np.full(len(y), 10**9)
for i in range(len(sid)):
    idx = np.flatnonzero(sess == i); yy = y[idx]
    ch = np.flatnonzero(yy[1:] != yy[:-1]) + 1
    if len(ch):
        t = np.arange(len(yy)); dd = np.min(np.abs(t[:, None] - ch[None, :]), axis=1); dist[idx] = dd
print(f"errors total {err.sum()} s")
for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120), (120, 10**9)]:
    m = err & (dist >= lo) & (dist < hi)
    print(f"  errors {lo:>4}-{hi if hi < 10**9 else 'inf':>4} s from a true boundary: {m.sum():6d} ({m.sum() / err.sum():.0%})")
rows = []
for i in range(len(sid)):
    m = (sess == i) & ok
    e = (pred[m] != y[m]).sum()
    rows.append((e, i))
print("\nworst sessions (seconds wrong / scored, true template -> chosen):")
for e, i in sorted(rows, reverse=True)[:15]:
    m = (sess == i) & ok
    yy, pp = y[m], pred[m]
    detail = " ".join(f"{a}->{b}:{((yy == a) & (pp == b)).sum()}" for a in range(3) for b in range(3) if a != b and ((yy == a) & (pp == b)).sum() > 0)
    print(f"  {prot[i]:9s} {sid[i]}  {e:5d}/{m.sum():5d}  {keys[i][0]}/{keys[i][1]} -> {chosen[i][0]}/{chosen[i][1]}   [{detail}]   (0=R 1=S 2=A)")
wrong_t = [i for i in range(len(sid)) if chosen[i] != keys[i]]
we = sum(((sess == i) & err).sum() for i in wrong_t)
print(f"\nsessions with wrong template: {len(wrong_t)}; their errors {we} s = {we / err.sum():.0%} of all errors")
from collections import Counter
print("  confusions:", Counter(f"{keys[i][0]}/{keys[i][1]}->{chosen[i][0]}/{chosen[i][1]}" for i in wrong_t).most_common())
