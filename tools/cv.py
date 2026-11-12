"""Cross-validation of the 1 Hz feature model.

Two splits:
  subject : all sessions of a subject held out together (unseen person).
  session : single sessions held out, the person's other protocols stay in training
            (matches the inferred make-up of the hidden evaluation set).
Usage: python tools/cv.py [--groups local,rel,sess] [--iters 200] [--split subject,session]
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--groups", default="local,rel,sess")
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--leaves", type=int, default=15)
ap.add_argument("--split", default="subject,session")
ap.add_argument("--folds", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smooth", default="0,30,60")
ap.add_argument("--save", default="")
ap.add_argument("--sub", type=int, default=5, help="train on every k-th second (neighbouring seconds are near-duplicates)")
a = ap.parse_args()

d = C.build_dataset()
X, y, sess, win, names = d["X"], d["y"], d["sess"], d["win"], d["names"]
sid_of_sess = d["sid"]


keep = np.array([C.M.group_of(n) in a.groups.split(",") for n in names])
X = X[:, keep]
print(f"features: {keep.sum()} ({a.groups})  rows: {len(X)}  scored: {(y >= 0).sum()}", flush=True)


def folds(kind):
    return C.fold_of_session(kind, sid_of_sess, a.folds, a.seed)


for kind in a.split.split(","):
    fold_of_sess = folds(kind)
    prob = np.zeros((len(X), 3))
    t = time.time()
    for k in range(a.folds):
        te = fold_of_sess[sess] == k
        tr = ~te & (y >= 0) & (np.arange(len(y)) % a.sub == 0)
        clf = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=a.lr, max_leaf_nodes=a.leaves,
                                             class_weight="balanced", random_state=a.seed)
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])
        print(f"  [{kind}] fold {k} done {time.time() - t:.0f}s", flush=True)
    for s in [int(v) for v in a.smooth.split(",")]:
        pred = C.M.postprocess(prob, smooth_s=0, bias=(0, 0, 0)) if s == 0 else None
        if s:
            ps = np.empty_like(prob)
            for j in np.unique(sess):           # smooth inside each session only
                m = sess == j
                ps[m] = C.M._roll(prob[m], s)[0]
            pred = ps.argmax(axis=1)
        mf, f, cm = C.window_score(y, pred, sess, win)
        print(f"{kind:8s} smooth={s:3d}s  macro-F1={mf:.4f}  quality={mf * 80:.1f}/80  "
              f"F1 Rest/Stress/Act={f.round(3)}", flush=True)
        if s == 0:
            print("  confusion (rows = true Rest/Stress/Activity):\n" + str(cm), flush=True)
    if a.save:
        np.savez_compressed(f"{a.save}_{kind}.npz", prob=prob, y=y, sess=sess, win=win)
