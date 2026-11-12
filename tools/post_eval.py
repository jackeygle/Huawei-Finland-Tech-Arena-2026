"""Post-processing experiments on saved out-of-fold probabilities (no retraining).

  gate   : decide per input whether it is an exercise recording (mean P(Activity) > thr);
           exercise -> Stress impossible, otherwise -> Activity impossible.
  bias   : add a constant to log P(Stress).
  stride : run the classifier only every k-th second and repeat the label.
"""
import sys
import numpy as np
import common as C

z = np.load(sys.argv[1]); prob, y, sess, win = z["prob"], z["y"], z["sess"], z["win"]


def gate(p, thr=0.3):
    q = p.copy()
    for k in np.unique(sess):
        m = sess == k
        if p[m, 2].mean() > thr: q[m, 1] = 0
        else: q[m, 2] = 0
    return q


def stride(p, k):
    q = p.copy()
    for s in np.unique(sess):
        idx = np.flatnonzero(sess == s); n = len(idx)
        src = np.minimum((np.arange(n) // k) * k + k // 2, n - 1)
        q[idx] = p[idx[src]]
    return q


def show(name, p, bias=(0, 0, 0)):
    pred = (np.log(p + 1e-9) + np.array(bias)).argmax(1)
    mf, f, cm = C.window_score(y, pred, sess, win)
    print(f"{name:34s} macro-F1={mf:.4f} quality={mf*80:5.1f}  F1={f.round(3)}", flush=True)
    return mf


show("baseline", prob)
ex = np.array([prob[sess == k, 2].mean() for k in np.unique(sess)])
d = C.build_dataset(); is_ex = d["prot"] != "STRESS"
print(f"mean P(Act) per session: exercise min {ex[is_ex].min():.2f}  stress max {ex[~is_ex].max():.2f}")
g = gate(prob); show("gate", g)
for b in (0.5, 1.0, 1.5, 2.0):
    show(f"gate + stress bias {b}", g, (0, b, 0))
for k in (5, 10):
    show(f"gate + stride {k}s", stride(g, k))
