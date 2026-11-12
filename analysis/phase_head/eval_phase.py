"""Decode out-of-fold outputs with the parallel chat's decoder (acc_sync/decode4.py, read-only import) and
compare the phase-aux runs with the anchor u4b. Per seed and 3-seed mean, subject and session splits.
Usage: ../../env313/bin/python eval_phase.py <tag-prefix> [<tag-prefix> ...]   (e.g. ph05 u4b)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "acc_sync"))
import decode4 as D  # noqa: E402
from scoring import fold_of_session, window_score  # noqa: E402
from template_decode import build_templates  # noqa: E402

CFG = dict(beta=0.05, u_emit=True)


def oof(tag, kind):
    for base in (os.path.join(HERE, "oof"), os.path.join(HERE, "..", "acc_sync", "oof")):
        f = os.path.join(base, f"nn_{tag}_{kind}.npz")
        if os.path.exists(f):
            return np.load(f)
    return None


for prefix in sys.argv[1:]:
    for kind in ("subject", "session"):
        res = []
        for s in range(3):
            z = oof(f"{prefix}_s{s}", kind)
            if z is None:
                continue
            prob, y, sess, win = z["prob"], z["y"], z["sess"], z["win"]
            fos = fold_of_session(kind, D.sid)
            pred = prob[:, :3].argmax(1).copy()
            for k in range(6):
                tr = np.flatnonzero(fos != k)
                T = build_templates([y[sess == i] for i in tr], [D.keys[i] for i in tr])
                for i in np.flatnonzero(fos == k):
                    m = sess == i
                    pred[m] = D.decode4(prob[m], T, D.keys[i][1], 4, 1, **CFG)
            r = window_score(y, pred, sess, win)
            res.append((r[0], r[1][1]))
            print(f"{prefix} s{s} [{kind}] macro-F1 {r[0]:.4f}  F1 {r[1].round(3)}", flush=True)
        if res:
            a = np.array(res)
            print(f"{prefix} MEAN of {len(a)} [{kind}] macro-F1 {a[:, 0].mean():.4f}  stress F1 {a[:, 1].mean():.3f}", flush=True)
