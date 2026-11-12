"""Error decomposition of the current best pipeline (TCN probs + template decoder + version from date).
Read-only w.r.t. the rest of the project. Usage (from tools/):
    ../env313/bin/python ../analysis/acc_sync/err_decomp.py nn_fu_steps3000_s0 [more tags averaged]
For each split: overall macro-F1, template-choice accuracy, and where the wrong 10 s windows sit
(distance to the nearest true label change), per protocol."""
import sys
import datetime as dt
import collections
import numpy as np

sys.path.insert(0, ".")
from scoring import fold_of_session, window_score, majority  # noqa: E402
from template_decode import build_templates, decode  # noqa: E402

tags = sys.argv[1:]
d = np.load("../runs/dataset_v4.npz", allow_pickle=True)
sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]


def win_table(y, p, sess, win):
    key = sess.astype(np.int64) * 10_000_000_000 + win
    u, g = np.unique(key, return_inverse=True)
    ng = g.max() + 1
    yw = majority(y, g, ng)
    pw = majority(np.where(y >= 0, p, -1), g, ng)
    first = np.zeros(ng, dtype=np.int64)
    first[g[::-1]] = np.arange(len(g))[::-1]
    return yw, pw, first


for kind in ("subject", "session"):
    Z = [np.load(f"../runs/{t}_{kind}.npz") for t in tags]
    prob = np.mean([z["prob"] for z in Z], axis=0)
    y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
    fos = fold_of_session(kind, sid)
    preds = {}
    for mode in ("free", "date", "oracle"):
        pred = prob.argmax(1).copy()
        hit = 0
        for k in range(6):
            tr = np.flatnonzero(fos != k)
            T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
            for i in np.flatnonzero(fos == k):
                m = sess == i
                if mode == "oracle":
                    pred[m] = decode(prob[m], {keys[i]: T[keys[i]]}, 0.002, 10); key = keys[i]
                else:
                    pred[m], key = decode(prob[m], T, 0.002, 10, return_key=True,
                                          version=keys[i][1] if mode == "date" else None)
                hit += key == keys[i]
        r = window_score(y, pred, sess, win)
        preds[mode] = pred
        print(f"[{kind}] {mode:6s} macro-F1 {r[0]:.4f} F1 {r[1].round(3)} template exact {hit}/{len(sid)}")
    print(f"[{kind}] argmax only macro-F1 {window_score(y, prob.argmax(1), sess, win)[0]:.4f}")

    # where are the errors of the 'date' pipeline?
    pred = preds["date"]
    # distance (s) of every second to the nearest change of the scored label sequence within its session
    dist = np.full(len(y), 10 ** 6)
    for i in range(len(sid)):
        idx = np.flatnonzero(sess == i)
        yy = y[idx]
        sc = np.flatnonzero(yy >= 0)
        ys = yy[sc]
        ch = sc[1:][ys[1:] != ys[:-1]]               # first second of a new scored class
        if len(ch):
            t = np.arange(len(idx))
            dist[idx] = np.min(np.abs(t[:, None] - ch[None, :]), axis=1)
    yw, pw, first = win_table(y, pred, sess, win)
    ok = yw >= 0
    wd = dist[first]
    ws = sess[first]
    wp = np.array([prot[s] for s in ws])
    bad = ok & (yw != pw)
    print(f"[{kind}] wrong windows {bad.sum()} of {ok.sum()}")
    for p in ("STRESS", "AEROBIC", "ANAEROBIC"):
        b = bad & (wp == p)
        bins = [(0, 15), (15, 45), (45, 120), (120, 10 ** 7)]
        s = "  ".join(f"d{lo}-{hi if hi < 1e6 else 'inf'}:{(b & (wd >= lo) & (wd < hi)).sum()}" for lo, hi in bins)
        cm = collections.Counter(zip(yw[b], pw[b]))
        print(f"   {p:9s} wrong {b.sum():4d}  by distance to true change: {s}  (true,pred): {dict(cm)}")
    # per-session worst offenders
    per = [(int((bad & (ws == i)).sum()), f"{prot[i]}/{sid[i]}") for i in range(len(sid))]
    per.sort(reverse=True)
    print(f"[{kind}] worst sessions:", per[:10])
    np.savez_compressed(f"../analysis/acc_sync/pred_date_{'+'.join(tags)}_{kind}.npz", pred=pred)
