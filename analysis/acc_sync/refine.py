"""Two-stage decoding: global template alignment (as now), then local re-positioning of every Stress run.
Each S run (start and end move together) is shifted by delta in [-W, W] s; the seconds it vacates go to the
neighbouring segment on that side, the score is alpha2 * sum(change of log-likelihood ratio) - 0.5 (delta/sig)^2.
U seconds have emission 0. Usage (from tools/): ../env313/bin/python ../analysis/acc_sync/refine.py TAG[+TAG]"""
import sys
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "../analysis/acc_sync")
from scoring import fold_of_session, window_score  # noqa: E402
from template_decode import build_templates  # noqa: E402
from adm_model import align, _key_parts  # noqa: E402
from decode_variants import sid, prot, keys  # noqa: E402


def stage1(prob, templates, alpha, res, version):
    lp = np.log(np.clip(prob, 1e-6, 1.0)) + np.log(3.0)
    best = None
    for key, segs in templates.items():
        if _key_parts(key)[1] != version:
            continue
        sc, seg = align(lp, segs, alpha, res)
        if best is None or sc > best[0]:
            best = (sc, seg, key, segs)
    _, seg, key, segs = best
    cls = np.array([g["c"] for g in segs])
    c_of = np.where(seg >= 0, cls[np.maximum(seg, 0)], -1)
    return c_of, seg, lp, key


def refine(c_of, seg, lp, W, alpha2, sig):
    c = c_of.copy()
    T = len(c)
    ids = [s for s in np.unique(seg[seg >= 0]) if c_of[seg == s][0] == 1]
    for s in ids:
        idx = np.flatnonzero(seg == s)
        a, b = idx[0], idx[-1] + 1
        left_c = c[a - 1] if a > 0 else -1
        right_c = c[b] if b < T else -1
        # limits: do not run into another S run on either side
        lo_lim = a - 1
        while lo_lim >= 0 and c[lo_lim] != 1 and a - lo_lim <= W:
            lo_lim -= 1
        hi_lim = b
        while hi_lim < T and c[hi_lim] != 1 and hi_lim - b < W:
            hi_lim += 1
        e = lambda t0, t1, k: 0.0 if k < 0 or t1 <= t0 else lp[t0:t1, k].sum()  # noqa: E731
        best, bd = 0.0, 0
        for dlt in range(-W, W + 1):
            na, nb = a + dlt, b + dlt
            if na <= lo_lim or nb > hi_lim or na < 0 or nb > T:
                continue
            if dlt > 0:   # vacate [a, na) to the left neighbour, take [b, nb) from the right one
                g = e(a, na, left_c) - e(a, na, 1) + e(b, nb, 1) - e(b, nb, right_c)
            elif dlt < 0:
                g = e(na, a, 1) - e(na, a, left_c) + e(nb, b, right_c) - e(nb, b, 1)
            else:
                g = 0.0
            sc = alpha2 * g - 0.5 * (dlt / sig) ** 2
            if sc > best:
                best, bd = sc, dlt
        if bd:
            na, nb = a + bd, b + bd
            new = c.copy()
            if bd > 0:
                new[a:na] = left_c
            else:
                new[nb:b] = right_c
            new[na:nb] = 1
            c = new
    return c


def fill_neighbour(c_of, prob):
    out = np.where(c_of >= 0, c_of, prob.argmax(1))
    idx = np.flatnonzero(c_of >= 0)
    if len(idx) == 0:
        return out
    T = len(out)
    pi = np.searchsorted(idx, np.arange(T), side="right") - 1
    ni = np.searchsorted(idx, np.arange(T), side="left")
    pc = np.where(pi >= 0, c_of[idx[np.maximum(pi, 0)]], -1)
    nc = np.where(ni < len(idx), c_of[idx[np.minimum(ni, len(idx) - 1)]], -1)
    allowed = np.zeros((T, 3), bool)
    allowed[np.arange(T)[pc >= 0], pc[pc >= 0]] = True
    allowed[np.arange(T)[nc >= 0], nc[nc >= 0]] = True
    u = c_of < 0
    out[u] = np.where(allowed, prob, -1)[u].argmax(1)
    return out


def evaluate(tagset, kind, W=0, alpha2=0.0, sig=1e9, res=5):
    Z = [np.load(f"../runs/{t}_{kind}.npz") for t in tagset.split("+")]
    prob = np.mean([z["prob"] for z in Z], axis=0)
    y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
    fos = fold_of_session(kind, sid)
    pred = prob.argmax(1).copy()
    for k in range(6):
        tr = np.flatnonzero(fos != k)
        T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i
            c_of, seg, lp, key = stage1(prob[m], T, 0.002, res, keys[i][1])
            if W:
                c_of = refine(c_of, np.repeat(seg, 1), lp, W, alpha2, sig)
            pred[m] = fill_neighbour(c_of, prob[m])
    return window_score(y, pred, sess, win)


if __name__ == "__main__":
    tagset = sys.argv[1]
    for kind in ("subject", "session"):
        r = evaluate(tagset, kind)
        print(f"[{kind}] stage1 only            macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
        for W in (30, 60, 120):
            for a2 in (0.05, 0.2, 1.0):
                for sig in (30, 1e9):
                    r = evaluate(tagset, kind, W, a2, sig)
                    print(f"[{kind}] W={W:3d} alpha2={a2:<4} sig={sig:<6g} macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
