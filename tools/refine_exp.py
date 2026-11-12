"""Local boundary refinement after template decoding: each boundary between two scored segments of different
classes moves within +-W s (not past neighbouring boundaries, min segment 5 s) to maximise the summed
log-probability of the two classes, minus lam * |shift| (s)."""
import sys, datetime as dt
import numpy as np
from scoring import fold_of_session, window_score
from template_decode import build_templates, decode


def refine(prob, lab, W=60, lam=0.0, minlen=5):
    lp = np.log(np.clip(prob, 1e-6, 1))
    out = lab.copy()
    ch = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.r_[0, ch, len(out)]
    for j in range(1, len(bounds) - 1):
        a, b = out[bounds[j] - 1], out[bounds[j]]
        if a < 0 or b < 0 or a == b:
            continue
        lo = max(bounds[j - 1] + minlen, bounds[j] - W); hi = min(bounds[j + 1] - minlen, bounds[j] + W)
        if hi <= lo:
            continue
        cand = np.arange(lo, hi + 1)
        ca, cb = np.cumsum(lp[:, a]), np.cumsum(lp[:, b])
        # score(c) = sum_{t in [lo, c)} lp_a + sum_{t in [c, hi)} lp_b ; relative terms only
        sa = ca[cand - 1] - (ca[lo - 1] if lo > 0 else 0.0)
        sb = ca[hi - 1] * 0 + (cb[hi - 1] - cb[cand - 1])
        sc = sa + sb - lam * np.abs(cand - bounds[j])
        c = int(cand[np.argmax(sc)])
        out[bounds[j]:c] = a if c > bounds[j] else out[bounds[j]:c]
        out[c:bounds[j]] = b
        bounds[j] = c
    return out


if __name__ == "__main__":
    tag = sys.argv[1]; settings = [tuple(map(float, v.split(","))) for v in sys.argv[2:]]
    d = np.load("../runs/dataset_v4.npz", allow_pickle=True); sid, prot = d["sid"], d["prot"]
    keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
    start = np.array([d["win"][d["sess"] == i][0] * 10 for i in range(len(sid))])
    ver = ["V1" if dt.datetime.fromtimestamp(int(t), dt.UTC).month <= 4 else "V2" for t in start]
    for kind in ("subject", "session"):
        z = np.load(f"../runs/{tag}_{kind}.npz"); prob, y, sess, win = z["prob"], z["y"], z["sess"], z["win"]
        fos = fold_of_session(kind, sid)
        base = prob.argmax(1).copy()
        for k in range(6):
            tr = np.flatnonzero(fos != k)
            T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
            for i in np.flatnonzero(fos == k):
                m = sess == i
                base[m] = decode(prob[m], T, 0.002, 10, version=ver[i])
        r = window_score(y, base, sess, win)
        print(f"[{kind}] template only            macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
        for W, lam in settings:
            p = base.copy()
            for i in range(len(sid)):
                m = sess == i
                p[m] = refine(prob[m], base[m], int(W), lam)
            r = window_score(y, p, sess, win)
            print(f"[{kind}] refine W={int(W):3d} lam={lam:<5} macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
