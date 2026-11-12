"""Decoder variants on saved OOF probabilities (date-based version, as v6). Does not touch submission/.
Usage (from tools/): ../env313/bin/python ../analysis/acc_sync/decode_variants.py TAG[+TAG...] [grid]
ufill:  argmax    = current (U seconds keep the model argmax)
        neighbour = U seconds take argmax restricted to the classes of the nearest scored segments before/after
        prev      = U seconds take the class of the previous scored segment (next one if none before)"""
import sys
import numpy as np

sys.path.insert(0, ".")
from scoring import fold_of_session, window_score  # noqa: E402
from template_decode import build_templates  # noqa: E402
from adm_model import align, _key_parts  # noqa: E402

d = np.load("../runs/dataset_v4.npz", allow_pickle=True)
sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]


def decode_v(prob, templates, alpha, res, version, ufill="argmax", shift=0, skip_pen=50.0, sd_scale=1.0):
    p = prob
    if shift > 0:
        p = np.concatenate([prob[shift:], np.repeat(prob[-1:], shift, axis=0)])
    lp = np.log(np.clip(p, 1e-6, 1.0)) + np.log(3.0)
    cand = {k: v for k, v in templates.items() if _key_parts(k)[1] == version}
    best = None
    for key, segs in cand.items():
        if sd_scale != 1.0:
            segs = [dict(g, sd=g["sd"] * sd_scale) for g in segs]
        sc, seg = align(lp, segs, alpha, res, skip_pen)
        if best is None or sc > best[0]:
            best = (sc, seg, key, segs)
    _, seg, key, segs = best
    cls = np.array([g["c"] for g in segs])
    c_of = np.where(seg >= 0, cls[np.maximum(seg, 0)], -1)
    out = np.where(c_of >= 0, c_of, prob.argmax(1))
    if ufill != "argmax":
        T = len(out)
        idx = np.flatnonzero(c_of >= 0)
        if len(idx):
            # previous / next scored class for every second
            prev_i = np.searchsorted(idx, np.arange(T), side="right") - 1
            next_i = np.searchsorted(idx, np.arange(T), side="left")
            pc = np.where(prev_i >= 0, c_of[idx[np.maximum(prev_i, 0)]], -1)
            nc = np.where(next_i < len(idx), c_of[idx[np.minimum(next_i, len(idx) - 1)]], -1)
            u = c_of < 0
            if ufill == "prev":
                out[u] = np.where(pc[u] >= 0, pc[u], nc[u])
            else:
                allowed = np.zeros((T, 3), bool)
                allowed[np.arange(T)[pc >= 0], pc[pc >= 0]] = True
                allowed[np.arange(T)[nc >= 0], nc[nc >= 0]] = True
                pm = np.where(allowed, prob, -1)
                out[u] = pm[u].argmax(1)
    return out, key


def run(tagset, kind, **kw):
    Z = [np.load(f"../runs/{t}_{kind}.npz") for t in tagset.split("+")]
    prob = np.mean([z["prob"] for z in Z], axis=0)
    y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
    fos = fold_of_session(kind, sid)
    pred = prob.argmax(1).copy()
    alpha, res = kw.pop("alpha", 0.002), kw.pop("res", 10)
    for k in range(6):
        tr = np.flatnonzero(fos != k)
        T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i
            pred[m], _ = decode_v(prob[m], T, alpha, res, keys[i][1], **kw)
    return window_score(y, pred, sess, win)


if __name__ == "__main__":
    tagset = sys.argv[1]
    grid = len(sys.argv) > 2 and sys.argv[2] == "grid"
    for kind in ("subject", "session"):
        for uf in ("argmax", "neighbour", "prev"):
            r = run(tagset, kind, ufill=uf)
            print(f"[{kind}] ufill={uf:9s} macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
        if grid:
            for a in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
                for res in (5, 10):
                    r = run(tagset, kind, ufill="neighbour", alpha=a, res=res)
                    print(f"[{kind}] neighbour alpha={a:<6} res={res:2d} macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
            for sh in (5, 10, 20):
                r = run(tagset, kind, ufill="neighbour", shift=sh)
                print(f"[{kind}] neighbour shift={sh:2d} macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
            for sc in (0.5, 2.0):
                r = run(tagset, kind, ufill="neighbour", sd_scale=sc)
                print(f"[{kind}] neighbour sd_scale={sc} macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
