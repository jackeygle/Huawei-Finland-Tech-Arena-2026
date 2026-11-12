"""Decoder for the 4-class (+ boundary head) TCN outputs, built on align2 (end anchoring cap) + neighbour U fill.
prob columns: 0 R, 1 S, 2 A, [3 U], [last: boundary probability if bnd].
  u_emit : U segments use log p(U) + log 4 as emission (else 0, like the 3-class decoder)
  beta   : weight of the boundary evidence added at every segment start (block log-odds vs session median)
Usage (from analysis/acc_sync/): ../../env313/bin/python decode4.py TAG[+TAG] ncls bnd"""
import os
import sys

import numpy as np

ACC = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(os.path.dirname(ACC)), "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, ACC)
from scoring import fold_of_session, window_score  # noqa: E402
from template_decode import build_templates  # noqa: E402
from adm_model import _dur_table, _key_parts  # noqa: E402
from refine import fill_neighbour  # noqa: E402

d = np.load(os.path.join(os.path.dirname(TOOLS), "runs", "dataset_v4.npz"), allow_pickle=True)
sid, prot = d["sid"], d["prot"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]


def align4(logp, segs, alpha, res=10, skip_pen=50.0, cap=4.0, bblk=None, beta=0.0, u_emit=False):
    """logp: (T, 4) log-ratios (column 3 = U). bblk: (n,) boundary score per block (or None)."""
    T = len(logp); n = -(-T // res)
    pad = np.zeros((n * res, logp.shape[1])); pad[:T] = logp
    blk = pad.reshape(n, res, -1).sum(axis=1) * alpha
    C = np.vstack([np.zeros((1, blk.shape[1])), np.cumsum(blk, axis=0)])
    B = np.zeros(n + 1)
    if bblk is not None and beta:
        B[:n] = beta * bblk
        B[0] = 0.0
    S = len(segs); NEG = -1e18
    F = np.full((S + 1, n + 1), NEG); F[0, 0] = 0.0
    arg = np.zeros((S, n + 1), dtype=np.int64); t = np.arange(n + 1)
    best_end, best_s, best_d = NEG, -1, 0

    def col(c):
        return c if c >= 0 else (3 if u_emit else None)
    for s, seg in enumerate(segs):
        dd_, lp, mu, sd = _dur_table(seg, res, n + 1)
        st = t[None, :] - dd_[:, None]; valid = st >= 0; stc = np.clip(st, 0, n)
        c = col(seg["c"])
        em = (C[t, c][None, :] - C[stc, c]) if c is not None else 0.0
        bterm = np.where(dd_[:, None] > 0, B[stc], 0.0)
        cand = np.where(valid, F[s][stc] + em + lp[:, None] + bterm, NEG)
        k = cand.argmax(axis=0); F[s + 1] = cand[k, t]; arg[s] = dd_[k]
        left = sum(1 for g in segs[s + 1:] if not g["opt"])
        dd = np.arange(1 if not seg["opt"] else 0, n + 1); st2 = n - dd
        em2 = (C[n, c] - C[st2, c]) if c is not None else 0.0
        z2 = 0.5 * ((dd - mu) / sd) ** 2
        lp2 = np.where(dd <= mu, -np.minimum(z2, cap), -z2)
        e = F[s][st2] + em2 + lp2 - skip_pen * left + np.where(dd > 0, B[st2], 0.0)
        j = int(e.argmax())
        if e[j] > best_end:
            best_end, best_s, best_d = float(e[j]), s, int(dd[j])
    lab = np.full(n, -1, dtype=np.int64); s, end, dcur = best_s, n, best_d
    while s >= 0 and end > 0:
        lab[end - dcur:end] = s; end -= dcur; s -= 1
        if s >= 0:
            dcur = int(arg[s][end])
    return best_end, np.repeat(lab, res)[:T]


def decode4(prob, templates, version, ncls, bnd, alpha=0.002, res=10, cap=4.0, beta=0.0, u_emit=False):
    pc = prob[:, :ncls]
    if ncls == 4 and not u_emit:          # renormalise the scored classes, behave like the 3-class decoder
        p3 = pc[:, :3] / pc[:, :3].sum(1, keepdims=True)
        lp = np.c_[np.log(np.clip(p3, 1e-6, 1)) + np.log(3.0), np.zeros(len(p3))]
    elif ncls == 4:
        lp = np.log(np.clip(pc, 1e-6, 1)) + np.log(4.0)
    else:
        lp = np.c_[np.log(np.clip(pc, 1e-6, 1)) + np.log(3.0), np.zeros(len(pc))]
    bblk = None
    if bnd and beta:
        pb = prob[:, -1]
        T = len(pb); n = -(-T // res)
        padb = np.full(n * res, pb[-1]); padb[:T] = pb
        mx = padb.reshape(n, res).max(1)
        lo = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - x, 1e-4, 1))  # noqa: E731
        bblk = lo(mx) - np.median(lo(mx))
    best = None
    for key, segs in templates.items():
        if _key_parts(key)[1] != version:
            continue
        sc, seg = align4(lp, segs, alpha, res, 50.0, cap, bblk, beta, u_emit and ncls == 4)
        if best is None or sc > best[0]:
            best = (sc, seg, key, segs)
    _, seg, key, segs = best
    cls = np.array([g["c"] for g in segs])
    c_of = np.where(seg >= 0, cls[np.maximum(seg, 0)], -1)
    p3 = pc[:, :3] / pc[:, :3].sum(1, keepdims=True)
    return fill_neighbour(c_of, p3)


def evaluate(tagset, kind, ncls, bnd, **kw):
    Z = [np.load(os.path.join(ACC, "oof", f"{t}_{kind}.npz")) if os.path.exists(os.path.join(ACC, "oof", f"{t}_{kind}.npz"))
         else np.load(os.path.join(os.path.dirname(TOOLS), "runs", f"{t}_{kind}.npz")) for t in tagset.split("+")]
    prob = np.mean([z["prob"] for z in Z], axis=0)
    y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
    fos = fold_of_session(kind, sid)
    pred = prob[:, :3].argmax(1).copy()
    arg = window_score(y, pred, sess, win)[0]
    for k in range(6):
        tr = np.flatnonzero(fos != k)
        T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i
            pred[m] = decode4(prob[m], T, keys[i][1], ncls, bnd, **kw)
    r = window_score(y, pred, sess, win)
    return r, arg


if __name__ == "__main__":
    tagset, ncls, bnd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfgs = [dict()]
    if ncls == 4:
        cfgs += [dict(u_emit=True)]
    if bnd:
        cfgs += [dict(beta=b, u_emit=ue) for ue in ([False, True] if ncls == 4 else [False]) for b in (0.02, 0.05, 0.1, 0.2)]
    for kind in ("subject", "session"):
        for c in cfgs:
            r, arg = evaluate(tagset, kind, ncls, bnd, **c)
            print(f"[{kind}] {str(c):32s} argmax {arg:.4f} | decoded macro-F1 {r[0]:.4f} F1 {r[1].round(3)}", flush=True)
