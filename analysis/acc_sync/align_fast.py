"""Faster align4: one gather per segment instead of three, sentinel index instead of np.where(valid, ...)."""
import numpy as np


def align4_fast(logp, segs, alpha, res=10, skip_pen=50.0, cap=4.0, bblk=None, beta=0.0, u_emit=False,
                onblk=None, offblk=None, beta_so=0.0, _dur_table=None):
    T = len(logp); n = -(-T // res)
    pad = np.zeros((n * res, logp.shape[1])); pad[:T] = logp
    blk = pad.reshape(n, res, -1).sum(axis=1) * alpha
    C = np.vstack([np.zeros((1, blk.shape[1])), np.cumsum(blk, axis=0)])
    B = np.zeros(n + 1)
    if bblk is not None and beta:
        B[:n] = beta * bblk; B[0] = 0.0
    def blk_bonus(x):
        z = np.zeros(n + 1)
        if x is not None and beta_so:
            z[:n] = beta_so * x; z[0] = 0.0
        return z
    BON, BOFF = blk_bonus(onblk), blk_bonus(offblk)
    S = len(segs); NEG = -1e18
    F = np.full((S + 1, n + 1), NEG); F[0, 0] = 0.0
    arg = np.zeros((S, n + 1), dtype=np.int64); t = np.arange(n + 1)
    best_end, best_s, best_d = NEG, -1, 0
    G = np.empty(n + 2); G[n + 1] = NEG                       # index n+1 = invalid start
    for s, seg in enumerate(segs):
        dd_, lp, mu, sd = _dur_table(seg, res, n + 1)
        c = seg["c"] if seg["c"] >= 0 else (3 if u_emit else None)
        Bs = B
        if seg["c"] == 1: Bs = Bs + BON
        if s > 0 and segs[s - 1]["c"] == 1: Bs = Bs + BOFF
        st = t[None, :] - dd_[:, None]
        st[st < 0] = n + 1
        Fs = F[s]
        if c is not None:
            Ct = C[:, c]
            G[:n + 1] = Fs - Ct + Bs
            cand = G[st] + Ct[None, :]
        else:
            G[:n + 1] = Fs + Bs
            cand = G[st]
        if dd_[0] == 0:                                        # zero duration: no boundary bonus, no emission
            cand[0] = Fs
        cand += lp[:, None]
        k = cand.argmax(axis=0); F[s + 1] = cand[k, t]; arg[s] = dd_[k]
        left = sum(1 for g in segs[s + 1:] if not g["opt"])
        dd = np.arange(1 if not seg["opt"] else 0, n + 1); st2 = n - dd
        em2 = (C[n, c] - C[st2, c]) if c is not None else 0.0
        z2 = 0.5 * ((dd - mu) / sd) ** 2
        lp2 = np.where(dd <= mu, -np.minimum(z2, cap), -z2)
        e = Fs[st2] + em2 + lp2 - skip_pen * left + np.where(dd > 0, Bs[st2], 0.0)
        j = int(e.argmax())
        if e[j] > best_end:
            best_end, best_s, best_d = float(e[j]), s, int(dd[j])
    lab = np.full(n, -1, dtype=np.int64); s, end, dcur = best_s, n, best_d
    while s >= 0 and end > 0:
        lab[end - dcur:end] = s; end -= dcur; s -= 1
        if s >= 0: dcur = int(arg[s][end])
    return best_end, np.repeat(lab, res)[:T]
