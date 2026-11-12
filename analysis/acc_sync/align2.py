"""align() from submission/adm_model.py with one change: when the recording ends inside segment s, a duration
shorter than the mean is penalised as -min(0.5 z^2, cap) instead of 0 (cap=0 reproduces the original)."""
import numpy as np
from adm_model import _dur_table


def align2(logp, segs, alpha, res=5, skip_pen=50.0, cap=0.0):
    T = len(logp); n = -(-T // res)
    pad = np.zeros((n * res, 3)); pad[:T] = logp
    blk = pad.reshape(n, res, 3).sum(axis=1) * alpha
    C = np.vstack([np.zeros((1, 3)), np.cumsum(blk, axis=0)])
    S = len(segs); NEG = -1e18
    F = np.full((S + 1, n + 1), NEG); F[0, 0] = 0.0
    arg = np.zeros((S, n + 1), dtype=np.int64); t = np.arange(n + 1)
    best_end, best_s, best_d = NEG, -1, 0
    for s, seg in enumerate(segs):
        d, lp, mu, sd = _dur_table(seg, res, n + 1)
        st = t[None, :] - d[:, None]; valid = st >= 0; stc = np.clip(st, 0, n)
        em = (C[t, seg["c"]][None, :] - C[stc, seg["c"]]) if seg["c"] >= 0 else 0.0
        cand = np.where(valid, F[s][stc] + em + lp[:, None], NEG)
        k = cand.argmax(axis=0); F[s + 1] = cand[k, t]; arg[s] = d[k]
        left = sum(1 for g in segs[s + 1:] if not g["opt"])
        dd = np.arange(1 if not seg["opt"] else 0, n + 1); st2 = n - dd
        em2 = (C[n, seg["c"]] - C[st2, seg["c"]]) if seg["c"] >= 0 else 0.0
        z2 = 0.5 * ((dd - mu) / sd) ** 2
        lp2 = np.where(dd <= mu, -np.minimum(z2, cap), -z2)
        e = F[s][st2] + em2 + lp2 - skip_pen * left
        j = int(e.argmax())
        if e[j] > best_end:
            best_end, best_s, best_d = float(e[j]), s, int(dd[j])
    lab = np.full(n, -1, dtype=np.int64); s, end, d = best_s, n, best_d
    while s >= 0 and end > 0:
        lab[end - d:end] = s; end -= d; s -= 1
        if s >= 0: d = int(arg[s][end])
    return best_end, np.repeat(lab, res)[:T]
