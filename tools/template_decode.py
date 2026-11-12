"""Protocol-template alignment (explicit-duration / semi-Markov Viterbi) on top of per-second class probabilities.

Each protocol version (STRESS/AEROBIC/ANAEROBIC x V1/V2) follows a fixed order of label segments
(R rest, S stress, A activity, U unscored) with fairly stable durations. Given per-second probabilities from
a sequence model, we find for every template the segmentation that maximises
    alpha * sum log p(class of segment | second)  +  sum log N(duration; mean, sd)
pick the best-scoring template and label each second by its segment (U segments keep the model's argmax).
Unscored (U) segments have neutral emission and may be skipped (duration 0). The recording may end inside
any segment (truncated protocol); every non-optional segment left out costs `skip_pen`.

Evaluation on saved out-of-fold probabilities, templates estimated from the training folds only:
    python template_decode.py nn_tcnA_anchor [--alpha 0.2 --res 5]
"""
import argparse
import collections
import itertools

import numpy as np

from scoring import fold_of_session, window_score
import common as C  # noqa: F401  (puts submission/ on sys.path)
from adm_model import align, decode  # noqa: E402,F401  (single implementation, shared with the submission)

CODE = {-1: "U", 0: "R", 1: "S", 2: "A"}
CLS = {"R": 0, "S": 1, "A": 2, "U": -1}


def runs_of(y, min_len=5):
    r = [(CODE[int(v)], len(list(g))) for v, g in itertools.groupby(y)]
    return [(c, n) for c, n in r if n >= min_len]


def build_templates(ys, keys):
    """ys: list of per-second label arrays; keys: list of (protocol, version). Returns {key: segments}.
    A segment is dict(c=class code, mean, sd, opt=bool). Leading / trailing unscored stretches that are not
    part of the most common pattern get their duration distribution from the data (zeros included), so a
    template cannot absorb scored seconds into a free unscored segment."""
    by = collections.defaultdict(list)
    edge = collections.defaultdict(lambda: ([], []))
    for y, k in zip(ys, keys):
        by[k].append(runs_of(y))
        raw = [(int(v), len(list(g))) for v, g in itertools.groupby(y)]
        edge[k][0].append(raw[0][1] if raw[0][0] == -1 else 0)
        edge[k][1].append(raw[-1][1] if raw[-1][0] == -1 else 0)
    T = {}
    for k, rr in by.items():
        pats = collections.Counter("".join(c for c, _ in r) for r in rr)
        P = pats.most_common(1)[0][0]
        D = np.array([[n for _, n in r] for r in rr if "".join(c for c, _ in r) == P], dtype=float)
        lead, trail = np.array(edge[k][0], float), np.array(edge[k][1], float)
        segs = []
        if P[0] != "U":
            segs.append(dict(c=-1, mean=lead.mean(), sd=lead.std(), opt=True))
        for j, ch in enumerate(P):
            segs.append(dict(c=CLS[ch], mean=D[:, j].mean(), sd=D[:, j].std(), opt=(ch == "U")))
        if P[-1] != "U":
            segs.append(dict(c=-1, mean=trail.mean(), sd=trail.std(), opt=True))
        T[k] = segs
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="OOF prefixes in runs/, e.g. nn_tcnA_anchor (several = averaged)")
    ap.add_argument("--alpha", default="0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--skip_pen", type=float, default=50.0)
    a = ap.parse_args()
    d = np.load("../runs/dataset_v4.npz", allow_pickle=True)
    sid, prot = d["sid"], d["prot"]
    keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
    for kind in ("subject", "session"):
        Z = [np.load(f"../runs/{t}_{kind}.npz") for t in a.tags]
        prob = np.mean([z["prob"] for z in Z], axis=0)
        y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
        base = window_score(y, prob.argmax(1), sess, win)
        print(f"[{kind}] model argmax           macro-F1 {base[0]:.4f}  F1 {base[1].round(3)}", flush=True)
        fos = fold_of_session(kind, sid)
        for alpha in [float(v) for v in a.alpha.split(",")]:
            pred = prob.argmax(1).copy()
            hit = hit_type = 0
            for k in range(6):
                tr = np.flatnonzero(fos != k)
                tmpl = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
                for i in np.flatnonzero(fos == k):
                    m = sess == i
                    pred[m], key = decode(prob[m], tmpl, alpha, a.res, a.skip_pen, return_key=True)
                    hit += key == keys[i]
                    hit_type += key[0] == keys[i][0] or (key[0] != "STRESS" and keys[i][0] != "STRESS")
            r = window_score(y, pred, sess, win)
            print(f"[{kind}] template alpha={alpha:<5} macro-F1 {r[0]:.4f}  F1 {r[1].round(3)}  "
                  f"template exact {hit}/{len(sid)}, stress-vs-exercise {hit_type}/{len(sid)}", flush=True)
        if kind == "session":
            print("  confusion (last alpha):\n" + str(r[2]))


if __name__ == "__main__":
    main()
