"""Template choice with protocol-head evidence: score(template) = alignment score + gamma * mean_t log p(protocol).
prob columns: 0-3 classes, 4-6 boundary heads, 7-9 protocol head (STRESS, AEROBIC, ANAEROBIC).
Usage (from tools/): ../env313/bin/python ../analysis/acc_sync/eval_proto.py p10pr   (3 seeds + their ensemble)"""
import os, sys
import numpy as np
ACC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, "."); sys.path.insert(0, ACC)
import decode4 as D
from refine import fill_neighbour
from scoring import fold_of_session, window_score
from template_decode import build_templates

PI = {"STRESS": 7, "AEROBIC": 8, "ANAEROBIC": 9}


def blk(pb, res=10):
    T = len(pb); n = -(-T // res)
    padb = np.full(n * res, pb[-1]); padb[:T] = pb
    mx = padb.reshape(n, res).max(1)
    lo = np.log(np.clip(mx, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - mx, 1e-4, 1))
    return lo - np.median(lo)


def decode_p(p, T, ver, gamma):
    lp = np.log(np.clip(p[:, :4], 1e-6, 1)) + np.log(4.0)
    b, on, off = blk(p[:, 4]), blk(p[:, 5]), blk(p[:, 6])
    mlp = {k: float(np.mean(np.log(np.clip(p[:, c], 1e-6, 1)))) for k, c in PI.items()}
    best = None
    for key, segs in T.items():
        if key[1] != ver:
            continue
        sc, seg = D.align4(lp, segs, 0.002, 10, 50.0, 4.0, b, 0.05, True, on, off, 0.1)
        sc += (gamma * mlp[key[0]]) if gamma < 1e3 else (0.0 if key[0] == max(mlp, key=mlp.get) else -1e9)
        if best is None or sc > best[0]:
            best = (sc, seg, key, segs)
    _, seg, key, segs = best
    cls = np.array([g["c"] for g in segs]); c_of = np.where(seg >= 0, cls[np.maximum(seg, 0)], -1)
    return fill_neighbour(c_of, p[:, :3] / p[:, :3].sum(1, keepdims=True)), key


def run(tags, kind, gamma):
    Z = [np.load(os.path.join(ACC, "oof", f"nn_{t}_{kind}.npz")) for t in tags]
    prob = np.mean([z["prob"] for z in Z], 0); y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
    fos = fold_of_session(kind, D.sid); pred = prob[:, :3].argmax(1).copy(); wrong = 0; pacc = []
    for k in range(6):
        tr = np.flatnonzero(fos != k); T = build_templates([y[sess == i] for i in tr], [D.keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i
            pred[m], key = decode_p(prob[m], T, D.keys[i][1], gamma)
            wrong += key != D.keys[i]
            mlp = {q: np.mean(np.log(np.clip(prob[m][:, c], 1e-6, 1))) for q, c in PI.items()}
            pacc.append(max(mlp, key=mlp.get) == D.keys[i][0])
    r = window_score(y, pred, sess, win)
    return r[0], r[1][1], wrong, np.mean(pacc)


if __name__ == "__main__":
    pre = sys.argv[1]
    sets = [[f"{pre}_s{s}"] for s in range(3)] + [[f"{pre}_s{s}" for s in range(3)]]
    for kind in ("subject", "session"):
        if not all(os.path.exists(os.path.join(ACC, "oof", f"nn_{t}_{kind}.npz")) for t in sets[-1]):
            continue
        for gamma in (0.0, 2.0, 5.0, 10.0, 20.0, 1e3):
            rs = [run(t, kind, gamma) for t in sets]
            print(f"[{kind}] gamma={gamma:<6g} single mean {np.mean([r[0] for r in rs[:3]]):.4f} | ensemble {rs[3][0]:.4f} "
                  f"stressF1 {rs[3][1]:.3f} wrong templates {rs[3][2]} | protocol-head session accuracy {rs[3][3]:.3f}", flush=True)
