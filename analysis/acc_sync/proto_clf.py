"""Session-level AEROBIC vs ANAEROBIC classifier for exercise recordings (physiology only), and its effect on the
template decoder (u4b, beta 0.05, U emission). The classifier is trained on the training folds only.
Usage (from tools/): ../env313/bin/python ../analysis/acc_sync/proto_clf.py"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ACC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ".")
sys.path.insert(0, ACC)
from scoring import fold_of_session, window_score  # noqa: E402
from template_decode import build_templates  # noqa: E402
import decode4 as D  # noqa: E402

d = np.load("../runs/dataset_v4.npz", allow_pickle=True)
X, sess, sid, prot, names = d["X"], d["sess"], d["sid"], d["prot"], list(d["names"])
keys = D.keys
col = {n: names.index(n) for n in ("hr_m10", "hr_m30", "hr_m120", "eda_m30", "bvp_std_m30", "bvp_dstd_m30")}


def sess_feats(Xs):
    hr10, hr30, hr120 = Xs[:, col["hr_m10"]], Xs[:, col["hr_m30"]], Xs[:, col["hr_m120"]]
    n = len(hr10)
    hf = hr30 - hr120                                     # fast fluctuations (sprints) around the slow trend
    z = (hf - hf.mean()) / (hf.std() + 1e-6)
    ac = lambda x, L: float(np.corrcoef(x[:-L], x[L:])[0, 1]) if len(x) > L + 10 else 0.0  # noqa: E731
    peaks = np.sum((z[1:-1] > 1.5) & (z[1:-1] >= z[:-2]) & (z[1:-1] >= z[2:]))
    mid = slice(n // 4, 3 * n // 4)
    t = np.arange(n)[mid]
    slope = np.polyfit(t, hr120[mid], 1)[0] if n > 100 else 0.0
    bv = Xs[:, col["bvp_std_m30"]]
    return np.array([np.log(n), hf.std() / (hr120.std() + 1e-6), ac(hf, 120), ac(hf, 240), peaks / n * 1000,
                     slope * 600, (hr10.max() - np.median(hr10)) / (hr10.std() + 1e-6),
                     np.std(bv - np.convolve(bv, np.ones(120) / 120, "same")) / (bv.std() + 1e-6)])


Fs = np.array([sess_feats(X[sess == i]) for i in range(len(sid))])
ex = np.flatnonzero(prot != "STRESS")
yA = (prot == "ANAEROBIC").astype(int)
CFG = dict(beta=0.05, u_emit=True)

for kind in ("subject", "session"):
    fos = fold_of_session(kind, sid)
    pA = np.full(len(sid), 0.5)
    for k in range(6):
        tr = [i for i in ex if fos[i] != k]; te = [i for i in ex if fos[i] == k]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
        clf.fit(Fs[tr], yA[tr]); pA[te] = clf.predict_proba(Fs[te])[:, 1]
    acc = ((pA[ex] > 0.5) == yA[ex]).mean()
    wrong = [f"{prot[i][:4]}/{sid[i]}:{pA[i]:.2f}" for i in ex if (pA[i] > 0.5) != yA[i]]
    print(f"[{kind}] exercise protocol accuracy {acc:.3f} ({int(round(acc * len(ex)))}/{len(ex)}) wrong: {wrong}", flush=True)
    for gamma in (0.0, 1.0, 3.0, 1e3):
        sc = []
        nwrong = []
        for s in (0, 1, 2):
            Z = np.load(os.path.join(ACC, "oof", f"nn_u4b_s{s}_{kind}.npz"))
            prob, y, ss, win = Z["prob"], Z["y"], Z["sess"], Z["win"]
            pred = prob[:, :3].argmax(1).copy(); nw = 0
            for k in range(6):
                tr = np.flatnonzero(fos != k)
                T = build_templates([y[ss == i] for i in tr], [keys[i] for i in tr])
                for i in np.flatnonzero(fos == k):
                    m = ss == i
                    # template prior: + gamma * log p(protocol) for exercise templates, STRESS unchanged
                    best = None
                    for key in T:
                        if key[1] != keys[i][1]:
                            continue
                        pr = 0.0
                        if key[0] == "ANAEROBIC": pr = gamma * np.log(max(pA[i], 1e-6)) if gamma < 1e3 else (0 if pA[i] > 0.5 else -1e9)
                        if key[0] == "AEROBIC": pr = gamma * np.log(max(1 - pA[i], 1e-6)) if gamma < 1e3 else (0 if pA[i] <= 0.5 else -1e9)
                        if prot[i] == "STRESS": pr = 0.0
                        # score of this template = alignment score + prior (via a one-template decode)
                        lp4 = np.log(np.clip(prob[m][:, :4], 1e-6, 1)) + np.log(4.0)
                        pb = prob[m][:, -1]
                        res = 10; n = -(-len(pb) // res)
                        padb = np.full(n * res, pb[-1]); padb[:len(pb)] = pb
                        mx = padb.reshape(n, res).max(1)
                        lo = np.log(np.clip(mx, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - mx, 1e-4, 1))
                        sc_, seg = D.align4(lp4, T[key], 0.002, res, 50.0, 4.0, lo - np.median(lo), 0.05, True)
                        if best is None or sc_ + pr > best[0]:
                            best = (sc_ + pr, key)
                    nw += best[1] != keys[i]
                    pred[m] = D.decode4(prob[m], {best[1]: T[best[1]]}, keys[i][1], 4, 1, **CFG)
            sc.append(window_score(y, pred, ss, win)[0]); nwrong.append(nw)
        print(f"[{kind}] gamma={gamma:<6g} macro-F1 3-seed mean {np.mean(sc):.4f} (seeds {np.round(sc, 4)})  wrong templates {nwrong}", flush=True)
