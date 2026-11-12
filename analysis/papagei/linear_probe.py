"""Linear probe on 10 s segments (the PaPaGei paper's evaluation protocol): frozen features -> standardise ->
logistic regression, subject-grouped 6-fold CV (same fold assignment as everything else).
Feature sets: PaPaGei embedding (512-d), our hand-crafted PPG features, all our 175 features, and combinations.
Score: macro-F1 over 10 s segments (segment label = majority of scored seconds, needs >= 50% scored), plus
per-class F1 and stress-vs-rest AUC inside STRESS recordings. Usage: linear_probe.py <emb.npz> [<emb.npz> ...]"""
import os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
ROOT = "/scratch/work/zhangx29/huawei_tech_arena_2026"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from scoring import fold_of_session  # noqa: E402

d = np.load(os.path.join(ROOT, "runs/dataset_v4.npz"), allow_pickle=True)
X, y, sess, sid, prot, names = d["X"], d["y"], d["sess"], d["sid"], d["prot"], list(d["names"])
G6 = ("local", "rel", "sess", "eda2", "hrv", "spec")
def grp(n):
    for p, g in (("scr_", "eda2"), ("spec_", "spec"), ("hrv_", "hrv"), ("pos_", "pos"), ("resp_", "resp"), ("pulse_", "pulse"), ("rec_", "recov")):
        if n.startswith(p): return g
    if n.endswith(("_sessrange", "_sesshi", "_relrange")): return "sess2"
    if n.endswith(("_rel", "_z", "_rank")): return "rel"
    if n.endswith(("_sessmed", "_sessstd")): return "sess"
    return "local"
ALL = np.array([grp(n) in G6 for n in names])
PPG = np.array([(n.startswith(("bvp_", "spec_")) and grp(n) in G6) for n in names])
print(f"our features: all {ALL.sum()}, PPG-only {PPG.sum()}")
for embf in sys.argv[1:]:
    E = np.load(embf)
    rows = {"emb": [], "all": [], "ppg": [], "y": [], "sess": []}
    for k in range(len(sid)):
        if f"emb_{k}" not in E.files: continue
        idx = np.flatnonzero(sess == k); yy = y[idx]
        for e, c in zip(E[f"emb_{k}"], E[f"centre_{k}"]):
            lo, hi = c - 5, c + 5
            if hi > len(idx): continue
            seg = yy[lo:hi]; sc = seg[seg >= 0]
            if len(sc) < 5: continue
            lab = np.bincount(sc, minlength=3).argmax()
            rows["emb"].append(e); rows["all"].append(X[idx[c], ALL]); rows["ppg"].append(X[idx[c], PPG])
            rows["y"].append(lab); rows["sess"].append(k)
    F = {k: np.array(v) for k, v in rows.items()}
    yv, sv = F["y"], F["sess"]
    print(f"\n== {os.path.basename(embf)}: {len(yv)} segments, class counts {np.bincount(yv)}")
    sets = {"PaPaGei embedding (512)": F["emb"], "PaPaGei embedding PCA-32": None, "our PPG features": F["ppg"],
            "all our features": F["all"], "all ours + PaPaGei PCA-32": None}
    for kind in ("subject",):
        fos = fold_of_session(kind, sid)[sv]
        for name, Xf in sets.items():
            pred = np.zeros(len(yv), int); ps = np.zeros(len(yv))
            for k in range(6):
                tr, te = fos != k, fos == k
                if name.endswith("PCA-32") or "PCA-32" in name:
                    pca = PCA(32, random_state=0).fit(F["emb"][tr])
                    Z = pca.transform(F["emb"])
                    Xf_ = Z if name.startswith("PaPaGei") else np.c_[F["all"], Z]
                else:
                    Xf_ = Xf
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
                clf.fit(Xf_[tr], yv[tr]); pred[te] = clf.predict(Xf_[te]); ps[te] = clf.predict_proba(Xf_[te])[:, 1]
            f = f1_score(yv, pred, average=None, labels=[0, 1, 2])
            st = np.isin(sv, [i for i in range(len(sid)) if prot[i] == "STRESS"]) & (yv < 2)
            auc = roc_auc_score(yv[st] == 1, ps[st])
            print(f"   [{kind}] {name:28s} macro-F1 {f.mean():.4f}  F1 R/S/A {f.round(3)}  stress-vs-rest AUC (stress recs) {auc:.3f}", flush=True)
