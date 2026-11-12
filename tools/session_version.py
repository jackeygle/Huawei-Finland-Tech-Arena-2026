"""Recording-level classifier for the protocol version (V1/V2) of exercise recordings, from physiology only.
Features: change between the first 270 s and the next 270 s (V2 starts with 270 s seated rest), level at the
start, and recording length. Logistic regression, evaluated with the same fold assignment as everything else."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scoring import fold_of_session

CH = ["hr_m30", "bvp_dstd_m30", "bvp_std_m30", "spec_hifrac_m30", "spec_dom_m30", "eda_m30", "eda_ad_m30",
      "temp_m30", "ibi_cnt_m30", "scr_cnt_m30"]


def session_features(X, names):
    idx = [list(names).index(c) for c in CH]
    A = X[:, idx]
    n = len(A)
    a, b = A[:270].mean(0), A[270:540].mean(0) if n > 540 else A[n // 2:].mean(0)
    early_slope = A[min(240, n - 1)] - A[0]
    return np.r_[b - a, a, early_slope, np.log(n)]


if __name__ == "__main__":
    d = np.load("../runs/dataset_v4.npz", allow_pickle=True)
    X, sess, sid, prot, names = d["X"], d["sess"], d["sid"], d["prot"], d["names"]
    ver = np.array([0 if s.startswith("S") else 1 for s in sid])
    ex = np.flatnonzero(prot != "STRESS")
    F = np.array([session_features(X[sess == i], names) for i in range(len(sid))])
    for kind in ("subject", "session"):
        fos = fold_of_session(kind, sid)
        pr = np.zeros(len(sid))
        for k in range(6):
            tr = [i for i in ex if fos[i] != k]; te = [i for i in ex if fos[i] == k]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
            clf.fit(F[tr], ver[tr]); pr[te] = clf.predict_proba(F[te])[:, 1]
        acc = ((pr[ex] > 0.5) == ver[ex]).mean()
        wrong = [f"{prot[i]}/{sid[i]}:{pr[i]:.2f}" for i in ex if (pr[i] > 0.5) != ver[i]]
        conf = (np.abs(pr[ex] - 0.5) > 0.3)
        print(f"[{kind}] exercise version accuracy {acc:.3f} ({int(acc * len(ex))}/{len(ex)}); "
              f"confident (p<0.2 or >0.8): {conf.sum()} of {len(ex)}, accuracy there {((pr[ex] > 0.5) == ver[ex])[conf].mean():.3f}")
        print("   wrong:", wrong)
