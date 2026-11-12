"""Quick difficulty probe: per-second features + rolling context, subject-grouped CV.

Scenario A ("local"):   features use only a centred context window (no session info).
Scenario B ("session"): A + features minus the per-session median (needs the whole session as input).
Scoring mimics the official one: macro-F1 over 10 s windows, majority label per window.
"""
import glob, re, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

DATA = sys.argv[1]
LABELS = ["Rest", "Stress", "Activity"]
STRESS_P = {"Stroop", "TMCT", "Real opinion", "Opposite opinion", "Subtraction"}


def phase_to_label(p):
    if p in STRESS_P: return 1
    if p in ("Warm-up", "Sprint", "Cooldown") or "rpm" in p: return 2
    if p in ("Baseline", "Rest"): return 0
    return -1


def per_second(df):
    t0 = df.index[0]
    sec = ((df.index - t0).total_seconds()).astype(np.int64)
    g = pd.DataFrame({"sec": sec})
    bvp = df.BVP.to_numpy()
    out = pd.DataFrame(index=pd.RangeIndex(sec.max() + 1, name="sec"))
    b = pd.DataFrame({"sec": sec, "bvp": bvp, "dbvp": np.r_[0, np.diff(bvp)]})
    gb = b.groupby("sec")
    out["bvp_std"] = gb.bvp.std()
    out["bvp_rng"] = gb.bvp.max() - gb.bvp.min()
    out["bvp_dstd"] = gb.dbvp.std()            # high-frequency energy (motion artefacts)
    for c, n in [("EDA", "eda"), ("Temperature", "temp"), ("HR", "hr"), ("IBI_duration", "ibi")]:
        s = df[c].to_numpy()
        m = ~np.isnan(s)
        out[n] = pd.Series(s[m]).groupby(sec[m]).mean()
    out.loc[out.temp > 45, "temp"] = np.nan   # sensor-fault values (~209)
    out["eda"] = np.log1p(out.eda.clip(lower=0))
    lab = pd.Series(df.Phase.map(phase_to_label).to_numpy()).groupby(sec).agg(lambda x: x.mode().iloc[0])
    out = out.interpolate(limit_direction="both")
    out["y"] = lab.reindex(out.index).fillna(-1).astype(int)   # seconds inside gaps -> unscored
    return out


def add_context(f, wins=(10, 30, 60, 120)):
    base = ["bvp_std", "bvp_rng", "bvp_dstd", "eda", "temp", "hr", "ibi"]
    cols = {}
    for w in wins:
        r = f[base].rolling(w, center=True, min_periods=1)
        m, s = r.mean(), r.std()
        for c in base:
            cols[f"{c}_m{w}"] = m[c]
            cols[f"{c}_s{w}"] = s[c]
        for c in ["eda", "temp", "hr"]:
            cols[f"{c}_slope{w}"] = f[c].shift(-w // 2) - f[c].shift(w // 2)
    ibi = f["ibi"]
    cols["ibi_rmssd30"] = ibi.diff().pow(2).rolling(30, center=True, min_periods=1).mean().pow(.5)
    return pd.DataFrame(cols)


t = time.time()
parts = []
for fn in sorted(glob.glob(f"{DATA}/*/*.parquet")):
    df = pd.read_parquet(fn)
    f = per_second(df)
    X = add_context(f)
    Xs = (X - X.median()).add_suffix("_rel")     # session-relative
    D = pd.concat([X, Xs], axis=1)
    D["y"] = f.y.values; D["sid"] = df.user_id.iloc[0]; D["prot"] = fn.split("/")[-2]
    D["win"] = np.arange(len(D)) // 10
    parts.append(D)
D = pd.concat(parts, ignore_index=True)
print("built", D.shape, f"{time.time()-t:.0f}s")

loc_cols = [c for c in D.columns if c not in ("y", "sid", "prot", "win") and not c.endswith("_rel")]
all_cols = [c for c in D.columns if c not in ("y", "sid", "prot", "win")]
lab = D.y.to_numpy()


def window_score(D, pred):
    d = pd.DataFrame({"k": D.sid + D.prot + D.win.astype(str), "y": D.y, "p": pred})
    d = d[d.y >= 0]
    agg = d.groupby("k").agg(y=("y", lambda x: x.mode().iloc[0]), p=("p", lambda x: x.mode().iloc[0]))
    return f1_score(agg.y, agg.p, average="macro"), confusion_matrix(agg.y, agg.p, labels=[0, 1, 2]), \
        f1_score(agg.y, agg.p, average=None, labels=[0, 1, 2])


gkf = GroupKFold(n_splits=6)
for name, cols in [("A_local", loc_cols), ("B_session", all_cols)]:
    pred = np.full(len(D), -1)
    for tr, te in gkf.split(D, groups=D.sid):
        tr = tr[lab[tr] >= 0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                             max_leaf_nodes=31, random_state=0)
        clf.fit(D.iloc[tr][cols], lab[tr])
        pred[te] = clf.predict(D.iloc[te][cols])
    mf1, cm, per = window_score(D, pred)
    print(f"\n{name}: macro-F1={mf1:.4f}  score={mf1*80:.1f}/80  per-class F1 (Rest,Stress,Act)={per.round(3)}")
    print("confusion (rows=true Rest/Stress/Act):\n", cm)
    D["pred_" + name] = pred
D[["sid", "prot", "win", "y", "pred_A_local", "pred_B_session"]].to_parquet("analysis/quick_baseline_preds.parquet")
