"""Where do the out-of-fold errors sit? Breaks 1 Hz OOF predictions down by protocol, phase and cohort."""
import sys
import numpy as np, pandas as pd
import common as C

z = np.load(sys.argv[1]); prob, y, sess = z["prob"], z["y"], z["sess"]
pred = prob.argmax(1)
rows = []
for k, fn in enumerate(C.session_files()):
    df = pd.read_parquet(fn, columns=["Phase", "user_id"])
    t = df.index.values.astype("datetime64[ns]").astype(np.int64); sec = (t - t[0]) // 1_000_000_000
    ph = df.Phase.to_numpy(); first = np.r_[True, sec[1:] != sec[:-1]]
    n = int((sess == k).sum())
    phase = np.full(n, "gap", dtype=object); phase[sec[first]] = ph[first]
    # seconds since the current phase started / until it ends
    chg = np.r_[True, phase[1:] != phase[:-1]]; start = np.maximum.accumulate(np.where(chg, np.arange(n), 0))
    endi = np.r_[np.flatnonzero(chg)[1:], n]; end = np.repeat(endi, np.diff(np.r_[np.flatnonzero(chg), n]))
    prev = np.concatenate([np.array(["start"], dtype=object), phase[:-1]])[start]
    rows.append(pd.DataFrame(dict(prot=fn.split("/")[-2], sid=df.user_id.iloc[0], phase=phase, prev=prev,
                                  since=np.arange(n) - start, until=end - np.arange(n))))
D = pd.concat(rows, ignore_index=True)
D["y"], D["p"] = y, pred
D["cohort"] = np.where(D.sid.str.startswith("S"), "V1", "V2")
D = D[D.y >= 0]; D["ok"] = D.y == D.p
L = np.array(["Rest", "Stress", "Act"])
pd.set_option("display.width", 200, "display.max_rows", 200)
print("== accuracy by protocol / cohort / phase (seconds)")
g = D.assign(phase=D.phase.where(~D.phase.str.contains("rpm"), "rpm")).groupby(["prot", "cohort", "phase"])
T = g.agg(n=("ok", "size"), acc=("ok", "mean"), pRest=("p", lambda x: (x == 0).mean()), pStress=("p", lambda x: (x == 1).mean()), pAct=("p", lambda x: (x == 2).mean()))
print(T.round(3).to_string())
print("\n== Rest phases by what came before")
R = D[D.y == 0]
print(R.groupby(["prot", "cohort", "phase", "prev"]).agg(n=("ok", "size"), acc=("ok", "mean"), pStress=("p", lambda x: (x == 1).mean()), pAct=("p", lambda x: (x == 2).mean())).round(3).to_string())
print("\n== Stress accuracy by seconds since phase start")
S = D[D.y == 1]
print(S.groupby(pd.cut(S.since, [-1, 10, 20, 30, 60, 120, 1e9])).ok.agg(["size", "mean"]).round(3).to_string())
print("\n== Stress accuracy by phase length")
S = S.assign(length=S.since + S.until)
print(S.groupby(pd.cut(S.length, [0, 40, 80, 150, 1e9])).ok.agg(["size", "mean"]).round(3).to_string())
print("\n== per-subject stress F1-ish: recall by subject (STRESS sessions)")
print(S.groupby("sid").ok.mean().round(2).to_string())
