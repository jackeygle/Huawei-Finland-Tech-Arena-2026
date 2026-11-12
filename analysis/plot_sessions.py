"""Plot full sessions (HR, EDA, TEMP, BVP envelope) with label shading for one subject per protocol."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "analysis")
DATA, sid, out = sys.argv[1], sys.argv[2], sys.argv[3]
COL = {"Rest": "#9ecae1", "Stress": "#fc9272", "Activity": "#a1d99b", None: "#eeeeee"}
SP = {"Stroop","TMCT","Real opinion","Opposite opinion","Subtraction"}
def lab(p): return "Stress" if p in SP else "Activity" if (p in ("Warm-up","Sprint","Cooldown") or "rpm" in p) else "Rest" if p in ("Baseline","Rest") else None
fig, axes = plt.subplots(5, 3, figsize=(16, 11), sharex="col")
for j, prot in enumerate(["STRESS", "AEROBIC", "ANAEROBIC"]):
    try: df = pd.read_parquet(f"{DATA}/{prot}/{sid}.parquet")
    except FileNotFoundError: continue
    tm = (df.index - df.index[0]).total_seconds() / 60
    L = df.Phase.map(lab)
    chg = np.r_[0, np.flatnonzero(L.values[1:] != L.values[:-1]) + 1, len(L)]
    bvp_env = df.BVP.rolling(64*2, center=True).std()
    series = [("HR [bpm]", df.HR), ("IBI [s]", df.IBI_duration), ("EDA [uS]", df.EDA),
              ("TEMP [C]", df.Temperature.where(df.Temperature < 45)), ("BVP 2s-std", bvp_env)]
    for i, (name, s) in enumerate(series):
        ax = axes[i, j]
        for a, b in zip(chg[:-1], chg[1:]): ax.axvspan(tm[a], tm[b-1], color=COL.get(L.iloc[a] if isinstance(L.iloc[a], str) else None, "#eeeeee"), lw=0)
        m = s.notna().to_numpy()
        ax.plot(tm[m], s.to_numpy()[m], "k", lw=.6, marker="." if name.startswith("IBI") else None, ms=1, ls="-" if not name.startswith("IBI") else "none")
        if j == 0: ax.set_ylabel(name)
    axes[0, j].set_title(f"{sid} {prot}"); axes[-1, j].set_xlabel("minutes")
from matplotlib.patches import Patch
fig.legend([Patch(color=c) for c in list(COL.values())[:4]], ["Rest", "Stress", "Activity", "unscored"], loc="upper right", ncol=4)
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(out, dpi=110)
