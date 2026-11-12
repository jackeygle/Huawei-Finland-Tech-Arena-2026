"""Visual check of the signal-processing channels on one STRESS and one AEROBIC session of a subject."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "tools"); import common as C
M = C.M
sid, out = sys.argv[1], sys.argv[2]
COL = {0: "#9ecae1", 1: "#fc9272", 2: "#a1d99b", -1: "#eeeeee"}
fig, axes = plt.subplots(5, 2, figsize=(16, 12), sharex="col")
for j, prot in enumerate(["STRESS", "AEROBIC"]):
    df, y_row, _, _ = C.load_session(f"{C.DATA}/{prot}/{sid}.parquet")
    B, sec = M.base_1hz(df); n = len(B); t = np.arange(n) / 60
    y = C.majority(y_row, sec, n)
    eda = df.EDA.dropna().to_numpy(); es = sec[~df.EDA.isna().to_numpy()]
    ph = M.sosfiltfilt(M._SOS_EDA_HP, M.sosfiltfilt(M._SOS_EDA_LP, np.maximum(eda, 0)))
    pk, _ = M.find_peaks(ph, prominence=M.SCR_MIN_AMP, distance=4)
    S = M.signal_features(df, sec, n, B[:, 5]); names = M.signal_feature_names()
    g = lambda nm: S[:, names.index(nm)]
    rows = [("EDA [uS] + SCR peaks", None), ("SCR per min (30 s)", g("scr_cnt_m30") * 60),
            ("RMSSD 60 s [s]", np.where(g("hrv_rmssd60") >= 0, g("hrv_rmssd60"), np.nan)),
            ("PPG dominant x60 vs HR", None), ("out-of-band energy share (30 s)", g("spec_hifrac_m30") + g("spec_lofrac_m30"))]
    for i, (name, v) in enumerate(rows):
        ax = axes[i, j]
        chg = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, n]
        for a, b in zip(chg[:-1], chg[1:]): ax.axvspan(t[a], t[b - 1], color=COL[y[a]], lw=0)
        if i == 0:
            ax.plot(es / 60, eda, "k", lw=.6); ax.plot(es[pk] / 60, eda[pk], "r.", ms=3)
        elif i == 3:
            ax.plot(t, g("spec_dom_m30") * 60, "b", lw=.7, label="PPG spectrum peak x60")
            ax.plot(t, B[:, 5], "k", lw=.7, label="device HR"); ax.legend(loc="upper left", fontsize=8)
        else:
            ax.plot(t, v, "k", lw=.7)
        if j == 0: ax.set_ylabel(name, fontsize=9)
    axes[0, j].set_title(f"{sid} {prot}"); axes[-1, j].set_xlabel("minutes")
fig.tight_layout(); fig.savefig(out, dpi=100)
