"""Visual check of the physiology channels (breathing, pulse amplitude, recovery) for one subject."""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "tools"); import common as C
M = C.M
sid, out = sys.argv[1], sys.argv[2]
prots = sys.argv[3].split(",") if len(sys.argv) > 3 else ["STRESS", "AEROBIC", "ANAEROBIC"]
COL = {0: "#9ecae1", 1: "#fc9272", 2: "#a1d99b", -1: "#eeeeee"}
names = M.physio_feature_names()
fig, axes = plt.subplots(5, len(prots), figsize=(6 * len(prots), 12), sharex="col", squeeze=False)
for j, prot in enumerate(prots):
    df, y_row, _, _ = C.load_session(f"{C.DATA}/{prot}/{sid}.parquet")
    B, sec = M.base_1hz(df); n = len(B); t = np.arange(n) / 60
    y = C.majority(y_row, sec, n)
    P = M.physio_features(df, sec, n, B); g = lambda nm: P[:, names.index(nm)]
    rows = [("breathing rate [/min]", None), ("breathing peak sharpness", None),
            ("pulse amplitude rel. own median (log)", g("pulse_amp_srel")),
            ("HR above own baseline [bpm]", None), ("HR decay rate [1/s]  (<0 = recovering)", g("rec_hr_decay"))]
    for i, (name, v) in enumerate(rows):
        ax = axes[i, j]
        chg = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, n]
        for a, b in zip(chg[:-1], chg[1:]): ax.axvspan(t[a], t[b - 1], color=COL[y[a]], lw=0)
        if i == 0:
            ax.plot(t, g("resp_riiv_bpm"), "b", lw=.6, label="baseline (RIIV)")
            ax.plot(t, g("resp_riav_bpm"), "k", lw=.6, label="amplitude (RIAV)"); ax.legend(fontsize=7, loc="upper left")
        elif i == 1:
            ax.plot(t, g("resp_riiv_peak"), "b", lw=.6); ax.plot(t, g("resp_riav_peak"), "k", lw=.6)
        elif i == 3:
            ax.plot(t, g("rec_hr_elev"), "k", lw=.7, label="elevation")
            ax.plot(t, g("rec_hr_frompeak"), "r", lw=.7, label="below 5-min peak"); ax.legend(fontsize=7, loc="upper left")
        else:
            ax.plot(t, v, "k", lw=.7)
        if j == 0: ax.set_ylabel(name, fontsize=8)
    axes[0, j].set_title(f"{sid} {prot}"); axes[-1, j].set_xlabel("minutes")
fig.tight_layout(); fig.savefig(out, dpi=95)
