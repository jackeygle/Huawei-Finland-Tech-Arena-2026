"""Why does the single TCN fail where it fails? Phase-level accuracy, event-related averages of the model
output and of the physiology around stress onsets/offsets, and per-subject response vs recall.
Uses out-of-fold TCN probabilities only (no template, no ensemble)."""
import sys, collections
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else "nn_fu_steps3000_s0"
d = np.load("../runs/dataset_v4.npz", allow_pickle=True)
X, y, sess, sid, prot, names = d["X"], d["y"], d["sess"], d["sid"], d["prot"], list(d["names"])
P = np.load("../runs/phase_v1.npz")
z = np.load(f"../runs/{tag}_session.npz"); prob = z["prob"]
phase = np.concatenate([P[f"phase_{k}"] for k in range(len(sid))])
ver = np.array(["V1" if sid[s].startswith("S") else "V2" for s in sess])
pred = prob.argmax(1)
LAB = {0: "Rest", 1: "Stress", 2: "Activity"}

# 1. accuracy per protocol phase (scored seconds only)
print("== 1. TCN per phase (session split, out-of-fold): seconds, accuracy, mean P(true class)")
st = prot[sess] == "STRESS"
for v in ("V1", "V2"):
    for ph in ["Baseline", "Stroop", "TMCT", "Real opinion", "Opposite opinion", "Subtraction", "Rest"]:
        m = st & (ver == v) & (phase == ph) & (y >= 0)
        if m.sum() == 0: continue
        print(f"   STRESS {v} {ph:17s} n={m.sum():6d}  acc {(pred[m] == y[m]).mean():.2f}  P(true) {prob[m, y[m]].mean():.2f}  "
              f"pred as R/S/A: {np.bincount(pred[m], minlength=3) / m.sum()}")

# 2. event-related averages around stress onsets and offsets
def runs(yy):
    ch = np.r_[0, np.flatnonzero(yy[1:] != yy[:-1]) + 1, len(yy)]
    return [(yy[a], a, b) for a, b in zip(ch[:-1], ch[1:])]
CH = {"P(stress)": None, "HR [bpm]": "hr_m10", "log EDA": "eda_m10", "SCR /min": "scr_cnt_m30",
      "RMSSD 60s": "hrv_rmssd60", "skin temp": "temp_m10"}
pre, post = 120, 240
ev = {("onset", "long"): [], ("onset", "short"): [], ("offset", "long"): [], ("offset", "short"): []}
for k in range(len(sid)):
    if prot[k] != "STRESS": continue
    idx = np.flatnonzero(sess == k); yy = y[idx]
    for c, a, b in runs(yy):
        if c != 1 or b - a < 15: continue
        kind = "long" if b - a >= 100 else "short"
        for which, t0 in (("onset", a), ("offset", b)):
            lo, hi = t0 - pre, t0 + post
            if lo < 0 or hi > len(idx): continue
            rows = []
            for nm, col in CH.items():
                v = prob[idx[lo:hi], 1] if col is None else X[idx[lo:hi], names.index(col)].astype(float)
                base = v[pre - 60:pre].mean() if col is not None and nm != "SCR /min" else 0.0
                vv = v - base
                if nm == "SCR /min": vv = v * 60
                if nm == "RMSSD 60s": vv = np.where(v >= 0, v - base, np.nan)
                rows.append(vv)
            ev[(which, kind)].append(np.array(rows))
fig, ax = plt.subplots(2, len(CH), figsize=(4 * len(CH), 7), sharex="row")
t = np.arange(-pre, post)
summary = {}
for r, which in enumerate(("onset", "offset")):
    for kind, col in (("long", "#d62728"), ("short", "#1f77b4")):
        A = np.array(ev[(which, kind)])
        if len(A) == 0: continue
        M = np.nanmean(A, axis=0)
        summary[(which, kind)] = (len(A), M)
        for j, nm in enumerate(CH):
            ax[r, j].plot(t, M[j], color=col, label=f"{kind} tasks (n={len(A)})")
    for j, nm in enumerate(CH):
        ax[r, j].axvline(0, color="k", lw=.8); ax[r, j].set_title(f"{nm} around stress {which}", fontsize=9)
    ax[r, 0].legend(fontsize=7)
for a_ in ax[-1]: a_.set_xlabel("seconds from event")
fig.suptitle("Event-related averages (change vs the 60 s before the event; SCR and P(stress) absolute)", fontsize=10)
fig.tight_layout(); fig.savefig("../analysis/figs/why_errors_eventavg.png", dpi=90)

print("\n== 2. around stress ONSET (mean over tasks): value at +15/+30/+60/+120 s")
for kind in ("long", "short"):
    n_, M = summary[("onset", kind)]
    for j, nm in enumerate(CH):
        print(f"   {kind:5s} (n={n_:2d}) {nm:10s} " + "  ".join(f"+{s_}s:{M[j][pre + s_]:6.3f}" for s_ in (15, 30, 60, 120)))
print("\n== around stress OFFSET: value at -10/+15/+30/+60/+120 s (relative to 60 s before the end)")
for kind in ("long", "short"):
    n_, M = summary[("offset", kind)]
    for j, nm in enumerate(CH):
        print(f"   {kind:5s} (n={n_:2d}) {nm:10s} " + "  ".join(f"{s_:+d}s:{M[j][pre + s_]:6.3f}" for s_ in (-10, 15, 30, 60, 120)))

# 3. per subject: response size vs recall
print("\n== 3. per subject (STRESS recordings): stress recall vs physiological response (task mean - preceding 60 s)")
rows = []
for k in range(len(sid)):
    if prot[k] != "STRESS": continue
    idx = np.flatnonzero(sess == k); yy = y[idx]
    dhr, deda, dscr = [], [], []
    for c, a, b in runs(yy):
        if c != 1 or b - a < 15 or a < 60: continue
        hr = X[idx, names.index("hr_m10")]; eda = X[idx, names.index("eda_m10")]; scr = X[idx, names.index("scr_cnt_m30")]
        dhr.append(hr[a:b].mean() - hr[a - 60:a].mean()); deda.append(eda[a:b].mean() - eda[a - 60:a].mean())
        dscr.append((scr[a:b].mean() - scr[a - 60:a].mean()) * 60)
    m = (yy == 1)
    rows.append((sid[k], (pred[idx][m] == 1).mean(), np.mean(dhr), np.mean(deda), np.mean(dscr)))
rows.sort(key=lambda r: r[1])
for r in rows: print(f"   {r[0]}: stress recall {r[1]:.2f} | dHR {r[2]:+5.1f} bpm  dlogEDA {r[3]:+.3f}  dSCR {r[4]:+5.2f}/min")
R = np.array([r[1:] for r in rows])
for j, nm in enumerate(["dHR", "dlogEDA", "dSCR"]):
    print(f"   correlation(recall, {nm}) = {np.corrcoef(R[:, 0], R[:, j + 1])[0, 1]:+.2f}")
print("\nfigure: analysis/figs/why_errors_eventavg.png")
