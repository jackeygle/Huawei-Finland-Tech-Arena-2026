"""Three diagnostics on the single TCN (out-of-fold, session split), STRESS recordings only.
1. lag: AUC(stress vs rest) of a signal taken at t+shift, shift = -60..+120 s (signal relative to the recording median)
2. V2 TMCT: per-subject recall, response size and signal quality vs V1 TMCT
3. what the model relies on: per-subject effect size (stress vs rest) per signal vs per-subject recall; S13 in detail"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = np.load("../runs/dataset_v4.npz", allow_pickle=True)
X, y, sess, sid, prot, names = d["X"], d["y"], d["sess"], d["sid"], d["prot"], list(d["names"])
P = np.load("../runs/phase_v1.npz"); phase = np.concatenate([P[f"phase_{k}"] for k in range(len(sid))])
prob = np.load("../runs/nn_fu_steps3000_s0_session.npz")["prob"]; pred = prob.argmax(1)
SIG = {"HR": "hr_m10", "log EDA": "eda_m10", "SCR rate": "scr_cnt_m30", "EDA phasic act.": "eda_ad_m10",
       "skin temp": "temp_m10", "PPG amplitude": "bvp_std_m10", "PPG hi-freq (motion)": "bvp_dstd_m10",
       "PPG out-of-band": "spec_hifrac_m30", "beats detected/s": "ibi_cnt_m30", "RMSSD": "hrv_rmssd60"}
stress_rec = [k for k in range(len(sid)) if prot[k] == "STRESS"]


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    allv = np.r_[pos, neg]; r = allv.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ---- 1. lag
shifts = np.arange(-60, 121, 10)
res = {}
for nm, col in SIG.items():
    j = names.index(col); A = []
    for s_ in shifts:
        pos, neg = [], []
        for k in stress_rec:
            idx = np.flatnonzero(sess == k); v = X[idx, j].astype(float); yy = y[idx]
            if nm == "RMSSD": v = np.where(v >= 0, v, np.nan)
            v = v - np.nanmedian(v)
            t = np.arange(len(idx)); tt = np.clip(t + s_, 0, len(idx) - 1)
            vs = v[tt]; ok = ~np.isnan(vs)
            pos.append(vs[(yy == 1) & ok]); neg.append(vs[(yy == 0) & ok])
        a = auc(np.concatenate(pos), np.concatenate(neg)); A.append(max(a, 1 - a))
    res[nm] = np.array(A)
print("== 1. AUC(stress vs rest) of each signal taken at t+shift (0.5 = useless; direction-free)")
print("   shift(s):          " + " ".join(f"{s_:>5d}" for s_ in shifts))
for nm, A in res.items():
    best = shifts[A.argmax()]
    print(f"   {nm:20s} " + " ".join(f"{a:5.2f}" for a in A) + f"   best shift {best:+d}s  gain vs 0: {A.max() - A[list(shifts).index(0)]:+.3f}")
fig, ax = plt.subplots(figsize=(8, 5))
for nm, A in res.items(): ax.plot(shifts, A, marker=".", label=nm)
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("signal taken at t + shift (s)"); ax.set_ylabel("AUC stress vs rest")
ax.set_title("How much later does each signal best reflect the stress label?"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig("../analysis/figs/why_lag_auc.png", dpi=100)
# model output itself
A = []
for s_ in shifts:
    pos, neg = [], []
    for k in stress_rec:
        idx = np.flatnonzero(sess == k); v = prob[idx, 1]; yy = y[idx]
        tt = np.clip(np.arange(len(idx)) + s_, 0, len(idx) - 1); pos.append(v[tt][yy == 1]); neg.append(v[tt][yy == 0])
    A.append(auc(np.concatenate(pos), np.concatenate(neg)))
print(f"   {'TCN P(stress)':20s} " + " ".join(f"{a:5.2f}" for a in A) + f"   best shift {shifts[int(np.argmax(A))]:+d}s")

# ---- 2. V2 TMCT
def stats(k, ph):
    idx = np.flatnonzero(sess == k); pp = phase[idx]; yy = y[idx]
    m = pp == ph
    if m.sum() < 20: return None
    a = np.flatnonzero(m)[0]; pre = slice(max(a - 120, 0), max(a - 30, 1))       # rest before, skip instructions edge
    restm = (yy == 0)
    g = lambda col: X[idx, names.index(col)].astype(float)
    return dict(n=int(m.sum()), recall=(pred[idx][m] == 1).mean(),
                dHR=g("hr_m10")[m].mean() - g("hr_m10")[restm].mean(),
                dEDA=g("eda_m10")[m].mean() - g("eda_m10")[restm].mean(),
                dSCR=(g("scr_cnt_m30")[m].mean() - g("scr_cnt_m30")[restm].mean()) * 60,
                beats=g("ibi_cnt_m30")[m].mean(), beats_rest=g("ibi_cnt_m30")[restm].mean(),
                motion=g("bvp_dstd_m10")[m].mean() - g("bvp_dstd_m10")[restm].mean(),
                oob=g("spec_hifrac_m30")[m].mean() - g("spec_hifrac_m30")[restm].mean())
print("\n== 2. TMCT per subject: recall | response vs the recording's rest | signal quality")
for v in ("V1", "V2"):
    rows = [(sid[k], stats(k, "TMCT")) for k in stress_rec if (sid[k].startswith("S")) == (v == "V1")]
    rows = [(s, r) for s, r in rows if r]
    for s, r in sorted(rows, key=lambda x: x[1]["recall"]):
        print(f"   {v} {s}: recall {r['recall']:.2f} | dHR {r['dHR']:+5.1f}  dlogEDA {r['dEDA']:+.2f}  dSCR {r['dSCR']:+5.1f}/min | "
              f"beats/s task {r['beats']:.2f} rest {r['beats_rest']:.2f}  dMotion {r['motion']:+.2f}  dOutOfBand {r['oob']:+.3f}")
    R = np.array([[r["recall"], r["dHR"], r["dEDA"], r["dSCR"], r["beats"], r["motion"]] for _, r in rows])
    print(f"   {v} mean: recall {R[:,0].mean():.2f}  dHR {R[:,1].mean():+.1f}  dlogEDA {R[:,2].mean():+.2f}  dSCR {R[:,3].mean():+.1f}  beats/s {R[:,4].mean():.2f}  dMotion {R[:,5].mean():+.2f}")

# ---- 3. what does the model rely on
print("\n== 3. per-subject effect size d = (stress mean - rest mean)/rest sd, and its correlation with per-subject recall")
D, rec, who = [], [], []
for k in stress_rec:
    idx = np.flatnonzero(sess == k); yy = y[idx]
    row = []
    for nm, col in SIG.items():
        v = X[idx, names.index(col)].astype(float)
        if nm == "RMSSD": v = np.where(v >= 0, v, np.nan)
        a, b = v[yy == 1], v[yy == 0]
        row.append((np.nanmean(a) - np.nanmean(b)) / (np.nanstd(b) + 1e-6))
    D.append(row); rec.append((pred[idx][yy == 1] == 1).mean()); who.append(sid[k])
D, rec = np.array(D), np.array(rec)
for j, nm in enumerate(SIG):
    ok = ~np.isnan(D[:, j])
    print(f"   {nm:20s} mean d {np.nanmean(D[:, j]):+5.2f}  corr(d, recall) {np.corrcoef(D[ok, j], rec[ok])[0, 1]:+.2f}")
print("\n   d per signal for low-recall vs high-recall subjects:")
order = np.argsort(rec)
hdr = "   subject recall " + " ".join(f"{nm[:9]:>9s}" for nm in SIG)
print(hdr)
for i in list(order[:6]) + list(order[-4:]):
    print(f"   {who[i]:7s} {rec[i]:5.2f}  " + " ".join(f"{v:+9.2f}" for v in D[i]))
print("\nfigure: analysis/figs/why_lag_auc.png")
