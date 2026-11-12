"""B2: aerobic vs anaerobic from the heart-rate curve (anaerobic = 3-4 sprints = sharp HR peaks).
Compares with the template decoder's own protocol choice (version from the date)."""
import numpy as np
from scipy.signal import find_peaks
from scoring import fold_of_session
from template_decode import build_templates, decode
import datetime as dt

d = np.load("../runs/dataset_v4.npz", allow_pickle=True)
X, sess, sid, prot, names, y = d["X"], d["sess"], d["sid"], d["prot"], list(d["names"]), d["y"]
keys = [(str(p), "V1" if s.startswith("S") else "V2") for p, s in zip(prot, sid)]
hr_i = names.index("hr_m10")
ex = [i for i in range(len(sid)) if prot[i] != "STRESS"]
print("HR peaks (prominence >= P bpm, >= 60 s apart) per exercise recording:")
for P in (8, 12, 16):
    rows = {}
    for i in ex:
        h = X[sess == i, hr_i]
        pk, _ = find_peaks(h, prominence=P, distance=60)
        rows.setdefault(keys[i], []).append(len(pk))
    print(f"  P={P:2d}: " + "  ".join(f"{k[0][:4]}/{k[1]}: {sorted(v)}" for k, v in sorted(rows.items())))
z = np.load("../runs/nn_fu_steps3000_s0_session.npz"); prob = z["prob"]
fos = fold_of_session("session", sid)
start = np.array([d["win"][sess == i][0] * 10 for i in range(len(sid))])
ver = ["V1" if dt.datetime.fromtimestamp(int(t), dt.UTC).month <= 4 else "V2" for t in start]
wrong = []
for k in range(6):
    tr = np.flatnonzero(fos != k)
    T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
    for i in np.flatnonzero(fos == k):
        _, key = decode(prob[sess == i], T, 0.002, 10, return_key=True, version=ver[i])
        if key != keys[i]: wrong.append((keys[i], key, sid[i]))
print("\nwrong templates with date version (session split):", wrong)
