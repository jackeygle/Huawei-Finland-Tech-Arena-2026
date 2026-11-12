"""Per-second protocol phase names for every recording, aligned with dataset_v4 (runs/phase_v1.npz)."""
import numpy as np, pandas as pd
import common as C
out = {}
d = C.build_dataset()
for k, fn in enumerate(C.session_files()):
    df = pd.read_parquet(fn, columns=["Phase"])
    t = df.index.values.astype("datetime64[ns]").astype(np.int64)
    sec = (t - t[0]) // 1_000_000_000
    n = int((d["sess"] == k).sum())
    ph = df["Phase"].to_numpy().astype(str)
    first = np.r_[True, sec[1:] != sec[:-1]]
    names = np.full(n, "gap", dtype=object)
    names[sec[first]] = ph[first]
    for j in range(1, n):                               # fill seconds inside gaps with the previous phase
        if names[j] == "gap": names[j] = names[j - 1]
    out[f"phase_{k}"] = names.astype(str)
np.savez_compressed("../runs/phase_v1.npz", **out)
print("saved", len(out))
