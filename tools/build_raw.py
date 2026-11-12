"""Build runs/raw_v1.npz: raw-signal inputs (scheme C) for every session, aligned with dataset_v3 seconds."""
import os, sys, time
import numpy as np
import common as C
sys.path.insert(0, os.path.join(C.ROOT, "submission"))
from raw_inputs import raw_inputs

d = C.build_dataset()
out = {}
t = time.time()
for k, fn in enumerate(C.session_files()):
    df, _, _, _ = C.load_session(fn)
    b, s4, f1, _ = raw_inputs(df)
    n = int((d["sess"] == k).sum())
    assert f1.shape[1] == n and b.shape[0] == 16 * n and s4.shape[1] == 4 * n, (k, f1.shape, n)
    out[f"bvp_{k}"], out[f"slow_{k}"], out[f"fast_{k}"] = b, s4, f1
np.savez(os.path.join(C.ROOT, "runs", "raw_v1.npz"), **out)
print("sessions", len(C.session_files()), f"{time.time() - t:.0f}s")
ts = []
df, _, _, _ = C.load_session(C.session_files()[70])
for _ in range(10):
    t0 = time.perf_counter(); raw_inputs(df); ts.append(time.perf_counter() - t0)
print("raw_inputs ms on longest session: %.1f" % (1e3 * np.median(ts)))
