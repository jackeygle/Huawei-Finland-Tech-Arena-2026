"""Break run() time into stages for one recording (single thread)."""
import sys, time, importlib.util, os
import numpy as np, pandas as pd
import common as C
M = C.M
zipdir = sys.argv[1]                       # folder holding adm_model.py + adm_model.onnx of a packaged run
k = int(sys.argv[2]) if len(sys.argv) > 2 else 70
spec = importlib.util.spec_from_file_location("sub", os.path.join(zipdir, "adm_model.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
model = S.get_model()
df, _, _, _ = C.load_session(C.session_files()[k])
G = S.GROUPS
def t(f, n=20):
    f(); ts = []
    for _ in range(n):
        a = time.perf_counter(); f(); ts.append(time.perf_counter() - a)
    return 1e3 * np.median(ts)
B, sec = S.base_1hz(df); n = len(B)
X, _ = S.features(df)
rows = [
 ("run() total", lambda: model.run(df)),
 ("  features() total", lambda: S.features(df)),
 ("    index->sec + base_1hz", lambda: S.base_1hz(df)),
 ("    context_features", lambda: S.context_features(B)),
 ("    signal: eda", lambda: S._eda_channels(df, sec, n)),
 ("    signal: bvp spectrum", lambda: S._bvp_channels(df, sec, n, B[:, 5])),
 ("    signal: hrv", lambda: S._hrv_features(df, sec, n)),
 ("    signal_features total", lambda: S.signal_features(df, sec, n, B[:, 5])),
 ("  onnx", lambda: model.session.run([model.prob_name], {model.input_name: X.T[None]})),
 ("  output DataFrame", lambda: pd.DataFrame({"Activity": S.LABELS[np.zeros(len(sec), int)]}, index=df.index)),
 ("  df column to_numpy x6", lambda: [df[c].to_numpy(dtype=np.float64) for c in C.INPUT_COLS]),
 ("  index to int64", lambda: df.index.values.astype("datetime64[ns]").astype(np.int64)),
]
print(f"rows {len(df)}  seconds {n}  groups {G}")
for name, f in rows:
    print(f"{name:30s} {t(f):7.2f} ms")
