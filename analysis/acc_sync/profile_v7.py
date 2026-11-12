"""Stage timing of run() for a packaged u4so submission (single thread). Usage (from tools/):
   ../env313/bin/python ../analysis/acc_sync/profile_v7.py <dir with adm_model.py/.onnx> <session index> [...]"""
import sys, time, importlib.util, os
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import common as C
zipdir = sys.argv[1]
spec = importlib.util.spec_from_file_location("sub", os.path.join(zipdir, "adm_model.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
model = S.get_model()
def t(f, n=30):
    f(); ts = []
    for _ in range(n):
        a = time.perf_counter(); f(); ts.append(time.perf_counter() - a)
    return 1e3 * np.median(ts)
for k in [int(v) for v in sys.argv[2:]]:
    df, _, prot, sid = C.load_session(C.session_files()[k])
    B, sec = S.base_1hz(df); n = len(B)
    X, _ = S.features(df)
    prob = model.session.run([model.prob_name], {model.input_name: X.T[None]})[0][0].T
    ver = S.version_from_date(df.index)
    cls = S.decode_u4so(prob, S.TEMPLATES, ver)
    rows = [
     ("run() total", lambda: model.run(df)),
     ("  features() total", lambda: S.features(df)),
     ("    base_1hz (index->sec, 1 Hz channels)", lambda: S.base_1hz(df)),
     ("    context_features (rolling/rel/sess)", lambda: S.context_features(B)),
     ("    signal: eda (SCR)", lambda: S._eda_channels(df, sec, n)),
     ("    signal: bvp spectrum", lambda: S._bvp_channels(df, sec, n, B[:, 5])),
     ("    signal: hrv", lambda: S._hrv_features(df, sec, n)),
     ("  onnx TCN", lambda: model.session.run([model.prob_name], {model.input_name: X.T[None]})),
     ("  decode_u4so (date: 3 templates)", lambda: S.decode_u4so(prob, S.TEMPLATES, ver)),
     ("  decode_u4so (no date: 6 templates)", lambda: S.decode_u4so(prob, S.TEMPLATES, None)),
     ("  output DataFrame", lambda: pd.DataFrame(S.LABELS[cls[sec]][:, None], index=df.index, columns=["Activity"], dtype=object)),
    ]
    print(f"== session {k} {prot}/{sid}: rows {len(df)}  seconds {n}")
    for name, f in rows:
        print(f"{name:42s} {t(f):7.2f} ms", flush=True)
