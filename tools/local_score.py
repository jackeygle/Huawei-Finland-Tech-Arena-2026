"""Run a submission zip the way the official evaluator is described to, and report the score.

Unzips into a temp dir, imports the module, calls get_model(), then for every session:
run() on a DataFrame holding only the six input columns, timed over --runs calls.
Quality here is NOT a generalisation estimate when the model was trained on these sessions;
use tools/cv.py for that. This script checks the interface, the timing and the speed score.

Run single-threaded:  OMP_NUM_THREADS=1 python tools/local_score.py runs/v1/submission_v1.zip
"""
import argparse
import importlib.util
import os
import sys
import tempfile
import time
import zipfile

import numpy as np

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("zip")
ap.add_argument("--runs", type=int, default=50)
ap.add_argument("--sessions", type=int, default=0, help="limit number of sessions (0 = all)")
a = ap.parse_args()

tmp = tempfile.mkdtemp(dir=os.path.join(C.ROOT, "runs"))
with zipfile.ZipFile(a.zip) as z:
    z.extractall(tmp)
    names = z.namelist()
py = [n for n in names if n.endswith(".py")]
assert len(py) == 1 and py[0][:-3] + ".onnx" in names, f"zip must hold x.py and x.onnx, got {names}"
print(f"zip {os.path.getsize(a.zip) / 1e6:.2f} MB  files {names}")
spec = importlib.util.spec_from_file_location("submitted_model", os.path.join(tmp, py[0]))
mod = importlib.util.module_from_spec(spec)
sys.modules["submitted_model"] = mod
spec.loader.exec_module(mod)
model = mod.get_model()

files = C.session_files()
if a.sessions:
    files = files[:: max(len(files) // a.sessions, 1)]
VALID = {"Rest": 0, "Stress": 1, "Activity": 2}
ys, ps, ss, ws, times, rows = [], [], [], [], [], []
for k, fn in enumerate(files):
    df, y_row, prot, sid = C.load_session(fn)
    out = model.run(df)
    assert list(out.columns) == ["Activity"] and out.index.equals(df.index), "bad output frame"
    assert set(out["Activity"].unique()) <= set(VALID), f"invalid labels {set(out['Activity'].unique())}"
    t = []
    for _ in range(a.runs):
        t0 = time.perf_counter(); model.run(df); t.append(time.perf_counter() - t0)
    times.append(np.mean(t)); rows.append(len(df))
    ts = df.index.values.astype("datetime64[ns]").astype(np.int64) // 1_000_000_000
    ys.append(y_row); ps.append(out["Activity"].map(VALID).to_numpy()); ss.append(np.full(len(df), k)); ws.append(ts // 10)
y, p, s, w = map(np.concatenate, (ys, ps, ss, ws))
mf, f, cm = C.window_score(y, p, s, w)
times = np.array(times)
met = times.mean()
print(f"sessions {len(files)}  rows/session median {int(np.median(rows))} max {max(rows)}")
print(f"run() time per session: mean {met * 1e3:.1f} ms  median {np.median(times) * 1e3:.1f}  max {times.max() * 1e3:.1f}")
print(f"quality (on training sessions): macro-F1 {mf:.4f} -> {mf * 80:.1f}/80   F1 {f.round(3)}")
print(f"speed score if MET = mean session time: {max(0.0, (1 - met / 0.3) * 20):.2f}/20")
print(f"speed score if MET = slowest session:   {max(0.0, (1 - times.max() / 0.3) * 20):.2f}/20")
