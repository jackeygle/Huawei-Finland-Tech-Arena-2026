"""Package the u4so ONNX (train_full_u4so.py) with v7_src/adm_model.py as a submission zip (evaluation env).
Checks live features == training features and ONNX == torch on the parity recording, embeds templates from all
training data, sets MODEL_KIND seq, N_OUT_CLS 4, VERSION_BY_DATE True.
Usage (from tools/): ../env313/bin/python ../analysis/acc_sync/package_v7.py --name v7_u4so_model --out v7_u4so"""
import argparse, json, os, re, shutil, sys, zipfile
import numpy as np, onnxruntime as ort
sys.path.insert(0, ".")
import common as C
from template_decode import build_templates

ACC = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--by_date", type=int, default=1); ap.add_argument("--src", default="v7_src")
a = ap.parse_args()
src_dir = os.path.join(C.ROOT, "runs", a.name); out = os.path.join(C.ROOT, "runs", a.out); os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(src_dir, "train_config.json"))); groups = tuple(cfg["feature_groups"])
par = np.load(os.path.join(src_dir, "parity.npz")); k = int(par["session"])
df, _, _, _ = C.load_session(C.session_files()[k])
X, _ = C.M.features(df, groups=groups)
print(f"live features vs training features (session {k}): max abs diff {np.abs(X.T[None] - par['x']).max():.2e}  shape {X.shape}")
s = ort.InferenceSession(os.path.join(src_dir, "model.onnx"), providers=["CPUExecutionProvider"])
p = s.run(None, {"x": par["x"]})[0]
assert p.shape[1] == 7, p.shape
print(f"ONNX output {p.shape}; vs torch max abs diff {np.abs(p - par['prob']).max():.2e}")
src = open(os.path.join(ACC, a.src, "adm_model.py")).read()
subs = [(r"(?m)^GROUPS = \(.*?\)", "GROUPS = " + repr(groups)), (r'(?m)^MODEL_KIND = "\w+"', 'MODEL_KIND = "seq"'),
        (r"(?m)^N_OUT_CLS = 3", "N_OUT_CLS = 4"), (r"(?m)^DECODE_ALPHA = [0-9.e-]+", "DECODE_ALPHA = 0.002"),
        (r"(?m)^DECODE_RES = \d+", "DECODE_RES = 10")]
dd = C.build_dataset()
keys = [(str(p_), "V1" if s_.startswith("S") else "V2") for p_, s_ in zip(dd["prot"], dd["sid"])]
T = build_templates([dd["y"][dd["sess"] == i] for i in range(len(keys))], keys)
lit = {f"{k_[0]}/{k_[1]}": [dict(c=int(g["c"]), mean=round(float(g["mean"]), 2), sd=round(float(g["sd"]), 2),
                               opt=bool(g["opt"])) for g in v] for k_, v in sorted(T.items())}
subs.append((r"(?m)^TEMPLATES = None", "TEMPLATES = " + repr(lit)))
if a.by_date:
    subs.append((r"(?m)^VERSION_BY_DATE = False", "VERSION_BY_DATE = True"))
for pat, rep in subs:
    src, n = re.subn(pat, lambda m: rep, src, count=1); assert n == 1, pat
print("templates:", {k_: len(v) for k_, v in lit.items()}, "| version by date:", bool(a.by_date))
open(os.path.join(out, "adm_model.py"), "w").write(src)
shutil.copy(os.path.join(src_dir, "model.onnx"), os.path.join(out, "adm_model.onnx"))
z = os.path.join(out, f"submission_{a.out}.zip")
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    f.write(os.path.join(out, "adm_model.py"), "adm_model.py"); f.write(os.path.join(out, "adm_model.onnx"), "adm_model.onnx")
print(f"{z}  {os.path.getsize(z) / 1e6:.2f} MB")
