"""Package a sequence-model ONNX (from nn_train_full.py) as a submission zip and verify it (evaluation env).

Checks: (1) features computed live by the submission code equal the training features of the parity
recording; (2) ONNX probabilities equal the torch probabilities; then builds runs/<name>/submission_<name>.zip.
Usage: OMP_NUM_THREADS=1 python tools/package_nn.py --name v2_tcn
"""
import argparse, json, os, re, shutil, zipfile
import numpy as np, onnxruntime as ort
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
ap.add_argument("--templates", action="store_true", help="embed protocol templates (all training data)")
ap.add_argument("--alpha", type=float, default=0.002)
ap.add_argument("--out", default="", help="output folder name (default: same as --name)")
ap.add_argument("--by_date", action="store_true", help="choose the protocol version from the recording date")
a = ap.parse_args()
src_dir = os.path.join(C.ROOT, "runs", a.name)
out = os.path.join(C.ROOT, "runs", a.out or a.name)
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(src_dir, "train_config.json")))
groups = tuple(cfg["feature_groups"])
par = np.load(os.path.join(src_dir, "parity.npz"))
k = int(par["session"])
df, _, _, _ = C.load_session(C.session_files()[k])
X, _ = C.M.features(df, groups=groups)
dx = np.abs(X.T[None] - par["x"]).max()
print(f"live features vs training features (session {k}): max abs diff {dx:.2e}  shape {X.shape}")
s = ort.InferenceSession(os.path.join(src_dir, "model.onnx"), providers=["CPUExecutionProvider"])
p = s.run(None, {"x": par["x"]})[0]
print(f"ONNX vs torch prob: max abs diff {np.abs(p - par['prob']).max():.2e}, label agreement "
      f"{(p.argmax(1) == par['prob'].argmax(1)).mean():.5f}")
src = open(os.path.join(C.ROOT, "submission", "adm_model.py")).read()
src, n1 = re.subn(r"(?m)^GROUPS = \(.*?\)", "GROUPS = " + repr(groups), src, count=1)
src, n2 = re.subn(r'(?m)^MODEL_KIND = "\w+"', 'MODEL_KIND = "seq"', src, count=1)
assert n1 == 1 and n2 == 1
if a.templates:
    from template_decode import build_templates
    dd = C.build_dataset()
    keys = [(str(p), "V1" if s_.startswith("S") else "V2") for p, s_ in zip(dd["prot"], dd["sid"])]
    T = build_templates([dd["y"][dd["sess"] == i] for i in range(len(keys))], keys)
    lit = {f"{k[0]}/{k[1]}": [dict(c=int(g["c"]), mean=round(float(g["mean"]), 2), sd=round(float(g["sd"]), 2),
                                   opt=bool(g["opt"])) for g in v] for k, v in sorted(T.items())}
    src, n3 = re.subn(r"(?m)^TEMPLATES = None", "TEMPLATES = " + repr(lit), src, count=1)
    src, n4 = re.subn(r"(?m)^DECODE_ALPHA = [0-9.e-]+", f"DECODE_ALPHA = {a.alpha}", src, count=1)
    assert n3 == 1 and n4 == 1
    if a.by_date:
        src, n5 = re.subn(r"(?m)^VERSION_BY_DATE = False", "VERSION_BY_DATE = True", src, count=1)
        assert n5 == 1
        print("version from recording date: on")
    print("templates:", {k: len(v) for k, v in lit.items()}, "alpha", a.alpha)
open(os.path.join(out, "adm_model.py"), "w").write(src)
shutil.copy(os.path.join(src_dir, "model.onnx"), os.path.join(out, "adm_model.onnx"))
z = os.path.join(out, f"submission_{a.out or a.name}.zip")
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    f.write(os.path.join(out, "adm_model.py"), "adm_model.py")
    f.write(os.path.join(out, "adm_model.onnx"), "adm_model.onnx")
print(f"{z}  {os.path.getsize(z) / 1e6:.2f} MB")
