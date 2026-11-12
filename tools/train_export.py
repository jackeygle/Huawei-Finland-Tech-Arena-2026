"""Train the final model on all labelled data, export ONNX, build the submission zip.

Usage: python tools/train_export.py --name v1 [--iters 200] [--groups local,rel,sess]
Writes runs/<name>/adm_model.py, adm_model.onnx, submission_<name>.zip and train_config.json.
"""
import argparse
import json
import os
import re
import zipfile

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import HistGradientBoostingClassifier

import onnx.helper as _oh

import common as C

# skl2onnx 1.20 passes Python bools in integer-list attributes for HistGradientBoosting trees,
# which onnx 1.23 rejects; cast them to int.
_make_attribute = _oh.make_attribute


def _make_attribute_intsafe(key, value, *args, **kw):
    if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in value):
        value = [int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
    return _make_attribute(key, value, *args, **kw)


_oh.make_attribute = _make_attribute_intsafe

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--leaves", type=int, default=15)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--groups", default="local,rel,sess")
ap.add_argument("--stride", type=int, default=1, help="classify every k-th second at inference")
ap.add_argument("--sub", type=int, default=1, help="train on every k-th second")
a = ap.parse_args()
groups = tuple(a.groups.split(","))

d = C.build_dataset()
X, y = d["X"][:, C.M.group_mask(groups)], d["y"]
ok = (y >= 0) & (np.arange(len(y)) % a.sub == 0)
clf = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=a.lr, max_leaf_nodes=a.leaves,
                                     class_weight="balanced", random_state=a.seed)
clf.fit(X[ok], y[ok])

onx = convert_sklearn(clf, initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
                      options={id(clf): {"zipmap": False}}, target_opset=17)
out = os.path.join(C.ROOT, "runs", a.name)
os.makedirs(out, exist_ok=True)
onnx_path = os.path.join(out, "adm_model.onnx")
with open(onnx_path, "wb") as f:
    f.write(onx.SerializeToString())
# the submitted copy carries the feature groups and stride this model was trained for
src = open(os.path.join(C.ROOT, "submission", "adm_model.py")).read()
src, n1 = re.subn(r"(?m)^GROUPS = \(.*?\)", "GROUPS = " + repr(groups), src, count=1)
src, n2 = re.subn(r"(?m)^STRIDE = \d+", f"STRIDE = {a.stride}", src, count=1)
assert n1 == 1 and n2 == 1
open(os.path.join(out, "adm_model.py"), "w").write(src)

# parity check: ONNX against sklearn on a sample of rows
rng = np.random.RandomState(0)
idx = rng.choice(len(X), 20000, replace=False)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
p_onnx = sess.run(None, {"input": X[idx]})[-1]
p_skl = clf.predict_proba(X[idx])
agree = float((p_onnx.argmax(1) == p_skl.argmax(1)).mean())
maxdiff = float(np.abs(p_onnx - p_skl).max())
print(f"ONNX vs sklearn: label agreement {agree:.5f}, max prob diff {maxdiff:.2e}")

zpath = os.path.join(out, f"submission_{a.name}.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(out, "adm_model.py"), "adm_model.py")
    z.write(onnx_path, "adm_model.onnx")
json.dump(dict(vars(a), n_features=int(X.shape[1]), n_train=int(ok.sum()), onnx_agreement=agree,
               onnx_max_prob_diff=maxdiff), open(os.path.join(out, "train_config.json"), "w"), indent=1)
print(f"{zpath}  {os.path.getsize(zpath) / 1e6:.2f} MB (onnx {os.path.getsize(onnx_path) / 1e6:.2f} MB)")
