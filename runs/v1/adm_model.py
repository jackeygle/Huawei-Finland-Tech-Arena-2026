"""Huawei Finland Tech Arena 2026 submission: Rest / Stress / Activity from wrist physiology.

Self-contained on purpose: the submission zip holds only this file and the ONNX model, so the
feature code lives here and the training scripts import it from this file.

Pipeline: 64 Hz input -> 1 Hz base channels -> rolling context features -> ONNX classifier
-> optional probability smoothing -> one label per input row.
"""
import pathlib

import numpy as np
import onnxruntime as ort
import pandas as pd

try:  # the base class file may or may not sit next to the submission in the evaluator
    from adm_base_class import ADM_Base
except Exception:  # pragma: no cover
    ADM_Base = object

LABELS = np.array(["Rest", "Stress", "Activity"], dtype=object)
WINS = (10, 30, 60, 120, 300)
BASE = ("bvp_std", "bvp_dstd", "eda", "eda_ad", "temp", "hr", "ibi", "ibi_cnt")
SLOPE_CH = (2, 4, 5)                       # eda, temp, hr
DEFAULTS = (3.5, 2.0, 1.0, 0.0, 32.5, 85.0, 0.75, 0.0)   # used when a channel is absent in the whole input
SMOOTH_S = 0                               # centred box filter on class probabilities, seconds (0 = off)
LOGIT_BIAS = (0.0, 0.0, 0.0)               # added to log-probabilities before argmax


def _sec_mean(val, sec, n):
    """Per-second mean of the non-NaN samples of val; NaN where a second has none."""
    m = ~np.isnan(val)
    s = sec[m]
    cnt = np.bincount(s, minlength=n)
    tot = np.bincount(s, weights=val[m], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / cnt, cnt


def _fill(x, default):
    ok = ~np.isnan(x)
    if ok.all():
        return x
    if not ok.any():
        return np.full(len(x), default)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def base_1hz(df):
    """Return (B, sec): B is (n_seconds, 8) base channels, sec maps every input row to its second."""
    t = df.index.values.astype("datetime64[ns]").astype(np.int64)
    sec = ((t - t[0]) // 1_000_000_000).astype(np.int64)
    n = int(sec.max()) + 1

    bvp = df["BVP"].to_numpy(dtype=np.float64)
    d = np.empty_like(bvp)
    d[0] = 0.0
    d[1:] = bvp[1:] - bvp[:-1]
    m1, _ = _sec_mean(bvp, sec, n)
    m2, _ = _sec_mean(bvp * bvp, sec, n)
    bvp_std = np.log1p(np.sqrt(np.maximum(m2 - m1 * m1, 0.0)))
    bvp_dstd = np.log1p(np.sqrt(_sec_mean(d * d, sec, n)[0]))   # high-frequency energy

    eda_raw = df["EDA"].to_numpy(dtype=np.float64)
    me = ~np.isnan(eda_raw)
    ev = np.log1p(np.maximum(eda_raw[me], 0.0))
    es = sec[me]
    eda = _sec_mean(ev, es, n)[0]
    if len(ev) > 1:
        ead = np.empty_like(ev)
        ead[0] = 0.0
        ead[1:] = np.abs(ev[1:] - ev[:-1])
        eda_ad = _sec_mean(ead, es, n)[0]                         # phasic activity
    else:
        eda_ad = np.full(n, np.nan)

    temp_raw = df["Temperature"].to_numpy(dtype=np.float64).copy()
    temp_raw[temp_raw > 45.0] = np.nan                            # sensor fault (~209 C)
    temp = _sec_mean(temp_raw, sec, n)[0]
    hr = _sec_mean(df["HR"].to_numpy(dtype=np.float64), sec, n)[0]
    ibi, ibi_cnt = _sec_mean(df["IBI_duration"].to_numpy(dtype=np.float64), sec, n)

    cols = [bvp_std, bvp_dstd, eda, eda_ad, temp, hr, ibi]
    B = np.empty((n, len(BASE)))
    for j, c in enumerate(cols):
        B[:, j] = _fill(c, DEFAULTS[j])
    B[:, 7] = ibi_cnt                                             # beats detected per second (0 under motion)
    return B, sec


def _roll(x, w):
    """Centred rolling mean and std over about w rows (shrinks at the edges). x is (n, k)."""
    n = len(x)
    h = w // 2
    i = np.arange(n)
    lo = np.maximum(i - h, 0)
    hi = np.minimum(i + h + 1, n)
    z = np.zeros((1, x.shape[1]))
    c1 = np.concatenate([z, np.cumsum(x, axis=0)])
    c2 = np.concatenate([z, np.cumsum(x * x, axis=0)])
    cnt = (hi - lo)[:, None]
    m = (c1[hi] - c1[lo]) / cnt
    v = (c2[hi] - c2[lo]) / cnt - m * m
    return m, np.sqrt(np.maximum(v, 0.0))


def feature_names():
    names = []
    for w in WINS:
        names += [f"{c}_m{w}" for c in BASE] + [f"{c}_s{w}" for c in BASE]
        names += [f"{BASE[j]}_slope{w}" for j in SLOPE_CH]
    names += [f"{c}_rel" for c in BASE] + [f"{c}_z" for c in BASE] + [f"{c}_rank" for c in BASE]
    names += [f"{c}_sessmed" for c in BASE] + [f"{c}_sessstd" for c in BASE]
    return names


def context_features(B):
    """(n_seconds, n_features) float32 matrix from the 1 Hz base channels."""
    n = len(B)
    i = np.arange(n)
    Bc = B - B.mean(axis=0)                    # centring keeps the cumulative sums well conditioned
    out = []
    m30 = None
    for w in WINS:
        m, s = _roll(Bc, w)
        m = m + B.mean(axis=0)
        if w == 30:
            m30 = m
        h = w // 2
        a = np.minimum(i + h, n - 1)
        b = np.maximum(i - h, 0)
        out += [m, s, B[a][:, SLOPE_CH] - B[b][:, SLOPE_CH]]
    med = np.median(B, axis=0)
    sd = B.std(axis=0)
    rel = m30 - med
    rank = np.argsort(np.argsort(m30, axis=0), axis=0) / max(n - 1, 1)
    out += [rel, rel / (sd + 1e-3), rank, np.broadcast_to(med, B.shape), np.broadcast_to(sd, B.shape)]
    return np.concatenate(out, axis=1).astype(np.float32)


def features(df):
    B, sec = base_1hz(df)
    return context_features(B), sec


def postprocess(prob, smooth_s=SMOOTH_S, bias=LOGIT_BIAS):
    """Class index per second from (n, 3) probabilities."""
    p = np.asarray(prob, dtype=np.float64)
    if smooth_s and len(p) > 1:
        p, _ = _roll(p, smooth_s)
    return np.argmax(np.log(p + 1e-9) + np.asarray(bias), axis=1)


class StressSweatModel(ADM_Base):
    def load_model(self, onnx_file: pathlib.Path) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_file), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.prob_name = self.session.get_outputs()[-1].name

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        if len(input) == 0:
            return pd.DataFrame({"Activity": pd.Series([], dtype=object)}, index=input.index)
        X, sec = features(input)
        prob = self.session.run([self.prob_name], {self.input_name: X})[0]
        cls = postprocess(prob)
        return pd.DataFrame({"Activity": LABELS[cls[sec]]}, index=input.index)


def get_model():
    model = StressSweatModel()
    model.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return model
