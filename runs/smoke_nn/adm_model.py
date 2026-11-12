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
from scipy.signal import butter, find_peaks, sosfiltfilt

try:  # the base class file may or may not sit next to the submission in the evaluator
    from adm_base_class import ADM_Base
except Exception:  # pragma: no cover
    ADM_Base = object

LABELS = np.array(["Rest", "Stress", "Activity"], dtype=object)
WINS = (10, 30, 60, 120, 300)
BASE = ("bvp_std", "bvp_dstd", "eda", "eda_ad", "temp", "hr", "ibi", "ibi_cnt")
SLOPE_CH = (2, 4, 5)                       # eda, temp, hr
DEFAULTS = (3.5, 2.0, 1.0, 0.0, 32.5, 85.0, 0.75, 0.0)   # used when a channel is absent in the whole input
GROUPS = ('local', 'rel', 'sess', 'eda2', 'hrv', 'spec')          # feature groups fed to the model (set by tools/train_export.py)
MODEL_KIND = "seq"                        # "tree": (N, F) -> (N, 3);  "seq": (1, F, T) -> (1, 3, T)
STRIDE = 1                                 # classify every STRIDE-th second and repeat the label
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


# ---------------------------------------------------------------- signal-processing channels
# EDA: denoise -> high-pass -> phasic (SCR) component -> peak detection.
# IBI: artefact rejection -> time-domain HRV over 60/120 s.
# BVP: 8 s Hann-window spectrum -> cardiac-band shape and out-of-band (motion/noise) energy.
_SOS_EDA_LP = butter(2, 1.0, fs=4.0, output="sos")
_SOS_EDA_HP = butter(1, 0.05, btype="highpass", fs=4.0, output="sos")
SCR_MIN_AMP = 0.01           # uS, usual minimum SCR amplitude
DEC = 4                      # BVP 64 Hz -> 16 Hz by 4-sample averaging before the spectrum
NFFT = 128                   # 8 s at 16 Hz -> 0.125 Hz resolution, band up to 8 Hz
SPEC_STEP = 2                # spectrum every 2 s, repeated in between
_FREQ = np.fft.rfftfreq(NFFT, DEC / 64.0)
_BAND = (_FREQ >= 0.5) & (_FREQ <= 4.0)
_LO = (_FREQ > 0) & (_FREQ < 0.5)
_HI = _FREQ > 4.0
_HANN = np.hanning(NFFT)
SIG_CH = ("scr_cnt", "scr_amp", "scr_phpow",
          "spec_dom", "spec_peak", "spec_ent", "spec_lofrac", "spec_hifrac", "spec_hrgap", "spec_bandpow")
SIG_LOG = (1, 2)             # channels that get log1p(1e3 * x) after averaging
SIG_WINS = (30, 120)
HRV_WINS = (60, 120)
HRV_CH = ("hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_cov")


def _eda_channels(df, sec, n):
    v = df["EDA"].to_numpy(dtype=np.float64)
    m = ~np.isnan(v)
    x = np.maximum(v[m], 0.0)
    s = sec[m]
    out = np.zeros((n, 3))
    if len(x) < 40:
        return out
    ph = sosfiltfilt(_SOS_EDA_HP, sosfiltfilt(_SOS_EDA_LP, x))
    pk, prop = find_peaks(ph, prominence=SCR_MIN_AMP, distance=4)
    out[:, 0] = np.bincount(s[pk], minlength=n)[:n]
    out[:, 1] = np.bincount(s[pk], weights=prop["prominences"], minlength=n)[:n]
    p, _ = _sec_mean(ph * ph, s, n)
    out[:, 2] = np.nan_to_num(p)
    return out


def _bvp_channels(df, sec, n, hr):
    b = np.nan_to_num(df["BVP"].to_numpy(dtype=np.float64))
    m = len(b) // DEC
    b = b[: m * DEC].reshape(m, DEC).mean(axis=1)            # boxcar low-pass + decimate
    if len(b) < NFFT:
        b = np.pad(b, (0, NFFT - len(b)))
    centres = np.searchsorted(sec[::DEC][:m], np.arange(0, n, SPEC_STEP))
    start = np.clip(centres - NFFT // 2, 0, len(b) - NFFT)
    fr = b[start[:, None] + np.arange(NFFT)]
    fr = (fr - fr.mean(axis=1, keepdims=True)) * _HANN
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    Pb = P[:, _BAND]
    tot = P[:, 1:].sum(axis=1)
    bp = Pb.sum(axis=1)
    k = Pb.argmax(axis=1)
    dom = _FREQ[_BAND][k]
    j = np.arange(len(k))
    near = sum(Pb[j, np.clip(k + o, 0, Pb.shape[1] - 1)] for o in (-2, -1, 0, 1, 2))
    q = Pb / bp[:, None]
    ent = -(q * np.log(q)).sum(axis=1) / np.log(Pb.shape[1])
    S = np.stack([dom, near / bp, ent, P[:, _LO].sum(axis=1) / tot, P[:, _HI].sum(axis=1) / tot,
                  np.zeros(len(k)), np.log(bp)], axis=1)
    S = np.repeat(S, SPEC_STEP, axis=0)[:n]
    S[:, 5] = np.abs(S[:, 0] * 60.0 - hr)                     # PPG rhythm vs device HR
    return S


def _hrv_features(df, sec, n):
    rr_all = df["IBI_duration"].to_numpy(dtype=np.float64)
    m = ~np.isnan(rr_all)
    rr = rr_all[m]
    dt = df["IBI_dt"].to_numpy(dtype=np.float64)[m]
    s = sec[m]
    out = []
    P = np.zeros((n, 6))                                      # n_rr, sum_rr, sum_rr2, n_pair, sum_d2, n_nn50
    if len(rr) > 2:
        ok = (rr > 0.3) & (rr < 1.5)
        consec = np.abs(dt - rr) < 0.01                       # no missed beat before this one
        d = rr[1:] - rr[:-1]
        pair = ok[1:] & ok[:-1] & consec[1:] & (np.abs(d) < 0.2 * rr[:-1])
        sv, rv = s[ok], rr[ok]
        sp, dp = s[1:][pair], d[pair]
        P[:, 0] = np.bincount(sv, minlength=n)[:n]
        P[:, 1] = np.bincount(sv, weights=rv, minlength=n)[:n]
        P[:, 2] = np.bincount(sv, weights=rv * rv, minlength=n)[:n]
        P[:, 3] = np.bincount(sp, minlength=n)[:n]
        P[:, 4] = np.bincount(sp, weights=dp * dp, minlength=n)[:n]
        P[:, 5] = np.bincount(sp, weights=(np.abs(dp) > 0.05).astype(float), minlength=n)[:n]
    for w in HRV_WINS:
        m_, _ = _roll(P, w)
        cnt = np.minimum(np.arange(n) + w // 2 + 1, n) - np.maximum(np.arange(n) - w // 2, 0)
        T = m_ * cnt[:, None]                                 # window sums
        with np.errstate(invalid="ignore", divide="ignore"):
            rmssd = np.sqrt(T[:, 4] / T[:, 3])
            mean = T[:, 1] / T[:, 0]
            sdnn = np.sqrt(np.maximum(T[:, 2] / T[:, 0] - mean * mean, 0.0))
            pnn = T[:, 5] / T[:, 3]
        few_p, few_r = T[:, 3] < 5, T[:, 0] < 5
        rmssd[few_p] = -1.0; pnn[few_p] = -1.0; sdnn[few_r] = -1.0
        out.append(np.stack([rmssd, sdnn, pnn, T[:, 3] / cnt], axis=1))
    H = np.concatenate(out, axis=1)
    med = np.array([np.median(c[c >= 0]) if (c >= 0).any() else -1.0 for c in H[:, :2].T])
    rel = np.where(H[:, :2] >= 0, H[:, :2] - med, 0.0)      # relative to the recording (60 s RMSSD, SDNN)
    return np.concatenate([H, rel], axis=1)


def signal_features(df, sec, n, hr):
    S = np.concatenate([_eda_channels(df, sec, n), _bvp_channels(df, sec, n, hr)], axis=1)
    out = []
    mref = None
    for w in SIG_WINS:
        m, _ = _roll(S, w)
        m[:, SIG_LOG] = np.log1p(1e3 * m[:, SIG_LOG])
        if mref is None:
            mref = m
        out.append(m)
    out.append(mref - np.median(mref, axis=0))
    out.append(_hrv_features(df, sec, n))
    return np.concatenate(out, axis=1)


def signal_feature_names():
    names = []
    for w in SIG_WINS:
        names += [f"{c}_m{w}" for c in SIG_CH]
    names += [f"{c}_srel" for c in SIG_CH]
    for w in HRV_WINS:
        names += [f"{c}{w}" for c in HRV_CH]
    names += ["hrv_rmssd60_srel", "hrv_sdnn60_srel"]
    return names


def feature_names():
    names = []
    for w in WINS:
        names += [f"{c}_m{w}" for c in BASE] + [f"{c}_s{w}" for c in BASE]
        names += [f"{BASE[j]}_slope{w}" for j in SLOPE_CH]
    names += [f"{c}_rel" for c in BASE] + [f"{c}_z" for c in BASE] + [f"{c}_rank" for c in BASE]
    names += [f"{c}_sessmed" for c in BASE] + [f"{c}_sessstd" for c in BASE]
    names += [f"{c}_sessrange" for c in BASE] + [f"{c}_sesshi" for c in BASE] + [f"{c}_relrange" for c in BASE]
    names += ["pos_since_start", "pos_to_end", "pos_frac", "pos_duration"]
    return names + signal_feature_names()


def group_of(name):
    if name.startswith("scr_"):
        return "eda2"
    if name.startswith("spec_"):
        return "spec"
    if name.startswith("hrv_"):
        return "hrv"
    if name.startswith("pos_"):
        return "pos"
    if name.endswith(("_sessrange", "_sesshi", "_relrange")):
        return "sess2"
    if name.endswith(("_rel", "_z", "_rank")):
        return "rel"
    if name.endswith(("_sessmed", "_sessstd")):
        return "sess"
    return "local"


def group_mask(groups):
    return np.array([group_of(n) in groups for n in feature_names()])


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
        if w == 60:
            m60 = m
        h = w // 2
        a = np.minimum(i + h, n - 1)
        b = np.maximum(i - h, 0)
        out += [m, s, B[a][:, SLOPE_CH] - B[b][:, SLOPE_CH]]
    med = np.median(B, axis=0)
    sd = B.std(axis=0)
    rel = m30 - med
    rank = np.argsort(np.argsort(m30, axis=0), axis=0) / max(n - 1, 1)
    out += [rel, rel / (sd + 1e-3), rank, np.broadcast_to(med, B.shape), np.broadcast_to(sd, B.shape)]
    lo, hi = np.percentile(m60, [5, 95], axis=0)
    rng = hi - lo
    out += [np.broadcast_to(rng, B.shape), np.broadcast_to(hi - med, B.shape), (m30 - lo) / (rng + 1e-3)]
    pos = np.stack([i, n - 1 - i, i / max(n - 1, 1), np.full(n, n)], axis=1).astype(np.float64)
    out.append(pos)
    return np.concatenate(out, axis=1).astype(np.float32)


_MASK = None


def features(df, groups=None):
    """Feature matrix restricted to the given groups (default: GROUPS) and the row -> second map."""
    global _MASK
    B, sec = base_1hz(df)
    X = np.concatenate([context_features(B),
                        signal_features(df, sec, len(B), B[:, 5]).astype(np.float32)], axis=1)
    if groups is None:
        if _MASK is None:
            _MASK = group_mask(GROUPS)
        return X[:, _MASK], sec
    return X[:, group_mask(groups)], sec


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
        if MODEL_KIND == "seq":
            prob = self.session.run([self.prob_name], {self.input_name: X.T[None]})[0][0].T
        elif STRIDE > 1:
            n = len(X)
            src = np.minimum((np.arange(n) // STRIDE) * STRIDE + STRIDE // 2, n - 1)
            pick = np.unique(src)
            prob = self.session.run([self.prob_name], {self.input_name: X[pick]})[0]
            prob = prob[np.searchsorted(pick, src)]
        else:
            prob = self.session.run([self.prob_name], {self.input_name: X})[0]
        cls = postprocess(prob)
        return pd.DataFrame({"Activity": LABELS[cls[sec]]}, index=input.index)


def get_model():
    model = StressSweatModel()
    model.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return model
