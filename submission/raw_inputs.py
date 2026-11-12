"""Raw-signal inputs for the waveform model (scheme C), on the same 1 s grid as the feature pipeline.

Per input DataFrame returns float32 arrays
  bvp16 : (n*16,)   BVP averaged into 16 slots per second (64 Hz -> 16 Hz), per-recording robust z-score
  slow4 : (2, n*4)  log EDA and skin temperature, 4 slots per second, per-recording robust z-score
  fast1 : (4, n)    HR, IBI, beats per second, and EDA level before normalisation (log), per-recording z-score
and the row -> second map. Missing slots are linearly interpolated.
"""
import numpy as np


def _slots(val, t_ns, t0, per_s, n):
    m = ~np.isnan(val)
    rel = t_ns[m] - t0
    sec = rel // 1_000_000_000
    pos = (rel % 1_000_000_000) * per_s // 1_000_000_000
    k = sec * per_s + pos
    tot = np.bincount(k, weights=val[m], minlength=n * per_s)[: n * per_s]
    cnt = np.bincount(k, minlength=n * per_s)[: n * per_s]
    out = np.full(n * per_s, np.nan)
    ok = cnt > 0
    out[ok] = tot[ok] / cnt[ok]
    return out


def _interp(x, default=0.0):
    ok = ~np.isnan(x)
    if ok.all():
        return x
    if not ok.any():
        return np.full(len(x), default)
    i = np.arange(len(x))
    return np.interp(i, i[ok], x[ok])


def _rz(x):
    """Robust z-score along the last axis (median / IQR), clipped to +-8."""
    med = np.median(x, axis=-1, keepdims=True)
    q1, q3 = np.percentile(x, [25, 75], axis=-1, keepdims=True)
    return np.clip((x - med) / (q3 - q1 + 1e-3), -8, 8)


def raw_inputs(df):
    t = df.index.values.astype("datetime64[ns]").astype(np.int64)
    t0 = t[0]
    sec = (t - t0) // 1_000_000_000
    n = int(sec.max()) + 1
    bvp = _interp(_slots(df["BVP"].to_numpy(np.float64), t, t0, 16, n))
    eda = _interp(_slots(np.log1p(np.maximum(df["EDA"].to_numpy(np.float64), 0)), t, t0, 4, n))
    tmp = df["Temperature"].to_numpy(np.float64).copy()
    tmp[tmp > 45] = np.nan
    temp = _interp(_slots(tmp, t, t0, 4, n), 32.5)
    hr = _interp(_slots(df["HR"].to_numpy(np.float64), t, t0, 1, n), 85.0)
    ibi_raw = df["IBI_duration"].to_numpy(np.float64)
    ibi = _interp(_slots(ibi_raw, t, t0, 1, n), 0.75)
    m = ~np.isnan(ibi_raw)
    beats = np.bincount(sec[m], minlength=n)[:n].astype(np.float64)
    eda_level = eda.reshape(n, 4).mean(axis=1)
    bvp16 = _rz(bvp).astype(np.float32)
    slow4 = _rz(np.stack([eda, temp])).astype(np.float32)
    fast1 = np.stack([_rz(hr), _rz(ibi), beats, eda_level]).astype(np.float32)
    return bvp16, slow4, fast1, sec
