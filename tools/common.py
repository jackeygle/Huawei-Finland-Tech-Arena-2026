"""Shared helpers: label mapping, dataset building, official-style scoring."""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "submission"))
import adm_model as M  # noqa: E402
from scoring import f1_per_class, fold_of_session, majority, window_score  # noqa: E402,F401

DATA = os.path.join(ROOT, "downloads", "data")
INPUT_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
ALL_GROUPS = ("local", "rel", "sess", "sess2", "pos", "eda2", "spec", "hrv", "resp", "pulse", "recov")
STRESS_P = {"Stroop", "TMCT", "Real opinion", "Opposite opinion", "Subtraction"}


def phase_to_label(p):
    """0 Rest, 1 Stress, 2 Activity, -1 not scored (phases_to_activity.txt)."""
    if p in STRESS_P:
        return 1
    if p in ("Warm-up", "Sprint", "Cooldown") or "rpm" in p:
        return 2
    if p in ("Baseline", "Rest"):
        return 0
    return -1


def session_files():
    return sorted(glob.glob(os.path.join(DATA, "*", "*.parquet")))


def load_session(fn):
    df = pd.read_parquet(fn)
    y_row = df["Phase"].map(phase_to_label).to_numpy()
    return df[INPUT_COLS], y_row, fn.split(os.sep)[-2], str(df["user_id"].iloc[0])


def build_dataset(cache=os.path.join(ROOT, "runs", "dataset_v4.npz"), rebuild=False):
    """1 Hz feature matrix for all sessions. Returns dict of arrays (X, y, sess, sid, prot, win)."""
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    Xs, ys, sess, wins, meta = [], [], [], [], []
    for k, fn in enumerate(session_files()):
        df, y_row, prot, sid = load_session(fn)
        X, sec = M.features(df, groups=ALL_GROUPS)
        n = len(X)
        y = majority(y_row, sec, n)
        t0 = df.index.values.astype("datetime64[ns]").astype(np.int64)[0] // 1_000_000_000
        Xs.append(X); ys.append(y); sess.append(np.full(n, k)); wins.append((t0 + np.arange(n)) // 10)
        meta.append((prot, sid))
    d = dict(X=np.concatenate(Xs), y=np.concatenate(ys), sess=np.concatenate(sess), win=np.concatenate(wins),
             prot=np.array([m[0] for m in meta]), sid=np.array([m[1] for m in meta]),
             names=np.array(M.feature_names()))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **d)
    return d
