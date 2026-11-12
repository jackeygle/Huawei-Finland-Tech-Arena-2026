"""Official-style scoring and the shared CV fold assignment. Pure numpy, importable from any environment."""
import numpy as np


def majority(codes, groups, n_groups, n_classes=3):
    """Most frequent code (>=0) per group; -1 where a group has no scored element."""
    ok = codes >= 0
    cnt = np.zeros((n_groups, n_classes), dtype=np.int64)
    np.add.at(cnt, (groups[ok], codes[ok]), 1)
    out = cnt.argmax(axis=1)
    out[cnt.sum(axis=1) == 0] = -1
    return out


def f1_per_class(y, p, n_classes=3):
    f = []
    for c in range(n_classes):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return np.array(f)


def window_score(y, pred, sess, win):
    """Macro-F1 over 10 s windows, majority label per window (scored seconds only)."""
    key = sess.astype(np.int64) * 10_000_000_000 + win
    _, g = np.unique(key, return_inverse=True)
    ng = g.max() + 1
    yw = majority(y, g, ng)
    pw = majority(np.where(y >= 0, pred, -1), g, ng)
    ok = yw >= 0
    f = f1_per_class(yw[ok], pw[ok])
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (yw[ok], pw[ok]), 1)
    return f.mean(), f, cm


def fold_of_session(kind, sid_of_sess, n_folds=6, seed=0):
    """Fold id per session. 'subject': all sessions of a subject share a fold; 'session': each session alone.
    Identical to the assignment used by tools/cv.py for the tree models."""
    rng = np.random.RandomState(seed)
    if kind == "subject":
        units = np.unique(sid_of_sess)
        unit_of_sess = np.searchsorted(units, sid_of_sess)
    else:
        units = np.arange(len(sid_of_sess))
        unit_of_sess = units
    assign = rng.permutation(len(units)) % n_folds
    return assign[unit_of_sess]
