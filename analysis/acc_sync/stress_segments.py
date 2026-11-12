"""Wrong 10 s windows in STRESS sessions by true segment ordinal (neighbour fill, res 5, date version)."""
import sys, itertools, collections
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "../analysis/acc_sync")
from scoring import fold_of_session, majority
from template_decode import build_templates
from decode_variants import decode_v, sid, prot, keys
tag = sys.argv[1]
for kind in ("subject",):
    Z = np.load(f"../runs/{tag}_{kind}.npz"); prob, y, sess, win = Z["prob"], Z["y"], Z["sess"], Z["win"]
    fos = fold_of_session(kind, sid); pred = prob.argmax(1).copy()
    for k in range(6):
        tr = np.flatnonzero(fos != k)
        T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i; pred[m], _ = decode_v(prob[m], T, 0.002, 5, keys[i][1], ufill="neighbour")
    stats = collections.defaultdict(lambda: [0, 0])   # (ver, seg name) -> [wrong, total] windows
    onset = collections.defaultdict(list)             # signed boundary error (s) of predicted S onsets
    for i in range(len(sid)):
        if prot[i] != "STRESS": continue
        m = np.flatnonzero(sess == i); yy, pp, ww = y[m], pred[m], win[m]
        # segment ordinal per second (scored runs only, U skipped)
        lab = np.full(len(yy), "", dtype=object); o = collections.Counter()
        t = 0
        for v, g in itertools.groupby(yy):
            n = len(list(g))
            if v >= 0:
                nm = "RS"[v] + str(o["RS"[v]]); o["RS"[v]] += 1; lab[t:t + n] = nm
            t += n
        _, g = np.unique(ww, return_inverse=True); ng = g.max() + 1
        yw = majority(yy, g, ng); pw = majority(np.where(yy >= 0, pp, -1), g, ng)
        first = np.zeros(ng, int); first[g[::-1]] = np.arange(len(g))[::-1]
        # use label of the window's majority second
        for w in range(ng):
            if yw[w] < 0: continue
            secs = np.flatnonzero(g == w); nm = collections.Counter(lab[secs][yy[secs] == yw[w]]).most_common(1)[0][0]
            stats[(keys[i][1], nm)][1] += 1; stats[(keys[i][1], nm)][0] += int(pw[w] != yw[w])
        # onset error for each true S run: predicted S run start nearest
        ts = np.flatnonzero((yy[1:] == 1) & (yy[:-1] != 1)) + 1
        ps = np.flatnonzero((pp[1:] == 1) & (pp[:-1] != 1)) + 1
        for a in ts:
            if len(ps): onset[keys[i][1]].append(int(ps[np.argmin(np.abs(ps - a))] - a))
    for k in sorted(stats):
        w, n = stats[k]; print(f"{k[0]} {k[1]:4s} wrong {w:4d} / {n:5d}  ({w / n:.1%})")
    for v in ("V1", "V2"):
        o = np.array(onset[v]); print(v, "pred S onset - true onset (s): median", np.median(o), "MAE", np.abs(o).mean().round(1),
              "|err|>30s:", (np.abs(o) > 30).sum(), "/", len(o))
