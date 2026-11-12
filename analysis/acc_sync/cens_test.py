import sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "../analysis/acc_sync")
from scoring import fold_of_session, window_score
from template_decode import build_templates
from adm_model import _key_parts
from decode_variants import sid, prot, keys
from refine import fill_neighbour
from align2 import align2

def run(tagset, kind, cap, alpha=0.002, res=5, skip_pen=50.0, version_mode="date"):
    Z = [np.load(f"../runs/{t}_{kind}.npz") for t in tagset.split("+")]
    prob = np.mean([z["prob"] for z in Z], axis=0); y, sess, win = Z[0]["y"], Z[0]["sess"], Z[0]["win"]
    fos = fold_of_session(kind, sid); pred = prob.argmax(1).copy(); hit = 0
    for k in range(6):
        tr = np.flatnonzero(fos != k)
        T = build_templates([y[sess == i] for i in tr], [keys[i] for i in tr])
        for i in np.flatnonzero(fos == k):
            m = sess == i; lp = np.log(np.clip(prob[m], 1e-6, 1)) + np.log(3.0); best = None
            for key, segs in T.items():
                if version_mode == "date" and _key_parts(key)[1] != keys[i][1]: continue
                sc, seg = align2(lp, segs, alpha, res, skip_pen, cap)
                if best is None or sc > best[0]: best = (sc, seg, key, segs)
            _, seg, key, segs = best; hit += key == keys[i]
            cls = np.array([g["c"] for g in segs]); c_of = np.where(seg >= 0, cls[np.maximum(seg, 0)], -1)
            pred[m] = fill_neighbour(c_of, prob[m])
    r = window_score(y, pred, sess, win)
    st = np.isin(sess, np.flatnonzero(prot == "STRESS"))
    rx = window_score(y[~st], pred[~st], sess[~st], win[~st])
    return r, hit, rx

if __name__ == "__main__":
  tag = sys.argv[1]; vm = sys.argv[2] if len(sys.argv) > 2 else "date"
  for kind in ("subject", "session"):
    for cap in (0.0, 1.0, 2.0, 4.0, 8.0, 1e9):
        r, hit, rx = run(tag, kind, cap, version_mode=vm)
        print(f"[{kind}] {vm} cap={cap:<6g} macro-F1 {r[0]:.4f} F1 {r[1].round(3)} tmpl {hit}/85 | exercise-only F1 R/A {rx[1][[0,2]].round(3)}", flush=True)
