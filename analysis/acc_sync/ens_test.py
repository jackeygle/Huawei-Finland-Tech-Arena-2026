import sys
sys.path.insert(0, "."); sys.path.insert(0, "../analysis/acc_sync")
from cens_test import run
for tag in sys.argv[1:]:
    out = []
    for kind in ("subject", "session"):
        r, hit, rx = run(tag, kind, 4.0)
        out.append(f"{r[0]:.4f} F1 {r[1].round(3)}")
    print(f"{tag:70s} subject {out[0]} | session {out[1]}", flush=True)
