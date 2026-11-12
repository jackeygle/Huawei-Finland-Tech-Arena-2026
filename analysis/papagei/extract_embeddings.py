"""PaPaGei-S embeddings for every recording (offline analysis only, not used in the submission).
Preprocessing follows the PaPaGei repo (preprocessing/ppg.py via pyPPG): Chebyshev-II band-pass 0.5-12 Hz, order 4,
zero-phase; 50 ms moving-average smoothing; 10 s segments; per-segment z-score; resample 64 -> 125 Hz (1250 samples).
pyPPG is not installed, so the filter is reimplemented with scipy using the same parameters (stopband 20 dB assumed).
Segments start every HOP seconds; the embedding is assigned to the segment's centre second.
Usage: python extract_embeddings.py <weights.pt | RANDOM> <out.npz> [hop] [max_sessions]"""
import glob, importlib.util, os, sys, time
import numpy as np, pandas as pd, torch
from scipy.signal import cheby2, sosfiltfilt, resample_poly
from scipy.ndimage import uniform_filter1d

ROOT = "/scratch/work/zhangx29/huawei_tech_arena_2026"
REPO = os.path.join(ROOT, "external/papagei/repo/models/resnet.py")
spec = importlib.util.spec_from_file_location("papagei_resnet", REPO); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
wpath, out = sys.argv[1], sys.argv[2]
HOP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
MAXS = int(sys.argv[4]) if len(sys.argv) > 4 else 0
FS, FT, SEG = 64, 125, 10
SIGN = float(os.environ.get("PPG_SIGN", "1"))          # -1: invert polarity (Empatica BVP vs clinical PPG)
model = R.ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18, n_classes=512, n_experts=3)
if wpath != "RANDOM":
    sd = torch.load(wpath, map_location="cpu", weights_only=True)        # data only, no pickled code
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("loaded", wpath, "missing", missing, "unexpected", unexpected, flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev).eval()
SOS = cheby2(4, 20, [0.5, 12], btype="bandpass", fs=FS, output="sos")
files = sorted(glob.glob(os.path.join(ROOT, "downloads/data/*/*.parquet")))
if MAXS: files = files[:MAXS]
res = {}
t0 = time.time()
for k, fn in enumerate(files):
    df = pd.read_parquet(fn, columns=["BVP"])
    t = df.index.values.astype("datetime64[ns]").astype(np.int64); sec = (t - t[0]) // 1_000_000_000
    n = int(sec.max()) + 1
    x = SIGN * np.nan_to_num(df["BVP"].to_numpy(np.float64))
    x = uniform_filter1d(sosfiltfilt(SOS, x), size=max(1, int(round(0.05 * FS))))
    first = np.searchsorted(sec, np.arange(n))                           # first row of each second
    starts = np.arange(0, max(n - SEG, 0) + 1, HOP)
    segs = []
    for s0 in starts:
        a = first[s0]; seg = x[a:a + SEG * FS]
        if len(seg) < SEG * FS: seg = np.pad(seg, (0, SEG * FS - len(seg)), mode="edge")
        segs.append(seg)
    S = np.array(segs)
    S = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)
    S = resample_poly(S, FT, FS, axis=1).astype(np.float32)               # 640 -> 1250 samples
    embs = []
    with torch.no_grad():
        for b in range(0, len(S), 256):
            embs.append(model(torch.from_numpy(S[b:b + 256, None]).to(dev))[0].cpu().numpy())
    res[f"emb_{k}"] = np.concatenate(embs).astype(np.float32)
    res[f"centre_{k}"] = (starts + SEG // 2).astype(np.int64)
    if k % 10 == 0: print(f"  {k}/{len(files)} {os.path.basename(os.path.dirname(fn))}/{os.path.basename(fn)}: {len(S)} segments  {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(out, **res)
print("saved", out, len(files), "recordings", f"{time.time() - t0:.0f}s", flush=True)
