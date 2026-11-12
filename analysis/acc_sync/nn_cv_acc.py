"""Neural-network cross-validation on whole recordings, same folds and scoring as tools/cv.py.

Models (--model):
  tcn   : scheme A, dilated residual 1D-CNN over the 1 Hz feature sequence (receptive field ~ +-2 min)
  trans : scheme B, small transformer over the 1 Hz features pooled to 0.2 Hz; position enters only
          through a depthwise convolution (relative), never as absolute time
  raw   : scheme C, multi-branch 1D-CNN on raw waveforms (BVP 16 Hz, EDA/TEMP 4 Hz, HR/IBI 1 Hz) + TCN head

Every model maps a whole recording (B, C, T) to per-second logits (B, 3, T), so inference is one forward pass.
Out-of-fold probabilities are saved in the same format as tools/cv.py (prob, y, sess, win).
Usage: python nn_cv.py --model tcn --split subject,session --steps 1500
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACC = os.path.dirname(os.path.abspath(__file__))           # analysis/acc_sync (copy of tools/nn_cv.py)
ROOT = os.path.dirname(os.path.dirname(ACC))
HERE = os.path.join(ROOT, "tools")
OUT = os.path.join(ACC, "oof")
sys.path.insert(0, HERE)
from scoring import fold_of_session, window_score  # noqa: E402

FEATURE_GROUPS = ("local", "rel", "sess", "eda2", "hrv", "spec")


def group_of(name):
    """Mirror of submission/adm_model.group_of (that module needs onnxruntime, absent here)."""
    if name.startswith("resp_"): return "resp"
    if name.startswith("pulse_"): return "pulse"
    if name.startswith("rec_"): return "recov"
    if name.startswith("scr_"): return "eda2"
    if name.startswith("spec_"): return "spec"
    if name.startswith("hrv_"): return "hrv"
    if name.startswith("pos_"): return "pos"
    if name.endswith(("_sessrange", "_sesshi", "_relrange")): return "sess2"
    if name.endswith(("_rel", "_z", "_rank")): return "rel"
    if name.endswith(("_sessmed", "_sessstd")): return "sess"
    return "local"


# ------------------------------------------------------------------------------------------ models
class ResBlock(nn.Module):
    def __init__(self, ch, dil, drop):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.n1 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.c2(self.drop(F.gelu(self.n1(self.c1(x)))))


class TCN(nn.Module):
    def __init__(self, cin, ch=48, dils=(1, 2, 4, 8, 16, 32, 64), drop=0.2, nout=3):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 1)
        self.blocks = nn.Sequential(*[ResBlock(ch, d, drop) for d in dils])
        self.out = nn.Conv1d(ch, nout, 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


class Normalise(nn.Module):
    """Per-feature robust scaling stored inside the model, so the ONNX file takes raw features."""
    def __init__(self, med, iqr):
        super().__init__()
        self.register_buffer("med", torch.tensor(med, dtype=torch.float32)[None, :, None])
        self.register_buffer("iqr", torch.tensor(iqr, dtype=torch.float32)[None, :, None])

    def forward(self, x):
        return torch.clamp((x - self.med) / self.iqr, -10, 10)


class FeatTCN(nn.Module):
    """pool > 1: average the 1 Hz features over pool-second blocks, run the TCN at the block rate and repeat the
    outputs back to 1 Hz (inside the model, so the ONNX file still maps (1, F, T) -> (1, C, T))."""
    def __init__(self, med, iqr, pool=1, **kw):
        super().__init__()
        self.norm = Normalise(med, iqr)
        self.net = TCN(len(med), **kw)
        self.pool = pool

    def forward(self, x):
        h = self.norm(x)
        if self.pool == 1:
            return self.net(h)
        T = h.shape[-1]
        pad = (-T) % self.pool
        h = F.avg_pool1d(F.pad(h, (0, pad), mode="replicate"), self.pool)
        o = self.net(h)
        return torch.repeat_interleave(o, self.pool, dim=2)[..., :T]


class EncoderLayer(nn.Module):
    """Pre-norm transformer layer written with shape-agnostic reshapes, so the ONNX export keeps a
    dynamic sequence length (nn.MultiheadAttention bakes the export-time length into Reshape nodes)."""
    def __init__(self, d, heads, drop):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):                                       # x: (B, S, d)
        B = x.shape[0]
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.reshape(B, -1, self.h, self.dh).transpose(1, 2)    # (B, h, S, dh)
        k = k.reshape(B, -1, self.h, self.dh).transpose(1, 2)
        v = v.reshape(B, -1, self.h, self.dh).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / self.dh ** 0.5, dim=-1)
        o = (self.drop(att) @ v).transpose(1, 2).reshape(B, -1, self.h * self.dh)
        x = x + self.drop(self.proj(o))
        return x + self.drop(self.ff(self.n2(x)))


class FeatTransformer(nn.Module):
    def __init__(self, med, iqr, d=48, heads=4, layers=2, pool=5, drop=0.2):
        super().__init__()
        self.norm = Normalise(med, iqr)
        self.pool = pool
        self.inp = nn.Conv1d(len(med), d, 1)
        self.local = nn.Sequential(ResBlock(d, 1, drop), ResBlock(d, 2, drop))     # 1 Hz detail
        self.cpe = nn.Conv1d(d, d, 7, padding=3, groups=d)                         # relative position
        self.enc = nn.Sequential(*[EncoderLayer(d, heads, drop) for _ in range(layers)])
        self.out = nn.Conv1d(2 * d, 3, 1)

    def forward(self, x):
        h = self.local(self.inp(self.norm(x)))                  # (B, d, T)
        T = h.shape[-1]
        pad = (-T) % self.pool
        hp = F.avg_pool1d(F.pad(h, (0, pad), mode="replicate"), self.pool)
        hp = hp + self.cpe(hp)
        g = self.enc(hp.transpose(1, 2)).transpose(1, 2)
        g = torch.repeat_interleave(g, self.pool, dim=2)[..., :T]
        return self.out(torch.cat([h, g], dim=1))


class RawNet(nn.Module):
    def __init__(self, ch=48, drop=0.2):
        super().__init__()
        def down(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, stride=2, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU())
        self.bvp = nn.Sequential(down(1, 16, 7), down(16, 24, 5), down(24, 32, 5), down(32, 32, 3))   # 16 -> 1 Hz
        self.slow = nn.Sequential(down(2, 16, 5), down(16, 16, 3))                                   # 4 -> 1 Hz
        self.head = TCN(32 + 16 + 4, ch=ch, drop=drop)

    def forward(self, bvp, slow, fast):
        T = fast.shape[-1]
        a = self.bvp(bvp)[..., :T]
        b = self.slow(slow)[..., :T]
        return self.head(torch.cat([a, b, fast], dim=1))


# ------------------------------------------------------------------------------------------ data
def load(model, groups=FEATURE_GROUPS):
    d = np.load(os.path.join(ROOT, "runs", "dataset_v4.npz"), allow_pickle=True)
    names = list(d["names"])
    keep = np.array([group_of(n) in groups for n in names])
    X, y, sess, win, sid = d["X"][:, keep], d["y"], d["sess"], d["win"], d["sid"]
    n_sess = len(sid)
    seqs = []
    raw = np.load(os.path.join(ROOT, "runs", "raw_v1.npz")) if model == "raw" else None
    for k in range(n_sess):
        m = sess == k
        item = {"x": X[m].T.copy(), "y": y[m].copy()}
        if raw is not None:
            item.update(bvp=raw[f"bvp_{k}"][None], slow=raw[f"slow_{k}"], fast=raw[f"fast_{k}"])
        seqs.append(item)
    for k in range(n_sess):
        seqs[k]["prot"], seqs[k]["sid"] = str(d["prot"][k]), str(sid[k])
    return seqs, y, sess, win, sid, int(keep.sum())


PROTS = ("STRESS", "AEROBIC", "ANAEROBIC")


def crop(item, L, rng, model):
    T = item["y"].shape[0]
    if T <= L:
        s = 0; L = T
    else:
        s = rng.randint(0, T - L + 1)
    out = {"x": item["x"][:, s:s + L], "y": item.get("ytrain", item["y"])[s:s + L]}
    if "prot" in item:
        out["pid"] = PROTS.index(item["prot"])
    if model == "raw":
        out.update(bvp=item["bvp"][:, 16 * s:16 * (s + L)], slow=item["slow"][:, 4 * s:4 * (s + L)],
                   fast=item["fast"][:, s:s + L])
    return out


def batchify(items, model, dev, rng, train):
    L = max(i["y"].shape[0] for i in items)
    def pad(a, n):  # pad last axis by edge values
        return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])], mode="edge")
    y = np.stack([np.pad(i["y"], (0, L - len(i["y"])), constant_values=-100) for i in items])
    b = {"y": torch.tensor(y, dtype=torch.long, device=dev)}
    if model == "raw":
        for k, f in (("bvp", 16), ("slow", 4), ("fast", 1)):
            arr = np.stack([pad(i[k], f * L) for i in items]).astype(np.float32)
            if train and k != "fast":
                arr = arr * rng.uniform(0.7, 1.3, size=(len(items), arr.shape[1], 1)).astype(np.float32)
            b[k] = torch.tensor(arr, device=dev)
    else:
        x = np.stack([pad(i["x"], L) for i in items]).astype(np.float32)
        b["x"] = torch.tensor(x, device=dev)
    if all("pid" in i for i in items):
        b["pid"] = torch.tensor([i["pid"] for i in items], dtype=torch.long, device=dev)
    return b


def forward(net, b, model):
    return net(b["bvp"], b["slow"], b["fast"]) if model == "raw" else net(b["x"])


def build(model, Xtr, ch=48, drop=0.2, nout=3, pool=1, dils=(1, 2, 4, 8, 16, 32, 64)):
    if model == "raw":
        return RawNet(ch=ch, drop=drop)
    med = np.median(Xtr, axis=0)
    q1, q3 = np.percentile(Xtr, [25, 75], axis=0)
    iqr = np.maximum(q3 - q1, 1e-3)
    return FeatTCN(med, iqr, pool=pool, ch=ch, drop=drop, nout=nout, dils=tuple(dils)) if model == "tcn" else FeatTransformer(med, iqr, d=ch, drop=drop)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tcn", "trans", "raw"])
    ap.add_argument("--split", default="subject,session")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0, help="model init / sampling seed")
    ap.add_argument("--fold_seed", type=int, default=0, help="fold assignment seed (keep 0 to compare runs)")
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--proto", type=float, default=0.0, help=">0: per-second protocol head (STRESS/AEROBIC/ANAEROBIC), loss weight")
    ap.add_argument("--pool", type=int, default=1, help="run the TCN on pool-second blocks (speed)")
    ap.add_argument("--dils", default="1,2,4,8,16,32,64", help="TCN dilations (one residual block each)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=900)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--featdrop", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--u_class", type=int, default=0, help="1: unscored seconds are a 4th class")
    ap.add_argument("--u_weight", type=float, default=1.0, help="loss weight multiplier of the U class")
    ap.add_argument("--bnd", type=int, default=0, help="1: extra boundary head (label change within +-bnd_k s); 3: + Stress-onset and Stress-offset heads")
    ap.add_argument("--bnd_k", type=int, default=3)
    ap.add_argument("--bnd_weight", type=float, default=1.0)
    ap.add_argument("--groups", default=",".join(FEATURE_GROUPS))
    ap.add_argument("--eval_every", type=int, default=0, help="learning curve: score train/val every k steps")
    ap.add_argument("--exclude_train", default="", help="comma list PROT/SID removed from training folds only")
    ap.add_argument("--boundary_ignore", type=int, default=0, help="ignore labels within k s of a label change (training only)")
    ap.add_argument("--train_prot", default="", help="comma list of protocols used for training (e.g. STRESS); test folds unchanged")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    seqs, y, sess, win, sid, nf = load(a.model, tuple(a.groups.split(",")))
    excl = set(a.exclude_train.split(",")) - {""}
    if a.boundary_ignore > 0:                       # training copy of labels with transition zones masked
        for it in seqs:
            yy = it["y"]; ch = np.flatnonzero((yy[1:] != yy[:-1]) & (yy[1:] >= 0) & (yy[:-1] >= 0)) + 1
            yt = yy.copy()
            for c in ch:
                yt[max(c - a.boundary_ignore, 0):c + a.boundary_ignore] = -1
            it["ytrain"] = yt
    tag = a.tag or a.model
    print(f"model={a.model} features={nf} sessions={len(seqs)} device={dev}", flush=True)
    summary = {}
    for kind in a.split.split(","):
        fos = fold_of_session(kind, sid, a.folds, a.fold_seed)
        nC = 4 if a.u_class else 3
        nP = 3 if a.proto > 0 else 0
        prob = np.zeros((len(y), nC + a.bnd + nP), dtype=np.float32)
        n_ck = a.steps // a.eval_every if a.eval_every else 0
        curve_val = [np.zeros((len(y), 3), dtype=np.float32) for _ in range(n_ck)]
        curve_tr = [[{} for _ in range(a.folds)] for _ in range(n_ck)]
        t0 = time.time()
        for k in range(a.folds):
            tr = [i for i in range(len(seqs)) if fos[i] != k and f"{seqs[i]['prot']}/{seqs[i]['sid']}" not in excl
                  and (not a.train_prot or seqs[i]["prot"] in a.train_prot.split(","))]
            te = [i for i in range(len(seqs)) if fos[i] == k]
            rng = np.random.RandomState(a.seed * 100 + k)
            Xtr = np.concatenate([seqs[i]["x"].T for i in tr])
            torch.manual_seed(a.seed * 100 + k)
            net = build(a.model, Xtr, a.ch, a.drop, nout=nC + a.bnd + nP, pool=a.pool,
                        dils=[int(v) for v in a.dils.split(",")]).to(dev)
            ytr = np.concatenate([seqs[i]["y"] for i in tr])
            curve_steps = list(range(a.eval_every, a.steps + 1, a.eval_every)) if a.eval_every else []
            cnt = np.bincount(np.where(ytr >= 0, ytr, 3), minlength=4)[:nC]
            wn = np.where(cnt > 0, cnt.sum() / (nC * np.maximum(cnt, 1)), 0.0)
            if a.u_class:
                wn[3] *= a.u_weight
            w = torch.tensor(wn, dtype=torch.float32, device=dev)
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
            net.train()
            for step in range(a.steps):
                items = [crop(seqs[i], a.crop, rng, a.model) for i in rng.choice(tr, a.batch)]
                b = batchify(items, a.model, dev, rng, train=True)
                if a.model != "raw" and a.featdrop > 0:           # drop whole feature channels
                    keep = (torch.rand(b["x"].shape[0], b["x"].shape[1], 1, device=dev) > a.featdrop).float()
                    med = net.norm.med
                    b["x"] = b["x"] * keep + med * (1 - keep)
                out = forward(net, b, a.model)
                yy = b["y"]
                ycls = torch.where(yy == -1, 3, yy) if a.u_class else torch.where(yy == -1, -100, yy)
                loss = F.cross_entropy(out[:, :nC], ycls, weight=w, ignore_index=-100)
                if a.bnd:
                    valid = (yy != -100)
                    ch = torch.zeros_like(yy, dtype=torch.float32)
                    ch[:, 1:] = ((yy[:, 1:] != yy[:, :-1]) & valid[:, 1:] & valid[:, :-1]).float()
                    tgt = F.max_pool1d(ch[:, None], 2 * a.bnd_k + 1, stride=1, padding=a.bnd_k)[:, 0]
                    pw = torch.tensor(20.0, device=dev)
                    bl = F.binary_cross_entropy_with_logits(out[:, nC], tgt, pos_weight=pw, reduction="none")
                    loss = loss + a.bnd_weight * (bl * valid).sum() / valid.sum()
                    if a.bnd == 3:                                  # Stress onset / offset heads
                        both = valid[:, 1:] & valid[:, :-1]
                        for j, ev in ((1, (yy[:, 1:] == 1) & (yy[:, :-1] != 1)), (2, (yy[:, 1:] != 1) & (yy[:, :-1] == 1))):
                            e = torch.zeros_like(yy, dtype=torch.float32)
                            e[:, 1:] = (ev & both).float()
                            tj = F.max_pool1d(e[:, None], 2 * a.bnd_k + 1, stride=1, padding=a.bnd_k)[:, 0]
                            bj = F.binary_cross_entropy_with_logits(out[:, nC + j], tj, pos_weight=pw, reduction="none")
                            loss = loss + a.bnd_weight * (bj * valid).sum() / valid.sum()
                if a.proto > 0:                                     # per-second protocol head (session label)
                    tp = torch.where(yy != -100, b["pid"][:, None].expand_as(yy), torch.full_like(yy, -100))
                    loss = loss + a.proto * F.cross_entropy(out[:, nC + a.bnd:nC + a.bnd + 3], tp, ignore_index=-100)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
                if (step + 1) in curve_steps:
                    ci = curve_steps.index(step + 1)
                    net.eval()
                    with torch.no_grad():
                        for i in te:
                            bb = batchify([seqs[i]], a.model, dev, rng, train=False)
                            pp = torch.softmax(forward(net, bb, a.model)[:, :3], dim=1)[0].T.cpu().numpy()
                            curve_val[ci][sess == i] = pp[: (sess == i).sum()]
                        for i in tr[::3]:                       # every 3rd training recording
                            bb = batchify([seqs[i]], a.model, dev, rng, train=False)
                            pp = torch.softmax(forward(net, bb, a.model)[:, :3], dim=1)[0].T.cpu().numpy()
                            curve_tr[ci][k][i] = pp[: (sess == i).sum()].argmax(1)
                    net.train()
            net.eval()
            with torch.no_grad():
                for i in te:
                    b = batchify([seqs[i]], a.model, dev, rng, train=False)
                    o = forward(net, b, a.model)
                    pc = torch.softmax(o[:, :nC], dim=1)
                    if a.bnd:
                        pc = torch.cat([pc, torch.sigmoid(o[:, nC:nC + a.bnd])], dim=1)
                    if nP:
                        pc = torch.cat([pc, torch.softmax(o[:, nC + a.bnd:nC + a.bnd + 3], dim=1)], dim=1)
                    p = pc[0].T.cpu().numpy()
                    prob[sess == i] = p[: (sess == i).sum()]
            print(f"  [{kind}] fold {k} loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
        for ci in range(n_ck):
            vf = window_score(y, curve_val[ci].argmax(1), sess, win)[0]
            tf = []
            for k in range(a.folds):
                ids = sorted(curve_tr[ci][k])
                if ids:
                    m = np.concatenate([np.flatnonzero(sess == i) for i in ids])
                    pr = np.concatenate([curve_tr[ci][k][i] for i in ids])
                    tf.append(window_score(y[m], pr, sess[m], win[m])[0])
            print(f"  curve {kind} step {(ci + 1) * a.eval_every:5d}  train macro-F1 {np.mean(tf):.4f}  "
                  f"val macro-F1 {vf:.4f}", flush=True)
        mf, f, cm = window_score(y, prob[:, :3].argmax(1), sess, win)
        print(f"{tag} {kind:8s} macro-F1={mf:.4f} quality={mf * 80:.1f}/80 F1 Rest/Stress/Act={f.round(3)}", flush=True)
        print("  confusion (rows = true Rest/Stress/Activity):\n" + str(cm), flush=True)
        np.savez_compressed(os.path.join(OUT, f"nn_{tag}_{kind}.npz"), prob=prob, y=y, sess=sess, win=win)
        summary[kind] = dict(macro_f1=float(mf), f1=f.tolist())
    n_par = sum(p.numel() for p in net.parameters())
    summary.update(params=n_par, args=vars(a))
    json.dump(summary, open(os.path.join(OUT, f"nn_{tag}_summary.json"), "w"), indent=1)
    print(f"params {n_par}", flush=True)


if __name__ == "__main__":
    main()
