"""Train the u4so model (4 classes R/S/A/U + generic / Stress-onset / Stress-offset boundary heads) on ALL labelled
recordings with exactly the loss of nn_cv_acc.py, and export one ONNX file.
ONNX: input "x" (1, n_features, T) raw features -> output "prob" (1, 7, T) =
      softmax over [R, S, A, U] (4) + sigmoid of [generic, onset, offset] boundary logits (3).
Usage (torch env): python train_full_u4so.py --name v7_u4so --seed 0 --steps 3000
Writes runs/<name>/model.onnx, train_config.json, parity.npz (same layout as tools/nn_train_full.py)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn_cv_acc as N  # noqa: E402


class Export(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        o = self.net(x)
        return torch.cat([torch.softmax(o[:, :4], dim=1), torch.sigmoid(o[:, 4:7])], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=900)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--featdrop", type=float, default=0.1)
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--bnd_k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--dils", default="1,2,4,8,16,32,64")
    ap.add_argument("--bnd_weight", type=float, default=1.0)
    a = ap.parse_args()
    groups = N.FEATURE_GROUPS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs, y, sess, win, sid, nf = N.load("tcn", groups)
    allidx = list(range(len(seqs)))
    Xall = np.concatenate([s["x"].T for s in seqs])
    yall = np.concatenate([s["y"] for s in seqs])
    nC = 4
    cnt = np.bincount(np.where(yall >= 0, yall, 3), minlength=4)[:nC]
    w = torch.tensor(np.where(cnt > 0, cnt.sum() / (nC * np.maximum(cnt, 1)), 0.0), dtype=torch.float32, device=dev)
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    net = N.build("tcn", Xall, a.ch, a.drop, nout=nC + 3, pool=a.pool, dils=[int(v) for v in a.dils.split(",")]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    pw = torch.tensor(20.0, device=dev)
    t0 = time.time()
    net.train()
    for step in range(a.steps):
        items = [N.crop(seqs[i], a.crop, rng, "tcn") for i in rng.choice(allidx, a.batch)]
        b = N.batchify(items, "tcn", dev, rng, train=True)
        if a.featdrop > 0:
            keep = (torch.rand(b["x"].shape[0], b["x"].shape[1], 1, device=dev) > a.featdrop).float()
            b["x"] = b["x"] * keep + net.norm.med * (1 - keep)
        out = net(b["x"])
        yy = b["y"]
        ycls = torch.where(yy == -1, 3, yy)
        loss = F.cross_entropy(out[:, :nC], ycls, weight=w, ignore_index=-100)
        valid = (yy != -100)
        both = valid[:, 1:] & valid[:, :-1]
        events = ((0, yy[:, 1:] != yy[:, :-1]),
                  (1, (yy[:, 1:] == 1) & (yy[:, :-1] != 1)),
                  (2, (yy[:, 1:] != 1) & (yy[:, :-1] == 1)))
        for j, ev in events:
            e = torch.zeros_like(yy, dtype=torch.float32)
            e[:, 1:] = (ev & both).float()
            tj = F.max_pool1d(e[:, None], 2 * a.bnd_k + 1, stride=1, padding=a.bnd_k)[:, 0]
            bj = F.binary_cross_entropy_with_logits(out[:, nC + j], tj, pos_weight=pw, reduction="none")
            loss = loss + a.bnd_weight * (bj * valid).sum() / valid.sum()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1} loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
    exp = Export(net.eval().cpu()).eval()
    out_dir = os.path.join(N.ROOT, "runs", a.name)
    os.makedirs(out_dir, exist_ok=True)
    x0 = torch.tensor(seqs[0]["x"][None, :, :600])
    torch.onnx.export(exp, (x0,), os.path.join(out_dir, "model.onnx"), input_names=["x"], output_names=["prob"],
                      dynamic_axes={"x": {2: "t"}, "prob": {2: "t"}}, opset_version=17, dynamo=False)
    k = int(np.argmax([s["y"].shape[0] for s in seqs]))
    with torch.no_grad():
        p = exp(torch.tensor(seqs[k]["x"][None])).numpy()
    np.savez_compressed(os.path.join(out_dir, "parity.npz"), x=seqs[k]["x"][None], prob=p, session=k)
    cfg = dict(vars(a), model="tcn", u_class=1, bnd=3, n_features=nf, feature_groups=list(groups),
               params=sum(q.numel() for q in exp.parameters()), train_seconds=int((yall >= 0).sum()))
    json.dump(cfg, open(os.path.join(out_dir, "train_config.json"), "w"), indent=1)
    print(f"exported {out_dir}/model.onnx  params {cfg['params']}  output (1, 7, T)", flush=True)


if __name__ == "__main__":
    main()
