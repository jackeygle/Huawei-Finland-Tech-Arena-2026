"""Train sequence models on ALL labelled recordings and export one ONNX file (torch env).

Several seeds are wrapped into a single module that averages softmax probabilities, because the
submission may contain only one .onnx file. Input "x": (1, n_features, T) raw features, output
"prob": (1, 3, T).
Usage: python nn_train_full.py --name v2_tcn --seeds 0 [--ch 48 --drop 0.2 ...]
Writes runs/<name>/model.onnx, train_config.json and parity.npz (features + torch probabilities of one
recording, used by tools/package_nn.py to check the ONNX file in the evaluation environment).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn_cv as N  # noqa: E402


class Ensemble(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        p = None
        for net in self.nets:
            q = torch.softmax(net(x), dim=1)
            p = q if p is None else p + q
        return p / len(self.nets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="tcn", choices=["tcn", "trans"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=900)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--featdrop", type=float, default=0.1)
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--groups", default=",".join(N.FEATURE_GROUPS))
    a = ap.parse_args()
    groups = tuple(a.groups.split(","))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs, y, sess, win, sid, nf = N.load(a.model, groups)
    allidx = list(range(len(seqs)))
    Xall = np.concatenate([s["x"].T for s in seqs])
    yall = np.concatenate([s["y"] for s in seqs])
    cnt = np.bincount(yall[yall >= 0], minlength=3)
    w = torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float32, device=dev)
    nets = []
    t0 = time.time()
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        net = N.build(a.model, Xall, a.ch, a.drop).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
        net.train()
        for step in range(a.steps):
            items = [N.crop(seqs[i], a.crop, rng, a.model) for i in rng.choice(allidx, a.batch)]
            b = N.batchify(items, a.model, dev, rng, train=True)
            if a.featdrop > 0:
                keep = (torch.rand(b["x"].shape[0], b["x"].shape[1], 1, device=dev) > a.featdrop).float()
                b["x"] = b["x"] * keep + net.norm.med * (1 - keep)
            loss = F.cross_entropy(net(b["x"]), b["y"], weight=w, ignore_index=-1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
        print(f"seed {seed}: final loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
        nets.append(net.eval().cpu())
    ens = Ensemble(nets).eval()

    out = os.path.join(N.ROOT, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    x0 = torch.tensor(seqs[0]["x"][None, :, :600])
    torch.onnx.export(ens, (x0,), os.path.join(out, "model.onnx"), input_names=["x"], output_names=["prob"],
                      dynamic_axes={"x": {2: "t"}, "prob": {2: "t"}}, opset_version=17, dynamo=False)
    k = int(np.argmax([s["y"].shape[0] for s in seqs]))                 # longest recording for parity
    with torch.no_grad():
        p = ens(torch.tensor(seqs[k]["x"][None])).numpy()
    np.savez_compressed(os.path.join(out, "parity.npz"), x=seqs[k]["x"][None], prob=p, session=k)
    cfg = dict(vars(a), n_features=nf, feature_groups=list(groups),
               params=sum(q.numel() for q in ens.parameters()), train_seconds=int((yall >= 0).sum()))
    json.dump(cfg, open(os.path.join(out, "train_config.json"), "w"), indent=1)
    print(f"exported {out}/model.onnx  params {cfg['params']}", flush=True)


if __name__ == "__main__":
    main()
