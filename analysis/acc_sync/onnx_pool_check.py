"""Export a randomly initialised pooled FeatTCN with a dynamic length and save torch outputs for odd lengths."""
import numpy as np, torch, nn_cv_acc as N
torch.manual_seed(0)
X = np.random.RandomState(0).randn(5000, 175).astype(np.float32)
for pool, dils in ((5, "1,2,4,8,16"), (10, "1,2,4,8")):
    net = N.build("tcn", X, 48, 0.2, nout=7, pool=pool, dils=[int(v) for v in dils.split(",")]).eval()
    torch.onnx.export(net, (torch.tensor(X.T[None, :, :600]),), f"oof/pool{pool}_check.onnx", input_names=["x"],
                      output_names=["prob"], dynamic_axes={"x": {2: "t"}, "prob": {2: "t"}}, opset_version=17, dynamo=False)
    out = {}
    for T in (1001, 2003, 3146):
        with torch.no_grad():
            out[f"x{T}"] = X.T[None, :, :T]; out[f"y{T}"] = net(torch.tensor(out[f"x{T}"])).numpy()
    np.savez(f"oof/pool{pool}_check.npz", **out)
    print("exported pool", pool, flush=True)
