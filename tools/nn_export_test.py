"""Export untrained instances of the three networks to ONNX with a dynamic time axis (speed test only)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn_cv as N
out = os.path.join(N.ROOT, "runs", "onnx_speed")
T = 600
nf = 175
med, iqr = np.zeros(nf), np.ones(nf)
for name, net, args, names in [
    ("tcn", N.FeatTCN(med, iqr), (torch.randn(1, nf, T),), ["x"]),
    ("trans", N.FeatTransformer(med, iqr), (torch.randn(1, nf, T),), ["x"]),
    ("raw", N.RawNet(), (torch.randn(1, 1, 16 * T), torch.randn(1, 2, 4 * T), torch.randn(1, 4, T)), ["bvp", "slow", "fast"]),
]:
    net.eval()
    dyn = {k: {2: "t"} for k in names}
    dyn["logits"] = {2: "t"}
    torch.onnx.export(net, args, os.path.join(out, f"{name}.onnx"), input_names=names, output_names=["logits"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    print(name, "exported", sum(p.numel() for p in net.parameters()), "params")
