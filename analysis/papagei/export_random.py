"""Build PaPaGei-S (ResNet1DMoE, paper configuration) with RANDOM weights, export the embedding branch to ONNX.
Only for size / speed analysis: speed does not depend on weight values."""
import importlib.util, sys, torch, torch.nn as nn
REPO = "/scratch/work/zhangx29/huawei_tech_arena_2026/external/papagei/repo/models/resnet.py"
spec = importlib.util.spec_from_file_location("papagei_resnet", REPO); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
m = R.ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18, n_classes=512, n_experts=3).eval()
tot = sum(p.numel() for p in m.parameters())
enc = sum(p.numel() for n_, p in m.named_parameters() if not n_.startswith(("expert_layers", "gating")))
print(f"PaPaGei-S params: total {tot/1e6:.2f} M, encoder+dense (used for embeddings) {enc/1e6:.2f} M")


class Emb(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)[0]          # 512-d embedding (dense output)


x = torch.randn(4, 1, 1250)
torch.onnx.export(Emb(m), (x,), "papagei_s_random.onnx", input_names=["x"], output_names=["emb"],
                  dynamic_axes={"x": {0: "b"}, "emb": {0: "b"}}, opset_version=17, dynamo=False)
print("exported, output", Emb(m)(x).shape)
