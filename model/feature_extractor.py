from ultralytics import YOLO

feature = {}

def hook_fn(name):
    def hook(module, input, output):
        feature[name] = output
    return hook

model = YOLO("model/yolo26n-seg.pt")
for i, m in enumerate(model.model.model):
    print(f"Layer {i}: {type(m).__name__}")

# Layer 0: Conv
# Layer 1: Conv
# Layer 2: C3k2
# Layer 3: Conv
# Layer 4: C3k2
# Layer 5: Conv
# Layer 6: C3k2
# Layer 7: Conv
# Layer 8: C3k2
# Layer 9: SPPF
# Layer 10: C2PSA
# Layer 11: Upsample
# Layer 12: Concat
# Layer 13: C3k2
# Layer 14: Upsample
# Layer 15: Concat
# Layer 16: C3k2
# Layer 17: Conv
# Layer 18: Concat
# Layer 19: C3k2
# Layer 20: Conv
# Layer 21: Concat
# Layer 22: C3k2
# Layer 23: Segment26

for idx in [16, 19, 22]:
    model.model.model[idx].register_forward_hook(hook_fn(f"Layer{idx}"))

model.predict("https://ultralytics.com/images/bus.jpg", save=True)

for f in feature:
    print(f, feature[f].shape)

# Layer16 torch.Size([1, 64, 80, 60]) -> P3
# Layer19 torch.Size([1, 128, 40, 30]) -> P4
# Layer22 torch.Size([1, 256, 20, 15]) -> P5