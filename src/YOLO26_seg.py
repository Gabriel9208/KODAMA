from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

result = model.predict("https://ultralytics.com/images/bus.jpg", save=True)

show_result = result[0].show()

print("\n=== Bbox and Mask shape ===")
print("Bbox shape: " + str(result[0].boxes.shape if result[0].boxes is not None else "None"))
print("Mask shape: " + str(result[0].masks.shape if result[0].masks is not None else "None"))


print("\n=== Boxes Data ===")
print("First 2 rows of Boxes.data: ", result[0].boxes.data[:2])
print("Class id for each box: ", result[0].boxes.cls)

print("\n=== Masks Data ===")
print("Mask value range: ", result[0].masks.data.min().item(), result[0].masks.data.max().item())

# 3. ★ 最重要：prototype mask 在哪裡？
#    YOLO seg 的運作方式：
#    最終 mask = mask_coefficients (32維) @ prototypes (32, H, W)
print("\n=== prototypes (內部) ===")

# 用 hook 拿到 prototype tensor
protos = None
def hook_fn(module, input, output):
    global protos
    protos = output

# model.model 是 nn.Module，最後一個 segment head 會輸出 proto
handle = model.model.model[-1].proto.register_forward_hook(hook_fn)
model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)
handle.remove()

if protos is not None:
    print("proto shape:", protos.shape)
    # 預期：torch.Size([1, 32, 160, 120]) 或類似
