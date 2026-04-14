from ultralytics import YOLO

model = YOLO("model/yolo26n-seg.pt")

result = model.predict("https://ultralytics.com/images/bus.jpg", save=True)

# show_result = result[0].show()

print("\n=== Bbox ===")
print("shape: " + str(result[0].boxes.shape )) # (num_boxes, 6) ( [x1, y1, x2, y2, (optional) track_id, confidence, class] )
print("data: " + str(result[0].boxes.data )) # (num_boxes, 6)
print("orig_shape: " + str(result[0].boxes.orig_shape )) # (height, width)
print("is_track: " + str(result[0].boxes.is_track )) # boolean
print("xyxy: " + str(result[0].boxes.xyxy )) # (num_boxes, 4)
print("conf: " + str(result[0].boxes.conf )) # (num_boxes,)
print("cls: " + str(result[0].boxes.cls ))
print("id: " + str(result[0].boxes.id if result[0].boxes is not None else "None"))

print("\n=== Instance Mask ===")
print("data: " + str(result[0].masks.data.shape )) # (num_masks, height, width)
print("orig_shape: " + str(result[0].masks.orig_shape )) # (height, width)

print(model.model.yaml)
#print(model.model.model[-1])
feature = {}

def hook_fn(name):
    def hook(m, i, o):
        feature[name] = o.shape
    return hook

# Register hooks for P3, P4, P5 (Layers 16, 19, 22)
model.model.model[16].register_forward_hook(hook_fn("P3"))
model.model.model[19].register_forward_hook(hook_fn("P4"))
model.model.model[22].register_forward_hook(hook_fn("P5"))

result = model.predict("bus.jpg")

print("\n=== Feature Map Shapes ===")
for name, shape in feature.items():
    print(f"{name}: {shape}")
