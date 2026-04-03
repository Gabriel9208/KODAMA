from ultralytics import YOLO

def hook_fn(name):
    def hook(model, input, output):
        feature[name] = output
    return hook

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
