# Return type of `model("yolo26n-seg.pt")`: list[ultralytics.engine.results.Results]

## Attributes

| Name       | Type            | Description                                     |
| ---------- | --------------- | ----------------------------------------------- |
| orig_img   | np.ndarray      | The original image as a numpy array.            |
| orig_shape | tuple[int, int] | Original image shape in (height, width) format. |
| **boxes**  | Boxes           | None \| Detected bounding boxes.                |
| **masks**  | Masks           | None \| Segmentation masks.                     |

## ultralytics.engine.results.Results.update.boxes

### Attributes

| Name       | Type                       | Description                                                    |
| ---------- | -------------------------- | -------------------------------------------------------------- |
| data       | torch.Tensor \| np.ndarray | The raw tensor containing detection boxes and associated data. |
| orig_shape | tuple[int, int]            | The original image dimensions (height, width).                 |
| xyxy       | torch.Tensor \| np.ndarray | Boxes in [x1, y1, x2, y2] format.                              |
| conf       | torch.Tensor \| np.ndarray | Confidence scores for each box.                                |
| cls        | torch.Tensor \| np.ndarray | Class labels for each box.                                     |
| id         | torch.Tensor \| None       | Tracking IDs for each box (if available).                      |

## ultralytics.engine.results.Masks

| Name       | Type                       | Description                                     |
| ---------- | -------------------------- | ----------------------------------------------- |
| data       | torch.Tensor \| np.ndarray | The raw tensor or array containing mask data.   |
| orig_shape | tuple[int, int]            | Original image shape in (height, width) format. |
| xy         | list[np.ndarray]           | A list of segments in pixel coordinates.        |
| xyn        | list[np.ndarray]           | A list of normalized segments.                  |
