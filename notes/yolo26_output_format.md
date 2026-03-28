# Output format of YOLOv26

## Bbox (`results[0].boxes`)

| Attribute | Shape / Type | Description |
| :--- | :--- | :--- |
| `shape` | `(N, 6)` | Format: `[x1, y1, x2, y2, confidence, class]` (Tracking id included if active) |
| `data` | `(N, 6)` | Raw tensor containing box coordinates and scores |
| `orig_shape`| `(H, W)` | Original image dimensions |
| `is_track` | `bool` | True if tracking is enabled |
| `xyxy` | `(N, 4)` | Bounding box coordinates in `[x1, y1, x2, y2]` format |
| `conf` | `(N,)` | Confidence scores for each detection |
| `cls` | `(N,)` | Class indices for each detection |
| `id` | `(N,)` | Track IDs (returns `None` if `is_track=False`) |

## Instance Mask (`results[0].masks`)

| Attribute | Shape / Type | Description |
| :--- | :--- | :--- |
| `data` | `(N, H, W)` | Mask tensors (usually resized to inference size, e.g., 640x480) |
| `orig_shape`| `(H, W)` | Original image dimensions |
| `xy` | `list(array)` | List of segments in pixel coordinates (contours) |
