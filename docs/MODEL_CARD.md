# Model card: exp01_yolo11s_baseline

## Identity

| Field | Value |
|---|---|
| Task | Single-class object detection (building / house) |
| Architecture | YOLO11s, COCO-pretrained, fine-tuned |
| Parameters | 9,428,179 |
| Input | 640 x 640, RGB |
| Output | boxes `(x1, y1, x2, y2)`, confidence, class 0 |
| Files | `best.pt` (PyTorch, 19.18 MB), `best.onnx` (37.93 MB) |
| SHA256 (best.pt) | `b3df0c3fdb5f24ce0bad5fc90ee26807fa4ac1c594418f61654167a4b36116ae` |

## Training data

WHU Building Dataset, aerial subset (Christchurch, NZ), 5,732 training tiles of 512 px at 0.3 m.
Masks converted to boxes by connected components. Details: `docs/DATA.md`.

## Evaluation

Untouched test split, 1,228 images, 30,206 instances, conf 0.25, IoU 0.6:

| Metric | Value |
|---|---|
| Precision | 0.9323 |
| Recall | 0.9156 |
| F1 | 0.9239 |
| mAP@0.5 | 0.9342 |
| mAP@0.5:0.95 | 0.7965 |
| Inference | 9.5 ms per 640 px image on T4 |

Counting: MAE 1.79 houses per image, 11.12% mean count error, small-object recall 0.8557.

## Intended use

Counting and locating residential buildings in aerial or satellite orthophotos with ground
sampling distance near 0.3 m. For larger inputs, slice with SAHI using 512 px slices and 0.2
overlap.

## Out of scope

- Distinguishing houses from commercial buildings (one generic class).
- Dense row housing where roofs touch; these merge into one detection.
- Regions outside the training geography; no cross-domain evaluation was run.
- Oblique drone video without tracking; use the video module and expect drift.

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("image.png", imgsz=640, conf=0.25)
print(len(results[0].boxes))  # detections in this tile
```

For images much larger than 640 px, use `src/counting/house_counter.py` with `use_sahi=True`.
Measured on synthetic mosaics: at 2048 px, direct inference missed 232 of 463 houses while SAHI
missed 78 and kept the count error at 20.

## License caveat

Dataset terms come from the Hugging Face mirror card (CC BY 4.0), not the original WHU page.
Check the original terms before redistribution or commercial use.
