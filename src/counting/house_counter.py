"""House counting from aerial imagery: detect -> merge -> count unique boxes.

Counting always happens AFTER overlap merging (SAHI NMM or NMS), never on raw
tile predictions, so one house is not counted once per overlapping tile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def count_houses(
    image_path: str | Path,
    model: Any = None,
    detector: Any = None,
    conf: float = 0.25,
    iou: float = 0.5,
    use_sahi: bool = True,
    slice_size: int = 512,
    overlap: float = 0.2,
) -> dict:
    """Return count, boxes, scores and an annotated image for one aerial image.

    Args:
        image_path: image to analyse.
        model: Ultralytics YOLO model, used when ``use_sahi`` is False.
        detector: SAHI ``AutoDetectionModel``, required when ``use_sahi`` is True.
        conf: confidence threshold.
        iou: NMS / merge IoU threshold.
        use_sahi: sliced inference for high-resolution inputs.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    if use_sahi:
        if detector is None:
            raise ValueError("detector is required when use_sahi=True")
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            img,
            detector,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_match_threshold=iou,
            verbose=0,
        )
        boxes = [b.bbox.to_xyxy() for b in result.object_prediction_list]
        scores = [float(b.score.value) for b in result.object_prediction_list]
    else:
        if model is None:
            raise ValueError("model is required when use_sahi=False")
        pred = model.predict(img, imgsz=640, conf=conf, iou=iou, verbose=False)[0]
        boxes = pred.boxes.xyxy.cpu().numpy().tolist() if pred.boxes is not None else []
        scores = pred.boxes.conf.cpu().numpy().tolist() if pred.boxes is not None else []

    annotated = img.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        cv2.putText(
            annotated,
            f"{score:.2f}",
            (int(x1), max(int(y1) - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"Houses: {len(boxes)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return {"count": len(boxes), "boxes": boxes, "scores": scores, "annotated": annotated}
