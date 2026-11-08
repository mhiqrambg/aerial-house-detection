"""Per-image counting and error analysis on a YOLO label directory.

Mirrors the analysis that produced experiments/eval_test/counting_analysis.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.evaluation.metrics import counting_errors, iou_xyxy, match_predictions


def load_gt(txt: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    """YOLO label file -> xyxy boxes in pixel coordinates."""
    boxes = []
    for line in txt.read_text().splitlines():
        _, cx, cy, w, h = map(float, line.split())
        boxes.append(
            (
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            )
        )
    return boxes


def analyse(model: Any, images_dir: Path, labels_dir: Path, conf: float = 0.25,
            iou: float = 0.5, small_px: int = 32) -> dict:
    """Return TP/FP/FN/duplicates, count MAE, and small-object recall."""
    totals = {"tp": 0, "fp": 0, "fn": 0, "dup": 0, "small_tp": 0, "small_total": 0}
    pred_counts: list[int] = []
    gt_counts: list[int] = []

    for ip in sorted(images_dir.glob("*.png")):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(labels_dir / f"{ip.stem}.txt", w, h)
        result = model.predict(str(ip), imgsz=640, conf=conf, iou=iou, verbose=False)[0]
        preds = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []

        tp, fp, fn, dup = match_predictions(preds, gt, thr=0.5)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["dup"] += dup
        for g in gt:
            if (g[2] - g[0]) < small_px or (g[3] - g[1]) < small_px:
                totals["small_total"] += 1
                if any(iou_xyxy(p, g) >= 0.5 for p in preds):
                    totals["small_tp"] += 1
        pred_counts.append(len(preds))
        gt_counts.append(len(gt))

    res: dict = dict(totals)
    res.update(counting_errors(pred_counts, gt_counts))
    res["small_object_recall"] = totals["small_tp"] / max(totals["small_total"], 1)
    res["precision"] = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    res["recall"] = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    res["f1"] = 2 * res["precision"] * res["recall"] / max(res["precision"] + res["recall"], 1e-9)
    return res
