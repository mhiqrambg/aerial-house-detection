"""Tests for mask -> YOLO bbox conversion (geometric correctness).

Mirrors the assertions that ran in Colab Phase 4; they must pass before any
dataset conversion or training.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.convert_masks import boxes_to_yolo, mask_to_bboxes


def test_bboxes_match_known_rectangles():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:30, 20:50] = 255  # x=20 y=10 w=30 h=20
    mask[60:90, 70:95] = 255  # x=70 y=60 w=25 h=30
    mask[95:97, 2:4] = 255  # 4 px speck, must be filtered

    got = sorted(mask_to_bboxes(mask), key=lambda b: b[0])
    assert got == [(20, 10, 30, 20), (70, 60, 25, 30)]


def test_bbox_exactly_bounds_its_component():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:30, 20:50] = 255
    mask[60:90, 70:95] = 255

    n, labels, _, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    for x, y, w, h in mask_to_bboxes(mask):
        label = labels[y + h // 2, x + w // 2]
        assert label != 0
        gt = (labels == label).astype(np.uint8)
        pred = np.zeros_like(gt)
        pred[y : y + h, x : x + w] = 1
        iou = np.logical_and(gt, pred).sum() / np.logical_or(gt, pred).sum()
        assert iou == 1.0


def test_yolo_normalisation():
    lines = boxes_to_yolo([(20, 10, 30, 20)], 100, 100)
    assert len(lines) == 1
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0" and len(lines[0].split()) == 5
    assert abs(float(cx) - 0.35) < 1e-9
    assert abs(float(cy) - 0.20) < 1e-9
    assert abs(float(w) - 0.30) < 1e-9
    assert abs(float(h) - 0.20) < 1e-9
