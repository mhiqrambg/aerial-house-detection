"""Tests for counting/matching utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import counting_errors, iou_xyxy, match_predictions


def test_iou_identical_is_one():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_match_counts_duplicate_not_double_tp():
    gt = [(0, 0, 10, 10)]
    preds = [(0, 0, 10, 10), (0, 0, 10, 10)]  # same house twice
    tp, fp, fn, dup = match_predictions(preds, gt)
    assert (tp, fp, fn, dup) == (1, 0, 0, 1)


def test_counting_errors():
    res = counting_errors([10, 8], [10, 10])
    assert res["count_mae"] == 1.0
    assert abs(res["count_mae_pct"] - 10.0) < 1e-9
