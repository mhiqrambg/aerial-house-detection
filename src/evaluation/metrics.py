"""Counting and matching utilities shared by evaluation and inference."""

from __future__ import annotations

from typing import Sequence

Box = Sequence[float]  # x1, y1, x2, y2


def iou_xyxy(a: Box, b: Box) -> float:
    """IoU of two axis-aligned boxes in the same coordinate frame."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_predictions(
    preds: list[Box], gts: list[Box], thr: float = 0.5
) -> tuple[int, int, int, int]:
    """Greedy IoU matching. Returns (tp, fp, fn, duplicates)."""
    used: set[int] = set()
    tp = fp = dup = 0
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            v = iou_xyxy(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= thr:
            if best_j in used:
                dup += 1
            else:
                used.add(best_j)
                tp += 1
        else:
            fp += 1
    fn = len(gts) - len(used)
    return tp, fp, fn, dup


def counting_errors(pred_counts: list[int], gt_counts: list[int]) -> dict[str, float]:
    """Count MAE and mean percentage error, both per image."""
    if not pred_counts:
        return {"count_mae": 0.0, "count_mae_pct": 0.0}
    errs = [abs(p - g) for p, g in zip(pred_counts, gt_counts)]
    pct = [abs(p - g) / max(g, 1) for p, g in zip(pred_counts, gt_counts)]
    return {
        "count_mae": sum(errs) / len(errs),
        "count_mae_pct": 100.0 * sum(pct) / len(pct),
    }
