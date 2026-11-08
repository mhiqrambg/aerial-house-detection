"""Evaluate the trained model: standard metrics + counting analysis.

Usage: python scripts/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.count_analysis import analyse

ROOT = Path(__file__).resolve().parents[1]
best = ROOT / "experiments" / "exp01_yolo11s_baseline" / "weights" / "best.pt"
dataset = ROOT / "datasets" / "whu_yolo"

model = YOLO(str(best))
metrics = model.val(data=str(ROOT / "configs" / "dataset.yaml"), split="test",
                    imgsz=640, conf=0.25, iou=0.6, plots=True,
                    project=str(ROOT / "experiments"), name="eval_test", exist_ok=True)

out = {
    "precision": metrics.box.mp,
    "recall": metrics.box.mr,
    "map50": metrics.box.map50,
    "map50_95": metrics.box.map,
    "inference_ms": metrics.speed["inference"],
}
out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"] + 1e-9)
out["counting"] = analyse(model, dataset / "images" / "test", dataset / "labels" / "test")
(ROOT / "experiments" / "eval_test" / "summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
