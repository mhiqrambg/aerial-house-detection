"""Train the YOLO11s baseline from configs/train_baseline.yaml.

Usage: python scripts/train_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "train_baseline.yaml").read_text())

model = YOLO(cfg.pop("model"))
data = ROOT / cfg.pop("data")
project = ROOT / cfg.pop("project")
result = model.train(data=str(data), project=str(project), **cfg)
print("BEST:", result.save_dir)
