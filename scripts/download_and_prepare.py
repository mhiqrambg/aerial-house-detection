"""Download the WHU Building Dataset (CC BY 4.0 mirror) and convert to YOLO.

Usage: python scripts/download_and_prepare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.convert_masks import build_yolo_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = ROOT / "datasets" / "whu"
    if not (src / "train" / "Image").exists():
        snapshot_download(
            repo_id="giswqs/WHU-Building-Dataset",
            repo_type="dataset",
            local_dir=str(src),
        )
    stats = build_yolo_dataset(src, ROOT / "datasets" / "whu_yolo")
    out = ROOT / "datasets" / "conversion_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
