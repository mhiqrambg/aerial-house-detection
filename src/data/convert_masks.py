"""Convert WHU binary building masks to YOLO instance bounding boxes.

WHU ships raster masks (0/255), not instance polygons, so every connected
component becomes one bounding box. Touching houses therefore merge into a
single box; that limitation is documented in README and analysed in Phase 7.

Validated by tests/test_conversion.py (exact bbox match + IoU = 1.0 on
synthetic masks).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

MIN_AREA = 64  # px^2, drops specks; heuristic documented in the phase-1 report


def mask_to_bboxes(mask: np.ndarray, min_area: int = MIN_AREA) -> list[tuple[int, int, int, int]]:
    """Binary mask -> (x, y, w, h) axis-aligned boxes using 8-connectivity."""
    binary = (mask > 127).astype(np.uint8)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):  # label 0 is background
        x, y, w, h, area = stats[i]
        if area >= min_area and w > 0 and h > 0:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def boxes_to_yolo(boxes: list[tuple[int, int, int, int]], width: int, height: int) -> list[str]:
    """Pixel boxes -> normalized ``cls cx cy w h`` lines (single class 0)."""
    lines = []
    for x, y, w, h in boxes:
        lines.append(
            f"0 {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
            f"{w / width:.6f} {h / height:.6f}"
        )
    return lines


def build_yolo_dataset(src: Path, dst: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> dict:
    """Write images/{split} (symlinks) + labels/{split}/*.txt from WHU layout."""
    stats: dict[str, dict] = {}
    for split in splits:
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_img = n_box = n_empty = n_bad = 0
        for ip in sorted((src / split / "Image").glob("*.png")):
            mask_path = src / split / "Mask" / ip.name
            im = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
            if im is None or mask is None:
                n_bad += 1
                continue
            h, w = mask.shape
            lines = boxes_to_yolo(mask_to_bboxes(mask), w, h)
            dst_img = img_dir / ip.name
            if not dst_img.exists():
                os.symlink(ip, dst_img)
            (lbl_dir / f"{ip.stem}.txt").write_text("\n".join(lines))
            n_img += 1
            n_box += len(lines)
            n_empty += len(lines) == 0
        stats[split] = {
            "images": n_img,
            "boxes": n_box,
            "empty_images": n_empty,
            "corrupt": n_bad,
            "mean_boxes_per_img": round(n_box / max(n_img, 1), 2),
        }
    return stats


if __name__ == "__main__":
    src_dir = Path("datasets/whu")
    dst_dir = Path("datasets/whu_yolo")
    result = build_yolo_dataset(src_dir, dst_dir)
    print(json.dumps(result, indent=2))
