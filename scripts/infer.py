"""Count houses in aerial images with the trained weights.

Usage:
    python scripts/infer.py path/to/image.png
    python scripts/infer.py path/to/folder --json counts.json
    python scripts/infer.py image.png --no-sahi --conf 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predictor import DEFAULT_WEIGHTS, HouseDetector

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def collect(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file():
        return [source]
    raise FileNotFoundError(f"source not found: {source}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect and count houses in aerial imagery.")
    ap.add_argument("source", help="image file or directory")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--slice", type=int, default=512, help="SAHI slice size in pixels")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--no-sahi", action="store_true", help="disable slicing")
    ap.add_argument("--device", default=None, help="cpu or cuda:0; default is automatic")
    ap.add_argument("--out", default="outputs", help="directory for annotated images")
    ap.add_argument("--json", default=None, help="write per-image counts to this JSON file")
    ap.add_argument("--csv", default=None, help="write per-image counts to this CSV file")
    args = ap.parse_args()

    images = collect(Path(args.source))
    if not images:
        print("no images found")
        return

    detector = HouseDetector(
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        use_sahi=not args.no_sahi,
        slice_size=args.slice,
        overlap=args.overlap,
    )
    print(f"device: {detector.device} | sahi: {detector.use_sahi} | images: {len(images)}")

    rows = []
    for i, image_path in enumerate(images, 1):
        result = detector.detect_and_save(image_path, args.out)
        rows.append({"image": image_path.name, "count": result["count"], "output": result["output"]})
        print(f"[{i}/{len(images)}] {image_path.name}: {result['count']} houses -> {result['output']}")

    total = sum(r["count"] for r in rows)
    print(f"total detections: {total}")

    if args.json:
        Path(args.json).write_text(json.dumps({"total": total, "images": rows}, indent=2))
        print("json:", args.json)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "count", "output"])
            writer.writeheader()
            writer.writerows(rows)
        print("csv:", args.csv)


if __name__ == "__main__":
    main()
