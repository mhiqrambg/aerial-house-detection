"""Inference pipeline for the trained house detector.

Three entry points:
  - predict_image: one image, returns boxes, scores, count, annotated image
  - predict_folder: batch over a directory, writes annotated images and a CSV/JSON summary
  - predict_video: tracking-based unique count and annotated clip

SAHI slicing is chosen automatically when the image is much larger than the
training tile, which is the rule that came out of the high-resolution experiment
(1024 px direct was fine, 2048 px direct lost half the houses).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


class HouseDetectionPipeline:
    """Wraps YOLO11s with optional SAHI slicing and count reporting."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.5,
        slice_size: int = 512,
        overlap: float = 0.2,
        use_sahi: bool | None = None,
        auto_sahi_ratio: float = 1.5,
        device: str | None = None,
    ) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"weights not found: {self.weights}")
        self.conf = conf
        self.iou = iou
        self.slice_size = slice_size
        self.overlap = overlap
        self.use_sahi = use_sahi
        self.auto_sahi_ratio = auto_sahi_ratio
        self.device = device
        self._model = None
        self._detector = None

    # -- lazy loaders keep import time low for CLI use --
    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        return self._model

    @property
    def detector(self):
        if self._detector is None:
            from sahi import AutoDetectionModel

            device = self.device
            if device is None:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._detector = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(self.weights),
                confidence_threshold=self.conf,
                device=device,
            )
        return self._detector

    def _should_slice(self, width: int, height: int) -> bool:
        if self.use_sahi is not None:
            return self.use_sahi
        return max(width, height) > self.auto_sahi_ratio * self.slice_size

    def predict_image(self, image_path: str | Path) -> dict[str, Any]:
        """Detect houses in one image. Returns count, boxes, scores, annotated."""
        from src.counting.house_counter import count_houses

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        h, w = img.shape[:2]
        use_sahi = self._should_slice(w, h)

        t0 = time.perf_counter()
        res = count_houses(
            image_path,
            model=self.model,
            detector=self.detector if use_sahi else None,
            conf=self.conf,
            iou=self.iou,
            use_sahi=use_sahi,
            slice_size=self.slice_size,
            overlap=self.overlap,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "image": str(image_path),
            "width": w,
            "height": h,
            "sahi": use_sahi,
            "count": res["count"],
            "boxes": [[float(v) for v in b] for b in res["boxes"]],
            "scores": [float(s) for s in res["scores"]],
            "inference_ms": round(elapsed, 1),
            "annotated": res["annotated"],
        }

    def predict_folder(self, folder: str | Path, out_dir: str | Path) -> list[dict[str, Any]]:
        """Run over every image in a folder, write annotated images plus CSV/JSON."""
        folder = Path(folder)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXT:
                continue
            res = self.predict_image(path)
            dst = out_dir / f"{path.stem}_detected{path.suffix}"
            cv2.imwrite(str(dst), res["annotated"])
            records.append(
                {
                    "image": path.name,
                    "width": res["width"],
                    "height": res["height"],
                    "sahi": res["sahi"],
                    "count": res["count"],
                    "mean_score": round(
                        sum(res["scores"]) / len(res["scores"]), 4
                    ) if res["scores"] else 0.0,
                    "inference_ms": res["inference_ms"],
                    "boxes": res["boxes"],
                    "scores": res["scores"],
                }
            )

        self._write_summary(records, out_dir)
        return records

    def predict_video(
        self,
        video_path: str | Path,
        out_path: str | Path | None = None,
        tracker: str = "botsort.yaml",
    ) -> dict[str, Any]:
        """Unique house count over a clip. Never sums per-frame detections."""
        from src.inference.video import annotate_video, count_unique_houses_video

        result = count_unique_houses_video(video_path, self.model, tracker=tracker, conf=self.conf)
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            annotate_video(video_path, out_path, self.model, tracker=tracker, conf=self.conf)
            result["annotated_video"] = str(out_path)
        result["source"] = str(video_path)
        return result

    @staticmethod
    def _write_summary(records: list[dict[str, Any]], out_dir: Path) -> None:
        csv_path = out_dir / "predictions.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["image", "width", "height", "sahi", "count", "mean_score", "inference_ms"],
            )
            writer.writeheader()
            for r in records:
                writer.writerow({k: r[k] for k in writer.fieldnames})

        json_path = out_dir / "predictions.json"
        json_path.write_text(json.dumps(records, indent=2))

        total = sum(r["count"] for r in records)
        print(f"{len(records)} images, {total} houses total")
        print(f"summary: {csv_path} and {json_path}")
