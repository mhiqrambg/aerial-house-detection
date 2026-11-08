"""Local inference wrapper around the trained YOLO11s house detector.

Loads weights/best.pt, runs detection either on the whole image or with SAHI
slicing, and returns boxes, scores and a house count. Counting always happens
after slice merging.

Example:
    from src.inference.predictor import HouseDetector

    detector = HouseDetector()                       # CPU or GPU automatically
    result = detector.detect_file("aerial.png")      # SAHI slicing by default
    print(result["count"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.counting.house_counter import count_houses

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "best.pt"


class HouseDetector:
    """Thin, dependency-light wrapper so callers do not touch Ultralytics or SAHI."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        device: str | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        use_sahi: bool = True,
        slice_size: int = 512,
        overlap: float = 0.2,
    ) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"weights not found: {weights}")

        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.iou = iou
        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self.overlap = overlap
        self.model = YOLO(str(weights))
        self.detector: Any = None
        if use_sahi:
            from sahi import AutoDetectionModel

            self.detector = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(weights),
                confidence_threshold=conf,
                device=self.device,
            )

    def detect_file(self, image_path: str | Path) -> dict:
        """Detect houses in one image. Returns count, boxes, scores, annotated image."""
        return count_houses(
            image_path,
            model=self.model,
            detector=self.detector,
            conf=self.conf,
            iou=self.iou,
            use_sahi=self.use_sahi,
            slice_size=self.slice_size,
            overlap=self.overlap,
        )

    def detect_array(self, image: np.ndarray) -> dict:
        """Detect houses in an in-memory BGR image."""
        tmp = self._write_temp(image)
        try:
            return self.detect_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _write_temp(image: np.ndarray) -> Path:
        import tempfile

        fd = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fd.close()
        cv2.imwrite(fd.name, image)
        return Path(fd.name)

    def detect_and_save(self, image_path: str | Path, out_dir: str | Path = "outputs") -> dict:
        """Detect, write the annotated image, and return the result without the image array."""
        result = self.detect_file(image_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{Path(image_path).stem}_counted.png"
        cv2.imwrite(str(dst), result["annotated"])
        return {
            "image": str(image_path),
            "output": str(dst),
            "count": result["count"],
            "boxes": result["boxes"],
            "scores": result["scores"],
        }
