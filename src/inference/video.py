"""Video house counting via tracking: unique track IDs, never per-frame sums.

BoT-SORT is used because it compensates camera motion, which is the documented
recommendation for drone footage. No public UAV *house* video dataset exists in
the phase-1 findings, so this module is validated on synthetic panning footage
(pipeline mechanics only, not a count benchmark).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def count_unique_houses_video(
    video_path: str | Path,
    model: Any,
    tracker: str = "botsort.yaml",
    conf: float = 0.25,
) -> dict:
    """Count unique tracked houses across a clip.

    ``persist=True`` keeps tracker state between frames so one house keeps one
    ID. The returned ``unique_ids`` is the house estimate; the per-frame sum is
    returned only to show the duplication that naive counting would produce.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    ids_seen: set[int] = set()
    per_frame: list[int] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker=tracker, conf=conf, verbose=False)[0]
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            ids_seen.update(ids)
            per_frame.append(len(ids))
        else:
            per_frame.append(0)
    cap.release()

    return {
        "unique_ids": len(ids_seen),
        "frames": len(per_frame),
        "sum_per_frame_detections": int(sum(per_frame)),
        "note": "unique track IDs are the count; per-frame sum is a duplication lower bound",
    }


def annotate_video(video_path: str | Path, out_path: str | Path, model: Any,
                   tracker: str = "botsort.yaml", conf: float = 0.25) -> None:
    """Write a copy of the clip with boxes and track IDs drawn."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker=tracker, conf=conf, verbose=False)[0]
        annotated = result.plot()
        writer.write(annotated)
    cap.release()
    writer.release()
