# Experiment log

All runs executed on Google Colab, Tesla T4 (15.4 GB), PyTorch 2.14.1 + CUDA 13.0,
Ultralytics 8.4.208, SAHI 0.14.0.

## exp01_yolo11s_baseline

| Field | Value |
|---|---|
| Experiment ID | exp01_yolo11s_baseline |
| Model | YOLO11s, COCO-pretrained (`yolo11s.pt`) |
| Dataset | WHU aerial, YOLO conversion v1 (`datasets/conversion_stats.json`) |
| Split | train 5,732 / val 1,228 / test 1,228 (fixed) |
| Input size | 640 |
| Batch | 16 |
| Epochs | 30 (patience 10) |
| Optimizer | Ultralytics default (auto) |
| Seed | 42, deterministic |
| Augmentation | Ultralytics default (HSV, fliplr, flipud, scale, mosaic) |
| Hardware | Colab T4 16 GB, ~5.5 GB VRAM used |
| Training duration | 3,931 s total (about 65 min), ~2 min per epoch including validation |

### Result

| Metric | Validation (best, epoch 30) | Test |
|---|---|---|
| Precision | 0.9335 | 0.9323 |
| Recall | 0.9143 | 0.9156 |
| mAP@0.5 | 0.9697 | 0.9342 |
| mAP@0.5:0.95 | 0.8233 | 0.7965 |

Counting on test: MAE 1.79 houses per image (11.12% mean error), small-object recall 0.8557,
43 duplicate detections across 1,228 images.

The gap between validation and test mAP@0.5 (0.9697 vs 0.9342) is consistent with the test split
containing tiles from areas not seen in training, although the mirror does not expose coordinates
to confirm this. No tuning was done on test.

See README Results for the SAHI and video tables.

## Handled failures during the build

| Failure | Root cause | Fix |
|---|---|---|
| Conversion unit test failed (IoU 0.443) | test compared one bbox against the full mask containing two components | IoU computed against the matching connected component |
| EDA plotting `FileNotFoundError` | `outputs/` not created before `savefig` | `mkdir` before saving |
| SAHI `AttributeError: to_xyxy` | SAHI 0.14 exposes the box through `ObjectPrediction.bbox` | use `b.bbox.to_xyxy()` |
| MCP `run_code_cell` appeared to hang | the call waits synchronously for long cells | heavy work moved to background threads with flag files |
