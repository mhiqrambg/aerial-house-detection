# Aerial House Detection

Object detection of houses in top-view aerial imagery: bounding boxes, confidence scores and a
per-image house count. Built on YOLO11s with SAHI sliced inference for high-resolution inputs.

This repository contains the implementation of the technical plan produced in the research phase
(`phase1_literature/`, `phase1_dataset_evaluation/`, `phase1_model_stack_research/`).

## Research objective

The system detects individual houses in aerial or satellite imagery, draws a box around each one,
reports a confidence score per detection, and counts the unique houses per image. It processes
images larger than the detector input by slicing. A video path (detection plus tracking) exists for
later drone footage work.

## Dataset

WHU Building Dataset, aerial subset, downloaded from the Hugging Face mirror
`giswqs/WHU-Building-Dataset` (CC BY 4.0 per the dataset card). 8,188 tiles of 512 x 512 px at
0.3 m, fixed split of 5,732 / 1,228 / 1,228. The masks are raster, so each connected component is
converted to one bounding box.

Full provenance, statistics and limitations: `docs/DATA.md`.

## Model

| Item | Value |
|---|---|
| Detector | YOLO11s (Ultralytics), COCO-pretrained |
| Input size | 640 |
| Batch / epochs | 16 / 30 with early-stopping patience 10 |
| Seed | 42, deterministic |
| Augmentation | Ultralytics defaults (HSV, horizontal and vertical flip, scale, mosaic) |
| High-resolution inference | SAHI, 512 px slices, 0.2 overlap, NMM merge |

The decision to use YOLO11s plus SAHI rather than a transformer detector comes from the research
phase: the hard part is small objects in large images, which slicing addresses directly, and the
training budget is a single free-tier GPU.

## Installation

```bash
pip install -r requirements.txt
```

## Data preparation

```bash
python scripts/download_and_prepare.py
```

This downloads the mirror into `datasets/whu/` and writes the YOLO dataset to `datasets/whu_yolo/`
plus `datasets/conversion_stats.json`. Conversion logic is unit tested:

```bash
pytest tests/
```

## Training

```bash
python scripts/train_baseline.py
```

Configuration lives in `configs/train_baseline.yaml`. Checkpoints and curves go to
`experiments/exp01_yolo11s_baseline/` (weights, `results.csv`, `results.png`,
`training_history.png`).

## Run locally with the trained model

Colab was used only to train and export the model. Inference runs locally on CPU or GPU.

```bash
python3 -m venv .venv-infer          # Python 3.10 to 3.13
source .venv-infer/bin/activate
pip install -r requirements-infer.txt

# quick check with the bundled sample (76 houses)
python scripts/infer.py samples/demo_tile.png --no-sahi

# one image, SAHI slicing by default (recommended for large images)
python scripts/infer.py photo.png

# a folder, with a JSON and CSV summary
python scripts/infer.py photos/ --json counts.json --csv counts.csv

# whole-image inference, faster but weaker on large inputs
python scripts/infer.py photo.png --no-sahi

# force CPU
python scripts/infer.py photo.png --device cpu
```

Weights are in `weights/best.pt` (PyTorch) and `weights/best.onnx`. The CLI writes annotated
images to `outputs/` and prints the house count per image.

Python API:

```python
from src.inference.predictor import HouseDetector

detector = HouseDetector()                 # picks cuda if available
result = detector.detect_file("photo.png")
print(result["count"], result["scores"][:3])
```

## Evaluation

```bash
python scripts/evaluate.py
```

Writes standard metrics and the counting analysis to `experiments/eval_test/summary.json`. The test
split is not used for tuning.

## Inference and counting

```bash
python scripts/infer.py path/to/aerial_image.png            # SAHI slicing (default)
python scripts/infer.py path/to/aerial_image.png --no-sahi  # no slicing, for comparison
```

The annotated image is written to `outputs/`. Counting is always done after slice merging, so a
house seen in two overlapping tiles is counted once.

## Video (secondary)

`src/inference/video.py` counts unique track IDs with BoT-SORT, which compensates camera motion.
No public UAV house video dataset was found, so the video path is validated on synthetic panning
footage and is not a counting benchmark.

## Project structure

```text
aerial-house-detection/
├── configs/          dataset and training YAML
├── datasets/         downloaded data and YOLO conversion (gitignored)
├── docs/             dataset card, experiment log
├── notebooks/        Colab notebook exported verbatim (36 cells)
├── src/
│   ├── data/         mask to bbox conversion
│   ├── evaluation/   IoU, matching, counting errors, count analysis
│   ├── counting/     house counting after merge
│   └── inference/    video tracking and unique-ID counting
├── scripts/          download, train, evaluate, infer
├── tests/            conversion and counting unit tests
├── experiments/      metrics, curves, checkpoints (gitignored)
└── outputs/          annotated images and figures (gitignored)
```

## Documents

| File | Content |
|---|---|
| `docs/Laporan_Aerial_House_Detection.docx` | Full research report, 15 sections, 9 tables, 9 figures |
| `docs/Slides_Aerial_House_Detection.pptx` | 10-slide presentation |
| `docs/MODEL_CARD.md` | Model identity, metrics, intended use |
| `docs/DATA.md` | Dataset provenance and limitations |
| `docs/EXPERIMENTS.md` | Experiment log and handled failures |

Both the report and the slides are generated from run artifacts by
`scripts/build_report.py` and `scripts/build_slides.py`.

`notebooks/aerial_house_detection_colab.ipynb` is a verbatim export of the 36 cells that ran in
Colab. Its paths point to `/content/...`, so it is a record of the training run, not a local
runner. Local inference uses `scripts/infer.py` instead.

## Results

All numbers below come from the Colab run (T4, 30 epochs) and the untouched test split
(1,228 images, 30,206 instances at confidence 0.25, IoU 0.6 for `val`).

### Test-set metrics (exp01_yolo11s_baseline)

| Metric | Value |
|---|---|
| Precision | 0.9323 |
| Recall | 0.9156 |
| F1 | 0.9239 |
| mAP@0.5 | 0.9342 |
| mAP@0.5:0.95 | 0.7965 |
| Inference | 9.5 ms per 640 px image (T4) |

Best validation epoch was the last one: mAP@0.5 0.9697, mAP@0.5:0.95 0.8233. Train and validation
losses fell together, so no overfitting is visible in this run.

### Counting analysis (per image, conf 0.25, IoU 0.5)

| Metric | Value |
|---|---|
| True positives / false positives / false negatives | 27,944 / 3,338 / 2,262 |
| Duplicate detections | 43 |
| Count MAE | 1.79 houses per image |
| Mean count error | 11.12% |
| Small-object recall (box side < 32 px) | 0.8557 |

Precision here (0.8933) differs from the `val` precision (0.9323) because the analysis uses a
different matching path and confidence threshold. Both are reported; neither is adjusted.

### High-resolution inference: direct vs SAHI

Mosaics assembled from test tiles, matched to GT at IoU 0.5.

| Input | Method | Pred | TP | FP | FN | Count error | F1 |
|---|---|---|---|---|---|---|---|
| 1024 x 1024 (186 GT) | direct | 179 | 161 | 18 | 25 | 7 | 0.882 |
| 1024 x 1024 | SAHI 512 | 204 | 143 | 61 | 43 | 18 | 0.733 |
| 1024 x 1024 | SAHI 640 | 203 | 142 | 61 | 44 | 17 | 0.730 |
| 2048 x 2048 (463 GT) | direct | 257 | 231 | 26 | 232 | 206 | 0.642 |
| 2048 x 2048 | SAHI 512 | 483 | 385 | 98 | 78 | 20 | 0.814 |
| 2048 x 2048 | SAHI 640 | 492 | 365 | 127 | 98 | 29 | 0.764 |

At twice the tile size, direct inference is better. At four times, direct inference loses 232 of
463 houses while SAHI keeps the count error at 20. The practical rule from this run: slice when
the image is much larger than the training tile, not automatically.

### Video pipeline (synthetic panning clip)

40 frames, 3,745 per-frame detections, 188 unique track IDs. The per-frame sum is the duplication
a naive counter would report. The clip is synthetic and shows part of the scene, so 188 is a
pipeline check, not a counting benchmark.

### Figures

`outputs/eda_bbox_sizes.png`, `outputs/eda_gt_samples.png`, `outputs/eval_gt_vs_pred.png`,
`outputs/mosaic_1024_counted.png`, `experiments/exp01_yolo11s_baseline/training_history.png`.
PR, F1 and confusion-matrix plots are in `experiments/eval_test/`.

## Limitations

- Touching houses merge into one mask component and therefore one box, so dense row housing is
  under-counted.
- The class is generic "building". Residential tiles dominate, but commercial structures are
  included.
- Single city (Christchurch). Cross-region generalisation is untested.
- No UAV house video dataset was available; the video module is pipeline-validated only.

## References

- Ji, Wei & Lu. Fully Convolutional Networks for Multisource Building Extraction. IEEE TGRS 2019.
- Akyon, Altinuc & Temizel. Slicing Aided Hyper Inference and Fine-tuning for Small Object
  Detection. IEEE ICIP 2022. https://arxiv.org/abs/2202.06934
- Ultralytics YOLO11 documentation. https://docs.ultralytics.com/models/yolo11
- SAHI repository. https://github.com/obss/sahi
- Dataset mirror. https://huggingface.co/datasets/giswqs/WHU-Building-Dataset
