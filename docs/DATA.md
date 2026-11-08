# Dataset card: WHU Building Dataset (aerial subset, YOLO conversion)

## Source

| Field | Value |
|---|---|
| Dataset | WHU Building Dataset (aerial subset) |
| Original publication | Ji, Wei & Lu, "Fully Convolutional Networks for Multisource Building Extraction From Mismatched Cross-Resolution Satellite Imagery", IEEE TGRS 57(1), 2019 |
| Original page | https://gpcv.whu.edu.cn/data/building_dataset.html |
| Mirror used | https://huggingface.co/datasets/giswqs/WHU-Building-Dataset (license declared as **CC BY 4.0** on the dataset card) |
| Verified | 2026-11-09, snapshot downloaded from the mirror |

The original WHU page does not state a license. The CC BY 4.0 declaration comes from the
Hugging Face mirror card. Confirm terms with the original authors before redistribution.

## What the mirror contains

- 8,188 tiles, 512 x 512 px, RGB PNG at 0.3 m GSD
- one binary mask per tile (0 = background, 255 = building)
- fixed split: train 5,732 / val 1,228 / test 1,228
- no geographic coordinates per tile in the mirror, so the split is tile-level and geographic
  leakage cannot be re-checked from this mirror alone

## Conversion to object detection

WHU provides raster masks, not instance polygons, so each 8-connected component becomes one
bounding box (`src/data/convert_masks.py`). Components below 64 px² are dropped as noise.

Verified counts after conversion:

| Split | Images | Boxes | Empty images |
|---|---|---|---|
| train | 5,732 | 143,470 | 975 |
| val | 1,228 | 31,538 | 224 |
| test | 1,228 | 30,206 | 225 |

Dataset statistics from the EDA sample (900 masks, 22,781 instances):

- median building width 41 px, height 40 px at 512 px tiles
- p10 = 17 px, so roughly one in ten buildings is small enough to need slicing
- buildings cover 15.5% of pixels on average

## Known limitations

- **Attached houses merge.** Touching roofs form a single connected component and therefore a
  single box. Dense row housing is under-counted. This is the attached-house problem identified
  in the literature review and it is visible in the per-image error analysis.
- **"Building" is not "house".** The mask class is generic. Christchurch suburban tiles are
  dominated by residential buildings, but commercial and industrial structures are included.
- **One city.** The aerial subset covers Christchurch, New Zealand. Cross-region generalisation
  requires the SpaceNet/Inria comparison planned in the research phase and not yet run.
