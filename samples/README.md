# Sample input

`demo_tile.png` is tile `train_0014` from the WHU Building Dataset (Christchurch, 0.3 m),
downloaded from the Hugging Face mirror `giswqs/WHU-Building-Dataset`, license CC BY 4.0.
It is included only as a quick smoke test so a new user has a real image to run.

Expected result:

```bash
python scripts/infer.py samples/demo_tile.png --no-sahi
# 76 houses
```

The ground-truth mask for this tile contains 76 connected components, so the counts match.
This tile belongs to the **training split**, so the match is not evidence of generalisation.
Test-split performance is the figure reported in the README (mAP@0.5 0,9342, count MAE 1,79).

To run on your own imagery, replace the path:

```bash
python scripts/infer.py path/to/your_image.png
```
