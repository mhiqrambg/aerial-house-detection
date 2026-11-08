# Aerial House Detection — Model Architecture & Technology Stack Research

**Project context:** Object detection of houses in high-resolution aerial/satellite imagery (bbox + confidence + count per image), possible later extension to drone video. Training budget: Google Colab / Kaggle free GPU or single consumer GPU.

**Research date:** 2026-11-09 (all facts verified against official docs/repos on this date)
**Method:** Fact-check / quick-brief mode — official documentation and repository verification, followed by technical assessment. FACTS are cited; ESTIMATES/ASSESSMENTS are explicitly labeled.

---

## TASK A — Model Comparison

### A.1 Verified current state of the Ultralytics lineup (2026)

**FACT:** The current Ultralytics model lineup as of 2026 is:

| Model | Status (per official docs) | Notes |
|---|---|---|
| **YOLO26** | **Latest release, January 2026.** Official docs: "Start with YOLO26 for a new project: it is the latest Ultralytics release and the only one covering all seven tasks." | NMS-free end-to-end predictor, removed DFL, ProgLoss, **STAL (small-target-aware label assignment)**, MuSGD optimizer, up to 43% faster CPU/Jetson inference |
| **YOLO11** | "Mature alternative... stable production workloads, with pretrained checkpoints for every task" | C3k2 CSP bottlenecks, C2PSA spatial-attention module; 22% fewer params than YOLOv8m with +1.3 mAP on COCO |
| **YOLO12** | Attention-centric community release. Ultralytics itself warns it "may exhibit training instability, elevated memory consumption, and slower CPU throughput due to its heavy attention blocks" and recommends YOLO11/YOLO26 for production | Pretrained weights cover detection only |
| **YOLOE** | Open-vocabulary detection/segmentation (text/visual/prompt-free) | Niche; not needed for single-class house detection |

Sources: https://docs.ultralytics.com/models · https://docs.ultralytics.com/models/yolo26 · https://docs.ultralytics.com/models/yolo11 · https://docs.ultralytics.com/models/yolo12

**FACT (YOLO26 detection benchmarks, official GitHub README, COCO val, 640px):**

| Variant | mAP50-95 | Params (M) | FLOPs (B) | T4 TensorRT (ms) |
|---|---|---|---|---|
| YOLO26n | 40.9 | 2.4 | 5.5 | 1.7 |
| YOLO26s | 48.6 | 9.5 | 20.9 | 2.5 |
| YOLO26m | 53.1 | 20.4 | 68.4 | 4.7 |
| YOLO26l | 55.0 | 24.8 | 86.8 | 6.2 |
| YOLO26x | 57.5 | 55.7 | 194.4 | 11.8 |

Source: https://github.com/ultralytics/ultralytics

**FACT (community dissent, disclosed per contradiction rule):** Independent community benchmarks and third-party write-ups report YOLO11 sometimes matching or beating YOLO26 in latency/accuracy in practice (e.g., a Reddit r/computervision comparison reporting YOLO11 faster at ~70 FPS vs YOLO26), and a Medium review arguing YOLO26 underperforms YOLO12. YOLO26's advantages are strongest for edge deployment (NMS-free export) and small-target label assignment (STAL) — the latter is directly relevant to aerial house detection. Source: https://www.reddit.com/r/computervision/comments/1r8wahf/yolo_11_vs_yolo_26/ (Tier 3, anecdotal — treat as signal, not proof).

**FACT:** RT-DETR is fully integrated in Ultralytics: `rtdetr-l` (~45M params) and `rtdetr-x` (~86M params) pretrained weights, with Train/Val/Predict/Export modes supported. RT-DETRv2 is **not** integrated in Ultralytics (per official FAQ), though the upstream Baidu repos (Apache-2.0) cover v1/v2/v4.
Sources: https://docs.ultralytics.com/models/rtdetr · https://docs.ultralytics.com/models · https://github.com/lyuwenyu/RT-DETR

### A.2 Comparison table

Facts (specs, licenses, availability) are from official sources; the qualitative columns (advantages/limitations/suitability) are **ASSESSMENTS** grounded in those specs.

| Model | Architecture family | Advantages | Limitations | Complexity / training cost | Small-object suitability | High-res compatibility | Pretrained weights | Official repo |
|---|---|---|---|---|---|---|---|---|
| **YOLOv8** (Ultralytics) | One-stage CNN (CSPDarknet + C2f) | Extremely mature; huge community; every tutorial/dataset pipeline supports it; cheap to train | Superseded: YOLO11 ≥ YOLOv8 accuracy with fewer params (docs-verified) | **Low** — smallest of all listed | Moderate; standard 3-scale head struggles with very small objects | Via SAHI/tiling, or imgsz=1280 | COCO-pretrained n–x, all sizes | https://github.com/ultralytics/ultralytics |
| **YOLO11** (Ultralytics) | One-stage CNN (C3k2 + C2PSA attention) | Best maturity/stability balance; fewer params than v8m at +1.3 mAP; C2PSA helps spatial feature focus | AGPL-3.0 license (matters only for commercial products) | **Low** | Good (C2PSA attention); excellent with SAHI | Via SAHI/tiling, or imgsz=1280 | COCO-pretrained n–x | https://github.com/ultralytics/ultralytics |
| **YOLO26** (Ultralytics) | One-stage CNN, NMS-free end-to-end | Latest; **STAL small-target-aware label assignment** designed for small objects; NMS-free simplifies pipeline; edge-optimized | Newest = least community troubleshooting; community reports mixed vs YOLO11 (see A.1) | **Low** | **Good-by-design** (STAL); good with SAHI | Via SAHI/tiling, or imgsz=1280 | COCO-pretrained n–x | https://github.com/ultralytics/ultralytics |
| **RT-DETR** (Baidu, in Ultralytics) | Hybrid: CNN backbone (HGNetv2) + transformer encoder, DETR head | Real-time DETR; no NMS; strong COCO accuracy; integrated train/val/export in Ultralytics | Smallest pretrained variant is ~45M params (≈ 19× YOLO26n); transformers need more data/epochs to fine-tune well; higher VRAM | **Medium-high** — ~4–9× params of a YOLO s/m | Moderate (DETR queries favor larger objects; small-object aerial performance usually needs SAHI) | Via SAHI/tiling | rtdetr-l, rtdetr-x (COCO) in Ultralytics; r50vd variants upstream (Apache-2.0) | https://docs.ultralytics.com/models/rtdetr · https://github.com/lyuwenyu/RT-DETR |
| **Deformable DETR** | Sparse-attention transformer (deformable attention) | Efficient attention vs vanilla DETR; fast convergence; strong research baseline | Research code (fundamentalvision repo, minimal maintenance, open issues); custom COCO training scripts; no one-line fine-tune API;compile of CUDA ops (MSDeformAttn) is fiddly on Colab | **High** — research-grade tooling | Moderate-good (multi-scale deformable attention helps) | Multi-scale features help, but still typically trained ≤1333×800 | COCO-pretrained (ResNet-50) | https://github.com/fundamentalvision/Deformable-DETR |
| **DINO** (IDEA-Research) | DETR + denoising anchors (DN/DAB/Deformable stack), ICLR 2023 | SOTA-class COCO accuracy; strong foundation for fine-tuning | Heavy (large backbones, 12–36 epochs on COCO-class compute); MMDetection-based tooling; overkill for single-class counting on free GPUs | **Very high** | Good | Same as above | COCO-pretrained 4-scale variants | https://github.com/IDEA-Research/DINO |
| **Faster R-CNN** (torchvision) | Two-stage (RPN + RoI heads, ResNet50-FPN) | Strong localization accuracy; native variable image sizes (list of tensors, no resize to square); torchvision-integrated = one dependency | Slow inference (second stage); torchvision detection module officially "Beta"; more code to write than Ultralytics | **Medium** — heavier per-image than YOLO but trainable on T4/P100 at small batch | Moderate (FPN helps; aerial small objects still benefit from tiling) | **Best native** — accepts variable-size images without square resize | `fasterrcnn_resnet50_fpn(_v2)` COCO weights | https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html |
| **RetinaNet** (torchvision) | One-stage + FPN + focal loss | Simple, stable, torchvision-integrated; variable image sizes | Accuracy generally below modern YOLO/DETR; slower than YOLO | **Medium** | Moderate (FPN) | Good native (variable size) | `retinanet_resnet50_fpn(_v2)` COCO weights | https://docs.pytorch.org/vision/main/models/faster_rcnn.html (detection module) |
| **SAHI / tiling** | **Inference + fine-tuning technique** (not a model) | Slices large images (e.g., 512–1024px tiles with overlap), merges detections; **directly attacks the small-object + high-res problem**; model-agnostic | Slower inference (many tiles per image); hyperparameters (slice size, overlap) need tuning; double-detected objects in overlap zones must be merged (SAHI handles via NMM) | Low — inference-side; sliced fine-tuning optional | **Best-in-class for tiny objects in large images** | **Purpose-built for high-res** | N/A (wraps your model) | https://github.com/obss/sahi |

### A.3 Recommendation

**ASSESSMENT (technically argued, not popularity-driven):**

**Primary recommendation: YOLO11s or YOLO26s (Ultralytics) trained on SAHI-sliced crops, evaluated with SAHI sliced inference at full resolution.**

The argument, in order of what actually constrains this project:

1. **The problem is small objects in large images, and that is mostly an inference/data strategy, not an architecture choice.** Houses in high-res aerial imagery occupy a tiny fraction of pixels. Downsampling a 8000×6000 orthophoto to 640px destroys them. No backbone fixes this; slicing does. SAHI exists precisely for this, is model-agnostic, and has an official Ultralytics integration guide. This single decision matters more than YOLO-vs-DETR.

2. **Budget constraint eliminates the DETR family for the *primary* model.** RT-DETR's smallest Ultralytics weights are ~45M params vs YOLO26s at 9.5M (both docs-verified). DETR-family models are well known to need longer schedules and more data to converge (Deformable DETR's own motivation was DETR's slow convergence — repo cited above). On a free T4/P100 with ~10–50k images and one class, you get fewer experiment iterations per week with RT-DETR/DINO. Deformable DETR and DINO additionally carry research-grade tooling (manual CUDA op compilation, MMDetection configs) that burns free-GPU hours on setup, not science. They are reasonable *comparison baselines in the thesis*, not the daily driver.

3. **Why YOLO specifically (and it's not the popularity argument):** (a) lowest params/FLOPs per accuracy point of the verified table (YOLO26s: 48.6 mAP @ 9.5M params — better ratio than RT-DETR-l); (b) the only family among the candidates with a *one-line fine-tune API, native sliced-inference ecosystem (SAHI guide), integrated trackers for the video phase, and built-in mAP/confusion-matrix tooling* — this compresses engineering time, which is the true scarce resource on a free-tier budget; (c) YOLO26's STAL is explicitly a small-target label-assignment improvement, aligning the newest architecture with this exact problem.

4. **YOLO11 vs YOLO26 as the pick:** Either is defensible. YOLO11 is the "mature, stable" official recommendation; YOLO26 is the officially recommended choice for new projects and has STAL. **Pick: train YOLO11s first (maximum stability and troubleshooting resources for a student project), benchmark YOLO26s as a secondary experiment.** Do not use YOLO12 — Ultralytics itself flags its training instability and elevated memory on production workloads, which is the opposite of what a free-GPU budget wants. (Both docs cited in A.1.)

5. **Faster R-CNN is the strongest *fair-comparison* baseline** if the thesis needs a two-stage comparison: it is one `pip install` away inside torchvision, takes variable-size images natively, and has COCO weights. It is not the primary model because its two-stage inference is slow for tiled high-res inference.

**Rejected alternative worth recording:** DINO would give the best raw accuracy if budget were unlimited, but "best accuracy per GPU-hour" here belongs to YOLO+SAHI.

---

## TASK B — SAHI Specifics

**FACT — Repository:** SAHI (Slicing Aided Hyper Inference) by OBSS: https://github.com/obss/sahi — "Framework agnostic sliced/tiled inference + fine-tuning for small object detection."

**FACT — Paper:** Akyon, Altinuc, Temizel, *"Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection,"* IEEE ICIP 2022, pp. 966–970, DOI 10.1109/ICIP46576.2022.9897990. (Cited in the official Ultralytics SAHI guide: https://docs.ultralytics.com/guides/sahi-tiled-inference)

**FACT — Supported models (per repo/docs):** originally Detectron2, MMDetection, and YOLOv5 (per the paper abstract and repo docs); current releases support Ultralytics models via `AutoDetectionModel(model_type='ultralytics', ...)` — verified working in the official Ultralytics SAHI guide with YOLO26/YOLOv8 weights (e.g., `yolo26n.pt`, `-seg` variants).

**FACT — How slicing works:**

- **Inference** (`sahi.predict.get_sliced_prediction`): the large image is cut into overlapping slices (`slice_height`, `slice_width`, e.g., 512×512 with `overlap_height_ratio`/`overlap_width_ratio` = 0.2); the detector runs per-slice at its native resolution (so small objects stay large relative to the input); predictions are mapped back to full-image coordinates and overlapping duplicates merged (NMM/NMS). Full-size (untiled) prediction can also be blended in.
- **Training (sliced fine-tuning):** SAHI provides dataset slicing for fine-tuning the model on cropped tiles, matching the inference-time distribution — this is the "fine-tuning" half of the paper's title.

**FACT — Benefit for small objects:** The ICIP 2022 paper reports sliced inference **increased AP50 in all tested configurations**, with larger gains for YOLOv5 models; the technique is the standard approach for tiny-object aerial/drone detection and is integrated into Ultralytics' own official guide. Sources: repo above; Ultralytics guide above; paper DOI above.

**ASSESSMENT:** For house detection, SAHI is not optional — it is the core of the pipeline for full-resolution images. Practical setup (estimate, standard practice): slice 640–1024px with 0.2 overlap during training-crop generation; infer with the same slice geometry; count = number of merged detections above confidence threshold per image. Main cost: inference on a 8000×8000 image at 512/0.2 takes ~256 slice passes (est.) — acceptable offline, relevant for later video (mitigate with stride/ROI or larger slices).

---

## TASK C — Technology Stack Evaluation

Legend: **Required** = pipeline does not work without it. **Optional** = convenience/quality. **Alternative** = swap-in.

### Core

| Tool | Function in project | Status | Verdict |
|---|---|---|---|
| **Python 3** | Everything | Active | **Required** |
| **PyTorch** | Backend for Ultralytics/torchvision; CUDA training | Actively maintained, PyTorch Foundation | **Required** — https://github.com/pytorch/pytorch · https://pytorch.org/docs/stable/index.html |
| **Ultralytics** | YOLO11/26 + RT-DETR training, SAHI integration, tracking, val tooling | Very active (model tables on README current through YOLO26/YOLO27 discussion) | **Required** (if YOLO route) — https://github.com/ultralytics/ultralytics. Note **AGPL-3.0** license (docs cite AGPL-3.0 for YOLO11) — fine for academic use |
| **torchvision** | Faster R-CNN / RetinaNet baseline; transforms | Active, PyTorch Foundation (detection module flagged "Beta" in docs) | **Required for baselines**, else optional — https://docs.pytorch.org/vision/main/index.html |
| **OpenCV** (`opencv-python`) | Image I/O, tile extraction, video decode for later phase | Active | **Required** (tiles + video) — https://github.com/opencv/opencv |

### Data

| Tool | Function | Verdict |
|---|---|---|
| **NumPy** | Array ops everywhere | **Required** |
| **Pandas** | Per-image count tables, results aggregation | **Optional but cheap** (CSV reporting) |
| **Albumentations** | Augmentation pipelines (works with Ultralytics/bbox formats); aerial-specific augs (rotation, random crop) | **Optional** — Ultralytics has built-in aug; add Albumentations only for custom aerial augs — https://github.com/albumentations-team/albumentations |
| **Rasterio / GDAL** | Reading **GeoTIFF** orthophotos with georeferencing | **Conditional required**: needed *only* if source imagery is GeoTIFF (satellite orthophotos usually are); plain JPG/PNG drone/aerial datasets don't need it. GDAL installs are painful on some platforms; rasterio bundles wheels. If using GeoTIFF: rasterio required, GDAL via rasterio's bundled wheel, not direct. — https://github.com/rasterio/rasterio · https://gdal.org |

### Annotation

**FACT — all three verified current (2026):**

| Tool | Status | Verdict |
|---|---|---|
| **CVAT** | Open source, MIT, on GitHub since 2018, now under OpenCV ecosystem/cvat-ai; self-hosted is the full product | **Recommended primary** for bbox annotation; best video-annotation support for the later phase — https://github.com/cvat-ai/cvat · https://www.cvat.ai |
| **Label Studio** | Apache-2.0, active; broadest modality support; some QA/team features gated to Enterprise | **Alternative** (better for multimodal, this project is vision-only) — https://github.com/HumanSignal/label-studio |
| **Roboflow** | Proprietary SaaS; free tier; private data requires paying; no self-host of the platform | **Optional convenience** — dataset hosting/format conversion/preview auto-labeling; avoid for private data policy reasons — https://roboflow.com |

### Evaluation

**FACT — standard tooling, all built into Ultralytics `val` mode** (no extra dependency needed): COCO mAP (50-95, 50), precision, recall, IoU thresholds, per-class AP, and confusion matrix are all produced by `model.val()`; these are the COCO-standard metrics. For thesis-grade custom metrics (F1 per image, counting MAE), compute from predictions with NumPy. Sources: https://docs.ultralytics.com/modes/val · https://docs.ultralytics.com/usage/cfg (val settings incl. `confusion_matrix` plots). Add **pycocotools** (https://github.com/cocodataset/cocoapi) only if you want COCO-official evaluation independent of Ultralytics — optional.

### Experiment tracking

| Tool | Free tier | Verdict |
|---|---|---|
| **TensorBoard** | Fully free, local, built into Ultralytics (`settings tensorboard=True`) | **Required-by-default** (zero setup) — https://www.tensorflow.org/tensorboard |
| **Weights & Biases** | Free personal/academic tier (reported ~100 tracking hrs/month on free tier — third-party source) | **Optional**; nice run comparison UI — https://wandb.ai/site · https://github.com/wandb/wandb |
| **MLflow** | Fully free, open source, self-hosted (`uvx mlflow server`) | **Alternative** if you want local UI richer than TensorBoard — https://mlflow.org · https://github.com/mlflow/mlflow |

**ASSESSMENT:** TensorBoard alone is sufficient for a student project; add W&B only if comparing many runs across Colab/Kaggle (its cloud dashboard survives session resets better than local TensorBoard logs — a real practical issue on ephemeral machines).

---

## TASK D — Video Extension (Brief)

**FACT — Datasets verified:**
- **VisDrone**: 288 video clips (261,908 frames) + 10,209 static images, 14 cities in China; detection subset **VisDrone2019-DET** = 8,629 images (6,471 train / 548 val / 1,610 test-dev), 10 classes. Official repo: https://github.com/VisDrone/VisDrone-Dataset (TPAMI 2021 paper: Zhu et al., "Detection and Tracking Meet Drones Challenge"). Ultralytics has a built-in `VisDrone.yaml`: https://docs.ultralytics.com/datasets/detect/visdrone
- **UAVDT**: ~80,000 fully annotated frames from 10 hours of raw drone video, with 14 attribute types; supports object detection, single- and multi-object tracking. Listed in the official VisDrone "Awesome-VisDrone" index: https://github.com/VisDrone/Awesome-VisDrone · https://sites.google.com/site/daviddo0323/projects/uavdt

**FACT — Pipeline components verified:**
- Frame extraction → detection: OpenCV decode + Ultralytics predict (standard; no special tooling needed).
- Tracking: Ultralytics `model.track()` ships **six** trackers: BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker, and TrackTrack (default since v8.4.76, June 2026; previous default BoT-SORT). Official docs explicitly recommend: *"Handheld, drone, or moving-camera footage? → BoT-SORT (adds camera-motion compensation and optional ReID)."* Sources: https://docs.ultralytics.com/modes/track · https://docs.ultralytics.com/datasets/track
- **Duplicate suppression / unique counting:** tracking-based unique counting — count **unique track IDs** instead of per-frame detections. Official docs show exactly this pattern (`32 unique IDs` example in the tracker tuning guide). For fixed-wing/drone footage, BoT-SORT's camera-motion compensation (GMC) reduces ID churn; counts also need an ID-merge tolerance for tracker fragmentation (assessment).

**ASSESSMENT — recommendation:** **Yes, video is a later phase.** Reasons: (1) the image model is a strict prerequisite — the tracker consumes the detector's boxes, so detector quality bounds everything; (2) UAV house-counting video adds moving-camera geometry problems (GMC, parallax, re-detection of counted houses after camera revisits) that are a separate research problem from static counting; (3) the stack for it (VisDrone/UAVDT baselines + `model.track()` + BoT-SORT) is already integrated in the same Ultralytics dependency, so no new stack investment is needed now. Pin `tracker="bytetrack.yaml"` or `"botsort.yaml"` explicitly in code so a default-tracker change (it already changed once in June 2026) can't silently alter IDs.

---

## TASK E — Hardware / Training Estimates

> **Everything in this section is an ESTIMATE/ASSESSMENT.** Hardware *specs and quotas* are facts (cited); training times, VRAM headroom, and costs are derived estimates from verified model specs (params/FLOPs in A.1) and typical YOLO training throughput. No benchmark was run for this report.

### Verified hardware facts

**FACT — Google Colab free tier:** T4 GPU (16 GB VRAM) is the standard free-tier accelerator; ~12-hour max sessions; ~90-min idle disconnect; GPU access is dynamic/not guaranteed. Sources: https://www.gmicloud.ai/en/blog/best-free-gpu-cloud-options-for-ai-startups-and-researchers (summarizing Colab limits) · https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/

**FACT — Kaggle Notebooks:** ~30 GPU-hours/week quota, T4×2 or P100 (16 GB) hardware, 9–12 h sessions, no credit card. Sources: Kaggle community/docs as summarized in https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/ · https://www.kaggle.com/docs/efficient-gpu-usage

**FACT — RunPod (paid, no free tier for training):** per-second billing; RTX 4090 24 GB from $0.34/hr (Community) / $0.89 (Secure); A100 PCIe 80GB $1.19/hr Community; L4 24GB ~$0.39–0.59/hr. Source: https://www.runpod.io/pricing

### ESTIMATE — VRAM & training time, YOLO-class detector (s-variant ≈ 9.5M params, ~21 GFLOPs — specs verified in A.1)

Dataset: ~10–50k aerial images. Assumptions: images pre-sliced to tiles (640 or 1280 imgsz), single class, batch sized to fill GPU.

| Scenario | VRAM needed (est.) | Fits free tier? | Time per 100 epochs (est.) |
|---|---|---|---|
| YOLOs @ 640px, 10k imgs, Colab T4 | ~6–9 GB | ✅ T4 16 GB | ~4–8 h |
| YOLOs @ 640px, 50k imgs, Colab T4 | ~6–9 GB (batch-limited) | ✅ | ~20–40 h → split over sessions |
| YOLOs @ 1280px, 10k imgs, T4 | ~10–14 GB | ⚠️ tight, reduce batch to 4–8 | ~10–18 h |
| YOLOs @ 1280px, 50k imgs, Kaggle P100 | ~10–14 GB | ✅ but ~2 weeks of quota | ~35–70 h |
| YOLOm @ 1280px (secondary) | ~14–16+ GB | ⚠️ borderline on 16 GB | 2–3× YOLOs |
| RT-DETR-l fine-tune @ 640 (baseline only) | ~12–16 GB | ⚠️ | ~3–5× YOLOs (est.; 45M params vs 9.5M — both verified) |

**ESTIMATE — Storage:** 50k tiled 1280px JPGs ≈ 50–150 GB (est., JPEG q85) — exceeds Colab's ~100 GB ephemeral disk comfort zone and Kaggle's 20 GB working dir / 73 GB temp; plan for tiling on demand (stream from a mounted Google Drive of *original* images, tile per batch) or storing tiles on Drive/Kaggle Datasets.

**ASSESSMENT — platform comparison for this project:**

| Platform | Free capacity | Strength | Weakness |
|---|---|---|---|
| **Colab free** | T4, dynamic availability | Zero friction, Drive integration | No guaranteed quota; 12 h cap; session loss risk |
| **Kaggle** | ~30 GPU-h/wk, T4×2 or P100 | **Guaranteed quota** (publishable number), 20 GB persistent datasets | 9–12 h session; 20 GB working dir; slower I/O |
| **RunPod** | none free; 4090 from ~$0.34/hr | Full control, 24 GB VRAM, persistent volumes; a 40-h train ≈ **$14–35 (est.)** | Costs money; setup time |

**Practical plan (assessment):** prototype at 640px on Colab/Kaggle (2–3 weeks of free quota is enough for YOLOs-scale experiments per the table), and only rent a RunPod 4090 for the final 1280px full-dataset runs (~$20–40 total, est.) if free quotas prove insufficient. Combine Colab + Kaggle quotas ≈ 30–40+ free GPU-hrs/week (per the two FACT sources above).

---

## Summary of Recommendations

1. **Primary model:** YOLO11s (Ultralytics), secondary benchmark YOLO26s. Skip YOLO12 (officially flagged unstable). Current lineup verified: YOLO26 (Jan 2026, latest), YOLO11 (mature), YOLO12 (attention, community), YOLOE (open-vocab). RT-DETR available in Ultralytics (rtdetr-l/x) — use as comparison baseline only.
2. **Small-object strategy (the critical decision):** SAHI sliced training + sliced inference — verified, ICIP 2022, official Ultralytics guide.
3. **Baseline for thesis comparison:** torchvision Faster R-CNN R50-FPN (+ RetinaNet if a one-stage classic is wanted).
4. **Stack:** Python/PyTorch/Ultralytics/OpenCV required; rasterio(+GDAL) only if GeoTIFF; Albumentations/Pandas/W&B optional; CVAT for annotation; TensorBoard for tracking (built into Ultralytics).
5. **Video:** later phase, after image pipeline is validated; VisDrone/UAVDT as benchmarks; BoT-SORT for moving cameras via `model.track()`; unique-track-ID counting.

## Limitations of this research

- All training-time/VRAM/cost figures in Task E are **estimates from specs**, not measured benchmarks; validate with a 1-epoch smoke run on the target GPU before committing a schedule.
- YOLO26 vs YOLO11 relative performance has conflicting community reports (disclosed in A.1); no independent aerial-domain benchmark was found — hence the recommendation to run both as cheap experiments.
- Kaggle/Colab quotas and RunPod prices change frequently; figures verified 2026-11-09 from cited pages.

## AI Disclosure

This report was produced with AI-assisted research tooling (GLM 5.3 agent with web search/fetch verification). All FACT claims were verified against the cited official sources on 2026-11-09; ESTIMATES/ASSESSMENTS are labeled as such and were not externally verified.
