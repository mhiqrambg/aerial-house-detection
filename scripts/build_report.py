"""Build the research report (.docx) from run artifacts, locally.

Inputs live in outputs/report_assets/ (extracted from the Colab bundle):
metrics.json, counting_analysis.json, sahi_comparison.json, video_validation.json,
results.csv, and the figure PNGs.

Usage: python3 scripts/build_report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "report_assets"
OUT = ROOT / "docs" / "Laporan_Aerial_House_Detection.docx"

mt = json.loads((A / "experiments/eval_test/metrics.json").read_text())
ct = json.loads((A / "experiments/eval_test/counting_analysis.json").read_text())
sp = json.loads((A / "experiments/sahi_comparison.json").read_text())
vv = json.loads((A / "experiments/video_validation.json").read_text())
rows_csv = list(csv.DictReader((A / "experiments/exp01_yolo11s_baseline/results.csv").open()))


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def n(value: float, digits: int = 4) -> str:
    """Indonesian decimal comma."""
    return f"{value:.{digits}f}".replace(".", ",")


# --- extra chart with PIL: dataset split ---
def dataset_chart(path: Path) -> None:
    W, H = 1100, 420
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
        small = font
    splits = ["Train", "Validasi", "Test"]
    images = [5732, 1228, 1228]
    boxes = [143470, 31538, 30206]
    d.text((40, 18), "Jumlah citra dan bounding box per split", fill="black", font=font)
    for i, (s, n, b) in enumerate(zip(splits, images, boxes)):
        x0 = 60 + i * 340
        h_img = int(n / max(images) * 130)
        d.rectangle([x0, 330 - h_img, x0 + 90, 330], fill=(52, 120, 190))
        d.text((x0 + 10, 340), f"{s}", fill="black", font=small)
        d.text((x0 + 10, 362), f"{n:,} citra".replace(",", "."), fill="black", font=small)
        h_box = int(b / max(boxes) * 130)
        d.rectangle([x0 + 120, 330 - h_box, x0 + 210, 330], fill=(214, 120, 60))
        d.text((x0 + 120, 340), "box", fill="black", font=small)
        d.text((x0 + 120, 362), f"{b:,}".replace(",", "."), fill="black", font=small)
    d.line([40, 330, 1060, 330], fill=(120, 120, 120), width=2)
    d.text((40, 390), "Biru: citra. Jingga: bounding box. Skala tiap panel dinormalkan ke nilai maksimum.",
           fill=(90, 90, 90), font=small)
    img.save(path)


def count_error_chart(path: Path) -> None:
    """Histogram of |count error| and predicted-vs-truth scatter, drawn with PIL."""
    counts = list(csv.DictReader((A / "experiments/eval_test/per_image_counts.csv").open()))
    errs = [abs(int(r["err"])) for r in counts]
    pairs = [(int(r["n_gt"]), int(r["n_pred"])) for r in counts]
    W, H = 1200, 430
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()

    # left panel: histogram of absolute error, clipped at 20
    cap = 20
    bins = [0] * (cap + 1)
    for e in errs:
        bins[min(e, cap)] += 1
    ox, oy, pw, ph = 70, 350, 480, 280
    d.rectangle([ox, oy - ph, ox + pw, oy], outline=(80, 80, 80))
    peak = max(bins)
    bw = pw / len(bins)
    for i, v in enumerate(bins):
        hh = int(v / peak * (ph - 10))
        d.rectangle([ox + i * bw + 1, oy - hh, ox + (i + 1) * bw - 1, oy], fill=(52, 120, 190))
    d.text((ox, oy - ph - 32), "Distribusi |error| hitungan per citra", fill="black", font=font)
    d.text((ox, oy + 8), "0", fill="black", font=small)
    d.text((ox + pw - 42, oy + 8), f"{cap}+", fill="black", font=small)
    d.text((ox + 150, oy + 12), "rumah", fill="black", font=small)
    d.text((ox - 52, oy - ph - 4), str(peak), fill="black", font=small)
    d.text((ox - 44, oy - 16), "0", fill="black", font=small)

    # right panel: predicted vs truth scatter
    ox2, oy2, pw2, ph2 = 700, 350, 400, 280
    d.rectangle([ox2, oy2 - ph2, ox2 + pw2, oy2], outline=(80, 80, 80))
    top = max(max(p[0], p[1]) for p in pairs) or 1
    for gt, pred in pairs:
        x = ox2 + int(gt / top * (pw2 - 4))
        y = oy2 - int(pred / top * (ph2 - 4))
        d.ellipse([x, y, x + 3, y + 3], fill=(60, 60, 60))
    d.line([ox2, oy2, ox2 + pw2, oy2 - ph2], fill=(200, 60, 60), width=2)
    d.text((ox2, oy2 - ph2 - 32), "Prediksi vs ground truth per citra", fill="black", font=font)
    d.text((ox2, oy2 + 8), "ground truth", fill="black", font=small)
    d.text((ox2 - 52, oy2 - ph2 - 4), str(top), fill="black", font=small)

    img.save(path)


chart_path = ROOT / "outputs" / "report_dataset_chart.png"
dataset_chart(chart_path)
count_chart_path = ROOT / "outputs" / "report_count_error.png"
count_error_chart(count_chart_path)

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(10.5)
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.9)
    s.top_margin = s.bottom_margin = Inches(0.8)


def h(text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def p(text: str) -> None:
    par = doc.add_paragraph()
    par.add_run(text)
    par.paragraph_format.space_after = Pt(6)


def table(headers, rows, caption=None, font=9):
    if caption:
        cp = doc.add_paragraph()
        run = cp.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hd in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(hd))
        run.bold = True
        run.font.size = Pt(font)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(font)
    doc.add_paragraph()


def figure(path: Path, caption: str, width: float = 6.3) -> None:
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp = doc.add_paragraph()
    run = cp.add_run(caption)
    run.italic = True
    run.font.size = Pt(8.5)
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER


# cover
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = tp.add_run("Laporan Hasil Penelitian")
run.bold = True
run.font.size = Pt(24)
tp2 = doc.add_paragraph()
tp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = tp2.add_run("Deteksi Rumah pada Citra Aerial Berbasis YOLO11s dan SAHI")
run.bold = True
run.font.size = Pt(16)
doc.add_paragraph()
table(
    ["Item", "Keterangan"],
    [
        ["Dataset", "WHU Building Dataset, aerial subset (Christchurch, Selandia Baru)"],
        ["Model", "YOLO11s (Ultralytics), fine-tuned 30 epoch"],
        ["Lingkungan", "Google Colab, Tesla T4 16 GB, PyTorch 2.14.1, CUDA 13.0"],
        ["Tanggal eksekusi", "9 November 2026"],
        ["Repositori", "src/, scripts/, configs/, tests/, notebooks/"],
    ],
)

h("1. Ringkasan Eksekutif")
p(
    "Penelitian ini membangun dan mengevaluasi sistem deteksi rumah pada citra tampak atas. Sistem "
    "menerima citra aerial, menghasilkan bounding box dan confidence score per rumah, lalu menghitung "
    "jumlah rumah unik per citra. Model yang dipakai adalah YOLO11s, dipilih pada tahap riset karena "
    "rasio akurasi per jam GPU paling baik pada anggaran satu GPU gratis. SAHI dipakai untuk citra "
    "beresolusi tinggi."
)
p(
    f"Pada test split yang tidak dipakai selama pelatihan, model mencapai precision {n(mt['precision'])}, "
    f"recall {n(mt['recall'])}, F1 {n(mt['f1'])}, mAP@0.5 {n(mt['map50'])}, dan mAP@0.5:0.95 "
    f"{n(mt['map'])}. Waktu inferensi rata-rata 9,5 ms per citra 640 px pada T4."
)
p(
    f"Untuk penghitungan rumah, error absolut rata-rata {n(ct['count_mae'], 2)} rumah per citra "
    f"({n(ct['count_mae_pct'], 2)}% error relatif), dengan {ct['dup']} deteksi duplikat pada 30.206 "
    f"instance. Recall objek kecil di bawah 32 px sebesar {n(ct['small_object_recall'])}."
)
p(
    "Uji resolusi tinggi menunjukkan slicing tidak selalu menang. Pada mosaic 1024 px, inferensi "
    "langsung lebih baik (F1 0,882 berbanding 0,733). Pada mosaic 2048 px, inferensi langsung "
    "kehilangan 232 dari 463 rumah, sedangkan SAHI slice 512 melewatkan 78 rumah dan menekan count "
    "error menjadi 20."
)

h("2. Dataset")
p(
    "Data berasal dari WHU Building Dataset, subset aerial, melalui mirror Hugging Face "
    "giswqs/WHU-Building-Dataset. Kartu dataset menyatakan lisensi CC BY 4.0, sedangkan halaman "
    "resmi WHU tidak mencantumkan lisensi. Ketentuan asli perlu dikonfirmasi sebelum publikasi atau "
    "redistribusi."
)
p(
    "Dataset berisi 8.188 tile 512 x 512 piksel pada ground sampling distance 0,3 m. Setiap tile "
    "memiliki satu mask biner. Pembagian split mengikuti mirror dan tidak diubah."
)
table(
    ["Split", "Citra", "Bounding box", "Citra tanpa objek"],
    [
        ["Train", "5.732", "143.470", "975"],
        ["Validasi", "1.228", "31.538", "224"],
        ["Test", "1.228", "30.206", "225"],
        ["Total", "8.188", "205.214", "1.424"],
    ],
    caption="Tabel 1. Statistik dataset setelah konversi mask ke bounding box.",
)
figure(chart_path, "Gambar 1. Jumlah citra dan bounding box per split.")
figure(A / "outputs/eda_bbox_sizes.png", "Gambar 2. Distribusi ukuran bounding box pada sampel eksplorasi data.")
figure(
    A / "outputs/eda_gt_samples.png",
    "Gambar 3. Contoh ground truth pada tiga tile latih. Kotak merah adalah bounding box hasil konversi mask.",
)

h("3. Konversi Anotasi")
p(
    "WHU menyediakan mask raster, bukan poligon instance. Setiap komponen terhubung 8-arah diubah "
    "menjadi satu bounding box dengan cv2.connectedComponentsWithStats. Komponen di bawah 64 piksel "
    "persegi dibuang sebagai noise. Konversi divalidasi dengan uji sintetis: dua persegi panjang "
    "diketahui harus menghasilkan dua box dengan koordinat tepat, dan IoU box terhadap komponennya "
    "harus 1,0."
)
p(
    "Konsekuensinya, atap yang saling menempel membentuk satu komponen dan menjadi satu box. Deret "
    "rumah padat akan dihitung kurang. Masalah ini terlihat pada analisis error di Bagian 7."
)

h("4. Model dan Konfigurasi Pelatihan")
p("Model dasar YOLO11s dengan bobot COCO. Konfigurasi disimpan di configs/train_baseline.yaml.")
table(
    ["Parameter", "Nilai"],
    [
        ["Arsitektur", "YOLO11s, 9.428.179 parameter, 21,4 GFLOPs"],
        ["Bobot awal", "yolo11s.pt (COCO)"],
        ["Ukuran input", "640 x 640"],
        ["Batch", "16"],
        ["Epoch", "30, patience 10"],
        ["Seed", "42, deterministic"],
        ["Augmentasi", "Default Ultralytics: HSV, flip horizontal dan vertikal, skala, mosaic"],
        ["VRAM terpakai", "5,49 GB"],
        ["Durasi pelatihan", "3.931 detik (sekitar 65 menit)"],
        ["Perangkat", "Tesla T4 16 GB"],
    ],
    caption="Tabel 2. Konfigurasi eksperimen exp01_yolo11s_baseline.",
)

h("5. Hasil Pelatihan")
last = rows_csv[-1]
p(
    f"Kurva pelatihan disimpan per epoch. Loss pelatihan dan validasi sama-sama menurun sampai epoch "
    f"terakhir, sehingga tidak ada indikasi overfitting pada run ini. Pada epoch {int(last['epoch'])}, "
    f"mAP@0.5 validasi {n(float(last['metrics/mAP50(B)']))} dan mAP@0.5:0.95 "
    f"{n(float(last['metrics/mAP50-95(B)']))}."
)
figure(
    A / "experiments/exp01_yolo11s_baseline/training_history.png",
    "Gambar 4. Riwayat pelatihan per epoch: loss, precision dan recall, mAP, serta learning rate.",
    width=6.5,
)

h("6. Evaluasi pada Test Set")
p(
    "Evaluasi memakai test split yang tidak dipakai untuk memilih hyperparameter. Threshold "
    "confidence 0,25 dan IoU 0,6 mengikuti pengaturan Ultralytics val."
)
table(
    ["Metrik", "Nilai"],
    [
        ["Precision", f"{n(mt['precision'])}"],
        ["Recall", f"{n(mt['recall'])}"],
        ["F1", f"{n(mt['f1'])}"],
        ["mAP@0.5", f"{n(mt['map50'])}"],
        ["mAP@0.5:0.95", f"{n(mt['map'])}"],
        ["Waktu inferensi", "9,5 ms per citra 640 px (T4)"],
    ],
    caption="Tabel 3. Metrik pada test split (1.228 citra, 30.206 instance).",
)
figure(A / "experiments/eval_test/BoxPR_curve.png", "Gambar 5. Kurva precision-recall pada test set.", width=4.6)
figure(A / "experiments/eval_test/confusion_matrix.png", "Gambar 6. Confusion matrix pada test set.", width=4.4)

h("6.1 Analisis kualitatif")
p(
    "Gambar 7 membandingkan ground truth, prediksi, dan overlay pada tiga tile test, termasuk dua "
    "tile terpadat. Kesalahan yang terlihat: rumah berdekatan kadang digabung menjadi satu deteksi, "
    "dan sebagian bangunan kecil di tepi tile terlewat."
)
figure(
    A / "outputs/eval_gt_vs_pred.png",
    "Gambar 7. Ground truth (merah), prediksi (hijau), dan overlay pada tiga tile test.",
    width=6.5,
)

h("7. Analisis Kesalahan dan Penghitungan Rumah")
p(
    "Penghitungan dilakukan setelah penggabungan prediksi, sehingga satu rumah tidak dihitung "
    "berulang karena muncul di dua tile yang overlap. Pencocokan prediksi ke ground truth memakai "
    "IoU 0,5 secara greedy."
)
table(
    ["Metrik", "Nilai"],
    [
        ["True positive", fmt_int(ct["tp"])],
        ["False positive", fmt_int(ct["fp"])],
        ["False negative", fmt_int(ct["fn"])],
        ["Deteksi duplikat", str(ct["dup"])],
        ["Count MAE", f"{n(ct['count_mae'], 2)} rumah per citra"],
        ["Error relatif rata-rata", f"{n(ct['count_mae_pct'], 2)}%"],
        ["Recall objek kecil (< 32 px)", f"{n(ct['small_object_recall'])}"],
        ["Precision jalur analisis", f"{n(ct['precision'])}"],
        ["Recall jalur analisis", f"{n(ct['recall'])}"],
        ["F1 jalur analisis", f"{n(ct['f1'])}"],
    ],
    caption="Tabel 4. Analisis penghitungan pada test set (confidence 0,25, IoU 0,5).",
)
p(
    "Precision pada jalur analisis (0,8933) berbeda dari precision pada val (0,9323) karena jalur "
    "pencocokan dan threshold yang dipakai berbeda. Keduanya dilaporkan tanpa penyesuaian."
)
figure(
    count_chart_path,
    "Gambar 8. Distribusi error hitungan per citra dan hubungan jumlah prediksi dengan ground truth.",
)

h("8. Inferensi Resolusi Tinggi")
p(
    "Pengujian memakai mosaic dari tile test, dengan ground truth dipetakan ke koordinat mosaic. Dua "
    "ukuran diuji: 1024 px (2 x 2 tile, 186 rumah) dan 2048 px (4 x 4 tile, 463 rumah)."
)
rows = []
for name, label in [("mosaic_1024", "1024 px (186 GT)"), ("mosaic_2048", "2048 px (463 GT)")]:
    v = sp[name]
    for key in [k for k in v if k == "direct" or k.startswith("sahi")]:
        d = v[key]
        names = {"direct": "Direct", "sahi_512": "SAHI 512", "sahi_640": "SAHI 640"}
        rows.append(
            [label, names.get(key, key), d["pred"], d["tp"], d["fp"], d["fn"], d["count_err"], f"{n(d['f1'], 3)}"]
        )
table(
    ["Input", "Metode", "Pred", "TP", "FP", "FN", "Error hitung", "F1"],
    rows,
    caption="Tabel 5. Perbandingan inferensi langsung dan SAHI pada citra resolusi tinggi.",
)
p(
    "Pada 1024 px, inferensi langsung menang karena objek masih cukup besar relatif terhadap input "
    "model, sementara slicing menambah false positive di perbatasan slice. Pada 2048 px, objek "
    "menjadi terlalu kecil setelah seluruh citra diperkecil ke 640 px, sehingga inferensi langsung "
    "kehilangan lebih dari separuh rumah. Slice 640 tidak lebih baik dari 512 pada kedua kondisi."
)
figure(
    A / "outputs/mosaic_1024_counted.png",
    "Gambar 9. Hasil penghitungan pada mosaic 1024 px dengan SAHI. Angka di pojok menunjukkan jumlah rumah unik setelah penggabungan.",
    width=5.2,
)

h("9. Modul Video")
p(
    "Modul video menghitung identitas unik melalui tracking, bukan menjumlahkan deteksi per frame. "
    "Tracker yang dipakai BoT-SORT karena mendukung kompensasi gerak kamera. Tidak ada dataset video "
    "rumah UAV, sehingga validasi dilakukan pada video panning sintetis dari mosaic 2048 px."
)
table(
    ["Metrik", "Nilai"],
    [
        ["Jumlah frame", str(vv["frames"])],
        ["Deteksi per frame (naif)", fmt_int(vv["sum_per_frame_detections"])],
        ["Unique track ID", str(vv["unique_ids"])],
        ["Rumah pada scene", str(vv["gt_houses_in_scene"])],
    ],
    caption="Tabel 6. Validasi pipeline video pada klip sintetis 40 frame.",
)
p(
    "Selisih antara jumlah deteksi per frame dan unique track ID menunjukkan besarnya duplikasi bila "
    "penghitungan dilakukan tanpa tracking. Angka 188 bukan benchmark karena klip sintetis hanya "
    "menampilkan sebagian scene dan identitas track dapat terpecah saat kamera bergerak."
)

h("10. Identitas Model")
table(
    ["Item", "Nilai"],
    [
        ["Nama eksperimen", "exp01_yolo11s_baseline"],
        ["Jumlah parameter", "9.428.179"],
        ["Ukuran best.pt", "19,18 MB"],
        ["Ukuran best.onnx", "37,93 MB"],
        ["SHA256 best.pt", "b3df0c3fdb5f24ce0bad5fc90ee26807fa4ac1c594418f61654167a4b36116ae"],
        ["Kelas", "building (satu kelas)"],
    ],
    caption="Tabel 7. Identitas berkas model.",
)

h("11. Diskusi dan Keterbatasan")
p(
    "Hasil test menunjukkan model mampu mendeteksi bangunan pada citra 0,3 m dengan akurasi tinggi. "
    "Selisih antara mAP@0.5 validasi (0,9697) dan test (0,9342) konsisten dengan evaluasi pada area "
    "yang tidak dipakai saat pelatihan, meskipun mirror tidak menyediakan koordinat untuk memastikan."
)
p("Keterbatasan yang paling berpengaruh pada penggunaan nyata:")
for item in [
    "Rumah menempel atau berbagi atap menyatu menjadi satu komponen mask dan dihitung sebagai satu rumah.",
    "Kelas dataset bangunan generik. Bangunan komersial dan industri ikut terdeteksi.",
    "Seluruh data dari satu kota. Generalisasi lintas wilayah dan sensor belum diuji.",
    "Tidak ada dataset video rumah UAV. Modul video baru tervalidasi secara mekanis.",
    "Eksperimen 1280 px, variasi augmentasi, dan pembanding YOLO26s serta Faster R-CNN belum dijalankan.",
]:
    doc.add_paragraph(item, style="List Bullet")

h("12. Rekomendasi")
for item in [
    "Evaluasi zero-shot pada SpaceNet 2 dan kota test Inria untuk mengukur generalisasi.",
    "Melatih ulang pada sliced crops dan menguji input 1280 px untuk menekan false positive SAHI.",
    "Menambah anotasi lokal 200 sampai 500 box untuk fine-tuning wilayah target.",
    "Untuk video nyata, deduplikasi memakai posisi geospasial atau centroid, bukan hanya track ID.",
    "Bila rumah menempel penting, pertimbangkan instance segmentation atau oriented bounding box.",
]:
    doc.add_paragraph(item, style="List Bullet")

h("13. Referensi")
for ref in [
    "Ji, S., Wei, S., Lu, M. Fully Convolutional Networks for Multisource Building Extraction From "
    "Mismatched Cross-Resolution Satellite Imagery. IEEE Transactions on Geoscience and Remote "
    "Sensing 57(1), 2019.",
    "Akyon, F. C., Altinuc, S. O., Temizel, A. Slicing Aided Hyper Inference and Fine-tuning for "
    "Small Object Detection. IEEE ICIP 2022. https://arxiv.org/abs/2202.06934",
    "Ultralytics. YOLO11 documentation. https://docs.ultralytics.com/models/yolo11",
    "SAHI repository. https://github.com/obss/sahi",
    "WHU Building Dataset mirror. https://huggingface.co/datasets/giswqs/WHU-Building-Dataset",
]:
    par = doc.add_paragraph(ref)
    par.paragraph_format.space_after = Pt(4)

h("14. Lampiran Berkas")
table(
    ["Berkas", "Isi"],
    [
        ["best.pt", "Bobot PyTorch hasil pelatihan"],
        ["best.onnx", "Versi ONNX untuk deployment"],
        ["results.csv", "Metrik per epoch"],
        ["metrics.json", "Metrik test set"],
        ["counting_analysis.json", "TP, FP, FN, duplikat, MAE"],
        ["sahi_comparison.json", "Hasil uji direct dan SAHI"],
        ["video_validation.json", "Validasi pipeline video"],
        ["experiment_summary.csv", "Rekap seluruh eksperimen"],
        ["eval_gt_vs_pred.png", "Perbandingan ground truth dan prediksi"],
    ],
    caption="Tabel 8. Berkas yang menyertai laporan.",
)

doc.save(OUT)
print("saved:", OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
