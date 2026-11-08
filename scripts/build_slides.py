"""Build a 10-slide presentation from the report artifacts.

Usage: .venv/bin/python scripts/build_slides.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "outputs" / "report_assets"
OUT = ROOT / "docs" / "Slides_Aerial_House_Detection.pptx"

mt = json.loads((A / "experiments/eval_test/metrics.json").read_text())
ct = json.loads((A / "experiments/eval_test/counting_analysis.json").read_text())
sp = json.loads((A / "experiments/sahi_comparison.json").read_text())

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT2 = RGBColor(0xC5, 0x5A, 0x11)
GRAY = RGBColor(0x40, 0x40, 0x40)
LIGHT = RGBColor(0xEA, 0xF1, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def box(slide, left, top, width, height, fill=None, line=None):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    return shape


def text(slide, left, top, width, height, content, size=18, bold=False, color=GRAY,
         align=PP_ALIGN.LEFT, font="Calibri", spacing=6):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    lines = content if isinstance(content, list) else [content]
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(spacing)
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return tb


def bullets(slide, left, top, width, height, items, size=17, color=GRAY, spacing=10):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(spacing)
        run = para.add_run()
        run.text = "•  " + item
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return tb


def header(slide, title, subtitle=None, number=None):
    box(slide, 0, 0, 13.333, 0.12, fill=ACCENT)
    text(slide, 0.7, 0.35, 11.9, 0.6, title, size=28, bold=True, color=ACCENT)
    if subtitle:
        text(slide, 0.7, 0.95, 11.9, 0.4, subtitle, size=14, color=RGBColor(0x60, 0x60, 0x60))
    if number is not None:
        text(slide, 12.2, 7.05, 0.8, 0.3, str(number), size=11, color=RGBColor(0x90, 0x90, 0x90),
             align=PP_ALIGN.RIGHT)


def picture(slide, path, left, top, width=None, height=None):
    kw = {}
    if width:
        kw["width"] = Inches(width)
    if height:
        kw["height"] = Inches(height)
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kw)


def table(slide, headers, rows, left, top, width, height, font=12, col_widths=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tbl = shape.table
    for i, hd in enumerate(headers):
        cell = tbl.cell(0, i)
        cell.text = ""
        run = cell.text_frame.paragraphs[0].add_run()
        run.text = str(hd)
        run.font.size = Pt(font)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = "Calibri"
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            run = cell.text_frame.paragraphs[0].add_run()
            run.text = str(value)
            run.font.size = Pt(font)
            run.font.color.rgb = GRAY
            run.font.name = "Calibri"
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    return tbl


# ---------------- 1. cover ----------------
s = prs.slides.add_slide(BLANK)
box(s, 0, 0, 13.333, 7.5, fill=ACCENT)
box(s, 0, 4.55, 13.333, 0.06, fill=ACCENT2)
text(s, 0.9, 1.6, 11.5, 1.0, "Deteksi Rumah pada Citra Aerial", size=44, bold=True, color=WHITE)
text(s, 0.9, 2.7, 11.5, 0.6, "YOLO11s dengan SAHI untuk citra resolusi tinggi", size=24,
     color=RGBColor(0xC9, 0xD8, 0xE8))
text(s, 0.9, 3.5, 11.5, 0.5, "Laporan Hasil Penelitian", size=18, color=WHITE)
text(s, 0.9, 5.1, 11.5, 1.4,
     ["Dataset: WHU Building, aerial subset (Christchurch, 0,3 m)",
      "Model: 9,43 juta parameter  |  mAP@0.5 test 0,9342",
      "9 November 2026  ·  Google Colab T4"],
     size=15, color=RGBColor(0xD9, 0xE2, 0xEC), spacing=8)

# ---------------- 2. background ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Latar Belakang dan Tujuan", "Mengapa deteksi rumah dari udara masih sulit", 2)
bullets(s, 0.8, 1.55, 6.2, 4.6, [
    "Penghitungan rumah dari citra udara dibutuhkan untuk pemetaan cepat, namun penandaan manual lambat.",
    "Rumah menempati sebagian kecil piksel pada citra resolusi tinggi, sehingga objek kecil mudah hilang.",
    "Atap yang berdekatan menyatu, variasi bentuk dan warna atap menyulitkan pemisahan.",
    "Perbedaan sensor satelit dan drone menggeser distribusi gambar.",
], size=17)
box(s, 7.3, 1.6, 5.2, 4.2, fill=LIGHT)
text(s, 7.6, 1.85, 4.6, 0.4, "Tujuan penelitian", size=18, bold=True, color=ACCENT)
bullets(s, 7.6, 2.35, 4.6, 3.2, [
    "Deteksi setiap rumah dengan bounding box dan confidence score.",
    "Menghitung rumah unik per citra setelah penggabungan prediksi.",
    "Menangani citra berukuran jauh lebih besar dari input model.",
    "Mengevaluasi secara kuantitatif pada test set yang tidak dipakai saat pelatihan.",
], size=14, spacing=8)

# ---------------- 3. dataset ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Dataset", "WHU Building Dataset, subset aerial", 3)
bullets(s, 0.8, 1.5, 5.6, 4.6, [
    "8.188 tile 512 x 512 piksel, ground sampling distance 0,3 m.",
    "Mask biner per tile, dikonversi ke 205.214 bounding box.",
    "Split tetap: 5.732 train, 1.228 validasi, 1.228 test.",
    "Lisensi CC BY 4.0 dinyatakan pada mirror Hugging Face, bukan halaman resmi WHU.",
    "17% gambar tidak berisi objek, tetap dipakai sebagai contoh negatif.",
], size=16)
picture(s, ROOT / "outputs" / "report_dataset_chart.png", 6.7, 1.5, width=6.0)
text(s, 6.7, 4.35, 6.0, 1.4,
     "Median bangunan 41 x 40 px pada tile 512 px. Sepuluh persen bangunan lebih kecil dari 17 px, "
     "yang menjadi alasan pengujian slicing.",
     size=13, color=RGBColor(0x60, 0x60, 0x60))

# ---------------- 4. pipeline ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Alur Sistem", "Dari citra mentah ke jumlah rumah", 4)
steps = [
    ("Citra aerial", "JPG / PNG / GeoTIFF"),
    ("Praproses", "Tiling dan normalisasi"),
    ("YOLO11s", "Deteksi per tile"),
    ("Penggabungan", "NMM antar tile"),
    ("Hitung dan visualisasi", "Bounding box, count"),
]
for i, (title, sub) in enumerate(steps):
    left = 0.55 + i * 2.55
    box(s, left, 2.15, 2.2, 1.75, fill=LIGHT, line=ACCENT)
    text(s, left + 0.12, 2.35, 1.95, 0.6, title, size=14, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    text(s, left + 0.12, 2.95, 1.95, 0.6, sub, size=11, color=GRAY, align=PP_ALIGN.CENTER)
    if i < len(steps) - 1:
        text(s, left + 2.22, 2.8, 0.35, 0.4, "›", size=26, bold=True, color=ACCENT2)
box(s, 0.55, 4.45, 12.2, 1.15, fill=RGBColor(0xF7, 0xF9, 0xFB), line=RGBColor(0xD0, 0xD7, 0xDE))
text(s, 0.8, 4.62, 11.7, 0.95,
     ["Konversi mask ke bounding box memakai connected components 8-arah, dengan uji geometri: "
      "koordinat box harus tepat dan IoU terhadap komponen 1,0.",
      "Penghitungan dilakukan setelah penggabungan, sehingga rumah yang muncul di dua tile tidak dihitung dua kali."],
     size=13, color=GRAY, spacing=6)

# ---------------- 5. model config ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Model dan Konfigurasi", "YOLO11s, fine-tuning dari bobot COCO", 5)
table(s, ["Parameter", "Nilai"], [
    ["Arsitektur", "YOLO11s, 9.428.179 parameter, 21,4 GFLOPs"],
    ["Ukuran input", "640 x 640"],
    ["Batch / epoch", "16 / 30 (patience 10)"],
    ["Seed", "42, deterministic"],
    ["Augmentasi", "HSV, flip, skala, mosaic"],
    ["VRAM", "5,49 GB dari 16 GB"],
    ["Durasi", "3.931 detik (65 menit) pada T4"],
], left=0.8, top=1.55, width=7.6, height=3.9, font=13)
box(s, 8.8, 1.6, 4.0, 3.8, fill=LIGHT)
text(s, 9.05, 1.8, 3.5, 0.4, "Alasan pemilihan", size=16, bold=True, color=ACCENT)
text(s, 9.05, 2.3, 3.5, 2.9,
     ["Masalah utama adalah objek kecil pada gambar besar, dan itu diatasi oleh strategi slicing.",
      "Pada anggaran satu GPU gratis, YOLO memberi akurasi per jam GPU terbaik dibanding keluarga DETR.",
      "Ekosistem SAHI, tracking, dan evaluasi sudah terintegrasi."],
     size=12, color=GRAY, spacing=10)

# ---------------- 6. training ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Hasil Pelatihan", "Kurva per epoch, bukan hanya nilai akhir", 6)
picture(s, A / "experiments/exp01_yolo11s_baseline/training_history.png", 0.6, 1.45, width=8.6)
box(s, 9.5, 1.6, 3.3, 4.3, fill=LIGHT)
text(s, 9.7, 1.8, 3.0, 0.4, "Validasi epoch 30", size=15, bold=True, color=ACCENT)
text(s, 9.7, 2.25, 3.0, 0.5, "0,9697", size=32, bold=True, color=ACCENT)
text(s, 9.7, 2.75, 3.0, 0.3, "mAP@0.5", size=12, color=GRAY)
text(s, 9.7, 3.25, 3.0, 0.5, "0,8233", size=32, bold=True, color=ACCENT2)
text(s, 9.7, 3.75, 3.0, 0.3, "mAP@0.5:0.95", size=12, color=GRAY)
text(s, 9.7, 4.4, 3.0, 1.3,
     "Loss latih dan validasi turun bersama sampai epoch terakhir, tanpa indikasi overfitting.",
     size=12, color=GRAY)

# ---------------- 7. test metrics ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Evaluasi pada Test Set", "1.228 citra dan 30.206 instance yang tidak dipakai saat pelatihan", 7)
table(s, ["Metrik", "Nilai"], [
    ["Precision", fmt(mt["precision"], 4)],
    ["Recall", fmt(mt["recall"], 4)],
    ["F1", fmt(mt["f1"], 4)],
    ["mAP@0.5", fmt(mt["map50"], 4)],
    ["mAP@0.5:0.95", fmt(mt["map"], 4)],
    ["Inferensi", "9,5 ms per citra"],
], left=0.7, top=1.6, width=4.3, height=3.4, font=12)
picture(s, A / "experiments/eval_test/BoxPR_curve.png", 5.2, 1.5, height=3.3)
picture(s, A / "experiments/eval_test/confusion_matrix.png", 8.9, 1.5, height=3.3)
text(s, 0.7, 5.25, 11.9, 1.2,
     "Kurva PR dan confusion matrix disimpan dari Ultralytics val. Selisih mAP@0.5 validasi (0,9697) "
     "dan test (0,9342) konsisten dengan evaluasi pada area yang tidak dipakai saat pelatihan.",
     size=13, color=RGBColor(0x60, 0x60, 0x60))

# ---------------- 8. counting ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Penghitungan Rumah", "Akurasi hitungan, bukan hanya akurasi deteksi", 8)
picture(s, ROOT / "outputs" / "report_count_error.png", 0.45, 1.45, width=8.15)
kpis = [
    ("1,79", "MAE rumah per citra"),
    ("11,12%", "error relatif rata-rata"),
    ("43", "deteksi duplikat"),
    ("0,8557", "recall objek < 32 px"),
]
for i, (value, label) in enumerate(kpis):
    top = 1.55 + i * 1.25
    box(s, 8.7, top, 4.1, 1.05, fill=LIGHT)
    text(s, 8.9, top + 0.08, 1.9, 0.5, value, size=24, bold=True, color=ACCENT)
    text(s, 9.0, top + 0.62, 3.6, 0.35, label, size=11, color=GRAY)
text(s, 0.7, 4.55, 7.9, 0.8,
     "TP 27.944, FP 3.338, FN 2.262. Precision jalur analisis 0,8933, berbeda dari val karena "
     "pencocokan dan threshold berbeda.",
     size=11, color=RGBColor(0x60, 0x60, 0x60))
bullets(s, 0.7, 5.25, 7.9, 1.4, [
    "Sebagian besar citra salah hitung 0 sampai 2 rumah, ekor panjang berasal dari area padat.",
    "Prediksi mengikuti garis diagonal; bias utama adalah kurang hitung saat atap menyatu.",
], size=12, spacing=6)

# ---------------- 9. high res ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Inferensi Resolusi Tinggi", "Slicing tidak selalu menang", 9)
rows = []
for key, label in [("mosaic_1024", "1024 px"), ("mosaic_2048", "2048 px")]:
    v = sp[key]
    rows.append([label + " (GT " + str(v["n_gt"]) + ")", "Direct", v["direct"]["count_err"], fmt(v["direct"]["f1"], 3)])
    rows.append(["", "SAHI 512", v["sahi_512"]["count_err"], fmt(v["sahi_512"]["f1"], 3)])
t9 = table(s, ["Input", "Metode", "Error hitung", "F1"], rows,
           left=0.7, top=1.6, width=5.6, height=2.9, font=12)
t9.cell(1, 0).merge(t9.cell(2, 0))
t9.cell(3, 0).merge(t9.cell(4, 0))
picture(s, A / "outputs/mosaic_1024_counted.png", 6.6, 1.45, height=3.25)
box(s, 0.7, 4.9, 11.9, 1.5, fill=LIGHT)
text(s, 1.0, 5.05, 11.4, 1.2,
     ["Pada 1024 px, inferensi langsung lebih baik (F1 0,882 berbanding 0,733).",
      "Pada 2048 px, direct kehilangan 232 dari 463 rumah, SAHI melewatkan 78 dan menekan error hitung menjadi 20.",
      "Aturan praktis: slicing dipakai ketika citra jauh lebih besar daripada tile pelatihan."],
     size=13, color=GRAY, spacing=4)

# ---------------- 10. conclusion ----------------
s = prs.slides.add_slide(BLANK)
header(s, "Kesimpulan, Keterbatasan, dan Langkah Berikutnya", None, 10)
box(s, 0.7, 1.45, 3.9, 2.95, fill=LIGHT)
text(s, 0.95, 1.62, 3.4, 0.4, "Temuan", size=17, bold=True, color=ACCENT)
bullets(s, 0.95, 2.1, 3.4, 2.7, [
    "Baseline mencapai mAP@0.5 0,9342 pada test.",
    "Error hitungan 1,79 rumah per citra.",
    "SAHI penting hanya pada resolusi sangat tinggi.",
], size=12, spacing=8)
box(s, 4.75, 1.45, 3.9, 2.95, fill=LIGHT)
text(s, 5.0, 1.62, 3.4, 0.4, "Keterbatasan", size=17, bold=True, color=ACCENT2)
bullets(s, 5.0, 2.1, 3.4, 2.7, [
    "Rumah menempel menjadi satu box.",
    "Kelas bangunan generik, satu kota.",
    "Video belum diuji pada data UAV nyata.",
], size=12, spacing=8)
box(s, 8.8, 1.45, 3.9, 2.95, fill=LIGHT)
text(s, 9.05, 1.62, 3.4, 0.4, "Langkah berikutnya", size=17, bold=True, color=ACCENT)
bullets(s, 9.05, 2.1, 3.4, 2.7, [
    "Uji generalisasi ke SpaceNet dan Inria.",
    "Fine-tune pada sliced crops dan 1280 px.",
    "Deduplikasi video berbasis geolokasi.",
], size=12, spacing=8)
box(s, 0.7, 4.55, 11.9, 0.8, fill=ACCENT)
text(s, 0.95, 4.7, 11.4, 0.5,
     "Baseline layak untuk penghitungan kasar pada citra 0,3 m, belum untuk klaim kepadatan tinggi.",
     size=15, bold=True, color=WHITE)
text(s, 0.7, 5.55, 11.9, 1.2,
     "Semua angka berasal dari run tunggal pada test split WHU. Eksperimen pembanding YOLO26s "
     "dan Faster R-CNN, variasi resolusi, serta generalisasi lintas wilayah belum dilakukan.",
     size=13, color=RGBColor(0x60, 0x60, 0x60))

prs.save(OUT)
print("saved:", OUT, round(OUT.stat().st_size / 1e6, 2), "MB", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
