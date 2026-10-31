#!/usr/bin/env python3
"""Build a sourced PDF summary of the L-CAS thermal respiration project."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "thermal_respiration_project_report.pdf"
ASSETS = ROOT / "tmp" / "pdfs" / "thermal_respiration_report_assets"
SUBJECTS = ["anestis", "claudio", "jaime", "joao", "manuel"]
REGIONS = ["forehead", "nose", "mouth", "left_cheek", "right_cheek"]
COLORS = {
    "navy": "#16324F",
    "blue": "#2F6B9A",
    "teal": "#2A9D8F",
    "orange": "#E07A35",
    "red": "#C94C4C",
    "purple": "#8C62AA",
    "gold": "#D5A62E",
    "light": "#EAF0F5",
    "ink": "#21313F",
    "muted": "#667784",
    "gray": "#9AA8B2",
}
METHOD_COLORS = {"raw": COLORS["navy"], "pws": COLORS["teal"], "gtv": COLORS["orange"]}
REGION_COLORS = {
    "face": "#00B8B8",
    "forehead": "#FF7F0E",
    "nose": "#E31A1C",
    "mouth": "#D5A62E",
    "left_cheek": "#3264C8",
    "right_cheek": "#C51B8A",
}


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open("r", newline="", encoding="utf-8-sig") as file_obj:
        return list(csv.DictReader(file_obj))


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = ASSETS / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCD5DC")
    ax.spines["bottom"].set_color("#CCD5DC")
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.grid(axis="y", color="#E7EDF1", linewidth=0.8)
    ax.set_axisbelow(True)


def make_workflow_figure() -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 2.0))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 2.0)
    ax.axis("off")
    boxes = [
        (0.08, "ROS bags +\nreference CSVs", COLORS["navy"]),
        (2.22, "Decode + align\nimage timestamps", COLORS["blue"]),
        (4.36, "Track face +\nextract 5 ROIs", COLORS["teal"]),
        (6.50, "Filter / graph\nPWS + GTV", COLORS["orange"]),
        (8.64, "Rate + event\nevaluation", COLORS["purple"]),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch((x, 0.58), 1.8, 0.9, boxstyle="round,pad=0.08,rounding_size=0.08", facecolor=color, edgecolor="none")
        ax.add_patch(patch)
        ax.text(x + 0.9, 1.03, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    for (x, _, _), (nx, _, _) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((x + 1.83, 1.03), (nx - 0.08, 1.03), arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color=COLORS["muted"]))
    ax.text(5.4, 0.19, "Owner-confirmed event map: 0 = no event, 1 = inhalation, 2 = exhalation (not author-verified)", ha="center", va="center", fontsize=8, color=COLORS["muted"])
    return save_figure(fig, "project_workflow.png")


def make_roi_montage(review: dict[str, Any]) -> Path:
    font = ImageFont.load_default()
    tile_w, tile_h, top_h = 520, 356, 32
    canvas = PILImage.new("RGB", (tile_w * 2, tile_h * 3), "#F5F7F9")
    draw = ImageDraw.Draw(canvas)
    first_by_subject: dict[str, dict[str, Any]] = {}
    for item in review["annotations"]:
        first_by_subject.setdefault(item["subject"].casefold(), item)
    for i, subject in enumerate(SUBJECTS):
        item = first_by_subject[subject]
        frame = PILImage.open(ROOT / item["image_path"]).convert("RGB")
        frame.thumbnail((tile_w - 16, tile_h - top_h - 8), PILImage.Resampling.LANCZOS)
        x0 = (i % 2) * tile_w
        y0 = (i // 2) * tile_h
        draw.text((x0 + 10, y0 + 8), f"{subject.title()} | frame {item['frame_index']} | owner reference boxes", fill=COLORS["ink"], font=font)
        px = x0 + (tile_w - frame.width) // 2
        py = y0 + top_h
        canvas.paste(frame, (px, py))
        scale_x = frame.width / 382.0
        scale_y = frame.height / 288.0
        for region, box in item["boxes_xywh"].items():
            color = REGION_COLORS.get(region, "#FFFFFF")
            x1 = px + round(float(box["x"]) * scale_x)
            y1 = py + round(float(box["y"]) * scale_y)
            x2 = px + round((float(box["x"]) + float(box["width"])) * scale_x)
            y2 = py + round((float(box["y"]) + float(box["height"])) * scale_y)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3 if region == "face" else 2)
    legend_y = tile_h * 2 + 15
    legend_items = [("face", "Face"), ("forehead", "Forehead"), ("nose", "Nose"), ("mouth", "Mouth"), ("left_cheek", "Image-left cheek"), ("right_cheek", "Image-right cheek")]
    x = 12
    for key, label in legend_items:
        draw.rectangle((x, legend_y + 1, x + 13, legend_y + 14), outline=REGION_COLORS[key], width=3)
        draw.text((x + 19, legend_y), label, fill=COLORS["ink"], font=font)
        x += 165
    path = ASSETS / "manual_roi_reference_montage.png"
    canvas.save(path)
    return path


def make_tracker_iou_figure(tracker: dict[str, Any]) -> Path:
    summary = tracker["summary"]
    variants = list(summary)
    labels = ["Baseline", "Min neighbors = 2", "Scale 1.05 + min 2"]
    face = [summary[name]["by_region"]["face"]["mean_iou"] for name in variants]
    all_boxes = [summary[name]["mean_iou"] for name in variants]
    x = list(range(len(variants)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 2.55))
    bars1 = ax.bar([v - width / 2 for v in x], face, width, color=COLORS["blue"], label="Face boxes (n=20)")
    bars2 = ax.bar([v + width / 2 for v in x], all_boxes, width, color=COLORS["orange"], label="All visible boxes (n=115)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.68)
    ax.set_ylabel("Mean intersection over union")
    ax.set_title("Tracker agreement with the owner-labeled sample", loc="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for bars in (bars1, bars2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=COLORS["ink"])
    return save_figure(fig, "tracker_iou_comparison.png")


def make_region_iou_figure(tracker: dict[str, Any]) -> Path:
    baseline = tracker["summary"]["baseline"]["by_region"]
    labels = ["Face", "Forehead", "Left cheek", "Mouth", "Nose", "Right cheek"]
    keys = ["face", "forehead", "left_cheek", "mouth", "nose", "right_cheek"]
    values = [baseline[key]["mean_iou"] for key in keys]
    counts = [baseline[key]["box_count"] for key in keys]
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    y = list(range(len(keys)))
    bars = ax.barh(y, values, color=[COLORS["blue"]] + [COLORS["orange"]] * 5, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Mean IoU")
    ax.set_title("Baseline tracker agreement by manually labeled region", loc="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    style_axes(ax)
    ax.grid(axis="x", color="#E7EDF1", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.bar_label(bars, labels=[f"{v:.2f}  (n={n})" for v, n in zip(values, counts)], padding=3, fontsize=7.5, color=COLORS["ink"])
    return save_figure(fig, "tracker_iou_by_region.png")


def make_geometry_figure(geometry: dict[str, Any]) -> Path:
    modes = ["annotated_face_oracle", "baseline_tracker_face"]
    names = ["Manual face boxes", "Tracker face boxes"]
    fixed = [geometry["summary"][mode]["fixed_ratios"]["overall"]["mean_iou"] for mode in modes]
    learned = [geometry["summary"][mode]["loso_median_ratios"]["overall"]["mean_iou"] for mode in modes]
    x = [0, 1]
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.4, 2.75))
    b1 = ax.bar([v - w / 2 for v in x], fixed, w, label="Fixed geometry", color=COLORS["gray"])
    b2 = ax.bar([v + w / 2 for v in x], learned, w, label="LOSO learned ratios", color=COLORS["teal"])
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Mean ROI-box IoU (95 visible boxes)")
    ax.set_title("ROI geometry transfer to held-out subjects", loc="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=COLORS["ink"])
    ax.text(0.5, -0.28, "Ratio learning helps when the face box is manual; the tracker box error dominates otherwise.", transform=ax.transAxes, ha="center", fontsize=8, color=COLORS["muted"])
    return save_figure(fig, "roi_geometry_loso.png")


def make_graph_trace_figure(signals: list[dict[str, str]]) -> Path:
    rows = [row for row in signals if row["subject"].casefold() == "anestis" and row["region"].casefold() == "nose"]
    rows.sort(key=lambda row: int(row["frame_index"]))
    x = [float(row["seconds_since_first_image"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 2.55))
    for method, column in [("raw", "raw_temperature_c"), ("PWS", "pws_temperature_c"), ("GTV", "gtv_temperature_c")]:
        ax.plot(x, [float(row[column]) for row in rows], label=method, linewidth=0.85, alpha=0.88)
    ax.axvline(60, color=COLORS["red"], linestyle="--", linewidth=1, label="Movement phase starts")
    ax.set_xlabel("Seconds from first image-header timestamp")
    ax.set_ylabel("Nose ROI temperature (deg C)")
    ax.set_title("Anestis nose signal: raw and graph-filtered traces", loc="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    return save_figure(fig, "graph_trace_anestis.png")


def make_region_ablation_figure(ablation: dict[str, Any]) -> Path:
    summary = ablation["summary_by_event_span_rate"]
    x = list(range(len(REGIONS)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for j, method in enumerate(("raw", "pws", "gtv")):
        vals = [next(row["mean_absolute_error_bpm"] for row in summary if row["region"] == region and row["method"] == method) for region in REGIONS]
        bars = ax.bar([i + (j - 1) * width for i in x], vals, width, color=METHOD_COLORS[method], label=method.upper())
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=6.8, color=COLORS["ink"])
    ax.set_xticks(x, ["Forehead", "Nose", "Mouth", "Left cheek", "Right cheek"])
    ax.set_ylabel("Mean absolute rate error (BPM)")
    ax.set_ylim(0, 8)
    ax.set_title("ROI and graph-output ablation (5 subjects)", loc="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    return save_figure(fig, "region_graph_ablation.png")


def make_subject_rates_figure(ablation_rows: list[dict[str, str]]) -> Path:
    refs: dict[str, float] = {}
    estimates: dict[tuple[str, str], float] = {}
    for row in ablation_rows:
        if row["method"] == "raw":
            subject = row["subject"].casefold()
            refs[subject] = float(row["event_span_rate_bpm"])
            if row["region"] in ("nose", "mouth"):
                estimates[(subject, row["region"])] = float(row["estimated_rate_bpm"])
    x = list(range(len(SUBJECTS)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 2.75))
    b0 = ax.bar([v - width for v in x], [refs[s] for s in SUBJECTS], width, color=COLORS["navy"], label="Inhalation interval reference")
    b1 = ax.bar(x, [estimates[(s, "nose")] for s in SUBJECTS], width, color=COLORS["blue"], label="Raw nose ROI")
    b2 = ax.bar([v + width for v in x], [estimates[(s, "mouth")] for s in SUBJECTS], width, color=COLORS["orange"], label="Raw mouth ROI")
    ax.set_xticks(x, [s.title() for s in SUBJECTS])
    ax.set_ylabel("Whole-recording rate (BPM)")
    ax.set_ylim(0, 30)
    ax.set_title("Per-subject rates: reference vs. fixed spectral baseline", loc="left", fontsize=11, fontweight="bold", color=COLORS["ink"])
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")
    return save_figure(fig, "subject_rate_comparison.png")


def make_tcn_comparison_figure(tcn: dict[str, Any], event_rates: dict[str, Any]) -> Path:
    s = tcn["summary"]
    rate_summary = event_rates["summary"]
    methods = ["TCN", "Bandpass peak/valley"]
    f1 = [s["macro_f1"], s["peak_baseline_macro_f1"]]
    mae = [rate_summary["tcn"]["mean_absolute_error_event_span_rate_bpm"], rate_summary["bandpass_peak_valley"]["mean_absolute_error_event_span_rate_bpm"]]
    colors_list = [COLORS["purple"], COLORS["teal"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.65))
    for ax, vals, ylabel, title, ymax in [
        (axes[0], f1, "Macro F1 (higher is better)", "Held-out event timing", 0.42),
        (axes[1], mae, "Rate MAE (BPM; lower is better)", "Event-span rate agreement", 16),
    ]:
        bars = ax.bar(methods, vals, color=colors_list, width=0.55)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=COLORS["ink"])
        ax.tick_params(axis="x", labelrotation=8)
        style_axes(ax)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color=COLORS["ink"])
    fig.tight_layout(w_pad=2.0)
    return save_figure(fig, "tcn_vs_baseline.png")


def image_flowable(path: Path, max_w: float, max_h: float) -> Image:
    with PILImage.open(path) as img:
        w, h = img.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=colors.HexColor(COLORS["navy"]), alignment=TA_LEFT, spaceAfter=10),
        "subtitle": ParagraphStyle("ReportSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=12, leading=17, textColor=colors.HexColor(COLORS["muted"]), spaceAfter=12),
        "h1": ParagraphStyle("H1Custom", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=19, leading=22, textColor=colors.HexColor(COLORS["navy"]), spaceAfter=8),
        "h2": ParagraphStyle("H2Custom", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor(COLORS["blue"]), spaceBefore=5, spaceAfter=4),
        "body": ParagraphStyle("BodyCustom", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=colors.HexColor(COLORS["ink"]), spaceAfter=6),
        "small": ParagraphStyle("SmallCustom", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.6, textColor=colors.HexColor(COLORS["muted"]), spaceAfter=3),
        "caption": ParagraphStyle("CaptionCustom", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=7.6, leading=10, textColor=colors.HexColor(COLORS["muted"]), alignment=TA_LEFT, spaceBefore=3, spaceAfter=6),
        "callout": ParagraphStyle("CalloutCustom", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=colors.HexColor(COLORS["navy"]), spaceAfter=0),
        "table": ParagraphStyle("TableCustom", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.2, textColor=colors.HexColor(COLORS["ink"])),
        "table_head": ParagraphStyle("TableHeadCustom", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9.2, textColor=colors.white),
        "center": ParagraphStyle("CenterCustom", parent=base["BodyText"], fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor(COLORS["ink"]), alignment=TA_CENTER),
    }


def callout(text: str, styles: dict[str, ParagraphStyle], width: float) -> Table:
    table = Table([[p(text, styles["callout"])]], colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(COLORS["light"])),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7E1E8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def data_table(rows: list[list[Any]], widths: list[float], styles: dict[str, ParagraphStyle], header: bool = True) -> Table:
    converted: list[list[Any]] = []
    for i, row in enumerate(rows):
        converted.append([cell if isinstance(cell, (Paragraph, Image)) else p(escape(str(cell)), styles["table_head"] if (header and i == 0) else styles["table"]) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    rules = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E0E6")),
    ]
    if header:
        rules.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLORS["navy"])))
        if len(rows) > 1:
            rules.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]))
    table.setStyle(TableStyle(rules))
    return table


def page_header(canvas: Any, doc: Any) -> None:
    if doc.page == 1:
        return
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D8E0E6"))
    canvas.setLineWidth(0.55)
    canvas.line(doc.leftMargin, height - 0.43 * inch, width - doc.rightMargin, height - 0.43 * inch)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(colors.HexColor(COLORS["muted"]))
    canvas.drawString(doc.leftMargin, height - 0.33 * inch, "L-CAS Thermal Respiration Monitoring | Project Report")
    canvas.drawRightString(width - doc.rightMargin, 0.28 * inch, f"Page {doc.page}")
    canvas.drawString(doc.leftMargin, 0.28 * inch, "Exploratory research results - not a clinical performance claim")
    canvas.restoreState()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    tracker = load_json("data/annotations/owner_tracker_variant_comparison.json")
    review = load_json("data/annotations/roi_reference_review.json")
    geometry = load_json("data/annotations/roi_geometry_loso_comparison.json")
    graph_loso = load_json("data/processed/graph_denoising_rosbag_raw/leave_one_subject_out_tuning.json")
    tcn = load_json("data/processed/tcn_inhalation_event_detector/tcn_loso_report.json")
    event_rates = load_json("data/processed/tcn_inhalation_event_detector/tcn_event_interval_rate_comparison.json")
    ablation = load_json("data/processed/region_graph_ablations/region_graph_ablation_report.json")
    baseline_rows = load_csv("data/processed/respiration_baseline_rosbag_raw/rosbag_raw_estimates.csv")
    ablation_rows = load_csv("data/processed/region_graph_ablations/region_graph_ablation_metrics.csv")
    graph_rows = load_csv("data/processed/graph_denoising_rosbag_raw/timestamped_raw_graph_signals.csv")

    figures = {
        "workflow": make_workflow_figure(),
        "roi_montage": make_roi_montage(review),
        "tracker_iou": make_tracker_iou_figure(tracker),
        "region_iou": make_region_iou_figure(tracker),
        "geometry": make_geometry_figure(geometry),
        "graph_trace": make_graph_trace_figure(graph_rows),
        "region_ablation": make_region_ablation_figure(ablation),
        "subject_rates": make_subject_rates_figure(ablation_rows),
        "tcn": make_tcn_comparison_figure(tcn, event_rates),
    }
    roi_plot = ROOT / "data" / "processed" / "roi_signal_timestamp_plots" / "anestis_timestamped_raw_rois.png"
    styles = make_styles()
    page_w, _ = A4
    usable_w = page_w - 1.24 * inch
    story: list[Any] = []

    # Page 1: cover and executive summary.
    story += [Spacer(1, 0.32 * inch), p("PROJECT REPORT | 31 OCTOBER 2026", styles["small"]), Spacer(1, 0.08 * inch)]
    story += [p("Thermal Respiration<br/>Monitoring", styles["title"]), p("Reconstruction, signal processing, and baseline evaluation on the L-CAS dataset", styles["subtitle"])]
    metrics = [
        ["5", "recordings / subjects", "14,154", "thermal frames"],
        ["20", "owner-reviewed sample frames", "115", "visible reference boxes"],
    ]
    metric_cells = []
    for row in metrics:
        metric_cells.append([p(f"<b><font size='19' color='{COLORS['navy']}'>{row[0]}</font></b><br/><font size='8' color='{COLORS['muted']}'>{row[1]}</font>", styles["center"]), p(f"<b><font size='19' color='{COLORS['navy']}'>{row[2]}</font></b><br/><font size='8' color='{COLORS['muted']}'>{row[3]}</font>", styles["center"])])
    metric_table = Table(metric_cells, colWidths=[usable_w / 2] * 2, rowHeights=[0.7 * inch, 0.7 * inch])
    metric_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F6F8")), ("GRID", (0, 0), (-1, -1), 0.5, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [Spacer(1, 0.16 * inch), metric_table, Spacer(1, 0.22 * inch)]
    story += [callout("Main finding: the simple bandpass peak/valley detector outperformed the tested TCN event detector. A mouth-region spectral baseline also had lower error than the nose baseline in this five-subject exploratory comparison. Neither result is independently validated.", styles, usable_w), Spacer(1, 0.16 * inch)]
    story += [image_flowable(figures["workflow"], usable_w, 1.55 * inch), Spacer(1, 0.10 * inch)]
    story += [p("This report separates verified data-processing work from exploratory model results. The face-region labels are complete for the selected 20-frame sample; they do not validate the continuous face tracks across the full recordings.", styles["body"])]

    # Page 2: data and workflow.
    story += [PageBreak(), p("1. Dataset and reproducible processing", styles["h1"])]
    story += [p("The project combines five L-CAS thermal recordings with respiration and heartbeat reference CSVs. The local audit covers 14,154 thermal frames. The archived official ground-truth ZIP matches the local CSV files. Thermal matrices were decoded from the ROS bags, paired to PNGs by message order after count verification, and analyzed with image-header timestamps.", styles["body"])]
    story += [p("The observed image timing is irregular: median image intervals are about 37-38 ms and effective per-recording rates span 21.8-24.9 Hz. This measured timing was retained rather than assuming a nominal camera rate. Raw mono16 counts were preserved and Celsius was calculated with the dataset encoding formula, (count - 1000) / 10.", styles["body"])]
    story += [image_flowable(figures["workflow"], usable_w, 1.55 * inch), p("Figure 1. Processing and evaluation path implemented in the repository.", styles["caption"])]
    table_rows = [
        ["Stage", "Implemented", "Evidence / note"],
        ["Input audit", "5 ROS bags + official reference CSVs", "14,154 frames; local reference files match archived official source"],
        ["Time alignment", "Image-header timestamps", "Anestis reference pairing uses nearest time; ordinal pairing is invalid"],
        ["Temperature", "Raw counts + Celsius", "Counts preserved; official encoding conversion recorded"],
        ["Label map", "0 / 1 / 2 working map", "Owner-confirmed; authors' numeric codebook not located"],
    ]
    story += [data_table(table_rows, [1.0 * inch, 2.0 * inch, usable_w - 3.0 * inch], styles), Spacer(1, 0.08 * inch)]
    story += [p("The official dataset description gives a two-minute recording protocol and the camera encoding; the paper says respiration references were made by manually annotating inhalation times. Neither published description defines the numeric codes in the supplied CSV files. The report therefore treats 0 = no event, 1 = inhalation, 2 = exhalation as a project-owner-confirmed working convention, not an author-verified codebook.", styles["small"])]

    # Page 3: manual labels and tracker.
    story += [PageBreak(), p("2. Face and facial-region reference review", styles["h1"])]
    story += [p("The project owner completed the selected sample: 20 frames across five subjects, with face and five candidate facial regions reviewed on each frame. There are 115 visible boxes, five regions marked not visible, and no unresolved region decisions. This is a small, single-reviewer sample.", styles["body"])]
    story += [image_flowable(figures["roi_montage"], usable_w, 3.2 * inch), p("Figure 2. Owner-drawn face and facial-region boxes on one manually reviewed frame per subject. Colors identify the face, forehead, nose, mouth, and image-left/right cheeks.", styles["caption"])]
    story += [image_flowable(figures["tracker_iou"], usable_w, 2.05 * inch), p("Figure 3. Tracker agreement against all owner-labeled boxes in the 20-frame sample. Face-only IoU is higher than agreement over all facial subregions.", styles["caption"])]
    story += [p("Baseline mean IoU is 0.56 for the 20 face boxes, but 0.22 across all 115 visible face and subregion boxes. The lower-threshold candidates do not improve all-box agreement (0.21 and 0.22). These values describe agreement with one annotator's sparse reference sample; they are not full-recording accuracy.", styles["body"])]

    # Page 4: region IoU and geometry holdout.
    story += [PageBreak(), p("3. ROI geometry and localization", styles["h1"])]
    story += [p("Subregion agreement exposes a localization gap: the baseline face-box mean IoU is 0.56, while individual facial-region means are about 0.13-0.19. The main constraint is placing a small region inside the correct face area, not simply generating a box for every frame.", styles["body"])]
    story += [image_flowable(figures["region_iou"], usable_w, 2.55 * inch), p("Figure 4. Baseline mean IoU by region; the number beside each bar is the number of visible labeled boxes. The left cheek has 15 labels because five were marked not visible.", styles["caption"])]
    story += [image_flowable(figures["geometry"], usable_w, 2.55 * inch), p("Figure 5. Leave-one-subject-out comparison of fixed geometric ROI ratios with ratios learned on the other four subjects. Manual face boxes isolate geometry; tracked face boxes include detector error.", styles["caption"])]
    story += [p("With manually annotated face boxes, learned ratios raise overall mean IoU from 0.387 to 0.416. With baseline tracker boxes, IoU falls from 0.145 to 0.087. This shows the tracker box can dominate downstream ROI overlap; no ratio scheme is selected for production.", styles["body"])]

    # Page 5: signal quality and graph.
    story += [PageBreak(), p("4. Thermal signals and graph filtering", styles["h1"])]
    story += [p("Regional temperatures were extracted from each tracked ROI and plotted against measured time. The example below shows the tracked nose trace and a 10-40 BPM bandpass signal. Large excursions occur during the movement portion, so a full-recording spectral rate can be sensitive to motion and tracking shifts.", styles["body"])]
    story += [image_flowable(roi_plot, usable_w, 3.25 * inch), p("Figure 6. Anestis timestamped raw ROI signals and filtered nose trace. The plot is a diagnostic visualization; it does not by itself demonstrate event accuracy.", styles["caption"])]
    story += [image_flowable(figures["graph_trace"], usable_w, 2.25 * inch), p("Figure 7. Raw, PWS, and GTV nose traces for Anestis. The tested graph filters make only small changes to the dominant rate estimates.", styles["caption"])]
    graph = graph_loso["summary"]
    graph_rows_table = [
        ["Signal", "LOSO mean absolute rate error"],
        ["Raw", f"{graph['raw_mean_absolute_error_bpm']:.2f} BPM"],
        ["PWS", f"{graph['pws_mean_absolute_error_bpm']:.2f} BPM"],
        ["GTV", f"{graph['gtv_mean_absolute_error_bpm']:.2f} BPM"],
    ]
    story += [data_table(graph_rows_table, [2.0 * inch, usable_w - 2.0 * inch], styles), Spacer(1, 0.08 * inch)]
    story += [p("PWS lowers the five-subject LOSO rate MAE by only 0.10 BPM relative to raw (4.72 to 4.62 BPM); GTV is worse at 4.91 BPM. This is not strong evidence that graph filtering improves the baseline.", styles["body"])]

    # Page 6: rate baseline and region ablation.
    story += [PageBreak(), p("5. Baseline rates and region ablation", styles["h1"])]
    story += [p("The conventional baseline linearly resamples the measured timestamps, applies a fourth-order zero-phase Butterworth filter in the 10-40 BPM band, and selects the Hann-periodogram peak. Nose-region rates are compared with inhalation-interval rates under the working label map.", styles["body"])]
    story += [image_flowable(figures["subject_rates"], usable_w, 2.45 * inch), p("Figure 8. Reference event-span rate and two raw ROI spectral rates by subject. The mouth signal is shown as an exploratory candidate, not a held-out choice.", styles["caption"])]
    story += [image_flowable(figures["region_ablation"], usable_w, 2.65 * inch), p("Figure 9. Fixed 10-40 BPM spectral rate error by region and graph output across five subjects. Values are mean absolute errors against event-span rates.", styles["caption"])]
    best = min(ablation["summary_by_event_span_rate"], key=lambda row: row["mean_absolute_error_bpm"])
    nose_raw = next(row for row in ablation["summary_by_event_span_rate"] if row["region"] == "nose" and row["method"] == "raw")
    reduction = 100.0 * (nose_raw["mean_absolute_error_bpm"] - best["mean_absolute_error_bpm"]) / nose_raw["mean_absolute_error_bpm"]
    story += [p(f"The best observed region in this analysis is mouth/raw at {best['mean_absolute_error_bpm']:.2f} BPM MAE, compared with {nose_raw['mean_absolute_error_bpm']:.2f} BPM for nose/raw (about {reduction:.0f}% lower). Because the same five subjects were used to rank the regions, that reduction is exploratory and likely optimistic; it needs confirmation on new subjects.", styles["body"])]

    # Page 7: event model comparison.
    story += [PageBreak(), p("6. Event-level baseline comparison", styles["h1"])]
    story += [p("The TCN uses raw, PWS, and GTV nose signals to detect sparse inhalation timestamps. Outer leave-one-subject-out evaluation uses inner subject-wise validation for threshold selection. The comparison is against a fourth-order bandpass peak/valley detector with its polarity and prominence selected within the training subjects.", styles["body"])]
    story += [image_flowable(figures["tcn"], usable_w, 2.55 * inch), p("Figure 10. Macro event F1 and event-span rate MAE from held-out subjects. Higher F1 is better; lower rate MAE is better.", styles["caption"])]
    s = tcn["summary"]
    er = event_rates["summary"]
    event_table = [
        ["Metric", "TCN", "Bandpass peak/valley"],
        ["Macro event F1", f"{s['macro_f1']:.3f}", f"{s['peak_baseline_macro_f1']:.3f}"],
        ["Macro recall", f"{s['macro_recall']:.3f}", f"{s['peak_baseline_macro_recall']:.3f}"],
        ["Event-span rate MAE", f"{er['tcn']['mean_absolute_error_event_span_rate_bpm']:.2f} BPM", f"{er['bandpass_peak_valley']['mean_absolute_error_event_span_rate_bpm']:.2f} BPM"],
        ["Median-interval rate MAE", f"{er['tcn']['mean_absolute_error_median_interval_rate_bpm']:.2f} BPM", f"{er['bandpass_peak_valley']['mean_absolute_error_median_interval_rate_bpm']:.2f} BPM"],
        ["Count-rate MAE", f"{er['tcn']['mean_absolute_error_count_rate_bpm']:.2f} BPM", f"{er['bandpass_peak_valley']['mean_absolute_error_count_rate_bpm']:.2f} BPM"],
    ]
    story += [data_table(event_table, [2.25 * inch, 1.75 * inch, usable_w - 4.0 * inch], styles), Spacer(1, 0.08 * inch)]
    story += [p(f"Across all five held-out subjects, the TCN identified {s['total_true_positives']} events and missed {s['total_false_negatives']}; the comparator had higher macro F1 but still made substantial timing errors. The comparator's count-rate MAE is only {er['bandpass_peak_valley']['mean_absolute_error_count_rate_bpm']:.2f} BPM because the number of predicted events is close to the number of references. Rate agreement can therefore look good even when event-level F1 is modest.", styles["body"])]
    story += [callout("Interpretation: the current evidence favors the simple bandpass event baseline over the TCN. The TCN is not selected, and neither method is a validated clinical monitor.", styles, usable_w)]

    # Page 8: conclusions, limitations, references.
    story += [PageBreak(), p("7. Conclusions and next steps", styles["h1"])]
    conclusions = [
        "Data processing is traceable: the local reference CSVs were compared with the archived official source, raw bag frames were decoded, timestamps were retained, and all 14,154 PNG frames were audited.",
        "The face and facial-region annotation workflow is complete for the selected 20-frame sample. It contains 115 visible boxes and five explicit not-visible decisions.",
        "The owner-labeled sample exposes a real engineering issue: face-box agreement is moderate, while small facial-region overlap is low. Continuous ROI quality remains a limiting factor.",
        "The raw mouth-region baseline has the lowest in-sample region-ablation rate error, but the five-subject sample is too small to select a final ROI.",
        "The TCN event detector did not beat the bandpass peak/valley baseline under nested subject-wise evaluation. Graph filtering produced no convincing rate improvement over raw signals.",
    ]
    for i, item in enumerate(conclusions, 1):
        story.append(p(f"<b>{i}.</b> {item}", styles["body"]))
    story += [p("Open scientific requirements", styles["h2"])]
    open_rows = [
        ["Requirement", "Why it remains open"],
        ["Author-verified 0/1/2 semantics", "The published dataset description and paper do not define numeric CSV codes."],
        ["Independent ROI review", "Current labels have one annotator and cover 20 sampled frames, not full tracks."],
        ["Larger held-out evaluation", "Only five subjects are available; region ranking on them is exploratory."],
        ["Continuous-waveform TCN denoising", "The dataset supplies event timestamps, not a clean continuous respiration waveform."],
    ]
    story += [data_table(open_rows, [2.1 * inch, usable_w - 2.1 * inch], styles), Spacer(1, 0.12 * inch)]
    story += [p("Reproduce the report", styles["h2"]), p("From the project root, run <font name='Courier'>python scripts/build_project_report.py</font>. The report draws on the processed results under <font name='Courier'>data/processed/</font> and the owner review under <font name='Courier'>data/annotations/</font>.", styles["small"])]
    story += [p("Sources", styles["h2"])]
    sources = [
        "L-CAS, <link href='https://lcas.lincoln.ac.uk/wp/research/data-sets-software/lcas-thermal-physiological-monitoring-dataset/' color='#2F6B9A'>Thermal Physiological Monitoring Dataset</link> (official dataset description, recording protocol, and mono16 temperature conversion).",
        "S. Cosar et al., <link href='https://cpb-eu-w2.wpmucdn.com/blogs.lincoln.ac.uk/dist/2/1110/files/2020/06/Cosar2018.pdf' color='#2F6B9A'>Thermal Camera Based Physiological Monitoring with an Assistive Robot</link>, IEEE EMBC, 2018 (respiration reference based on manually marked inhalation instants).",
        "C. Yang et al., <link href='https://doi.org/10.1109/JIOT.2022.3150147' color='#2F6B9A'>Graph-Based Denoising for Respiration and Heart Rate Estimation During Sleep in Thermal Video</link>, IEEE Internet of Things Journal, 2022.",
        "B. Lokendra and G. Puneet, <link href='https://doi.org/10.1016/j.compbiomed.2021.105146' color='#2F6B9A'>AND-rPPG: A Novel Denoising-rPPG Network for Improving Remote Heart Rate Estimation</link>, Computers in Biology and Medicine, 2022.",
        "LCAS, <link href='https://github.com/LCAS/ENRICHME_Releases/blob/master/physiological_monitoring/src/physiological_monitoring.cpp' color='#2F6B9A'>official physiological monitoring ROS node</link> (rate estimator and 10-40 BPM search band).",
    ]
    for source in sources:
        story.append(p("- " + source, styles["small"]))
    story += [Spacer(1, 0.08 * inch), p("Report generated from local project outputs on 31 October 2026. Reported errors are descriptive research metrics under the project-owner-confirmed working code map; no clinical claims are made.", styles["small"])]

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        rightMargin=0.62 * inch, leftMargin=0.62 * inch,
        topMargin=0.62 * inch, bottomMargin=0.52 * inch,
        title="Thermal Respiration Monitoring - Project Report",
        author="Project analysis",
        subject="L-CAS thermal respiration reconstruction and baseline evaluation",
    )
    doc.build(story, onFirstPage=page_header, onLaterPages=page_header)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

