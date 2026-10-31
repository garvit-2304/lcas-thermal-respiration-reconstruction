#!/usr/bin/env python3
"""Evaluate fixed and calibrated facial ROI ratios with subject-wise holdout."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from detect_track_regions import ROI_RATIOS
from evaluate_roi_reference import box_scores, read_tracks, xywh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data" / "annotations" / "roi_reference_review.json"
DEFAULT_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "annotations"
METHODS = ("fixed_ratios", "loso_median_ratios")
FACE_MODES = ("annotated_face_oracle", "baseline_tracker_face")


def median_ratios(
    reference_rows: list[dict[str, Any]],
    training_subjects: set[str],
) -> dict[str, tuple[float, float, float, float]]:
    values: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for item in reference_rows:
        subject = str(item["subject"]).casefold()
        if subject not in training_subjects:
            continue
        face = item.get("boxes_xywh", {}).get("face")
        if not face:
            continue
        not_visible = set(item.get("not_visible_regions", []))
        for region, box in item.get("boxes_xywh", {}).items():
            if region == "face" or region in not_visible:
                continue
            fx, fy = float(face["x"]), float(face["y"])
            fw, fh = float(face["width"]), float(face["height"])
            x, y = float(box["x"]), float(box["y"])
            x2, y2 = x + float(box["width"]), y + float(box["height"])
            values[region].append(((x - fx) / fw, (y - fy) / fh, (x2 - fx) / fw, (y2 - fy) / fh))
    fitted = {}
    for region, boxes in values.items():
        fitted[region] = tuple(
            statistics.median(box[index] for box in boxes) for index in range(4)
        )
    return fitted


def project_box(
    face: dict[str, float],
    ratios: tuple[float, float, float, float],
) -> dict[str, float]:
    x, y, width, height = face["x"], face["y"], face["width"], face["height"]
    x1, y1, x2, y2 = ratios
    return {
        "x": x + x1 * width,
        "y": y + y1 * height,
        "width": (x2 - x1) * width,
        "height": (y2 - y1) * height,
    }


def summarize(scores: list[dict[str, float]]) -> dict[str, float | int]:
    if not scores:
        return {"box_count": 0}
    return {
        "box_count": len(scores),
        "mean_iou": statistics.mean(item["iou"] for item in scores),
        "median_iou": statistics.median(item["iou"] for item in scores),
        "median_center_error_pixels": statistics.median(item["center_error_pixels"] for item in scores),
    }


def run(reference_path: Path, tracks_path: Path, output_dir: Path, overwrite: bool) -> None:
    output_csv = output_dir / "roi_geometry_loso_comparison.csv"
    output_json = output_dir / "roi_geometry_loso_comparison.json"
    existing = [path for path in (output_csv, output_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"ROI geometry outputs already exist: {existing}; pass --overwrite")

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    annotations = reference.get("annotations", [])
    subjects = sorted({str(item["subject"]).casefold() for item in annotations})
    tracks = read_tracks(tracks_path)
    rows: list[dict[str, Any]] = []
    fitted_by_subject: dict[str, Any] = {}

    for heldout in subjects:
        training_subjects = set(subjects) - {heldout}
        fitted = median_ratios(annotations, training_subjects)
        fitted_by_subject[heldout] = {
            region: list(ratios) for region, ratios in sorted(fitted.items())
        }
        for item in annotations:
            subject = str(item["subject"]).casefold()
            if subject != heldout:
                continue
            manual_face = item.get("boxes_xywh", {}).get("face")
            if not manual_face:
                continue
            tracker_row = tracks.get((subject, int(item["frame_index"])))
            if tracker_row is None:
                raise ValueError(f"Missing tracker row for {subject} frame {item['frame_index']}")
            tracker_face = xywh(tracker_row, "face")
            not_visible = set(item.get("not_visible_regions", []))
            for region, manual_region in item.get("boxes_xywh", {}).items():
                if region == "face" or region in not_visible:
                    continue
                modes = {
                    "annotated_face_oracle": manual_face,
                    "baseline_tracker_face": tracker_face,
                }
                for mode, parent_face in modes.items():
                    if parent_face is None:
                        continue
                    ratios_to_test = {
                        "fixed_ratios": ROI_RATIOS[region],
                        "loso_median_ratios": fitted.get(region, ROI_RATIOS[region]),
                    }
                    for method, ratios in ratios_to_test.items():
                        prediction = project_box(parent_face, ratios)
                        scores = box_scores(manual_region, prediction)
                        rows.append({
                            "heldout_subject": subject,
                            "frame_index": int(item["frame_index"]),
                            "region": region,
                            "face_box_mode": mode,
                            "ratio_method": method,
                            "iou": scores["iou"],
                            "center_error_pixels": scores["center_error_pixels"],
                            "center_error_reference_diagonals": scores["center_error_reference_diagonals"],
                            "predicted_to_reference_area_ratio": scores["predicted_to_reference_area_ratio"],
                        })

    summary: dict[str, Any] = {}
    for mode in FACE_MODES:
        summary[mode] = {}
        for method in METHODS:
            subset = [row for row in rows if row["face_box_mode"] == mode and row["ratio_method"] == method]
            summary[mode][method] = {
                "overall": summarize([
                    {"iou": float(row["iou"]), "center_error_pixels": float(row["center_error_pixels"])}
                    for row in subset
                ]),
                "by_region": {},
            }
            for region in sorted({row["region"] for row in subset}):
                region_rows = [row for row in subset if row["region"] == region]
                summary[mode][method]["by_region"][region] = summarize([
                    {"iou": float(row["iou"]), "center_error_pixels": float(row["center_error_pixels"])}
                    for row in region_rows
                ])

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "heldout_subject", "frame_index", "region", "face_box_mode", "ratio_method",
        "iou", "center_error_pixels", "center_error_reference_diagonals",
        "predicted_to_reference_area_ratio",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "iou": f"{row['iou']:.8f}",
                "center_error_pixels": f"{row['center_error_pixels']:.6f}",
                "center_error_reference_diagonals": f"{row['center_error_reference_diagonals']:.8f}",
                "predicted_to_reference_area_ratio": f"{row['predicted_to_reference_area_ratio']:.8f}",
            })
    report = {
        "purpose": "Compare predefined geometric facial ROI ratios with median ratios calibrated on four subjects and evaluated on the held-out fifth subject.",
        "reference_input": str(reference_path),
        "annotator": reference.get("metadata", {}).get("annotator"),
        "reference_frames": len(annotations),
        "heldout_protocol": "Leave one subject out; learned ratios for each subject are fit only on the other four subjects.",
        "face_box_modes": {
            "annotated_face_oracle": "Region ratio applied to the manually annotated face box; isolates ROI geometry.",
            "baseline_tracker_face": "Region ratio applied to the baseline tracker face box; includes face-box error.",
        },
        "fixed_ratios": {name: list(values) for name, values in ROI_RATIOS.items()},
        "learned_ratios_by_heldout_subject": fitted_by_subject,
        "summary": summary,
        "limitations": [
            "Only 20 reference frames from five subjects are available; results are preliminary.",
            "Reference boxes have one annotator and need independent review.",
            "A region box IoU does not validate thermal signal quality or anatomical temperature suitability.",
            "Median ratios are not selected for production; a larger sample and second review are needed.",
        ],
        "outputs": {"csv": output_csv.name, "json": output_json.name},
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(rows)} ROI boxes across {len(subjects)} held-out subjects")
    print(f"Wrote {output_csv} and {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.reference.resolve(), args.tracks.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

