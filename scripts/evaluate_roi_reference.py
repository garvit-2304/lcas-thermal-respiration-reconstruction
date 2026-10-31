#!/usr/bin/env python3
"""Measure agreement between reviewed ROI boxes and a heuristic tracker."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_REFERENCE = ROOT / "data" / "annotations" / "roi_reference_review.json"
DEFAULT_OUTPUT = ROOT / "data" / "annotations"


def read_tracks(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    result: dict[tuple[str, int], dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        for row in csv.DictReader(file_obj):
            key = (row["subject"].casefold(), int(row["frame_index"]))
            if key in result:
                raise ValueError(f"Duplicate tracker row for {key}")
            result[key] = row
    return result


def xywh(row: dict[str, str], name: str) -> dict[str, float] | None:
    fields = (f"{name}_x", f"{name}_y", f"{name}_width", f"{name}_height")
    if any(not row.get(field, "") for field in fields):
        return None
    x, y, width, height = (float(row[field]) for field in fields)
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def box_scores(reference: dict[str, Any], predicted: dict[str, float]) -> dict[str, float]:
    rx, ry = float(reference["x"]), float(reference["y"])
    rw, rh = float(reference["width"]), float(reference["height"])
    px, py = predicted["x"], predicted["y"]
    pw, ph = predicted["width"], predicted["height"]
    left, top = max(rx, px), max(ry, py)
    right, bottom = min(rx + rw, px + pw), min(ry + rh, py + ph)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = rw * rh + pw * ph - intersection
    ref_center = (rx + rw / 2.0, ry + rh / 2.0)
    pred_center = (px + pw / 2.0, py + ph / 2.0)
    center_error = math.hypot(ref_center[0] - pred_center[0], ref_center[1] - pred_center[1])
    diagonal = math.hypot(rw, rh)
    return {
        "iou": intersection / union if union > 0 else 0.0,
        "center_error_pixels": center_error,
        "center_error_reference_diagonals": center_error / diagonal if diagonal > 0 else math.nan,
        "predicted_to_reference_area_ratio": (pw * ph) / (rw * rh),
    }


def evaluate(reference_path: Path, tracks_path: Path, output_dir: Path, overwrite: bool) -> None:
    output_csv = output_dir / "roi_reference_tracker_agreement.csv"
    output_json = output_dir / "roi_reference_tracker_agreement.json"
    existing = [str(path) for path in (output_csv, output_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Agreement outputs exist: {existing}; pass --overwrite")

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    track_rows = read_tracks(tracks_path)
    per_region: dict[str, list[dict[str, float]]] = defaultdict(list)
    output_rows: list[dict[str, Any]] = []
    missing_reference_boxes = 0
    missing_tracker_boxes = 0
    for item in reference.get("annotations", []):
        if not item.get("reviewed_regions"):
            continue
        subject = str(item["subject"])
        frame_index = int(item["frame_index"])
        track = track_rows.get((subject.casefold(), frame_index))
        if track is None:
            raise ValueError(f"Missing tracker row for {subject} frame {frame_index}")
        not_visible = set(item.get("not_visible_regions", []))
        reviewed = set(item.get("reviewed_regions", []))
        boxes = item.get("boxes_xywh", {})
        for region in sorted(reviewed - not_visible):
            manual_box = boxes.get(region)
            if manual_box is None:
                missing_reference_boxes += 1
                continue
            predicted = xywh(track, region)
            if predicted is None:
                missing_tracker_boxes += 1
            scores = box_scores(manual_box, predicted) if predicted else None
            row: dict[str, Any] = {
                "subject": subject,
                "frame_index": frame_index,
                "region": region,
                "tracker_box_available": predicted is not None,
                "iou": scores["iou"] if scores else "",
                "center_error_pixels": scores["center_error_pixels"] if scores else "",
                "center_error_reference_diagonals": scores["center_error_reference_diagonals"] if scores else "",
                "predicted_to_reference_area_ratio": scores["predicted_to_reference_area_ratio"] if scores else "",
            }
            output_rows.append(row)
            if scores:
                per_region[region].append(scores)

    fields = [
        "subject",
        "frame_index",
        "region",
        "tracker_box_available",
        "iou",
        "center_error_pixels",
        "center_error_reference_diagonals",
        "predicted_to_reference_area_ratio",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    region_summary = {}
    for region, values in sorted(per_region.items()):
        region_summary[region] = {
            "compared_box_count": len(values),
            "mean_iou": sum(item["iou"] for item in values) / len(values),
            "median_iou": statistics.median(item["iou"] for item in values),
            "median_center_error_pixels": statistics.median(
                item["center_error_pixels"] for item in values
            ),
        }
    report = {
        "reference_input": str(reference_path),
        "tracker_input": str(tracks_path),
        "reference_provenance": reference.get("metadata", {}).get("annotation_role", "unspecified"),
        "reviewed_reference_frames": sum(
            bool(item.get("reviewed_regions")) for item in reference.get("annotations", [])
        ),
        "complete_reference_frames": sum(
            item.get("review_status") == "complete" for item in reference.get("annotations", [])
        ),
        "partial_or_unreviewed_frame_count": sum(
            item.get("review_status") != "complete" for item in reference.get("annotations", [])
        ),
        "reviewed_boxes_without_reference_box": missing_reference_boxes,
        "reviewed_boxes_without_tracker_box": missing_tracker_boxes,
        "box_comparison_count": len(output_rows),
        "region_summary": region_summary,
        "interpretation": "Preliminary box agreement for the reviewed sample only. This does not establish dataset-wide accuracy or independent ground truth; have a second reviewer check the annotations before formal claims.",
        "outputs": {"csv": output_csv.name, "json": output_json.name},
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote preliminary agreement report for {len(output_rows)} reviewed boxes to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    evaluate(args.reference.resolve(), args.tracks.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

