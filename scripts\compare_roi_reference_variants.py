#!/usr/bin/env python3
"""Compare candidate face trackers against saved face-box reference annotations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_roi_reference import box_scores, read_tracks, xywh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data" / "annotations" / "roi_reference_review.json"
DEFAULT_BASELINE_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_MIN_NEIGHBORS_TRACKS = ROOT / "data" / "processed" / "face_tracking_min_neighbors_2" / "face_tracks.csv"
DEFAULT_SCALE_TRACKS = ROOT / "data" / "processed" / "face_tracking_scale_105_neighbors_2" / "face_tracks.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "annotations"


def summarize(values: list[dict[str, float]]) -> dict[str, float | int]:
    return {
        "box_count": len(values),
        "mean_iou": statistics.mean(item["iou"] for item in values),
        "median_iou": statistics.median(item["iou"] for item in values),
        "mean_center_error_pixels": statistics.mean(item["center_error_pixels"] for item in values),
        "median_center_error_pixels": statistics.median(item["center_error_pixels"] for item in values),
    }


def run(
    reference_path: Path,
    variant_paths: dict[str, Path],
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_csv = output_dir / "owner_tracker_variant_comparison.csv"
    output_json = output_dir / "owner_tracker_variant_comparison.json"
    existing = [path for path in (output_csv, output_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Variant comparison outputs already exist: {existing}; pass --overwrite")

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    tracker_sets = {name: read_tracks(path) for name, path in variant_paths.items()}
    comparisons: dict[str, list[dict[str, Any]]] = {name: [] for name in variant_paths}
    by_subject: dict[str, dict[str, list[dict[str, float]]]] = {
        name: defaultdict(list) for name in variant_paths
    }
    by_region: dict[str, dict[str, list[dict[str, float]]]] = {
        name: defaultdict(list) for name in variant_paths
    }
    missing_reference_boxes = 0

    for item in reference.get("annotations", []):
        subject = str(item["subject"])
        frame_index = int(item["frame_index"])
        reviewed = set(item.get("reviewed_regions", []))
        unavailable = set(item.get("not_visible_regions", []))
        for region in sorted(reviewed - unavailable):
            manual_box = item.get("boxes_xywh", {}).get(region)
            if manual_box is None:
                missing_reference_boxes += 1
                continue
            for variant, tracks in tracker_sets.items():
                row = tracks.get((subject.casefold(), frame_index))
                if row is None:
                    raise ValueError(f"Missing {variant} tracker row for {subject} frame {frame_index}")
                predicted = xywh(row, region)
                scores = box_scores(manual_box, predicted) if predicted else None
                record: dict[str, Any] = {
                    "variant": variant,
                    "subject": subject,
                    "frame_index": frame_index,
                    "region": region,
                    "tracker_box_available": predicted is not None,
                    "iou": scores["iou"] if scores else "",
                    "center_error_pixels": scores["center_error_pixels"] if scores else "",
                    "center_error_reference_diagonals": scores["center_error_reference_diagonals"] if scores else "",
                    "predicted_to_reference_area_ratio": scores["predicted_to_reference_area_ratio"] if scores else "",
                }
                comparisons[variant].append(record)
                if scores:
                    by_subject[variant][subject.casefold()].append(scores)
                    by_region[variant][region].append(scores)

    summary: dict[str, Any] = {}
    for variant, records in comparisons.items():
        available = [
            {key: float(row[key]) for key in ("iou", "center_error_pixels")}
            for row in records
            if row["tracker_box_available"]
        ]
        variant_summary = summarize(available) if available else {"box_count": 0}
        variant_summary["missing_tracker_boxes"] = len(records) - len(available)
        variant_summary["by_subject"] = {
            subject: summarize(scores)
            for subject, scores in sorted(by_subject[variant].items())
        }
        variant_summary["by_region"] = {
            region: summarize(scores)
            for region, scores in sorted(by_region[variant].items())
        }
        summary[variant] = variant_summary

    rows = [row for variant in variant_paths for row in comparisons[variant]]
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant", "subject", "frame_index", "region", "tracker_box_available", "iou",
        "center_error_pixels", "center_error_reference_diagonals", "predicted_to_reference_area_ratio",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "reference_input": str(reference_path),
        "reference_annotator": reference.get("metadata", {}).get("annotator"),
        "reference_provenance": reference.get("metadata", {}).get("annotation_role"),
        "complete_reference_frames": sum(item.get("review_status") == "complete" for item in reference.get("annotations", [])),
        "reference_box_count": sum(
            len(item.get("boxes_xywh", {})) for item in reference.get("annotations", [])
        ),
        "reviewed_regions_marked_not_visible": sum(
            len(item.get("not_visible_regions", [])) for item in reference.get("annotations", [])
        ),
        "reviewed_regions_missing_box": missing_reference_boxes,
        "interpretation": "Preliminary agreement on this single-annotator sample only; not dataset-wide accuracy or independent ground truth.",
        "variants": {name: str(path) for name, path in variant_paths.items()},
        "summary": summary,
        "outputs": {"csv": output_csv.name, "json": output_json.name},
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {sum(len(rows) for rows in comparisons.values())} tracker boxes across {len(variant_paths)} variants")
    print(f"Wrote {output_csv} and {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline-tracks", type=Path, default=DEFAULT_BASELINE_TRACKS)
    parser.add_argument("--min-neighbors-tracks", type=Path, default=DEFAULT_MIN_NEIGHBORS_TRACKS)
    parser.add_argument("--scale-tracks", type=Path, default=DEFAULT_SCALE_TRACKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.reference.resolve(),
        {
            "baseline": args.baseline_tracks.resolve(),
            "min_neighbors_2": args.min_neighbors_tracks.resolve(),
            "scale_1.05_min_neighbors_2": args.scale_tracks.resolve(),
        },
        args.output_dir.resolve(),
        args.overwrite,
    )


if __name__ == "__main__":
    main()

