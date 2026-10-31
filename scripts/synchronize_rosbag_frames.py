#!/usr/bin/env python3
"""Build a PNG-frame timestamp map and audit timing against L-CAS annotations."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAG_TIMESTAMPS = ROOT / "data" / "processed" / "rosbag_timestamps" / "rosbag_message_timestamps.csv"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "thermal_frames_manifest.csv"
DEFAULT_REFERENCE = ROOT / "data" / "processed" / "physiology_long.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "rosbag_timestamps"
RESPIRATION_CODE_LABELS = {
    "0": "no_event",
    "1": "inhalation",
    "2": "exhalation",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        return list(csv.DictReader(file_obj))


def source_timestamp_to_unix(timestamp: str) -> float:
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        # This interpretation is empirically supported by agreement with the
        # ROS epoch timestamps; the source CSV itself does not declare a zone.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def closest_reference(timestamp: float, ref_times: list[float]) -> int:
    insertion = bisect.bisect_left(ref_times, timestamp)
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(ref_times)]
    if not candidates:
        raise ValueError("No respiration reference timestamps available")
    return min(candidates, key=lambda index: (abs(ref_times[index] - timestamp), index))


def run(
    bag_timestamps_path: Path,
    manifest_path: Path,
    reference_path: Path,
    output_dir: Path,
    match_tolerance_ms: float,
    overwrite: bool,
) -> None:
    managed_names = {
        "frame_timestamps.csv",
        "frame_reference_nearest.csv",
        "frame_reference_synchronization.json",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = managed_names.intersection(path.name for path in output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"Synchronization outputs already exist under {output_dir}: {sorted(existing)}; "
            "pass --overwrite"
        )

    bag_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(bag_timestamps_path):
        if row["message_type"] != "sensor_msgs/Image":
            continue
        bag_groups[row["subject"].casefold()].append(row)
    for subject, rows in bag_groups.items():
        rows.sort(key=lambda row: int(row["message_index"]))

    png_counts: Counter[str] = Counter()
    manifest_indices: dict[str, list[int]] = defaultdict(list)
    for row in read_csv(manifest_path):
        subject = row["subject"].casefold()
        png_counts[subject] += 1
        manifest_indices[subject].append(int(row["frame_index"]))

    reference_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(reference_path):
        if row["stream"] == "respiration":
            subject = row["recording"].rsplit("-", 1)[-1].casefold()
            reference_groups[subject].append(row)
    for subject, rows in reference_groups.items():
        rows.sort(key=lambda row: int(row["sample_index"]))

    if not bag_groups:
        raise ValueError(f"No sensor_msgs/Image timestamps found in {bag_timestamps_path}")
    if set(bag_groups) != set(png_counts):
        raise ValueError(
            f"Bag subjects and PNG-manifest subjects differ: bags={sorted(bag_groups)}, "
            f"PNGs={sorted(png_counts)}"
        )

    frame_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    subject_reports: list[dict[str, object]] = []
    tolerance_s = match_tolerance_ms / 1000.0

    for subject in sorted(bag_groups):
        images = bag_groups[subject]
        indices = manifest_indices[subject]
        expected_indices = list(range(1, png_counts[subject] + 1))
        if indices != expected_indices:
            raise ValueError(f"PNG frame indices are not contiguous for {subject}")
        if len(images) != png_counts[subject]:
            raise ValueError(
                f"Bag/PNG count mismatch for {subject}: {len(images)} bag images, "
                f"{png_counts[subject]} PNGs"
            )
        message_indices = [int(row["message_index"]) for row in images]
        if message_indices != list(range(len(images))):
            raise ValueError(f"Bag message indices are not contiguous for {subject}")
        image_times = [float(row["image_header_stamp_unix"]) for row in images]
        if not all(timestamp > 0 for timestamp in image_times):
            raise ValueError(f"Missing image header timestamps for {subject}")
        intervals = [right - left for left, right in zip(image_times, image_times[1:])]
        if any(interval <= 0 for interval in intervals):
            raise ValueError(f"Image header timestamps are not strictly increasing for {subject}")
        duration = image_times[-1] - image_times[0]

        refs = reference_groups.get(subject, [])
        ref_times = [source_timestamp_to_unix(row["timestamp"]) for row in refs]
        same_index_deltas: list[float] = []
        if len(images) == len(refs):
            same_index_deltas = [image_times[i] - ref_times[i] for i in range(len(images))]

        nearest_indices: list[int] = []
        nearest_offsets: list[float] = []
        for row, timestamp in zip(images, image_times):
            message_index = int(row["message_index"])
            frame_index = message_index + 1
            frame_rows.append(
                {
                    "subject": subject,
                    "frame_index": frame_index,
                    "rosbag_message_index": message_index,
                    "topic": row["topic"],
                    "message_type": row["message_type"],
                    "bag_timestamp_unix": row["bag_timestamp_unix"],
                    "image_header_stamp_unix": row["image_header_stamp_unix"],
                    "seconds_since_first_image": f"{timestamp - image_times[0]:.9f}",
                }
            )

            if not refs:
                continue
            ref_index = closest_reference(timestamp, ref_times)
            reference = refs[ref_index]
            delta = ref_times[ref_index] - timestamp
            raw_code = reference["value"].strip()
            annotation_label = reference.get("value_label", "").strip()
            if not annotation_label:
                try:
                    numeric_code = float(raw_code)
                    normalized_code = str(int(numeric_code)) if numeric_code.is_integer() else raw_code
                except ValueError:
                    normalized_code = raw_code
                annotation_label = RESPIRATION_CODE_LABELS.get(normalized_code, "unmapped_code")
            nearest_indices.append(ref_index)
            nearest_offsets.append(delta)
            reference_rows.append(
                {
                    "subject": subject,
                    "frame_index": frame_index,
                    "image_header_stamp_unix": f"{timestamp:.9f}",
                    "nearest_reference_sample_index": reference["sample_index"],
                    "nearest_reference_timestamp": reference["timestamp"],
                    "nearest_reference_offset_seconds": f"{delta:.9f}",
                    "nearest_reference_offset_within_tolerance": str(abs(delta) <= tolerance_s).lower(),
                    "nearest_reference_annotation_code_raw": raw_code,
                    "nearest_reference_annotation_label": annotation_label,
                    "annotation_code_semantics": (
                        "project-owner-confirmed working map; raw code preserved; "
                        "nearest-by-time association only"
                    ),
                }
            )

        report: dict[str, object] = {
            "subject": subject,
            "png_frame_count": png_counts[subject],
            "rosbag_image_message_count": len(images),
            "frame_count_match": True,
            "duration_seconds": duration,
            "median_image_interval_seconds": statistics.median(intervals),
            "mean_image_interval_seconds": statistics.mean(intervals),
            "effective_mean_sampling_rate_hz": (len(images) - 1) / duration,
            "min_image_interval_seconds": min(intervals),
            "max_image_interval_seconds": max(intervals),
            "image_header_minus_bag_time_min_seconds": min(
                float(row["header_minus_bag_seconds"]) for row in images
            ),
            "image_header_minus_bag_time_max_seconds": max(
                float(row["header_minus_bag_seconds"]) for row in images
            ),
            "respiration_reference_sample_count": len(refs),
            "reference_timestamp_zone_assumption": "naive source timestamps interpreted as UTC for comparison with ROS Unix epoch",
        }
        if refs:
            report.update(
                {
                    "reference_duration_seconds": ref_times[-1] - ref_times[0],
                    "first_image_minus_first_reference_seconds": image_times[0] - ref_times[0],
                    "last_image_minus_last_reference_seconds": image_times[-1] - ref_times[-1],
                    "nearest_reference_unique_samples": len(set(nearest_indices)),
                    "nearest_reference_median_absolute_offset_seconds": statistics.median(
                        abs(value) for value in nearest_offsets
                    ),
                    "nearest_reference_max_absolute_offset_seconds": max(
                        abs(value) for value in nearest_offsets
                    ),
                    "nearest_reference_within_tolerance_count": sum(
                        abs(value) <= tolerance_s for value in nearest_offsets
                    ),
                    "nearest_reference_tolerance_ms": match_tolerance_ms,
                    "nearest_associated_raw_annotation_code_counts": dict(
                        Counter(refs[index]["value"] for index in nearest_indices)
                    ),
                }
            )
        if same_index_deltas:
            report["same_index_timestamp_match"] = {
                "count": len(same_index_deltas),
                "median_signed_delta_seconds": statistics.median(same_index_deltas),
                "median_absolute_delta_seconds": statistics.median(
                    abs(value) for value in same_index_deltas
                ),
                "max_absolute_delta_seconds": max(abs(value) for value in same_index_deltas),
                "within_20ms_count": sum(abs(value) <= 0.020 for value in same_index_deltas),
                "alignment_interpretation": "row order is independently corroborated by timestamps for this count-matched recording",
            }
        else:
            report["same_index_timestamp_match"] = None
        subject_reports.append(report)

    with (output_dir / "frame_timestamps.csv").open("w", newline="", encoding="utf-8") as file_obj:
        fields = [
            "subject",
            "frame_index",
            "rosbag_message_index",
            "topic",
            "message_type",
            "bag_timestamp_unix",
            "image_header_stamp_unix",
            "seconds_since_first_image",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    with (output_dir / "frame_reference_nearest.csv").open("w", newline="", encoding="utf-8") as file_obj:
        fields = [
            "subject",
            "frame_index",
            "image_header_stamp_unix",
            "nearest_reference_sample_index",
            "nearest_reference_timestamp",
            "nearest_reference_offset_seconds",
            "nearest_reference_offset_within_tolerance",
            "nearest_reference_annotation_code_raw",
            "nearest_reference_annotation_label",
            "annotation_code_semantics",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reference_rows)

    output_report = {
        "frame_timestamp_source": "sensor_msgs/Image header stamps recovered from original ROS1 bag records",
        "frame_to_png_mapping": "bag message order mapped to one-based PNG frame_index after per-subject counts matched the contiguous PNG manifest",
        "annotation_timestamp_comparison": "nearest source respiration timestamp; raw annotation codes retained and mapped to project-owner-confirmed working labels",
        "annotation_code_mapping": RESPIRATION_CODE_LABELS,
        "annotation_code_mapping_basis": "Project-owner-confirmed working interpretation on 2026-10-31; not an official dataset codebook.",
        "reference_timestamp_zone_assumption": "Source timestamps are timezone-naive; interpreted as UTC for Unix-epoch comparison because that yields sub-frame agreement with ROS timestamps",
        "subjects": subject_reports,
        "outputs": {
            "frame_timestamps_csv": "frame_timestamps.csv",
            "nearest_reference_csv": "frame_reference_nearest.csv",
        },
    }
    (output_dir / "frame_reference_synchronization.json").write_text(
        json.dumps(output_report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Mapped {len(frame_rows)} bag images to PNG frame indices.")
    print(f"Wrote synchronized timestamps and report under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-timestamps", type=Path, default=DEFAULT_BAG_TIMESTAMPS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--match-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.bag_timestamps.resolve(),
        args.manifest.resolve(),
        args.reference.resolve(),
        args.output_dir.resolve(),
        args.match_tolerance_ms,
        args.overwrite,
    )


if __name__ == "__main__":
    main()

