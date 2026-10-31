#!/usr/bin/env python3
"""Audit extracted L-CAS thermal PNG sequences without modifying source frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "original_dataset"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
FRAME_PATTERN = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)
TIMESTAMP_FORMAT = "%Y%m%d-%H:%M:%S.%f"


def read_reference(path: Path) -> dict[str, Any]:
    rows: list[list[str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        rows = [row for row in csv.reader(file_obj) if row]
    timestamps: list[datetime] = []
    for line_number, row in enumerate(rows, start=1):
        if len(row) != 2:
            raise ValueError(f"{path}:{line_number}: expected two columns")
        timestamps.append(datetime.strptime(row[0].strip(), TIMESTAMP_FORMAT))
    return {
        "row_count": len(rows),
        "start_time": timestamps[0].isoformat(timespec="milliseconds") if rows else None,
        "end_time": timestamps[-1].isoformat(timespec="milliseconds") if rows else None,
        "duration_seconds": round((timestamps[-1] - timestamps[0]).total_seconds(), 6)
        if len(timestamps) > 1
        else 0.0,
    }


def ordered_frames(subject_dir: Path) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    for path in subject_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_PATTERN.match(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])
    return frames


def frame_stats(image: np.ndarray) -> tuple[np.ndarray, bool, str]:
    if image.ndim == 2:
        return image, True, "grayscale"
    if image.ndim == 3 and image.shape[2] == 3:
        channels_identical = bool(
            np.array_equal(image[:, :, 0], image[:, :, 1])
            and np.array_equal(image[:, :, 1], image[:, :, 2])
        )
        if channels_identical:
            return image[:, :, 0], True, "RGB-replicated-grayscale"
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), False, "color"
    raise ValueError(f"Unsupported decoded image shape: {image.shape}")


def audit(input_dir: Path, output_dir: Path) -> None:
    manifest_path = output_dir / "thermal_frames_manifest.csv"
    alignment_path = output_dir / "frame_alignment_audit.csv"
    report_path = output_dir / "thermal_frame_audit.json"
    existing = [path for path in (manifest_path, alignment_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Audit outputs already exist: " + ", ".join(str(path) for path in existing)
        )

    annotation_paths = {
        path.stem.rsplit("-", 1)[-1].rsplit("_", 1)[0].casefold(): path
        for path in input_dir.glob("*_respiration.csv")
    }
    subject_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(f"No subject image folders found under {input_dir}")
    unknown_subjects = [path.name for path in subject_dirs if path.name.casefold() not in annotation_paths]
    if unknown_subjects:
        raise ValueError(f"Image folders have no respiration reference CSV: {unknown_subjects}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_fields = [
        "subject",
        "frame_index",
        "filename",
        "relative_path",
        "width",
        "height",
        "channels",
        "dtype",
        "pixel_encoding",
        "channels_identical",
        "pixel_min",
        "pixel_max",
        "mean_grayscale_intensity",
        "fraction_zero",
        "fraction_255",
    ]
    alignment_fields = [
        "subject",
        "image_frame_count",
        "respiration_annotation_rows",
        "count_difference_frames_minus_annotations",
        "frame_count_matches_annotation_rows",
        "alignment_status",
        "reference_start_time",
        "reference_end_time",
        "reference_duration_seconds",
    ]
    subject_reports: list[dict[str, Any]] = []
    total_frames = 0
    all_dtypes: Counter[str] = Counter()
    all_encodings: Counter[str] = Counter()
    all_shapes: Counter[str] = Counter()
    decoded_count = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file, alignment_path.open(
        "w", newline="", encoding="utf-8"
    ) as alignment_file:
        manifest_writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        alignment_writer = csv.DictWriter(alignment_file, fieldnames=alignment_fields)
        manifest_writer.writeheader()
        alignment_writer.writeheader()

        for subject_dir in subject_dirs:
            subject = subject_dir.name
            annotation_path = annotation_paths[subject.casefold()]
            reference = read_reference(annotation_path)
            frames = ordered_frames(subject_dir)
            if not frames:
                raise ValueError(f"No frame_*.png files found in {subject_dir}")

            indices = [index for index, _ in frames]
            expected_indices = list(range(indices[0], indices[-1] + 1))
            missing_indices = sorted(set(expected_indices) - set(indices))
            if indices[0] != 1 or missing_indices:
                raise ValueError(
                    f"{subject}: frame sequence must be contiguous from 1; "
                    f"first={indices[0]}, missing={missing_indices[:10]}"
                )

            shapes: Counter[str] = Counter()
            dtypes: Counter[str] = Counter()
            encodings: Counter[str] = Counter()
            identical_count = 0
            intensity_mins: list[int] = []
            intensity_maxs: list[int] = []
            intensity_means: list[float] = []
            failed_frames: list[str] = []

            for frame_index, path in frames:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    failed_frames.append(path.name)
                    continue
                grayscale, channels_identical, encoding = frame_stats(image)
                if channels_identical:
                    identical_count += 1
                dtype = str(image.dtype)
                shape = "x".join(str(value) for value in image.shape)
                pixel_min = int(grayscale.min())
                pixel_max = int(grayscale.max())
                pixel_mean = float(grayscale.mean())
                pixel_count = grayscale.size
                fraction_zero = float(np.count_nonzero(grayscale == 0)) / pixel_count
                fraction_255 = float(np.count_nonzero(grayscale == 255)) / pixel_count

                manifest_writer.writerow(
                    {
                        "subject": subject,
                        "frame_index": frame_index,
                        "filename": path.name,
                        "relative_path": path.relative_to(ROOT).as_posix(),
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "channels": 1 if image.ndim == 2 else image.shape[2],
                        "dtype": dtype,
                        "pixel_encoding": encoding,
                        "channels_identical": channels_identical,
                        "pixel_min": pixel_min,
                        "pixel_max": pixel_max,
                        "mean_grayscale_intensity": f"{pixel_mean:.6f}",
                        "fraction_zero": f"{fraction_zero:.8f}",
                        "fraction_255": f"{fraction_255:.8f}",
                    }
                )
                shapes[shape] += 1
                dtypes[dtype] += 1
                encodings[encoding] += 1
                intensity_mins.append(pixel_min)
                intensity_maxs.append(pixel_max)
                intensity_means.append(pixel_mean)
                decoded_count += 1

            if failed_frames:
                raise ValueError(f"{subject}: failed to decode frames {failed_frames[:10]}")

            frame_count = len(frames)
            total_frames += frame_count
            all_dtypes.update(dtypes)
            all_encodings.update(encodings)
            all_shapes.update(shapes)
            row_difference = frame_count - reference["row_count"]
            count_matches = row_difference == 0
            alignment_status = (
                "count_match_only_not_timestamp_verified"
                if count_matches
                else "unresolved_frame_annotation_count_mismatch"
            )
            alignment_writer.writerow(
                {
                    "subject": subject,
                    "image_frame_count": frame_count,
                    "respiration_annotation_rows": reference["row_count"],
                    "count_difference_frames_minus_annotations": row_difference,
                    "frame_count_matches_annotation_rows": count_matches,
                    "alignment_status": alignment_status,
                    "reference_start_time": reference["start_time"],
                    "reference_end_time": reference["end_time"],
                    "reference_duration_seconds": reference["duration_seconds"],
                }
            )
            subject_reports.append(
                {
                    "subject": subject,
                    "frame_count": frame_count,
                    "frame_index_first": indices[0],
                    "frame_index_last": indices[-1],
                    "sequence_contiguous": True,
                    "decoded_frame_count": sum(shapes.values()),
                    "annotation_file": annotation_path.relative_to(ROOT).as_posix(),
                    "annotation_row_count": reference["row_count"],
                    "frame_annotation_count_difference": row_difference,
                    "frame_count_matches_annotation_rows": count_matches,
                    "alignment_status": alignment_status,
                    "annotation_start_time": reference["start_time"],
                    "annotation_end_time": reference["end_time"],
                    "annotation_duration_seconds": reference["duration_seconds"],
                    "image_shapes": dict(shapes),
                    "image_dtypes": dict(dtypes),
                    "pixel_encodings": dict(encodings),
                    "all_channels_identical": identical_count == frame_count,
                    "pixel_minimum_over_frames": min(intensity_mins),
                    "pixel_maximum_over_frames": max(intensity_maxs),
                    "mean_of_frame_mean_intensities": round(
                        float(np.mean(intensity_means)), 6
                    ),
                }
            )

    report = {
        "dataset": "L-CAS extracted thermal image frames",
        "input_directory": input_dir.relative_to(ROOT).as_posix()
        if input_dir.is_relative_to(ROOT)
        else str(input_dir),
        "frame_count": total_frames,
        "decoded_frame_count": decoded_count,
        "subjects": subject_reports,
        "image_shapes": dict(all_shapes),
        "image_dtypes": dict(all_dtypes),
        "pixel_encodings": dict(all_encodings),
        "preprocessing_notes": [
            "Source PNG files were not modified.",
            "When channels are identical, the first channel is used only to summarize grayscale intensity.",
            "8-bit PNG grayscale values are image intensities, not calibrated uint16 Optris temperatures.",
            "Frame filenames provide ordinal indices only; no image timestamps were used or invented.",
            "Equal frame/reference row counts are reported as candidate index alignment only, not timestamp verification.",
        ],
        "manifest_csv": manifest_path.relative_to(ROOT).as_posix()
        if manifest_path.is_relative_to(ROOT)
        else str(manifest_path),
        "alignment_csv": alignment_path.relative_to(ROOT).as_posix()
        if alignment_path.is_relative_to(ROOT)
        else str(alignment_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Audited {decoded_count}/{total_frames} thermal PNG frames across {len(subject_reports)} subjects.")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {alignment_path}")
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit(args.input_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

