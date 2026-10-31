#!/usr/bin/env python3
"""Extract synchronized facial ROI signals from recovered mono16 ROS frames."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from extract_roi_signals import REGIONS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "processed" / "raw_thermal"
DEFAULT_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_NAME = "roi_signals_raw_mono16.csv"
REPORT_NAME = "roi_signals_raw_mono16_report.json"
TEMPERATURE_CONVERSION = "(mono16_count - 1000) / 10.0"
TEMPERATURE_SOURCE = "https://lcas.lincoln.ac.uk/wp/research/data-sets-software/lcas-thermal-physiological-monitoring-dataset/"


def extract(raw_dir: Path, tracks_path: Path, output_dir: Path, overwrite: bool) -> None:
    output_csv = output_dir / OUTPUT_NAME
    report_path = output_dir / REPORT_NAME
    if (output_csv.exists() or report_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Raw ROI outputs already exist in {output_dir}; choose another --output-dir."
        )
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Face-track CSV not found: {tracks_path}")

    tracks_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    with tracks_path.open("r", newline="", encoding="utf-8-sig") as track_file:
        for row in csv.DictReader(track_file):
            tracks_by_subject[row["subject"].casefold()].append(row)

    signal_fields = [
        "subject",
        "frame_index",
        "image_header_stamp_unix",
        "seconds_since_first_image",
        "tracking_status",
        "valid_roi",
        "face_mean_intensity",
        "face_median_intensity",
        "face_mean_temperature_c",
        "face_median_temperature_c",
        "face_pixel_count",
    ]
    for region in REGIONS:
        signal_fields.extend(
            [
                f"{region}_mean_intensity",
                f"{region}_median_intensity",
                f"{region}_mean_temperature_c",
                f"{region}_median_temperature_c",
                f"{region}_pixel_count",
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_reports: list[dict[str, Any]] = []
    processed_total = 0
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=signal_fields)
        writer.writeheader()

        for raw_path in sorted(raw_dir.glob("*_thermal_raw.h5")):
            subject = raw_path.name.removesuffix("_thermal_raw.h5").casefold()
            tracks = tracks_by_subject.get(subject)
            if not tracks:
                raise ValueError(f"No face tracks found for raw thermal subject {subject}")
            tracks.sort(key=lambda row: int(row["frame_index"]))

            with h5py.File(raw_path, "r") as thermal_file:
                frames = thermal_file["frames"]
                timestamps = thermal_file["image_header_stamp_unix"]
                frame_count = int(frames.shape[0])
                if len(tracks) != frame_count:
                    raise ValueError(
                        f"Track/raw frame count mismatch for {subject}: {len(tracks)} vs {frame_count}"
                    )
                expected_indices = list(range(1, frame_count + 1))
                track_indices = [int(row["frame_index"]) for row in tracks]
                if track_indices != expected_indices:
                    raise ValueError(f"Face track indices are not contiguous for {subject}")

                timestamp_origin = float(timestamps[0])
                valid_count = 0
                signal_values: dict[str, list[float]] = defaultdict(list)
                temperature_values: dict[str, list[float]] = defaultdict(list)
                for position, track in enumerate(tracks):
                    frame = frames[position]
                    row: dict[str, Any] = {
                        "subject": subject,
                        "frame_index": position + 1,
                        "image_header_stamp_unix": f"{float(timestamps[position]):.9f}",
                        "seconds_since_first_image": f"{float(timestamps[position]) - timestamp_origin:.9f}",
                        "tracking_status": track["tracking_status"],
                        "valid_roi": track["valid_roi"],
                    }
                    if track["valid_roi"].lower() == "true":
                        face = _crop(frame, track, "face", subject, position + 1)
                        face_mean = float(face.mean())
                        face_median = float(np.median(face))
                        row["face_mean_intensity"] = f"{face_mean:.6f}"
                        row["face_median_intensity"] = f"{face_median:.6f}"
                        row["face_mean_temperature_c"] = f"{(face_mean - 1000.0) / 10.0:.6f}"
                        row["face_median_temperature_c"] = f"{(face_median - 1000.0) / 10.0:.6f}"
                        row["face_pixel_count"] = int(face.size)
                        signal_values["face_mean_intensity"].append(face_mean)
                        temperature_values["face_mean_temperature_c"].append((face_mean - 1000.0) / 10.0)
                        valid_count += 1

                        for region in REGIONS:
                            crop = _crop(frame, track, region, subject, position + 1)
                            mean_value = float(crop.mean())
                            median_value = float(np.median(crop))
                            row[f"{region}_mean_intensity"] = f"{mean_value:.6f}"
                            row[f"{region}_median_intensity"] = f"{median_value:.6f}"
                            row[f"{region}_mean_temperature_c"] = f"{(mean_value - 1000.0) / 10.0:.6f}"
                            row[f"{region}_median_temperature_c"] = f"{(median_value - 1000.0) / 10.0:.6f}"
                            row[f"{region}_pixel_count"] = int(crop.size)
                            signal_values[f"{region}_mean_intensity"].append(mean_value)
                            temperature_values[f"{region}_mean_temperature_c"].append(
                                (mean_value - 1000.0) / 10.0
                            )
                    else:
                        for field in signal_fields:
                            row.setdefault(field, "")
                    writer.writerow(row)
                    processed_total += 1

                subject_reports.append(
                    {
                        "subject": subject,
                        "source_hdf5": raw_path.relative_to(ROOT).as_posix()
                        if raw_path.is_relative_to(ROOT)
                        else str(raw_path),
                        "frame_count": frame_count,
                        "valid_roi_frames": valid_count,
                        "raw_signal_unit": "raw mono16 sensor counts, preserved",
                        "temperature_signal_unit": "degrees Celsius using the official L-CAS mono16 encoding conversion",
                        "temperature_conversion": TEMPERATURE_CONVERSION,
                        "temperature_conversion_source": TEMPERATURE_SOURCE,
                        "timestamp_source": "sensor_msgs/Image header stamp stored with each ROS bag frame",
                        "tracking_source": "face_tracks.csv coordinates established on the matching 8-bit PNG sequence",
                        "signals": {
                            name: {
                                "sample_count": len(values),
                                "mean": float(statistics.mean(values)) if values else None,
                                "standard_deviation": float(statistics.pstdev(values)) if values else None,
                            }
                            for name, values in sorted(signal_values.items())
                        },
                        "temperature_signals": {
                            name: {
                                "sample_count": len(values),
                                "mean_c": float(statistics.mean(values)) if values else None,
                                "standard_deviation_c": float(statistics.pstdev(values)) if values else None,
                            }
                            for name, values in sorted(temperature_values.items())
                        },
                    }
                )

    if not subject_reports:
        raise FileNotFoundError(f"No *_thermal_raw.h5 files found in {raw_dir}")
    report = {
        "input_raw_dir": str(raw_dir),
        "face_tracks": str(tracks_path),
        "timestamp_handling": "Each signal row includes the ROS image header timestamp; no frame-rate assumption is used.",
        "pixel_semantics": "Original mono16 sensor counts are retained. Additional temperature columns use the official L-CAS encoding conversion (count - 1000) / 10.0 to degrees Celsius.",
        "temperature_conversion_source": TEMPERATURE_SOURCE,
        "roi_geometry": "Existing PNG-derived face and candidate facial-region rectangles are applied to the same-index raw ROS frame.",
        "total_rows": processed_total,
        "subjects": subject_reports,
        "output_csv": str(output_csv),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted raw mono16 ROI signals for {processed_total} frames.")
    print(f"Wrote {output_csv}")
    print(f"Wrote {report_path}")


def _crop(frame: np.ndarray, track: dict[str, str], name: str, subject: str, frame_index: int) -> np.ndarray:
    x = int(track[f"{name}_x"])
    y = int(track[f"{name}_y"])
    width = int(track[f"{name}_width"])
    height = int(track[f"{name}_height"])
    crop = frame[y : y + height, x : x + width]
    if crop.size == 0 or crop.shape != (height, width):
        raise ValueError(f"Invalid {name} ROI for {subject} frame {frame_index}")
    return crop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract(
        args.raw_dir.resolve(),
        args.tracks.resolve(),
        args.output_dir.resolve(),
        args.overwrite,
    )


if __name__ == "__main__":
    main()

