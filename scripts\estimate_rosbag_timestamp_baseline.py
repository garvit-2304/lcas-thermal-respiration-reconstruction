#!/usr/bin/env python3
"""Estimate a whole-recording respiration-rate baseline from timestamped raw ROIs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, periodogram, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNALS = ROOT / "data" / "processed" / "roi_signals_raw_mono16.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "respiration_baseline_rosbag_raw"
DEFAULT_SIGNAL_COLUMN = "nose_mean_temperature_c"
OUTPUTS = {
    "rosbag_raw_filtered_signals.csv",
    "rosbag_raw_estimates.csv",
    "rosbag_raw_report.json",
}


def load_signals(path: Path, column: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "frame_index", "image_header_stamp_unix", column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ROI signal CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row.get(column, ""):
                groups[row["subject"].casefold()].append(row)
    if not groups:
        raise ValueError(f"No usable signal values found in {path}")
    return groups


def run(
    signals_path: Path,
    output_dir: Path,
    signal_column: str,
    low_bpm: float,
    high_bpm: float,
    filter_order: int,
    overwrite: bool,
) -> None:
    if not 0 < low_bpm < high_bpm:
        raise ValueError("Require 0 < low_bpm < high_bpm")
    groups = load_signals(signals_path, signal_column)
    is_temperature = signal_column.endswith("_temperature_c")
    signal_unit = (
        "degrees Celsius using the official L-CAS mono16 encoding conversion"
        if is_temperature
        else "raw mono16 sensor counts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = OUTPUTS.intersection(path.name for path in output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"Baseline outputs already exist under {output_dir}: {sorted(existing)}; pass --overwrite"
        )

    low_hz, high_hz = low_bpm / 60.0, high_bpm / 60.0
    estimate_rows: list[dict[str, object]] = []
    report_subjects: list[dict[str, object]] = []
    filtered_path = output_dir / "rosbag_raw_filtered_signals.csv"
    with filtered_path.open("w", newline="", encoding="utf-8") as filtered_file:
        filtered_fields = [
            "subject",
            "frame_index",
            "image_header_stamp_unix",
            "seconds_since_first_image",
            "source_signal_column",
            "source_signal_value",
            "signal_unit",
            "uniform_time_s",
            "resampled_intensity",
            "butterworth_filtered_intensity",
            "effective_sample_rate_hz",
            "timestamp_source",
        ]
        filtered_writer = csv.DictWriter(filtered_file, fieldnames=filtered_fields)
        filtered_writer.writeheader()

        for subject, rows in sorted(groups.items()):
            rows.sort(key=lambda row: int(row["frame_index"]))
            frame_indices = np.array([int(row["frame_index"]) for row in rows], dtype=np.int64)
            if np.any(np.diff(frame_indices) <= 0):
                raise ValueError(f"Frame indices must be unique and increasing for {subject}")
            missing_roi_frames_between_observations = int(
                frame_indices[-1] - frame_indices[0] + 1 - len(frame_indices)
            )
            timestamps = np.array(
                [float(row["image_header_stamp_unix"]) for row in rows], dtype=np.float64
            )
            values = np.array([float(row[signal_column]) for row in rows], dtype=np.float64)
            if len(values) < 16:
                raise ValueError(f"Too few ROI samples for {subject}: {len(values)}")
            if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"Image header timestamps must increase for {subject}")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite ROI values for {subject}")

            elapsed = timestamps - timestamps[0]
            duration_s = float(elapsed[-1])
            sample_rate_hz = float((len(values) - 1) / duration_s)
            if high_hz >= sample_rate_hz / 2.0:
                raise ValueError(
                    f"Requested band high edge {high_hz:.5g} Hz exceeds Nyquist "
                    f"({sample_rate_hz / 2.0:.5g} Hz) for {subject}"
                )
            uniform_times = np.linspace(0.0, duration_s, len(values))
            resampled = np.interp(uniform_times, elapsed, values)
            sos = butter(
                filter_order,
                [low_hz, high_hz],
                btype="bandpass",
                fs=sample_rate_hz,
                output="sos",
            )
            filtered = sosfiltfilt(sos, resampled - np.mean(resampled))
            frequencies, power = periodogram(
                filtered,
                fs=sample_rate_hz,
                window="hann",
                detrend=False,
                scaling="spectrum",
            )
            in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
            if not np.any(in_band):
                raise ValueError(f"No periodogram bins in requested band for {subject}")
            candidate_indices = np.flatnonzero(in_band)
            peak_index = int(candidate_indices[np.argmax(power[in_band])])
            peak_hz = float(frequencies[peak_index])
            resolution_hz = float(sample_rate_hz / len(values))
            intervals = np.diff(timestamps)

            for i, row in enumerate(rows):
                filtered_writer.writerow(
                    {
                        "subject": subject,
                        "frame_index": int(frame_indices[i]),
                        "image_header_stamp_unix": f"{timestamps[i]:.9f}",
                        "seconds_since_first_image": f"{elapsed[i]:.9f}",
                        "source_signal_column": signal_column,
                        "source_signal_value": f"{values[i]:.8f}",
                        "signal_unit": signal_unit,
                        "uniform_time_s": f"{uniform_times[i]:.9f}",
                        "resampled_intensity": f"{resampled[i]:.8f}",
                        "butterworth_filtered_intensity": f"{filtered[i]:.10g}",
                        "effective_sample_rate_hz": f"{sample_rate_hz:.9g}",
                        "timestamp_source": "sensor_msgs/Image header stamp from original ROS bag",
                    }
                )

            estimate = {
                "subject": subject,
                "frame_count": len(rows),
                "missing_roi_frames_between_observations": missing_roi_frames_between_observations,
                "duration_seconds": duration_s,
                "effective_sample_rate_hz": sample_rate_hz,
                "median_image_interval_seconds": float(statistics.median(intervals)),
                "mean_image_interval_seconds": float(statistics.mean(intervals)),
                "signal_column": signal_column,
                "signal_unit": signal_unit,
                "band_hz": [low_hz, high_hz],
                "band_bpm": [low_bpm, high_bpm],
                "filter": {"type": "Butterworth bandpass", "order": filter_order, "zero_phase": True},
                "resampling": "linear interpolation to an evenly spaced grid over measured image-timestamp endpoints",
                "periodicity_estimator": "full-recording Hann periodogram peak",
                "dominant_frequency_hz": peak_hz,
                "estimated_rate_bpm": peak_hz * 60.0,
                "frequency_resolution_hz": resolution_hz,
                "timestamp_source": "sensor_msgs/Image header stamp from original ROS bag",
                "frame_to_png_mapping": "message order; one-to-one counts verified against PNG manifest",
                "annotation_codes_used": False,
                "accuracy_evaluated": False,
            }
            estimate_rows.append(estimate)
            report_subjects.append(
                {
                    **estimate,
                    "timestamp_start_unix": float(timestamps[0]),
                    "timestamp_end_unix": float(timestamps[-1]),
                    "minimum_image_interval_seconds": float(np.min(intervals)),
                    "maximum_image_interval_seconds": float(np.max(intervals)),
                }
            )
            print(
                f"{subject}: {peak_hz * 60.0:.2f} BPM, "
                f"{sample_rate_hz:.2f} effective Hz over {duration_s:.2f} s"
            )

    estimate_path = output_dir / "rosbag_raw_estimates.csv"
    estimate_fields = [
        "subject",
        "frame_count",
        "duration_seconds",
        "effective_sample_rate_hz",
        "median_image_interval_seconds",
        "mean_image_interval_seconds",
        "signal_column",
        "signal_unit",
        "band_bpm",
        "dominant_frequency_hz",
        "estimated_rate_bpm",
        "frequency_resolution_hz",
        "annotation_codes_used",
        "accuracy_evaluated",
    ]
    with estimate_path.open("w", newline="", encoding="utf-8") as estimate_file:
        writer = csv.DictWriter(estimate_file, fieldnames=estimate_fields)
        writer.writeheader()
        for row in estimate_rows:
            writer.writerow(
                {
                    "subject": row["subject"],
                    "frame_count": row["frame_count"],
                    "duration_seconds": f"{row['duration_seconds']:.6f}",
                    "effective_sample_rate_hz": f"{row['effective_sample_rate_hz']:.9g}",
                    "median_image_interval_seconds": f"{row['median_image_interval_seconds']:.9g}",
                    "mean_image_interval_seconds": f"{row['mean_image_interval_seconds']:.9g}",
                    "signal_column": row["signal_column"],
                    "signal_unit": row["signal_unit"],
                    "band_bpm": f"{low_bpm:g}-{high_bpm:g}",
                    "dominant_frequency_hz": f"{row['dominant_frequency_hz']:.9g}",
                    "estimated_rate_bpm": f"{row['estimated_rate_bpm']:.6f}",
                    "frequency_resolution_hz": f"{row['frequency_resolution_hz']:.9g}",
                    "annotation_codes_used": False,
                    "accuracy_evaluated": False,
                }
            )

    report = {
        "input_signals": str(signals_path),
        "signal_unit": signal_unit,
        "temperature_conversion": "(mono16_count - 1000) / 10.0 degrees Celsius" if is_temperature else None,
        "temperature_conversion_source": "https://lcas.lincoln.ac.uk/wp/research/data-sets-software/lcas-thermal-physiological-monitoring-dataset/" if is_temperature else None,
        "timestamp_source": "sensor_msgs/Image header stamps recovered from original ROS1 bags",
        "method": "irregular ROI samples are linearly resampled to a uniform grid; zero-phase Butterworth bandpass followed by a full-recording Hann periodogram",
        "respiration_band_bpm": [low_bpm, high_bpm],
        "annotation_codes_used": False,
        "accuracy_evaluated": False,
        "limitations": [
            "Geometric candidate regions and the face tracker have not been validated against landmarks or manual masks.",
            "No motion or ROI-quality rejection is applied; changes in face position or ROI geometry can affect the full-recording spectral peak.",
            "Missing ROI observations are omitted and bridged by linear interpolation during uniform-grid resampling.",
            "This baseline does not use respiration annotations; their project-owner-confirmed working map is documented separately and is not an official dataset codebook.",
            "Temperature values use the documented L-CAS count conversion; no independent camera-calibration accuracy assessment is included.",
        ],
        "subjects": report_subjects,
        "outputs": sorted(OUTPUTS),
    }
    (output_dir / "rosbag_raw_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote timestamped raw-sensor baseline under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signal-column", default=DEFAULT_SIGNAL_COLUMN)
    parser.add_argument("--low-bpm", type=float, default=10.0)
    parser.add_argument("--high-bpm", type=float, default=40.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.signals.resolve(),
        args.output_dir.resolve(),
        args.signal_column,
        args.low_bpm,
        args.high_bpm,
        args.filter_order,
        args.overwrite,
    )


if __name__ == "__main__":
    main()

