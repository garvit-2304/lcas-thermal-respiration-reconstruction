#!/usr/bin/env python3
"""Run an explicitly provisional baseline using count-matched reference times.

For recordings with equal PNG and respiration-row counts, this script assigns
reference sample timestamps to image frames by ordinal position. This is a
candidate alignment only; no image timestamps are available to verify it.
Unmatched recordings are excluded rather than force-joined.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, periodogram, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNALS = ROOT / "data" / "processed" / "roi_signals.csv"
DEFAULT_REFERENCE = ROOT / "data" / "processed" / "physiology_long.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "respiration_baseline_candidate_alignment"
ALIGNMENT_NOTE = (
    "provisional ordinal alignment: frame_index 1 is assigned respiration sample_index 0 "
    "because counts match; no shared image timestamp verifies this mapping"
)


def normalized_subject(name: str) -> str:
    return name.strip().casefold()


def load_inputs(
    signals_csv: Path,
    reference_csv: Path,
    signal_column: str,
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, list[tuple[int, float, str]]]]:
    image_rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with signals_csv.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "frame_index", signal_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ROI signals CSV is missing columns: {sorted(missing)}")
        for row in reader:
            subject = normalized_subject(row["subject"])
            value = float(row[signal_column])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite ROI value for {subject} frame {row['frame_index']}")
            image_rows[subject].append((int(row["frame_index"]), value))

    reference_rows: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    with reference_csv.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"recording", "stream", "sample_index", "relative_time_s", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row["stream"] != "respiration":
                continue
            subject = normalized_subject(row["recording"].rsplit("-", 1)[-1])
            reference_rows[subject].append(
                (int(row["sample_index"]), float(row["relative_time_s"]), row["value"])
            )

    for subject, rows in image_rows.items():
        rows.sort(key=lambda row: row[0])
        indices = [row[0] for row in rows]
        if len(set(indices)) != len(indices) or any(b - a != 1 for a, b in zip(indices, indices[1:])):
            raise ValueError(f"Duplicate or non-contiguous image frame indices for {subject}")
    for subject, rows in reference_rows.items():
        rows.sort(key=lambda row: row[0])
        indices = [row[0] for row in rows]
        if len(set(indices)) != len(indices) or any(b - a != 1 for a, b in zip(indices, indices[1:])):
            raise ValueError(f"Duplicate or non-contiguous reference indices for {subject}")

    if not image_rows or not reference_rows:
        raise ValueError("No usable ROI signals or respiration reference rows were found")
    return image_rows, reference_rows


def estimate(
    values: np.ndarray,
    timestamps_s: np.ndarray,
    low_hz: float,
    high_hz: float,
    filter_order: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    if len(values) != len(timestamps_s):
        raise ValueError("Image and timestamp arrays must have equal length")
    if len(values) < 16:
        raise ValueError("Too few aligned samples for a useful rate estimate")
    if not np.isfinite(timestamps_s).all() or np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("Candidate timestamps must be finite and strictly increasing")

    duration_s = float(timestamps_s[-1] - timestamps_s[0])
    if duration_s <= 0:
        raise ValueError("Candidate timestamp duration must be positive")
    fps = float((len(values) - 1) / duration_s)
    if not 0 < low_hz < high_hz < fps / 2.0:
        raise ValueError(
            f"Band must satisfy 0 < low < high < Nyquist ({fps / 2.0:.6g} Hz)"
        )

    uniform_times = np.linspace(timestamps_s[0], timestamps_s[-1], len(timestamps_s))
    uniform_values = np.interp(uniform_times, timestamps_s, values)
    sos = butter(filter_order, [low_hz, high_hz], btype="bandpass", fs=fps, output="sos")
    filtered = sosfiltfilt(sos, uniform_values - np.mean(uniform_values))
    frequencies, power = periodogram(
        filtered,
        fs=fps,
        window="hann",
        detrend=False,
        scaling="spectrum",
    )
    in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(in_band):
        raise ValueError("No periodogram bins fall within the requested band")
    band_indices = np.flatnonzero(in_band)
    peak_index = int(band_indices[np.argmax(power[in_band])])
    peak_hz = float(frequencies[peak_index])
    resolution_hz = float(fps / len(values))
    median_interval = float(np.median(np.diff(timestamps_s)))
    return (
        uniform_values,
        filtered,
        fps,
        duration_s,
        median_interval,
        peak_hz,
        resolution_hz,
    )


def run(
    signals_csv: Path,
    reference_csv: Path,
    output_dir: Path,
    signal_column: str,
    low_bpm: float,
    high_bpm: float,
    filter_order: int,
    overwrite: bool,
) -> None:
    image_rows, reference_rows = load_inputs(signals_csv, reference_csv, signal_column)
    managed = {"candidate_aligned_filtered_signals.csv", "candidate_aligned_estimates.csv", "candidate_aligned_report.json"}
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir() if path.is_file()}
        unexpected = existing - managed
        if unexpected:
            raise FileExistsError(f"Output directory has unmanaged files: {sorted(unexpected)}")
        if existing and not overwrite:
            raise FileExistsError(f"Outputs already exist under {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    low_hz, high_hz = low_bpm / 60.0, high_bpm / 60.0
    estimates: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    filtered_path = output_dir / "candidate_aligned_filtered_signals.csv"
    with filtered_path.open("w", newline="", encoding="utf-8") as file_obj:
        fields = [
            "subject",
            "frame_index",
            "candidate_reference_time_s",
            "uniform_time_s",
            "source_signal_column",
            "raw_image_intensity",
            "resampled_image_intensity",
            "butterworth_filtered_intensity",
            "effective_sample_rate_hz",
            "alignment_assumption",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()

        for subject, frames in sorted(image_rows.items()):
            refs = reference_rows.get(subject)
            if refs is None:
                skipped[subject] = "no respiration reference rows"
                continue
            if len(frames) != len(refs):
                skipped[subject] = (
                    f"count mismatch: {len(frames)} image frames versus {len(refs)} respiration rows"
                )
                continue

            sample_indices = np.array([row[0] for row in refs], dtype=np.int64)
            if not np.array_equal(sample_indices, np.arange(len(refs), dtype=np.int64)):
                skipped[subject] = "reference sample indices do not start at zero and remain contiguous"
                continue
            timestamps = np.array([row[1] for row in refs], dtype=np.float64)
            frames_index = np.array([row[0] for row in frames], dtype=np.int64)
            values = np.array([row[1] for row in frames], dtype=np.float64)

            (
                uniform_values,
                filtered,
                fps,
                duration_s,
                median_interval,
                peak_hz,
                resolution_hz,
            ) = estimate(values, timestamps, low_hz, high_hz, filter_order)
            uniform_times = np.linspace(timestamps[0], timestamps[-1], len(timestamps))
            codes = Counter(row[2] for row in refs)

            for index in range(len(frames)):
                writer.writerow(
                    {
                        "subject": subject,
                        "frame_index": int(frames_index[index]),
                        "candidate_reference_time_s": f"{timestamps[index]:.6f}",
                        "uniform_time_s": f"{uniform_times[index]:.6f}",
                        "source_signal_column": signal_column,
                        "raw_image_intensity": f"{values[index]:.8f}",
                        "resampled_image_intensity": f"{uniform_values[index]:.8f}",
                        "butterworth_filtered_intensity": f"{filtered[index]:.10g}",
                        "effective_sample_rate_hz": f"{fps:.8g}",
                        "alignment_assumption": ALIGNMENT_NOTE,
                    }
                )

            estimates.append(
                {
                    "subject": subject,
                    "frame_count": int(len(frames)),
                    "signal_column": signal_column,
                    "candidate_time_start_s": float(timestamps[0]),
                    "candidate_time_end_s": float(timestamps[-1]),
                    "duration_s": duration_s,
                    "effective_sample_rate_hz": fps,
                    "median_reference_interval_s": median_interval,
                    "resampling": "linear interpolation to an evenly spaced grid over candidate timestamp endpoints",
                    "filter": {"type": "Butterworth bandpass", "order": filter_order, "zero_phase": True},
                    "band_hz": [low_hz, high_hz],
                    "band_bpm": [low_bpm, high_bpm],
                    "periodicity_estimator": "full-recording Hann periodogram peak",
                    "frequency_resolution_hz": resolution_hz,
                    "dominant_frequency_hz": peak_hz,
                    "estimated_rate_bpm": peak_hz * 60.0,
                    "alignment_status": "candidate_count_match_ordinal_unverified",
                    "raw_annotation_code_counts_ignored": dict(codes),
                    "annotation_code_mapping_applied": False,
                    "accuracy_evaluated": False,
                }
            )

    if not estimates:
        raise ValueError("No count-matched recordings were available for candidate alignment")

    estimate_fields = [
        "subject",
        "frame_count",
        "duration_s",
        "effective_sample_rate_hz",
        "median_reference_interval_s",
        "signal_column",
        "low_bpm",
        "high_bpm",
        "dominant_frequency_hz",
        "estimated_rate_bpm",
        "frequency_resolution_hz",
        "alignment_status",
        "annotation_code_mapping_applied",
        "accuracy_evaluated",
    ]
    with (output_dir / "candidate_aligned_estimates.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=estimate_fields)
        writer.writeheader()
        for item in estimates:
            writer.writerow(
                {
                    "subject": item["subject"],
                    "frame_count": item["frame_count"],
                    "duration_s": f"{item['duration_s']:.6f}",
                    "effective_sample_rate_hz": f"{item['effective_sample_rate_hz']:.8g}",
                    "median_reference_interval_s": f"{item['median_reference_interval_s']:.8g}",
                    "signal_column": item["signal_column"],
                    "low_bpm": low_bpm,
                    "high_bpm": high_bpm,
                    "dominant_frequency_hz": f"{item['dominant_frequency_hz']:.8g}",
                    "estimated_rate_bpm": f"{item['estimated_rate_bpm']:.8g}",
                    "frequency_resolution_hz": f"{item['frequency_resolution_hz']:.8g}",
                    "alignment_status": item["alignment_status"],
                    "annotation_code_mapping_applied": False,
                    "accuracy_evaluated": False,
                }
            )

    report = {
        "method": "timestamp-aware conventional baseline using candidate ordinal frame/reference matching",
        "input_signals_csv": str(signals_csv),
        "input_reference_csv": str(reference_csv),
        "signal_column": signal_column,
        "sampling_rate_source": "respiration CSV relative timestamps assigned by ordinal position to image frames when counts match",
        "alignment_assumption": ALIGNMENT_NOTE,
        "respiration_band": {
            "low_bpm": low_bpm,
            "high_bpm": high_bpm,
            "source": "defaults in official LCAS physiological_monitoring ROS node",
        },
        "annotation_values": "codes are counted for reporting only; no code semantics are assumed or used",
        "limitations": [
            "Count equality does not prove that image frames and reference rows share timestamps; every estimate is provisional.",
            "Subjects with frame/reference count mismatch are skipped; no partial alignment is attempted.",
            "The PNG input contains 8-bit image intensity, not calibrated temperature.",
            "The full recording is summarized with one spectral peak; time-varying respiration is not estimated.",
            "No formal event-level accuracy is reported: this earlier ordinal-alignment result is superseded by timestamp alignment, and its values are not evaluated against mapped reference events.",
        ],
        "skipped_subjects": skipped,
        "recordings": estimates,
    }
    with (output_dir / "candidate_aligned_report.json").open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
        file_obj.write("\n")

    print(
        f"Wrote provisional baseline estimates for {len(estimates)} matched subjects; "
        f"skipped {len(skipped)} unmatched subjects under {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signal-column", default="nose_mean_intensity")
    parser.add_argument("--low-bpm", type=float, default=10.0)
    parser.add_argument("--high-bpm", type=float, default=40.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.low_bpm < args.high_bpm:
        parser.error("require 0 < --low-bpm < --high-bpm")
    if args.filter_order < 1:
        parser.error("--filter-order must be at least 1")

    run(
        args.signals_csv.resolve(),
        args.reference_csv.resolve(),
        args.output_dir.resolve(),
        args.signal_column,
        args.low_bpm,
        args.high_bpm,
        args.filter_order,
        args.overwrite,
    )


if __name__ == "__main__":
    main()

