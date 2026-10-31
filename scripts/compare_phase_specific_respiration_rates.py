#!/usr/bin/env python3
"""Compare thermal respiration rates between the protocol's static and motion phases.

This is a descriptive window-level rate comparison, not event-by-event detection
accuracy. It uses the project-owner-confirmed working annotation mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, periodogram, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNALS = ROOT / "data" / "processed" / "roi_signals_raw_mono16.csv"
DEFAULT_ANNOTATIONS = ROOT / "data" / "processed" / "physiology_long.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "phase_specific_respiration"
CSV_NAME = "phase_specific_respiration_rates.csv"
JSON_NAME = "phase_specific_respiration_rates.json"
MAPPING = {"0": "no_event", "1": "inhalation", "2": "exhalation"}


def read_signals(path: Path, signal_column: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "frame_index", "image_header_stamp_unix", "valid_roi", signal_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ROI signal file is missing columns: {sorted(missing)}")
        for row in reader:
            if row[signal_column].strip() and row["valid_roi"].strip().casefold() == "true":
                groups[row["subject"].casefold()].append(row)
    if not groups:
        raise ValueError(f"No valid ROI samples found in {path}")
    for subject in groups:
        groups[subject].sort(key=lambda row: int(row["frame_index"]))
    return groups


def parse_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def read_inhalations(path: Path) -> dict[str, list[float]]:
    events: dict[str, list[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"recording", "stream", "timestamp", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation file is missing columns: {sorted(missing)}")
        for row in reader:
            if row["stream"].casefold() != "respiration":
                continue
            raw_code = row["value"].strip()
            label = row.get("value_label", "").strip() or MAPPING.get(raw_code, "unmapped_code")
            if label == "inhalation":
                subject = row["recording"].rsplit("-", 1)[-1].casefold()
                events[subject].append(parse_timestamp(row["timestamp"]))
    for subject in events:
        events[subject].sort()
    return events


def estimate_window_rate(
    rows: list[dict[str, str]],
    signal_column: str,
    start_s: float,
    end_s: float,
    low_bpm: float,
    high_bpm: float,
    filter_order: int,
) -> dict[str, Any]:
    first_stamp = float(rows[0]["image_header_stamp_unix"])
    selected = [
        row for row in rows
        if start_s <= float(row["image_header_stamp_unix"]) - first_stamp < end_s
    ]
    if len(selected) < 64:
        raise ValueError(f"Only {len(selected)} ROI samples in {start_s:g}-{end_s:g}s window")

    timestamps = np.array([float(row["image_header_stamp_unix"]) for row in selected])
    values = np.array([float(row[signal_column]) for row in selected])
    elapsed = timestamps - timestamps[0]
    duration_s = float(elapsed[-1])
    sample_rate_hz = float((len(values) - 1) / duration_s)
    low_hz, high_hz = low_bpm / 60.0, high_bpm / 60.0
    if high_hz >= sample_rate_hz / 2.0:
        raise ValueError(f"Band edge exceeds Nyquist in {start_s:g}-{end_s:g}s window")

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
        raise ValueError(f"No periodogram bins in {low_bpm:g}-{high_bpm:g} BPM band")
    candidate_indices = np.flatnonzero(in_band)
    peak_index = int(candidate_indices[np.argmax(power[in_band])])
    intervals = np.diff(timestamps)
    return {
        "roi_sample_count": len(selected),
        "roi_window_duration_seconds": duration_s,
        "effective_sample_rate_hz": sample_rate_hz,
        "median_image_interval_seconds": float(statistics.median(intervals)),
        "thermal_rate_bpm": float(frequencies[peak_index] * 60.0),
        "frequency_resolution_bpm": float(sample_rate_hz / len(values) * 60.0),
    }


def event_rate(events: list[float], origin: float, start_s: float, end_s: float) -> dict[str, Any]:
    start_time, end_time = origin + start_s, origin + end_s
    selected = [stamp for stamp in events if start_time <= stamp < end_time]
    if len(selected) < 2:
        return {
            "inhalation_event_count": len(selected),
            "inter_inhalation_interval_count": max(len(selected) - 1, 0),
            "inhalation_rate_from_event_span_bpm": None,
            "inhalation_rate_from_median_interval_bpm": None,
        }
    intervals = np.diff(np.asarray(selected, dtype=np.float64))
    span = selected[-1] - selected[0]
    return {
        "inhalation_event_count": len(selected),
        "inter_inhalation_interval_count": len(intervals),
        "inhalation_event_span_seconds": span,
        "inhalation_rate_from_event_span_bpm": 60.0 * (len(selected) - 1) / span,
        "inhalation_rate_from_median_interval_bpm": 60.0 / float(np.median(intervals)),
    }


def run(
    signals_path: Path,
    annotations_path: Path,
    output_dir: Path,
    signal_column: str,
    low_bpm: float,
    high_bpm: float,
    filter_order: int,
    phase_duration_s: float,
    overwrite: bool,
) -> None:
    if not 0 < low_bpm < high_bpm:
        raise ValueError("Require 0 < low_bpm < high_bpm")
    if phase_duration_s <= 0:
        raise ValueError("phase_duration_s must be positive")
    output_csv, output_json = output_dir / CSV_NAME, output_dir / JSON_NAME
    if (output_csv.exists() or output_json.exists()) and not overwrite:
        raise FileExistsError(f"Phase comparison outputs already exist under {output_dir}; pass --overwrite")

    signal_groups = read_signals(signals_path, signal_column)
    annotation_groups = read_inhalations(annotations_path)
    if set(signal_groups) != set(annotation_groups):
        raise ValueError(
            f"Signal and annotation subjects differ: {sorted(signal_groups)} vs {sorted(annotation_groups)}"
        )

    rows: list[dict[str, Any]] = []
    for subject in sorted(signal_groups):
        samples = signal_groups[subject]
        origin = float(samples[0]["image_header_stamp_unix"])
        recording_duration_s = float(samples[-1]["image_header_stamp_unix"]) - origin
        phase_specs = [
            ("static_first_minute", 0.0, min(phase_duration_s, recording_duration_s)),
            (
                "head_movement_second_minute",
                phase_duration_s,
                min(2.0 * phase_duration_s, recording_duration_s),
            ),
        ]
        for phase, start_s, end_s in phase_specs:
            if end_s <= start_s:
                continue
            thermal = estimate_window_rate(
                samples,
                signal_column,
                start_s,
                end_s,
                low_bpm,
                high_bpm,
                filter_order,
            )
            reference = event_rate(annotation_groups[subject], origin, start_s, end_s)
            ref_rate = reference["inhalation_rate_from_median_interval_bpm"]
            span_rate = reference["inhalation_rate_from_event_span_bpm"]
            thermal_rate = thermal["thermal_rate_bpm"]
            rows.append(
                {
                    "subject": subject,
                    "phase": phase,
                    "phase_start_seconds": start_s,
                    "phase_end_seconds": end_s,
                    **thermal,
                    **reference,
                    "thermal_minus_event_span_rate_bpm": (
                        thermal_rate - span_rate if span_rate is not None else None
                    ),
                    "absolute_difference_from_event_span_rate_bpm": (
                        abs(thermal_rate - span_rate) if span_rate is not None else None
                    ),
                    "thermal_minus_median_interval_rate_bpm": (
                        thermal_rate - ref_rate if ref_rate is not None else None
                    ),
                    "absolute_difference_from_median_interval_rate_bpm": (
                        abs(thermal_rate - ref_rate) if ref_rate is not None else None
                    ),
                    "mapping": "0=no event, 1=inhalation, 2=exhalation",
                    "mapping_basis": "project-owner-confirmed working interpretation; not an official dataset codebook",
                }
            )

    phase_summary: list[dict[str, Any]] = []
    for phase in ("static_first_minute", "head_movement_second_minute"):
        comparable = [
            float(row["absolute_difference_from_event_span_rate_bpm"])
            for row in rows
            if row["phase"] == phase and row["absolute_difference_from_event_span_rate_bpm"] is not None
        ]
        median_interval_errors = [
            float(row["absolute_difference_from_median_interval_rate_bpm"])
            for row in rows
            if row["phase"] == phase and row["absolute_difference_from_median_interval_rate_bpm"] is not None
        ]
        phase_summary.append(
            {
                "phase": phase,
                "recordings_compared": len(comparable),
                "mean_absolute_difference_from_event_span_rate_bpm": statistics.mean(comparable) if comparable else None,
                "median_absolute_difference_from_event_span_rate_bpm": statistics.median(comparable) if comparable else None,
                "mean_signed_difference_from_event_span_rate_bpm": statistics.mean(
                    float(row["thermal_minus_event_span_rate_bpm"])
                    for row in rows
                    if row["phase"] == phase and row["thermal_minus_event_span_rate_bpm"] is not None
                ) if comparable else None,
                "mean_absolute_difference_from_median_interval_rate_bpm": statistics.mean(median_interval_errors) if median_interval_errors else None,
                "median_absolute_difference_from_median_interval_rate_bpm": statistics.median(median_interval_errors) if median_interval_errors else None,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        fields = list(rows[0])
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "purpose": "Descriptive window-level comparison of thermal respiration rates with mapped inhalation-event intervals, split by the L-CAS static/movement protocol.",
        "phase_protocol": {
            "static": f"0 to {phase_duration_s:g} seconds after the first thermal image",
            "head_movement": f"{phase_duration_s:g} to {2 * phase_duration_s:g} seconds after the first thermal image",
            "basis": "The L-CAS protocol describes the first minute as static and the second minute as head movement.",
        },
        "annotation_mapping": MAPPING,
        "mapping_basis": "Project-owner-confirmed working interpretation; dataset-author numeric codebook not found.",
        "thermal_method": {
            "signal_column": signal_column,
            "band_bpm": [low_bpm, high_bpm],
            "filter": f"{filter_order}-order zero-phase Butterworth bandpass",
            "rate_estimator": "Hann periodogram peak independently estimated within each phase window",
            "timestamp_source": "sensor_msgs/Image header timestamps recovered from original ROS bags",
        },
        "annotation_rate_methods": [
            "Event-span rate: 60 times the number of complete inter-inhalation intervals divided by the span from first to last inhale marker within the phase.",
            "Median-interval rate: 60 divided by the median inter-inhalation interval within the phase.",
        ],
        "phase_summary": phase_summary,
        "limitations": [
            "This is a window-level rate comparison, not event-by-event detection precision/recall.",
            "The code mapping is project-owner-confirmed but is not published by the dataset authors.",
            "Only Joao has code-2 exhalation markers; the primary comparison uses code-1 inhalation events.",
            "Face/ROI tracking has not been validated against complete manual boxes or landmarks; movement-phase signal errors may reflect ROI drift.",
            "No motion rejection or manually reviewed ROI correction is applied to the rate estimates.",
            "A 60-second spectral window has finite frequency resolution; reported rates are quantized to the periodogram bin spacing.",
        ],
        "inputs": {"signals": str(signals_path), "annotations": str(annotations_path)},
        "outputs": {"csv": CSV_NAME, "json": JSON_NAME},
        "recordings": rows,
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote phase-specific rate comparison for {len(rows)} windows under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-column", default="nose_mean_temperature_c")
    parser.add_argument("--low-bpm", type=float, default=10.0)
    parser.add_argument("--high-bpm", type=float, default=40.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--phase-duration-s", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.signals.resolve(),
        args.annotations.resolve(),
        args.output_dir.resolve(),
        args.signal_column,
        args.low_bpm,
        args.high_bpm,
        args.filter_order,
        args.phase_duration_s,
        args.overwrite,
    )


if __name__ == "__main__":
    main()

