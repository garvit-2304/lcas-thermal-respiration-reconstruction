#!/usr/bin/env python3
"""Compare facial-region and graph-signal ablations against inhalation intervals.

All five recordings are evaluated with the same fixed signal-processing
settings. No parameters are fit on the evaluated subject. Results are
exploratory because the respiration-code map is owner-confirmed, and the
full-recording tracker ROIs have not been independently validated.
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
DEFAULT_SIGNALS = ROOT / "data" / "processed" / "graph_denoising_rosbag_raw" / "timestamped_raw_graph_signals.csv"
DEFAULT_ANNOTATIONS = ROOT / "data" / "processed" / "physiology_long.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "region_graph_ablations"
REGIONS = ("forehead", "nose", "mouth", "left_cheek", "right_cheek")
SIGNALS = {
    "raw": "raw_temperature_c",
    "pws": "pws_temperature_c",
    "gtv": "gtv_temperature_c",
}
MAPPING = {"0": "no_event", "1": "inhalation", "2": "exhalation"}
LOW_BPM = 10.0
HIGH_BPM = 40.0
FILTER_ORDER = 4
CSV_NAME = "region_graph_ablation_metrics.csv"
JSON_NAME = "region_graph_ablation_report.json"


def parse_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def load_signals(path: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "frame_index", "image_header_stamp_unix", "region", *SIGNALS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Signal CSV is missing columns: {sorted(missing)}")
        for row in reader:
            subject = row["subject"].casefold()
            region = row["region"].casefold()
            if region in REGIONS:
                groups[subject][region].append(row)
    for subject in groups:
        for region in groups[subject]:
            groups[subject][region].sort(key=lambda row: int(row["frame_index"]))
    return groups


def load_inhalation_events(path: Path) -> dict[str, list[float]]:
    events: dict[str, list[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"recording", "stream", "timestamp", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row["stream"].casefold() != "respiration":
                continue
            raw_code = row["value"].strip()
            label = row.get("value_label", "").strip() or MAPPING.get(raw_code, "unmapped_code")
            if label == "inhalation":
                subject = row["recording"].rsplit("-", 1)[-1].casefold()
                events[subject].append(parse_timestamp(row["timestamp"]))
    return {subject: sorted(stamps) for subject, stamps in events.items()}


def reference_rate(event_stamps: list[float]) -> dict[str, float | int]:
    if len(event_stamps) < 2:
        raise ValueError("At least two inhalation annotations are required")
    intervals = np.diff(np.asarray(event_stamps, dtype=np.float64))
    span = event_stamps[-1] - event_stamps[0]
    return {
        "inhalation_event_count": len(event_stamps),
        "event_span_rate_bpm": 60.0 * (len(event_stamps) - 1) / span,
        "median_interval_rate_bpm": 60.0 / float(np.median(intervals)),
    }


def estimate_rate(timestamps: np.ndarray, values: np.ndarray) -> dict[str, float]:
    if len(values) < 64 or len(timestamps) != len(values):
        raise ValueError("At least 64 signal samples are required")
    if not np.isfinite(timestamps).all() or not np.isfinite(values).all():
        raise ValueError("Signal contains non-finite timestamps or values")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0):
        raise ValueError("Image-header timestamps must increase strictly")
    duration_s = float(timestamps[-1] - timestamps[0])
    sample_rate_hz = float((len(values) - 1) / duration_s)
    low_hz, high_hz = LOW_BPM / 60.0, HIGH_BPM / 60.0
    if high_hz >= sample_rate_hz / 2:
        raise ValueError("Requested band exceeds signal Nyquist frequency")

    uniform_time = np.linspace(0.0, duration_s, len(values))
    resampled = np.interp(uniform_time, timestamps - timestamps[0], values)
    sos = butter(FILTER_ORDER, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, resampled - np.mean(resampled))
    frequencies, power = periodogram(filtered, fs=sample_rate_hz, window="hann", detrend=False, scaling="spectrum")
    in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(in_band):
        raise ValueError("No periodogram bins fall in the requested respiration band")
    candidates = np.flatnonzero(in_band)
    peak_index = int(candidates[np.argmax(power[in_band])])
    return {
        "estimated_rate_bpm": float(frequencies[peak_index] * 60.0),
        "effective_sample_rate_hz": sample_rate_hz,
        "duration_seconds": duration_s,
    }


def summarize(rows: list[dict[str, Any]], rate_key: str, reference_key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in SIGNALS:
        for region in REGIONS:
            selected = [row for row in rows if row["method"] == method and row["region"] == region]
            errors = [float(row[rate_key]) - float(row[reference_key]) for row in selected]
            if not errors:
                continue
            absolute = [abs(value) for value in errors]
            output.append({
                "method": method,
                "region": region,
                "subject_count": len(selected),
                "mean_absolute_error_bpm": statistics.mean(absolute),
                "median_absolute_error_bpm": statistics.median(absolute),
                "mean_signed_error_bpm": statistics.mean(errors),
            })
    return output


def run(signals_path: Path, annotations_path: Path, output_dir: Path, overwrite: bool) -> None:
    csv_path, json_path = output_dir / CSV_NAME, output_dir / JSON_NAME
    if not overwrite and (csv_path.exists() or json_path.exists()):
        raise FileExistsError(f"Outputs already exist in {output_dir}; pass --overwrite")
    signals = load_signals(signals_path)
    events = load_inhalation_events(annotations_path)
    if set(signals) != set(events):
        raise ValueError(f"Signal and annotation subjects differ: {sorted(signals)} vs {sorted(events)}")

    rows: list[dict[str, Any]] = []
    for subject in sorted(signals):
        reference = reference_rate(events[subject])
        for region in REGIONS:
            region_rows = signals[subject].get(region, [])
            if not region_rows:
                raise ValueError(f"Missing {region} signal for {subject}")
            timestamps = np.asarray([float(row["image_header_stamp_unix"]) for row in region_rows])
            for method, column in SIGNALS.items():
                values = np.asarray([float(row[column]) for row in region_rows])
                rate = estimate_rate(timestamps, values)
                rows.append({
                    "subject": subject,
                    "region": region,
                    "method": method,
                    **reference,
                    **rate,
                    "error_vs_event_span_rate_bpm": rate["estimated_rate_bpm"] - float(reference["event_span_rate_bpm"]),
                    "absolute_error_vs_event_span_rate_bpm": abs(rate["estimated_rate_bpm"] - float(reference["event_span_rate_bpm"])),
                    "error_vs_median_interval_rate_bpm": rate["estimated_rate_bpm"] - float(reference["median_interval_rate_bpm"]),
                    "absolute_error_vs_median_interval_rate_bpm": abs(rate["estimated_rate_bpm"] - float(reference["median_interval_rate_bpm"])),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_by_event_span = summarize(rows, "estimated_rate_bpm", "event_span_rate_bpm")
    summary_by_median = summarize(rows, "estimated_rate_bpm", "median_interval_rate_bpm")
    report = {
        "purpose": "Fixed-configuration region and graph-signal ablation for respiration-rate estimation across all five subjects.",
        "fold_protocol": "No model parameters are fit on these subjects; each subject is evaluated once with the same fixed preprocessing and 10-40 BPM search band.",
        "method_columns": SIGNALS,
        "regions": list(REGIONS),
        "reference_rate": "Inhalation timestamps under 0=no event, 1=inhalation, 2=exhalation; event-span and median-interval rates are both reported.",
        "reference_mapping_basis": "Project-owner-confirmed working interpretation; not an author-published numeric codebook.",
        "filter": {"type": "fourth-order zero-phase Butterworth bandpass", "band_bpm": [LOW_BPM, HIGH_BPM], "periodicity_estimator": "Hann periodogram peak"},
        "subjects": sorted(signals),
        "row_count": len(rows),
        "summary_by_event_span_rate": summary_by_event_span,
        "summary_by_median_interval_rate": summary_by_median,
        "limitations": [
            "Only five subjects are available; summaries have high uncertainty.",
            "The numeric annotation-code mapping is owner-confirmed, not author-verified.",
            "The thermal face tracker and full-recording ROI trajectories are not independently validated.",
            "The 20-frame manual ROI sample does not validate the remaining continuous ROI tracks.",
            "These are exploratory rate errors, not clinical accuracy claims.",
        ],
        "inputs": {"signals": str(signals_path), "annotations": str(annotations_path)},
        "outputs": {"csv": csv_path.name, "json": json_path.name},
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    best = min(summary_by_event_span, key=lambda item: item["mean_absolute_error_bpm"])
    print(f"Wrote {len(rows)} subject/region/method comparisons across {len(signals)} subjects.")
    print(f"Lowest exploratory event-span MAE: {best['region']} / {best['method']} = {best['mean_absolute_error_bpm']:.2f} BPM")
    print(f"Wrote {csv_path} and {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.signals.resolve(), args.annotations.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

