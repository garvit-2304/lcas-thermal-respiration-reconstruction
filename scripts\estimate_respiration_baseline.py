#!/usr/bin/env python3
"""Estimate an exploratory whole-recording respiration rate from an ROI trace.

This baseline requires an explicit sampling rate and frequency band. It does
not join or evaluate against respiration annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, periodogram, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "roi_signals.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "respiration_baseline"
MANAGED_OUTPUTS = {
    "baseline_filtered_signals.csv",
    "baseline_estimates.csv",
    "baseline_report.json",
}


def load_recordings(input_csv: Path, signal_column: str) -> dict[str, list[tuple[int, float]]]:
    if not input_csv.is_file():
        raise FileNotFoundError(f"ROI signal CSV not found: {input_csv}")

    recordings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with input_csv.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "frame_index", signal_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            subject = row["subject"]
            try:
                frame_index = int(row["frame_index"])
                value = float(row[signal_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid frame index or signal value for {subject}: {row}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite {signal_column} value at {subject} frame {frame_index}; "
                    "missing frames cannot be silently dropped from a sampled signal."
                )
            recordings[subject].append((frame_index, value))

    if not recordings:
        raise ValueError(f"No recordings found in {input_csv}")

    for subject, samples in recordings.items():
        samples.sort(key=lambda item: item[0])
        indices = [item[0] for item in samples]
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate frame indices found for {subject}")
        if any(right - left != 1 for left, right in zip(indices, indices[1:])):
            raise ValueError(
                f"Non-contiguous frame indices found for {subject}; establish a missing-frame "
                "policy before frequency estimation."
            )
    return recordings


def estimate_recording(
    values: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> tuple[np.ndarray, float, float, float]:
    if len(values) < 4:
        raise ValueError("At least four samples are required for filtering.")

    nyquist = fps / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"Frequency band must satisfy 0 < low_hz < high_hz < Nyquist ({nyquist:g} Hz)."
        )

    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fps, output="sos")
    try:
        filtered = sosfiltfilt(sos, values - np.mean(values))
    except ValueError as exc:
        raise ValueError(
            f"Recording has {len(values)} samples, too few for zero-phase filtering "
            f"with order {order}."
        ) from exc

    frequencies, power = periodogram(
        filtered,
        fs=fps,
        window="hann",
        detrend=False,
        scaling="spectrum",
    )
    in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(in_band):
        raise ValueError(
            "No Fourier bins fall inside the requested band; use a longer recording "
            "or a wider band."
        )
    band_indices = np.flatnonzero(in_band)
    peak_index = int(band_indices[np.argmax(power[in_band])])
    peak_hz = float(frequencies[peak_index])
    resolution_hz = float(fps / len(values))
    return filtered, peak_hz, peak_hz * 60.0, resolution_hz


def write_outputs(
    input_csv: Path,
    output_dir: Path,
    signal_column: str,
    fps: float,
    fps_source: str,
    low_hz: float,
    high_hz: float,
    order: int,
    overwrite: bool,
) -> None:
    recordings = load_recordings(input_csv, signal_column)
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir() if path.is_file()}
        unexpected = existing - MANAGED_OUTPUTS
        if unexpected:
            raise FileExistsError(
                f"Output directory contains files not managed by this script: {sorted(unexpected)}"
            )
        if existing and not overwrite:
            raise FileExistsError(
                f"Baseline outputs already exist under {output_dir}; pass --overwrite to refresh them."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    estimates: list[dict[str, object]] = []
    filtered_path = output_dir / "baseline_filtered_signals.csv"
    with filtered_path.open("w", newline="", encoding="utf-8") as file_obj:
        fieldnames = [
            "subject",
            "frame_index",
            "source_signal_column",
            "raw_intensity",
            "filtered_signal",
            "fps_assumed",
            "sampling_rate_source",
            "low_hz",
            "high_hz",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()

        for subject, samples in sorted(recordings.items()):
            indices = np.array([sample[0] for sample in samples], dtype=np.int64)
            raw = np.array([sample[1] for sample in samples], dtype=np.float64)
            filtered, peak_hz, rate_bpm, resolution_hz = estimate_recording(
                raw, fps, low_hz, high_hz, order
            )
            for frame_index, raw_value, filtered_value in zip(indices, raw, filtered):
                writer.writerow(
                    {
                        "subject": subject,
                        "frame_index": int(frame_index),
                        "source_signal_column": signal_column,
                        "raw_intensity": f"{raw_value:.8f}",
                        "filtered_signal": f"{filtered_value:.10g}",
                        "fps_assumed": f"{fps:.8g}",
                        "sampling_rate_source": fps_source,
                        "low_hz": f"{low_hz:.8g}",
                        "high_hz": f"{high_hz:.8g}",
                    }
                )
            estimates.append(
                {
                    "subject": subject,
                    "signal_column": signal_column,
                    "sample_count": int(len(raw)),
                    "duration_seconds_assuming_fps": float((len(raw) - 1) / fps),
                    "fps_assumed": float(fps),
                    "sampling_rate_source": fps_source,
                    "frequency_band_hz": [float(low_hz), float(high_hz)],
                    "filter": {"type": "Butterworth bandpass", "order": int(order)},
                    "periodicity_estimator": "full-recording Hann periodogram peak",
                    "frequency_resolution_hz": resolution_hz,
                    "dominant_frequency_hz": peak_hz,
                    "estimated_rate_bpm": rate_bpm,
                    "annotation_joined": False,
                    "evaluation_performed": False,
                }
            )

    estimates_path = output_dir / "baseline_estimates.csv"
    estimate_fields = [
        "subject",
        "signal_column",
        "sample_count",
        "duration_seconds_assuming_fps",
        "fps_assumed",
        "sampling_rate_source",
        "low_hz",
        "high_hz",
        "frequency_resolution_hz",
        "dominant_frequency_hz",
        "estimated_rate_bpm",
        "annotation_joined",
        "evaluation_performed",
    ]
    with estimates_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=estimate_fields)
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(
                {
                    "subject": estimate["subject"],
                    "signal_column": estimate["signal_column"],
                    "sample_count": estimate["sample_count"],
                    "duration_seconds_assuming_fps": f"{estimate['duration_seconds_assuming_fps']:.6f}",
                    "fps_assumed": f"{estimate['fps_assumed']:.8g}",
                    "sampling_rate_source": estimate["sampling_rate_source"],
                    "low_hz": f"{estimate['frequency_band_hz'][0]:.8g}",
                    "high_hz": f"{estimate['frequency_band_hz'][1]:.8g}",
                    "frequency_resolution_hz": f"{estimate['frequency_resolution_hz']:.8g}",
                    "dominant_frequency_hz": f"{estimate['dominant_frequency_hz']:.8g}",
                    "estimated_rate_bpm": f"{estimate['estimated_rate_bpm']:.8g}",
                    "annotation_joined": False,
                    "evaluation_performed": False,
                }
            )

    report = {
        "input_csv": str(input_csv),
        "signal_column": signal_column,
        "sampling_rate_hz_assumed": float(fps),
        "sampling_rate_source": fps_source,
        "frequency_band_hz": [float(low_hz), float(high_hz)],
        "filter": {"type": "Butterworth bandpass", "order": int(order), "zero_phase": True},
        "periodicity_estimator": "full-recording Hann periodogram peak",
        "limitations": [
            "Sampling rate is user-supplied and is not verified against image timestamps.",
            "Frequency band is user-supplied and must be justified for the protocol.",
            "Signals are 8-bit image intensities, not calibrated temperatures.",
            "No respiration annotations are joined and no accuracy evaluation is performed.",
            "Estimates cover each entire recording and do not resolve time-varying respiration.",
        ],
        "recordings": estimates,
    }
    with (output_dir / "baseline_report.json").open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
        file_obj.write("\n")
    print(f"Wrote baseline outputs for {len(estimates)} recordings under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signal-column", default="face_mean_intensity")
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Explicitly chosen samples per second for the PNG sequence.",
    )
    parser.add_argument(
        "--fps-source",
        required=True,
        help="Provenance or assumption for --fps; recorded with every output.",
    )
    parser.add_argument(
        "--low-hz",
        type=float,
        required=True,
        help="Lower respiration-band edge in Hz; choose from protocol/validated references.",
    )
    parser.add_argument(
        "--high-hz",
        type=float,
        required=True,
        help="Upper respiration-band edge in Hz; choose from protocol/validated references.",
    )
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.filter_order < 1:
        parser.error("--filter-order must be at least 1")
    if not args.fps_source.strip():
        parser.error("--fps-source must not be empty")
    if not 0 < args.low_hz < args.high_hz:
        parser.error("require 0 < --low-hz < --high-hz")
    if args.high_hz >= args.fps / 2.0:
        parser.error("--high-hz must be below the Nyquist frequency (--fps / 2)")

    write_outputs(
        args.input_csv.resolve(),
        args.output_dir.resolve(),
        args.signal_column,
        args.fps,
        args.fps_source.strip(),
        args.low_hz,
        args.high_hz,
        args.filter_order,
        args.overwrite,
    )


if __name__ == "__main__":
    main()

