#!/usr/bin/env python3
"""Run exploratory fixed-graph PWS/GTV denoising on timestamped raw ROIs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, periodogram, sosfiltfilt

from graph_denoise_signals import REGIONS, solve_graph_filter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNALS = ROOT / "data" / "processed" / "roi_signals_raw_mono16.csv"
DEFAULT_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "graph_denoising_rosbag_raw"
GRAPH_OUTPUTS = {
    "timestamped_raw_graph_signals.csv",
    "timestamped_raw_graph_rates.csv",
    "timestamped_raw_graph_report.json",
}
MANAGED_OUTPUTS = GRAPH_OUTPUTS | {
    "parameter_sensitivity.csv",
    "parameter_sensitivity_summary.json",
    "leave_one_subject_out_tuning.csv",
    "leave_one_subject_out_tuning.json",
}


def read_groups(path: Path, key: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        for row in csv.DictReader(file_obj):
            groups[row[key].casefold()].append(row)
    return groups


def prepare_subjects(
    signals_path: Path,
    tracks_path: Path,
    image_width: int,
    image_height: int,
) -> dict[str, dict[str, np.ndarray]]:
    signals = read_groups(signals_path, "subject")
    tracks = read_groups(tracks_path, "subject")
    if set(signals) != set(tracks):
        raise ValueError(
            f"Signal and track subjects differ: {sorted(signals)} vs {sorted(tracks)}"
        )

    subjects: dict[str, dict[str, np.ndarray]] = {}
    for subject, signal_rows in sorted(signals.items()):
        track_rows = tracks[subject]
        signal_rows.sort(key=lambda row: int(row["frame_index"]))
        track_rows.sort(key=lambda row: int(row["frame_index"]))
        if len(signal_rows) != len(track_rows):
            raise ValueError(
                f"Signal/track row mismatch for {subject}: {len(signal_rows)} vs {len(track_rows)}"
            )
        frame_indices = np.array([int(row["frame_index"]) for row in signal_rows], dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(1, len(frame_indices) + 1)):
            raise ValueError(f"Frame indices must be contiguous from 1 for {subject}")
        track_indices = [int(row["frame_index"]) for row in track_rows]
        if track_indices != frame_indices.tolist():
            raise ValueError(f"Signal and track frame indices differ for {subject}")
        if any(row["valid_roi"].strip().casefold() != "true" for row in track_rows):
            raise ValueError(f"Invalid candidate region in tracking data for {subject}")

        values = np.array(
            [
                [float(row[f"{region}_mean_intensity"]) for region in REGIONS]
                for row in signal_rows
            ],
            dtype=np.float64,
        )
        positions = np.empty((len(signal_rows), len(REGIONS), 2), dtype=np.float64)
        for time_index, row in enumerate(track_rows):
            for region_index, region in enumerate(REGIONS):
                x = float(row[f"{region}_x"])
                y = float(row[f"{region}_y"])
                width = float(row[f"{region}_width"])
                height = float(row[f"{region}_height"])
                if not np.isfinite([x, y, width, height]).all() or width <= 0 or height <= 0:
                    raise ValueError(f"Invalid {region} box for {subject} frame {time_index + 1}")
                positions[time_index, region_index] = (
                    (x + width / 2.0) / image_width,
                    (y + height / 2.0) / image_height,
                )
        timestamps = np.array(
            [float(row["image_header_stamp_unix"]) for row in signal_rows], dtype=np.float64
        )
        if not np.isfinite(values).all() or not np.isfinite(timestamps).all():
            raise ValueError(f"Non-finite signal or timestamp for {subject}")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Image timestamps must be strictly increasing for {subject}")
        subjects[subject] = {
            "frame_indices": frame_indices,
            "timestamps": timestamps,
            "values": values,
            "positions": positions,
        }
    return subjects


def build_graph(
    values: np.ndarray,
    positions: np.ndarray,
    timestamps: np.ndarray,
    spatial_scale: float,
    intensity_scale: float,
    temporal_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    frame_count, region_count = values.shape
    left: list[int] = []
    right: list[int] = []
    weights: list[float] = []
    spatial_edges = 0
    temporal_edges = 0

    def add_edge(time_a: int, region_a: int, time_b: int, region_b: int) -> None:
        nonlocal spatial_edges, temporal_edges
        node_a = time_a * region_count + region_a
        node_b = time_b * region_count + region_b
        position_delta = positions[time_a, region_a] - positions[time_b, region_b]
        intensity_delta = values[time_a, region_a] - values[time_b, region_b]
        distance = float(np.dot(position_delta, position_delta)) / (spatial_scale**2)
        distance += float(intensity_delta**2) / (intensity_scale**2)
        if time_a == time_b:
            spatial_edges += 1
        else:
            time_delta = timestamps[time_b] - timestamps[time_a]
            distance += (float(time_delta) / temporal_scale) ** 2
            temporal_edges += 1
        weight = float(np.exp(-min(distance, 50.0)))
        if weight > 1e-12:
            left.append(node_a)
            right.append(node_b)
            weights.append(weight)

    for time_index in range(frame_count):
        for region_a in range(region_count):
            for region_b in range(region_a + 1, region_count):
                add_edge(time_index, region_a, time_index, region_b)
    for time_index in range(frame_count - 1):
        for region_index in range(region_count):
            add_edge(time_index, region_index, time_index + 1, region_index)

    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        {
            "candidate_spatial_edges": spatial_edges,
            "candidate_temporal_edges": temporal_edges,
            "positive_weight_edge_count": len(weights),
        },
    )


def denoise_subject(
    values_raw: np.ndarray,
    positions: np.ndarray,
    timestamps: np.ndarray,
    spatial_scale: float,
    intensity_scale: float,
    temporal_scale_factor: float,
    pws_strength: float,
    gtv_strength: float,
    gtv_epsilon: float,
    gtv_iterations: int,
    convergence_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], np.ndarray]:
    lower = float(np.percentile(values_raw, 1.0))
    upper = float(np.percentile(values_raw, 99.0))
    robust_range = upper - lower
    if robust_range <= np.finfo(float).eps:
        raise ValueError("Raw ROI signals have no usable robust dynamic range")
    values = np.clip((values_raw - lower) / robust_range, 0.0, 1.0)
    median_interval = float(np.median(np.diff(timestamps)))
    temporal_scale = temporal_scale_factor * median_interval
    left, right, base_weights, graph_counts = build_graph(
        values, positions, timestamps, spatial_scale, intensity_scale, temporal_scale
    )
    if not len(base_weights):
        raise ValueError("Time-aware graph construction produced no positive-weight edges")

    pws = solve_graph_filter(values, left, right, base_weights, pws_strength)
    gtv = values.copy()
    converged = False
    relative_change = float("inf")
    iterations_run = 0
    for iteration in range(gtv_iterations):
        edge_difference = np.abs(gtv.reshape(-1)[left] - gtv.reshape(-1)[right])
        reweighted = base_weights / np.maximum(edge_difference, gtv_epsilon)
        updated = solve_graph_filter(values, left, right, reweighted, gtv_strength)
        relative_change = float(
            np.linalg.norm(updated - gtv)
            / max(np.linalg.norm(gtv), np.finfo(float).eps)
        )
        gtv = updated
        iterations_run = iteration + 1
        if relative_change <= convergence_tolerance:
            converged = True
            break

    pws_raw = pws * robust_range + lower
    gtv_raw = gtv * robust_range + lower
    graph_report: dict[str, object] = {
        **graph_counts,
        "node_count": int(values.size),
        "positive_weight_min": float(np.min(base_weights)),
        "positive_weight_median": float(np.median(base_weights)),
        "positive_weight_max": float(np.max(base_weights)),
        "normalization": {
            "method": "subject-wide 1st/99th percentile robust scaling across five raw ROI signals",
            "lower_count": lower,
            "upper_count": upper,
        },
        "median_frame_interval_seconds": median_interval,
        "temporal_scale_seconds": temporal_scale,
        "gtv_iterations_run": iterations_run,
        "gtv_converged": converged,
        "gtv_relative_change_last": relative_change,
    }
    return pws_raw, gtv_raw, graph_report, values


def estimate_rate(values: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    elapsed = timestamps - timestamps[0]
    duration = float(elapsed[-1])
    sample_rate = float((len(values) - 1) / duration)
    uniform_times = np.linspace(0.0, duration, len(values))
    uniform_values = np.interp(uniform_times, elapsed, values)
    sos = butter(4, [10.0 / 60.0, 40.0 / 60.0], btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, uniform_values - np.mean(uniform_values))
    frequencies, power = periodogram(
        filtered, fs=sample_rate, window="hann", detrend=False, scaling="spectrum"
    )
    in_band = (frequencies >= 10.0 / 60.0) & (frequencies <= 40.0 / 60.0)
    if not np.any(in_band):
        raise ValueError("No periodogram bins in the 10–40 BPM band")
    candidates = np.flatnonzero(in_band)
    peak = int(candidates[np.argmax(power[in_band])])
    return {
        "dominant_frequency_hz": float(frequencies[peak]),
        "estimated_rate_bpm": float(frequencies[peak] * 60.0),
        "frequency_resolution_hz": float(sample_rate / len(values)),
        "effective_sample_rate_hz": sample_rate,
    }


def run(
    signals_path: Path,
    tracks_path: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    spatial_scale: float,
    intensity_scale: float,
    temporal_scale_factor: float,
    pws_strength: float,
    gtv_strength: float,
    gtv_epsilon: float,
    gtv_iterations: int,
    convergence_tolerance: float,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected = existing - MANAGED_OUTPUTS
    if unexpected:
        raise FileExistsError(f"Output directory contains unmanaged files: {sorted(unexpected)}")
    if existing and not overwrite:
        raise FileExistsError(f"Graph outputs already exist under {output_dir}; pass --overwrite")

    subjects = prepare_subjects(signals_path, tracks_path, image_width, image_height)
    signal_rows: list[dict[str, object]] = []
    rate_rows: list[dict[str, object]] = []
    subject_reports: list[dict[str, object]] = []
    for subject, data in sorted(subjects.items()):
        timestamps = data["timestamps"]
        values_raw = data["values"]
        pws_raw, gtv_raw, graph_report, normalized = denoise_subject(
            values_raw,
            data["positions"],
            timestamps,
            spatial_scale,
            intensity_scale,
            temporal_scale_factor,
            pws_strength,
            gtv_strength,
            gtv_epsilon,
            gtv_iterations,
            convergence_tolerance,
        )
        region_rates: dict[str, dict[str, dict[str, float]]] = {}
        for region_index, region in enumerate(REGIONS):
            region_rates[region] = {
                "raw": estimate_rate(values_raw[:, region_index], timestamps),
                "pws": estimate_rate(pws_raw[:, region_index], timestamps),
                "gtv": estimate_rate(gtv_raw[:, region_index], timestamps),
            }
            for method_name, signal in (
                ("raw", values_raw[:, region_index]),
                ("pws_fixed_graph", pws_raw[:, region_index]),
                ("gtv_irls_fixed_graph", gtv_raw[:, region_index]),
            ):
                rate = estimate_rate(signal, timestamps)
                rate_rows.append(
                    {
                        "subject": subject,
                        "region": region,
                        "method": method_name,
                        **rate,
                        "accuracy_evaluated": False,
                        "annotation_codes_used": False,
                    }
                )

        elapsed = timestamps - timestamps[0]
        frame_indices = data["frame_indices"]
        for time_index in range(len(frame_indices)):
            for region_index, region in enumerate(REGIONS):
                signal_rows.append(
                    {
                        "subject": subject,
                        "frame_index": int(frame_indices[time_index]),
                        "image_header_stamp_unix": f"{timestamps[time_index]:.9f}",
                        "seconds_since_first_image": f"{elapsed[time_index]:.9f}",
                        "region": region,
                        "raw_mono16_count": f"{values_raw[time_index, region_index]:.8f}",
                        "raw_temperature_c": f"{(values_raw[time_index, region_index] - 1000.0) / 10.0:.8f}",
                        "robust_normalized_raw": f"{normalized[time_index, region_index]:.8f}",
                        "pws_fixed_graph_count": f"{pws_raw[time_index, region_index]:.8f}",
                        "pws_temperature_c": f"{(pws_raw[time_index, region_index] - 1000.0) / 10.0:.8f}",
                        "gtv_irls_fixed_graph_count": f"{gtv_raw[time_index, region_index]:.8f}",
                        "gtv_temperature_c": f"{(gtv_raw[time_index, region_index] - 1000.0) / 10.0:.8f}",
                    }
                )
        subject_reports.append(
            {
                "subject": subject,
                "frame_count": int(len(frame_indices)),
                "duration_seconds": float(elapsed[-1]),
                "effective_sample_rate_hz": float((len(frame_indices) - 1) / elapsed[-1]),
                "graph": graph_report,
                "rates_bpm": {
                    region: {
                        method: result["estimated_rate_bpm"]
                        for method, result in methods.items()
                    }
                    for region, methods in region_rates.items()
                },
            }
        )
        nose = region_rates["nose"]
        print(
            f"{subject}: nose raw/PWS/GTV = "
            f"{nose['raw']['estimated_rate_bpm']:.2f}/"
            f"{nose['pws']['estimated_rate_bpm']:.2f}/"
            f"{nose['gtv']['estimated_rate_bpm']:.2f} BPM"
        )

    signal_path = output_dir / "timestamped_raw_graph_signals.csv"
    with signal_path.open("w", newline="", encoding="utf-8") as file_obj:
        fields = [
            "subject",
            "frame_index",
            "image_header_stamp_unix",
            "seconds_since_first_image",
            "region",
            "raw_mono16_count",
            "raw_temperature_c",
            "robust_normalized_raw",
            "pws_fixed_graph_count",
            "pws_temperature_c",
            "gtv_irls_fixed_graph_count",
            "gtv_temperature_c",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(signal_rows)

    rates_path = output_dir / "timestamped_raw_graph_rates.csv"
    with rates_path.open("w", newline="", encoding="utf-8") as file_obj:
        fields = [
            "subject",
            "region",
            "method",
            "effective_sample_rate_hz",
            "dominant_frequency_hz",
            "estimated_rate_bpm",
            "frequency_resolution_hz",
            "accuracy_evaluated",
            "annotation_codes_used",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rate_rows:
            writer.writerow(
                {
                    **row,
                    "effective_sample_rate_hz": f"{row['effective_sample_rate_hz']:.9g}",
                    "dominant_frequency_hz": f"{row['dominant_frequency_hz']:.9g}",
                    "estimated_rate_bpm": f"{row['estimated_rate_bpm']:.6f}",
                    "frequency_resolution_hz": f"{row['frequency_resolution_hz']:.9g}",
                }
            )

    report = {
        "method": "exploratory fixed-graph Laplacian PWS and iteratively reweighted GTV adaptation",
        "input_signals": str(signals_path),
        "timestamp_source": "sensor_msgs/Image header stamps recovered from original ROS bags",
        "raw_signal_unit": "mono16 sensor counts are preserved; temperature columns use the official L-CAS encoding conversion",
        "temperature_conversion": "(mono16_count - 1000) / 10.0 degrees Celsius",
        "temperature_conversion_source": "https://lcas.lincoln.ac.uk/wp/research/data-sets-software/lcas-thermal-physiological-monitoring-dataset/",
        "normalization": "Within each recording, all five ROI signals are scaled by the 1st and 99th percentiles into [0,1], clipping outside those quantiles; outputs are mapped back to raw-count units.",
        "graph": {
            "nodes": "one node per frame and each of five candidate facial regions",
            "spatial_edges": "all unordered region pairs within each frame",
            "temporal_edges": "each region to itself in the adjacent frame",
            "weight": "Gaussian on normalized ROI-center distance, robust-normalized intensity difference, and measured temporal gap for temporal edges",
            "temporal_scale": f"{temporal_scale_factor:g} times the per-recording median image interval",
        },
        "parameters": {
            "image_width": image_width,
            "image_height": image_height,
            "spatial_scale": spatial_scale,
            "intensity_scale_normalized": intensity_scale,
            "temporal_scale_factor": temporal_scale_factor,
            "pws_strength": pws_strength,
            "gtv_strength": gtv_strength,
            "gtv_epsilon": gtv_epsilon,
            "gtv_iterations": gtv_iterations,
            "convergence_tolerance": convergence_tolerance,
            "rate_band_bpm": [10, 40],
        },
        "limitations": [
            "This is a fixed-feature adaptation, not an exact reproduction of the paper's learned Mahalanobis metric or trace constraint.",
            "The five geometric regions and tracker have not been validated; no motion or ROI-quality rejection is applied.",
            "The configured graph strengths are exploratory defaults; a separate leave-one-subject-out selection is reported but did not establish a reliable improvement or final setting.",
            "Respiration codes use a project-owner-confirmed working map that is not author-documented; no formal accuracy is claimed.",
            "Temperature columns use the official L-CAS encoding conversion; no additional camera-specific calibration is applied.",
        ],
        "parameter_sensitivity": {
            "script": "scripts/sweep_timestamped_raw_graph_parameters.py",
            "summary_json": "parameter_sensitivity_summary.json",
            "grid_csv": "parameter_sensitivity.csv",
            "parameter_selection": "exploratory leave-one-subject-out tuning is reported separately in leave_one_subject_out_tuning.json; no final setting was selected",
        },
        "subjects": subject_reports,
        "outputs": sorted(GRAPH_OUTPUTS),
    }
    (output_dir / "timestamped_raw_graph_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote raw, timestamp-aware graph outputs under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-width", type=int, default=382)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--spatial-scale", type=float, default=0.15)
    parser.add_argument("--intensity-scale", type=float, default=0.20)
    parser.add_argument("--temporal-scale-factor", type=float, default=2.0)
    parser.add_argument("--pws-strength", type=float, default=0.05)
    parser.add_argument("--gtv-strength", type=float, default=0.002)
    parser.add_argument("--gtv-epsilon", type=float, default=0.01)
    parser.add_argument("--gtv-iterations", type=int, default=10)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    positive = {
        "image-width": args.image_width,
        "image-height": args.image_height,
        "spatial-scale": args.spatial_scale,
        "intensity-scale": args.intensity_scale,
        "temporal-scale-factor": args.temporal_scale_factor,
        "pws-strength": args.pws_strength,
        "gtv-strength": args.gtv_strength,
        "gtv-epsilon": args.gtv_epsilon,
        "convergence-tolerance": args.convergence_tolerance,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be greater than zero")
    if args.gtv_iterations < 1:
        parser.error("--gtv-iterations must be at least 1")
    run(
        args.signals.resolve(),
        args.tracks.resolve(),
        args.output_dir.resolve(),
        args.image_width,
        args.image_height,
        args.spatial_scale,
        args.intensity_scale,
        args.temporal_scale_factor,
        args.pws_strength,
        args.gtv_strength,
        args.gtv_epsilon,
        args.gtv_iterations,
        args.convergence_tolerance,
        args.overwrite,
    )


if __name__ == "__main__":
    main()

