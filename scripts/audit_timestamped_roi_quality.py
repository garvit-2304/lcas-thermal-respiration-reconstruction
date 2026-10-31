#!/usr/bin/env python3
"""Audit tracker motion and raw ROI jumps without automatically dropping frames."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_SIGNALS = ROOT / "data" / "processed" / "roi_signals_raw_mono16.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "roi_quality_audit"
METRICS_NAME = "timestamped_roi_quality_metrics.csv"
REPORT_NAME = "timestamped_roi_quality_report.json"
PLOT_DIR_NAME = "plots"


def grouped_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        for row in csv.DictReader(file_obj):
            groups[row["subject"].casefold()].append(row)
    return groups


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def run(tracks_path: Path, signals_path: Path, output_dir: Path, overwrite: bool) -> None:
    metrics_path = output_dir / METRICS_NAME
    report_path = output_dir / REPORT_NAME
    plot_dir = output_dir / PLOT_DIR_NAME
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"ROI quality outputs already exist under {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = grouped_rows(tracks_path)
    signals = grouped_rows(signals_path)
    if set(tracks) != set(signals):
        raise ValueError("Face-track and raw-signal subject sets differ")

    all_rows: list[dict[str, object]] = []
    subject_reports: list[dict[str, object]] = []
    plots: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = []
    for subject, track_rows in sorted(tracks.items()):
        signal_rows = signals[subject]
        track_rows.sort(key=lambda row: int(row["frame_index"]))
        signal_rows.sort(key=lambda row: int(row["frame_index"]))
        if len(track_rows) != len(signal_rows):
            raise ValueError(f"Track/raw sample count mismatch for {subject}")
        frame_indices = np.array([int(row["frame_index"]) for row in track_rows], dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(1, len(frame_indices) + 1)):
            raise ValueError(f"Frame indices must be contiguous for {subject}")
        if [int(row["frame_index"]) for row in signal_rows] != frame_indices.tolist():
            raise ValueError(f"Track and raw signal frame indices differ for {subject}")

        timestamps = np.array(
            [float(row["image_header_stamp_unix"]) for row in signal_rows], dtype=np.float64
        )
        elapsed = timestamps - timestamps[0]
        nose = np.array(
            [float(row["nose_mean_intensity"]) if row["nose_mean_intensity"] else np.nan for row in signal_rows],
            dtype=np.float64,
        )
        def optional_float(row: dict[str, str], field: str) -> float:
            return float(row[field]) if row[field] else np.nan

        x = np.array([optional_float(row, "face_x") for row in track_rows])
        y = np.array([optional_float(row, "face_y") for row in track_rows])
        width = np.array([optional_float(row, "face_width") for row in track_rows])
        height = np.array([optional_float(row, "face_height") for row in track_rows])
        box_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(width) & np.isfinite(height)
        if np.any(box_valid & ((width <= 0) | (height <= 0))):
            raise ValueError(f"Non-positive face box dimensions for {subject}")
        declared_valid = np.array(
            [row["valid_roi"].strip().casefold() == "true" for row in track_rows], dtype=bool
        )
        if np.any(declared_valid != box_valid):
            raise ValueError(f"Face-box coordinates and valid_roi disagree for {subject}")
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        area = width * height

        center_speed = np.full(len(track_rows), np.nan, dtype=np.float64)
        area_log_change = np.full(len(track_rows), np.nan, dtype=np.float64)
        area_fraction_change = np.full(len(track_rows), np.nan, dtype=np.float64)
        nose_abs_delta = np.full(len(track_rows), np.nan, dtype=np.float64)
        if len(track_rows) > 1:
            dt = np.diff(timestamps)
            if np.any(dt <= 0):
                raise ValueError(f"Non-increasing image timestamps for {subject}")
            box_pairs = box_valid[1:] & box_valid[:-1]
            box_pair_indices = np.flatnonzero(box_pairs) + 1
            displacement = np.hypot(np.diff(center_x)[box_pairs], np.diff(center_y)[box_pairs])
            center_speed[box_pair_indices] = displacement / dt[box_pairs]
            area_log_change[box_pair_indices] = np.abs(
                np.log(area[1:][box_pairs] / area[:-1][box_pairs])
            )
            area_fraction_change[box_pair_indices] = np.abs(
                area[1:][box_pairs] / area[:-1][box_pairs] - 1.0
            )
            signal_pairs = np.isfinite(nose[1:]) & np.isfinite(nose[:-1])
            signal_pair_indices = np.flatnonzero(signal_pairs) + 1
            nose_abs_delta[signal_pair_indices] = np.abs(np.diff(nose)[signal_pairs])

        speed_p95 = percentile(center_speed, 95)
        speed_p99 = percentile(center_speed, 99)
        area_p95 = percentile(area_log_change, 95)
        area_p99 = percentile(area_log_change, 99)
        signal_p95 = percentile(nose_abs_delta, 95)
        signal_p99 = percentile(nose_abs_delta, 99)
        motion_flag = (center_speed > speed_p99) | (area_log_change > area_p99)
        jump_flag = nose_abs_delta > signal_p99
        status_counts = Counter(row["tracking_status"] for row in track_rows)
        valid_count = int(np.count_nonzero(box_valid))
        summary = {
            "subject": subject,
            "frame_count": len(track_rows),
            "valid_roi_frame_count": valid_count,
            "missing_roi_frame_count": len(track_rows) - valid_count,
            "tracking_status_counts": dict(status_counts),
            "face_center_speed_px_s_p95": speed_p95,
            "face_center_speed_px_s_p99": speed_p99,
            "face_center_speed_px_s_max": float(np.nanmax(center_speed)),
            "face_area_log_change_p95": area_p95,
            "face_area_log_change_p99": area_p99,
            "face_area_log_change_max": float(np.nanmax(area_log_change)),
            "nose_abs_delta_count_p95": signal_p95,
            "nose_abs_delta_count_p99": signal_p99,
            "nose_abs_delta_count_max": float(np.nanmax(nose_abs_delta)),
            "frames_over_face_motion_p99": int(np.count_nonzero(motion_flag)),
            "frames_over_nose_jump_p99": int(np.count_nonzero(jump_flag)),
            "flags_are_within_subject_empirical_percentiles": True,
            "automatic_frame_exclusion": False,
        }
        subject_reports.append(summary)
        thresholds = {
            "speed_p99": speed_p99,
            "area_log_change_p99": area_p99,
            "nose_abs_delta_p99": signal_p99,
        }
        plots.append((subject, elapsed, nose, center_speed, nose_abs_delta, thresholds))

        for i, track in enumerate(track_rows):
            all_rows.append(
                {
                    "subject": subject,
                    "frame_index": int(frame_indices[i]),
                    "image_header_stamp_unix": f"{timestamps[i]:.9f}",
                    "seconds_since_first_image": f"{elapsed[i]:.9f}",
                    "tracking_status": track["tracking_status"],
                    "valid_roi": track["valid_roi"],
                    "face_center_x_px": "" if not np.isfinite(center_x[i]) else f"{center_x[i]:.5f}",
                    "face_center_y_px": "" if not np.isfinite(center_y[i]) else f"{center_y[i]:.5f}",
                    "face_width_px": "" if not np.isfinite(width[i]) else f"{width[i]:.5f}",
                    "face_height_px": "" if not np.isfinite(height[i]) else f"{height[i]:.5f}",
                    "face_center_speed_px_s": "" if not np.isfinite(center_speed[i]) else f"{center_speed[i]:.8f}",
                    "face_area_abs_log_change": "" if not np.isfinite(area_log_change[i]) else f"{area_log_change[i]:.8f}",
                    "face_area_fraction_change": "" if not np.isfinite(area_fraction_change[i]) else f"{area_fraction_change[i]:.8f}",
                    "nose_raw_count": "" if not np.isfinite(nose[i]) else f"{nose[i]:.8f}",
                    "nose_abs_delta_count": "" if not np.isfinite(nose_abs_delta[i]) else f"{nose_abs_delta[i]:.8f}",
                    "face_motion_outlier_p99": str(bool(motion_flag[i])).lower(),
                    "nose_signal_jump_p99": str(bool(jump_flag[i])).lower(),
                }
            )

    fields = [
        "subject",
        "frame_index",
        "image_header_stamp_unix",
        "seconds_since_first_image",
        "tracking_status",
        "valid_roi",
        "face_center_x_px",
        "face_center_y_px",
        "face_width_px",
        "face_height_px",
        "face_center_speed_px_s",
        "face_area_abs_log_change",
        "face_area_fraction_change",
        "nose_raw_count",
        "nose_abs_delta_count",
        "face_motion_outlier_p99",
        "nose_signal_jump_p99",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    plot_dir.mkdir(parents=True, exist_ok=True)
    for subject, elapsed, nose, speed, jump, thresholds in plots:
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
        axes[0].plot(elapsed, nose, color="#2a9d8f", linewidth=0.65)
        axes[0].set_ylabel("Nose raw count")
        axes[0].grid(True, alpha=0.2)
        axes[1].plot(elapsed, speed, color="#457b9d", linewidth=0.65)
        axes[1].axhline(thresholds["speed_p99"], color="#d62828", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Face-center speed\n(px/s)")
        axes[1].grid(True, alpha=0.2)
        axes[2].plot(elapsed, jump, color="#8d5a97", linewidth=0.65)
        axes[2].axhline(thresholds["nose_abs_delta_p99"], color="#d62828", linestyle="--", linewidth=0.8)
        axes[2].set_ylabel("Abs. nose change\n(count/frame)")
        axes[2].set_xlabel("Seconds from first image-header timestamp")
        axes[2].grid(True, alpha=0.2)
        fig.suptitle(f"{subject.title()} — exploratory ROI quality audit", fontsize=15)
        fig.savefig(plot_dir / f"{subject}_roi_quality.png", dpi=150)
        plt.close(fig)

    report = {
        "input_tracks": str(tracks_path),
        "input_signals": str(signals_path),
        "timestamp_source": "sensor_msgs/Image header stamps from the original ROS bags",
        "metrics": {
            "face_center_speed": "consecutive tracked face-box center displacement divided by measured timestamp interval",
            "face_area_change": "absolute log change in tracked face-box area between consecutive frames",
            "nose_signal_change": "absolute consecutive-frame difference in raw mono16 nose-region mean",
        },
        "outlier_policy": "Within-subject 99th-percentile flags are descriptive only; no frame is dropped and the thresholds are not validated quality cutoffs.",
        "limitations": [
            "Face boxes are heuristic and are not compared with manual landmarks or masks.",
            "Motion flags are empirical outliers, not calibrated tracker confidence or a clinical signal-quality score.",
            "Large raw ROI changes may reflect real scene/subject movement, box updates, or sensor behavior.",
        ],
        "subjects": subject_reports,
        "outputs": {
            "metrics_csv": METRICS_NAME,
            "plots_directory": PLOT_DIR_NAME,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audited timestamped ROI quality for {len(subject_reports)} subjects ({len(all_rows)} frames).")
    print(f"Wrote {metrics_path}, {report_path}, and plots under {plot_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.tracks.resolve(), args.signals.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

