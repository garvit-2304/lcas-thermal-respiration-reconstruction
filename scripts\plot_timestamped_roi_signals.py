#!/usr/bin/env python3
"""Plot raw mono16 facial ROI signals against recovered image timestamps."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNALS = ROOT / "data" / "processed" / "roi_signals_raw_mono16.csv"
DEFAULT_FILTERED = (
    ROOT
    / "data"
    / "processed"
    / "respiration_baseline_rosbag_raw"
    / "rosbag_raw_filtered_signals.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "roi_signal_timestamp_plots"
REGIONS = ("forehead", "nose", "mouth", "left_cheek", "right_cheek")
COLORS = {
    "forehead": "#e76f51",
    "nose": "#2a9d8f",
    "mouth": "#e9c46a",
    "left_cheek": "#457b9d",
    "right_cheek": "#8d5a97",
}


def load_rows(path: Path, group_key: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        for row in csv.DictReader(file_obj):
            groups[row[group_key].casefold()].append(row)
    return groups


def plot(signals_path: Path, filtered_path: Path, output_dir: Path) -> None:
    signals = load_rows(signals_path, "subject")
    filtered = load_rows(filtered_path, "subject")
    if set(signals) != set(filtered):
        raise ValueError(
            f"Subject mismatch between raw ROI signals and filtered baseline: "
            f"{sorted(signals)} vs {sorted(filtered)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject in sorted(signals):
        raw_rows = sorted(signals[subject], key=lambda row: int(row["frame_index"]))
        filtered_rows = sorted(filtered[subject], key=lambda row: int(row["frame_index"]))
        if len(raw_rows) != len(filtered_rows):
            raise ValueError(f"Raw and filtered row counts differ for {subject}")
        frame_indices = [int(row["frame_index"]) for row in raw_rows]
        filtered_indices = [int(row["frame_index"]) for row in filtered_rows]
        if frame_indices != filtered_indices:
            raise ValueError(f"Raw and filtered frame indices differ for {subject}")
        time_s = np.array([float(row["seconds_since_first_image"]) for row in raw_rows])
        filtered_time_s = np.array(
            [float(row["uniform_time_s"]) for row in filtered_rows]
        )
        fig, (axis_rois, axis_nose) = plt.subplots(
            2,
            1,
            figsize=(13, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.1, 1]},
            constrained_layout=True,
        )

        for region in REGIONS:
            values = np.array(
                [float(row[f"{region}_mean_intensity"]) for row in raw_rows]
            )
            axis_rois.plot(
                time_s,
                values,
                color=COLORS[region],
                linewidth=0.65,
                alpha=0.8,
                label=region.replace("_", " "),
            )
        axis_rois.set_ylabel("Mean raw mono16 count")
        axis_rois.set_title("Timestamped facial-region signals")
        axis_rois.legend(loc="upper right", ncol=3, frameon=False)
        axis_rois.grid(True, alpha=0.2)

        nose_raw = np.array([float(row["nose_mean_intensity"]) for row in raw_rows])
        nose_filtered = np.array(
            [float(row["butterworth_filtered_intensity"]) for row in filtered_rows]
        )
        axis_nose.plot(time_s, nose_raw, color="#9aa0a6", linewidth=0.55, label="nose raw")
        axis_nose.set_ylabel("Raw nose count", color="#626a73")
        filtered_axis = axis_nose.twinx()
        filtered_axis.plot(
            filtered_time_s,
            nose_filtered,
            color="#d62828",
            linewidth=0.85,
            label="nose band-pass",
        )
        filtered_axis.set_ylabel("Filtered count variation", color="#d62828")
        axis_nose.set_xlabel("Seconds from first image-header timestamp")
        axis_nose.set_title("Nose ROI and 10–40 BPM baseline filter")
        axis_nose.grid(True, alpha=0.2)
        handles, labels = axis_nose.get_legend_handles_labels()
        other_handles, other_labels = filtered_axis.get_legend_handles_labels()
        axis_nose.legend(handles + other_handles, labels + other_labels, loc="upper right", frameon=False)

        fig.suptitle(f"{subject.title()} — raw thermal signals ({len(raw_rows)} frames)", fontsize=15)
        output_path = output_dir / f"{subject}_timestamped_raw_rois.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--filtered", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.signals.resolve(), args.filtered.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

