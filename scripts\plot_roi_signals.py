#!/usr/bin/env python3
"""Plot extracted L-CAS region-intensity signals against frame index."""

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
DEFAULT_INPUT = ROOT / "data" / "processed" / "roi_signals.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "roi_signal_plots"
SIGNALS = {
    "face": "face_mean_intensity",
    "forehead": "forehead_mean_intensity",
    "nose": "nose_mean_intensity",
    "mouth": "mouth_mean_intensity",
    "left cheek": "left_cheek_mean_intensity",
    "right cheek": "right_cheek_mean_intensity",
}


def plot_signals(input_csv: Path, output_dir: Path, smooth_window: int, overwrite: bool) -> None:
    if not input_csv.is_file():
        raise FileNotFoundError(f"ROI signal CSV not found: {input_csv}")

    recordings: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_csv.open("r", newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            recordings[row["subject"]].append(row)

    expected_files = {f"{subject}_roi_signals.png" for subject in recordings}
    if output_dir.exists():
        existing_files = {path.name for path in output_dir.iterdir() if path.is_file()}
        unexpected = existing_files - expected_files
        if unexpected:
            raise FileExistsError(
                f"Plot directory contains files not managed by this script: {sorted(unexpected)}"
            )
        if existing_files and not overwrite:
            raise FileExistsError(f"Plots already exist under {output_dir}; pass --overwrite to refresh them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject, rows in sorted(recordings.items()):
        rows.sort(key=lambda row: int(row["frame_index"]))
        frame_indices = np.array([int(row["frame_index"]) for row in rows])
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        for axis, (label, column) in zip(axes.flat, SIGNALS.items()):
            values = np.array([float(row[column]) if row[column] else np.nan for row in rows])
            axis.plot(frame_indices, values, color="#8aa5c8", linewidth=0.55, alpha=0.8, label="Frame mean")
            finite = np.isfinite(values)
            effective_window = min(smooth_window, len(values))
            kernel = np.ones(effective_window, dtype=np.float64)
            numerator = np.convolve(np.nan_to_num(values), kernel, mode="same")
            denominator = np.convolve(finite.astype(np.float64), kernel, mode="same")
            rolling = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )
            axis.plot(
                frame_indices,
                rolling,
                color="#c45243",
                linewidth=1.1,
                label=f"{effective_window}-frame moving mean",
            )
            axis.set_title(label)
            axis.set_ylabel("8-bit intensity")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right", fontsize=7)
        for axis in axes[-1, :]:
            axis.set_xlabel("Frame index (capture timestamps unavailable)")
        fig.suptitle(f"{subject}: face and candidate-region image intensities")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / f"{subject}_roi_signals.png", dpi=150)
        plt.close(fig)

    print(f"Wrote {len(recordings)} plots under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smooth-window", type=int, default=27)
    parser.add_argument("--overwrite", action="store_true", help="Refresh plots previously created by this script.")
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window must be at least 1")
    plot_signals(
        args.input_csv.resolve(),
        args.output_dir.resolve(),
        args.smooth_window,
        args.overwrite,
    )


if __name__ == "__main__":
    main()

