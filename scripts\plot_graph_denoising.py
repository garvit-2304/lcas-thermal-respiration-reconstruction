#!/usr/bin/env python3
"""Plot raw and fixed-graph-denoised ROI intensity traces by frame index."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "graph_denoising" / "graph_denoised_signals.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "graph_denoising" / "plots"
REGIONS = ("forehead", "nose", "mouth", "left_cheek", "right_cheek")


def plot_graph_signals(input_csv: Path, output_dir: Path, overwrite: bool) -> None:
    if not input_csv.is_file():
        raise FileNotFoundError(f"Graph-denoised CSV not found: {input_csv}")

    recordings: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_csv.open("r", newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            recordings[row["subject"]].append(row)
    if not recordings:
        raise ValueError(f"No graph-denoised rows found in {input_csv}")

    expected_files = {f"{subject}_graph_denoising.png" for subject in recordings}
    if output_dir.exists():
        existing_files = {path.name for path in output_dir.iterdir() if path.is_file()}
        unexpected = existing_files - expected_files
        if unexpected:
            raise FileExistsError(
                f"Plot directory contains files not managed by this script: {sorted(unexpected)}"
            )
        if existing_files and not overwrite:
            raise FileExistsError(
                f"Plots already exist under {output_dir}; pass --overwrite to refresh them."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject, rows in sorted(recordings.items()):
        rows.sort(key=lambda row: (int(row["frame_index"]), row["region"]))
        fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
        for axis, region in zip(axes.flat, REGIONS):
            region_rows = [row for row in rows if row["region"] == region]
            frame_indices = [int(row["frame_index"]) for row in region_rows]
            axis.plot(
                frame_indices,
                [float(row["raw_mean_intensity"]) for row in region_rows],
                color="#777777",
                linewidth=0.5,
                alpha=0.6,
                label="Raw",
            )
            axis.plot(
                frame_indices,
                [float(row["mpt_fixed_graph_intensity"]) for row in region_rows],
                color="#3977b8",
                linewidth=0.9,
                label="Fixed-graph PWS",
            )
            axis.plot(
                frame_indices,
                [float(row["mgt_irls_fixed_graph_intensity"]) for row in region_rows],
                color="#c45243",
                linewidth=0.9,
                label="IRLS-GTV",
            )
            axis.set_title(region.replace("_", " "))
            axis.set_ylabel("8-bit intensity")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right", fontsize=7)
        axes[2, 0].set_xlabel("Frame index (timestamps unavailable)")
        axes[2, 1].set_visible(False)
        fig.suptitle(
            f"{subject}: exploratory fixed-graph filtering (not evaluated against respiration labels)"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / f"{subject}_graph_denoising.png", dpi=150)
        plt.close(fig)

    print(f"Wrote {len(recordings)} graph-filter comparison plots under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plot_graph_signals(args.input_csv.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

