#!/usr/bin/env python3
"""Plot raw and graph-denoised raw-count traces against ROS timestamps."""

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
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "graph_denoising_rosbag_raw"
    / "timestamped_raw_graph_signals.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "graph_denoising_rosbag_raw" / "plots"
REGIONS = ("forehead", "nose", "mouth", "left_cheek", "right_cheek")
METHODS = (
    ("raw_mono16_count", "Raw"),
    ("pws_fixed_graph_count", "PWS"),
    ("gtv_irls_fixed_graph_count", "GTV"),
)


def plot(input_path: Path, output_dir: Path) -> None:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with input_path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        for row in csv.DictReader(file_obj):
            grouped[row["subject"].casefold()][row["region"]].append(row)
    if not grouped:
        raise ValueError(f"No graph-denoising rows found in {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject, region_rows in sorted(grouped.items()):
        fig, axes = plt.subplots(
            len(REGIONS),
            len(METHODS),
            figsize=(15, 11),
            sharex="col",
            sharey="row",
            constrained_layout=True,
        )
        frame_count = 0
        for region_index, region in enumerate(REGIONS):
            rows = sorted(region_rows.get(region, []), key=lambda row: int(row["frame_index"]))
            if not rows:
                raise ValueError(f"Missing {region} graph output for {subject}")
            frame_count = max(frame_count, len(rows))
            elapsed = np.array(
                [float(row["seconds_since_first_image"]) for row in rows], dtype=np.float64
            )
            for method_index, (column, label) in enumerate(METHODS):
                values = np.array([float(row[column]) for row in rows], dtype=np.float64)
                axis = axes[region_index, method_index]
                axis.plot(elapsed, values, linewidth=0.65, color="#2a6f97")
                axis.grid(True, alpha=0.18)
                if method_index == 0:
                    axis.set_ylabel(f"{region.replace('_', ' ')}\nraw counts")
                if region_index == 0:
                    axis.set_title(label)
                if region_index == len(REGIONS) - 1:
                    axis.set_xlabel("Seconds from first image-header timestamp")
        fig.suptitle(
            f"{subject.title()} — timestamp-aware raw graph denoising ({frame_count} frames)",
            fontsize=15,
        )
        output_path = output_dir / f"{subject}_raw_graph_comparison.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.input.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

