#!/usr/bin/env python3
"""Plot baseline signals against provisional reference-derived timestamps."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "respiration_baseline_candidate_alignment"
    / "candidate_aligned_filtered_signals.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "plots"


def plot(input_csv: Path, output_dir: Path, overwrite: bool) -> None:
    if not input_csv.is_file():
        raise FileNotFoundError(f"Candidate-aligned signal CSV not found: {input_csv}")
    recordings: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_csv.open("r", newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            recordings[row["subject"]].append(row)
    if not recordings:
        raise ValueError("No candidate-aligned signal rows found")

    expected = {f"{subject}_candidate_aligned_baseline.png" for subject in recordings}
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir() if path.is_file()}
        unexpected = existing - expected
        if unexpected:
            raise FileExistsError(f"Plot directory contains unmanaged files: {sorted(unexpected)}")
        if existing and not overwrite:
            raise FileExistsError(f"Plots already exist under {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject, rows in sorted(recordings.items()):
        rows.sort(key=lambda row: int(row["frame_index"]))
        times = [float(row["candidate_reference_time_s"]) for row in rows]
        raw = [float(row["raw_image_intensity"]) for row in rows]
        raw_mean = sum(raw) / len(raw)
        raw_centered = [value - raw_mean for value in raw]
        filtered = [float(row["butterworth_filtered_intensity"]) for row in rows]
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        axes[0].plot(times, raw, color="#9aa9bb", linewidth=0.6, alpha=0.75)
        axes[0].set_ylabel("8-bit intensity")
        axes[0].set_title("Raw nose-region image intensity")
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(
            times,
            raw_centered,
            color="#9aa9bb",
            linewidth=0.55,
            alpha=0.65,
            label="Raw, mean-centered",
        )
        axes[1].plot(times, filtered, color="#c45243", linewidth=1.0, label="Butterworth filtered")
        axes[1].set_title("Band-passed intensity signal")
        axes[1].set_xlabel("Candidate reference time (seconds from recording start)")
        axes[1].set_ylabel("Centered intensity")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="upper right")
        fig.suptitle(f"{subject}: provisional ordinal timestamp alignment; not validated")
        fig.tight_layout()
        fig.savefig(output_dir / f"{subject}_candidate_aligned_baseline.png", dpi=150)
        plt.close(fig)
    print(f"Wrote {len(recordings)} provisional baseline plots under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plot(args.input_csv.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

