#!/usr/bin/env python3
"""Extract exploratory intensity series from tracked face and candidate ROIs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "original_dataset"
DEFAULT_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_ALIGNMENT = ROOT / "data" / "processed" / "frame_alignment_audit.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
REGIONS = ("forehead", "nose", "mouth", "left_cheek", "right_cheek")


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"sample_count": 0, "temporal_mean": None, "temporal_std": None}
    return {
        "sample_count": len(values),
        "temporal_mean": round(float(np.mean(values)), 6),
        "temporal_std": round(float(np.std(values)), 6),
    }


def extract(input_dir: Path, tracks_path: Path, alignment_path: Path, output_dir: Path) -> None:
    output_csv = output_dir / "roi_signals.csv"
    report_path = output_dir / "roi_signal_report.json"
    if output_csv.exists() or report_path.exists():
        raise FileExistsError(
            f"ROI signal outputs already exist in {output_dir}; choose another --output-dir."
        )
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Face-track CSV not found: {tracks_path}")

    alignment = {}
    if alignment_path.is_file():
        with alignment_path.open("r", newline="", encoding="utf-8") as file_obj:
            alignment = {row["subject"].casefold(): row for row in csv.DictReader(file_obj)}

    signal_fields = [
        "subject",
        "frame_index",
        "image_path",
        "tracking_status",
        "valid_roi",
        "face_mean_intensity",
        "face_median_intensity",
        "face_pixel_count",
    ]
    for region in REGIONS:
        signal_fields.extend(
            [
                f"{region}_mean_intensity",
                f"{region}_median_intensity",
                f"{region}_pixel_count",
            ]
        )

    by_subject: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    frame_counts: dict[str, int] = defaultdict(int)
    valid_counts: dict[str, int] = defaultdict(int)
    processed_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with tracks_path.open("r", newline="", encoding="utf-8") as tracks_file, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        reader = csv.DictReader(tracks_file)
        writer = csv.DictWriter(output_file, fieldnames=signal_fields)
        writer.writeheader()

        for track in reader:
            subject = track["subject"]
            frame_index = int(track["frame_index"])
            frame_counts[subject] += 1
            image_path = input_dir / subject / f"frame_{frame_index:06d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not decode source frame: {image_path}")

            row: dict[str, Any] = {
                "subject": subject,
                "frame_index": frame_index,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "tracking_status": track["tracking_status"],
                "valid_roi": track["valid_roi"],
            }
            if track["valid_roi"].lower() == "true":
                x = int(track["face_x"])
                y = int(track["face_y"])
                w = int(track["face_width"])
                h = int(track["face_height"])
                face_crop = image[y : y + h, x : x + w]
                if face_crop.size == 0:
                    raise ValueError(f"Empty face ROI at {subject} frame {frame_index}")
                row["face_mean_intensity"] = f"{float(face_crop.mean()):.6f}"
                row["face_median_intensity"] = f"{float(np.median(face_crop)):.6f}"
                row["face_pixel_count"] = int(face_crop.size)
                by_subject[subject]["face"].append(float(face_crop.mean()))

                for region in REGIONS:
                    rx = int(track[f"{region}_x"])
                    ry = int(track[f"{region}_y"])
                    rw = int(track[f"{region}_width"])
                    rh = int(track[f"{region}_height"])
                    crop = image[ry : ry + rh, rx : rx + rw]
                    if crop.size == 0:
                        raise ValueError(
                            f"Empty {region} ROI at {subject} frame {frame_index}"
                        )
                    mean = float(crop.mean())
                    median = float(np.median(crop))
                    row[f"{region}_mean_intensity"] = f"{mean:.6f}"
                    row[f"{region}_median_intensity"] = f"{median:.6f}"
                    row[f"{region}_pixel_count"] = int(crop.size)
                    by_subject[subject][f"{region}_mean"].append(mean)
                valid_counts[subject] += 1
            else:
                for field in signal_fields:
                    if field not in row:
                        row[field] = ""
            writer.writerow(row)
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"processed {processed_count} frames")

    subjects_report = []
    for subject in sorted(frame_counts):
        streams = by_subject[subject]
        subjects_report.append(
            {
                "subject": subject,
                "frame_count": frame_counts[subject],
                "valid_roi_frames": valid_counts[subject],
                "reference_alignment": alignment.get(subject.casefold(), {}).get(
                    "alignment_status", "not_assessed"
                ),
                "signals": {
                    "face_mean_intensity": summarize(streams.get("face", [])),
                    **{
                        f"{region}_mean_intensity": summarize(
                            streams.get(f"{region}_mean", [])
                        )
                        for region in REGIONS
                    },
                },
            }
        )

    report = {
        "source_tracks": tracks_path.relative_to(ROOT).as_posix()
        if tracks_path.is_relative_to(ROOT)
        else str(tracks_path),
        "source_frames": input_dir.relative_to(ROOT).as_posix()
        if input_dir.is_relative_to(ROOT)
        else str(input_dir),
        "signal_unit": "8-bit grayscale image intensity (not calibrated temperature)",
        "timestamp_handling": "No image timestamps available; signals use subject and frame_index only.",
        "label_handling": "Respiration labels are not joined to these signals.",
        "statistics": "Per-frame mean and median over the original grayscale image pixels in each rectangle.",
        "roi_regions": list(REGIONS),
        "subjects": subjects_report,
        "total_rows": sum(frame_counts.values()),
        "output_csv": output_csv.relative_to(ROOT).as_posix()
        if output_csv.is_relative_to(ROOT)
        else str(output_csv),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted ROI intensity signals for {sum(frame_counts.values())} frames.")
    print(f"Wrote {output_csv}")
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    extract(
        args.input_dir.resolve(),
        args.tracks.resolve(),
        args.alignment.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()

