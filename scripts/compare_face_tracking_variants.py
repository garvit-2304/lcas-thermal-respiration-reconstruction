#!/usr/bin/env python3
"""Compare the reference tracker with an exploratory lower-scale-step run.

This report measures implementation sensitivity only. It does not determine
which tracker is more accurate because neither run has manual face/ROI labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data" / "processed"
DEFAULT_CANDIDATE = ROOT / "data" / "processed" / "face_tracking_scale_105_neighbors_2"
CSV_NAME = "tracking_variant_comparison.csv"
JSON_NAME = "tracking_variant_comparison.json"


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        return {row[key].casefold(): row for row in csv.DictReader(file_obj)}


def read_quality(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["subject"].casefold(): row for row in data["subjects"]}


def compare(reference_dir: Path, candidate_dir: Path, overwrite: bool) -> None:
    output_csv = candidate_dir / CSV_NAME
    output_json = candidate_dir / JSON_NAME
    existing = [str(path) for path in (output_csv, output_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Comparison outputs exist: {existing}; pass --overwrite")

    reference_tracks = json.loads((reference_dir / "face_tracking_report.json").read_text(encoding="utf-8"))
    candidate_tracks = json.loads((candidate_dir / "face_tracking_report.json").read_text(encoding="utf-8"))
    reference_quality = read_quality(reference_dir / "roi_quality_audit" / "timestamped_roi_quality_report.json")
    candidate_quality = read_quality(candidate_dir / "roi_quality_audit" / "timestamped_roi_quality_report.json")
    reference_rates = read_csv(
        reference_dir / "respiration_baseline_rosbag_raw" / "rosbag_raw_estimates.csv", "subject"
    )
    candidate_rates = read_csv(candidate_dir / "respiration_baseline" / "rosbag_raw_estimates.csv", "subject")
    subjects = sorted(reference_rates)
    if not (set(subjects) == set(candidate_rates) == set(reference_quality) == set(candidate_quality)):
        raise ValueError("Subject sets differ among tracker, quality, and baseline reports")

    reference_detection = {
        "min_neighbors": reference_tracks.get("cascade_min_neighbors", 4),
        "scale_factor": reference_tracks.get("cascade_scale_factor", 1.10),
        "redetection_interval_frames": reference_tracks.get("redetection_interval_frames", 15),
    }
    candidate_detection = {
        "min_neighbors": candidate_tracks.get("cascade_min_neighbors"),
        "scale_factor": candidate_tracks.get("cascade_scale_factor"),
        "redetection_interval_frames": candidate_tracks.get("redetection_interval_frames"),
    }

    rows: list[dict[str, object]] = []
    for subject in subjects:
        ref_q = reference_quality[subject]
        cand_q = candidate_quality[subject]
        ref_rate = float(reference_rates[subject]["estimated_rate_bpm"])
        cand_rate = float(candidate_rates[subject]["estimated_rate_bpm"])
        rows.append(
            {
                "subject": subject,
                "reference_valid_roi_frames": ref_q["valid_roi_frame_count"],
                "candidate_valid_roi_frames": cand_q["valid_roi_frame_count"],
                "candidate_missing_roi_frames": cand_q["missing_roi_frame_count"],
                "reference_face_speed_p99_px_s": ref_q["face_center_speed_px_s_p99"],
                "candidate_face_speed_p99_px_s": cand_q["face_center_speed_px_s_p99"],
                "reference_nose_jump_p99_count": ref_q["nose_abs_delta_count_p99"],
                "candidate_nose_jump_p99_count": cand_q["nose_abs_delta_count_p99"],
                "reference_nose_baseline_bpm": ref_rate,
                "candidate_nose_baseline_bpm": cand_rate,
                "candidate_minus_reference_bpm": cand_rate - ref_rate,
            }
        )

    fields = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "purpose": "Sensitivity comparison of two heuristic Haar/optical-flow face-track runs.",
        "reference_detector": reference_detection,
        "candidate_detector": candidate_detection,
        "candidate_directory": str(candidate_dir),
        "interpretation": "No variant is selected as more accurate. The candidate improves the manually viewed first-frame box for Jaime, but reviewed turn/motion previews show displaced or incomplete face boxes in other samples. Candidate nose-rate differences demonstrate sensitivity to ROI geometry, not performance improvement.",
        "selection": "The established default outputs remain the reference run. The candidate run is retained separately for inspection and is not fed into the main project outputs.",
        "accuracy_evaluated": False,
        "limitations": [
            "Neither tracker run has manually annotated face boxes or facial-region masks for accuracy measurement.",
            "Face-motion and nose-jump summaries are descriptive diagnostics, not accuracy or signal-quality metrics.",
            "Candidate frames with no usable box are omitted from its spectral input; interpolation bridges those timestamp gaps.",
            "Respiration codes have a project-owner-confirmed working map, but spectral-rate differences are not accuracy measures and the map is not an official dataset codebook.",
        ],
        "recordings": rows,
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Compared tracker variants for {len(rows)} recordings; no candidate was selected.")
    print(f"Wrote {output_csv} and {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compare(args.reference_dir.resolve(), args.candidate_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

