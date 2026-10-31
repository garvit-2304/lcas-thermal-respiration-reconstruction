#!/usr/bin/env python3
"""Compare the thermal baseline with inhalation-event intervals under the working code map.

The working mapping was confirmed by the project owner, but the dataset authors'
numeric codebook was not located. Outputs remain descriptive, not formal accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHYSIOLOGY = ROOT / "data" / "processed" / "physiology_long.csv"
DEFAULT_BASELINE = ROOT / "data" / "processed" / "respiration_baseline_rosbag_raw" / "rosbag_raw_estimates.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
CSV_NAME = "provisional_code1_event_rate_comparison.csv"
JSON_NAME = "provisional_code1_event_rate_comparison.json"


def load_inhalation_events(path: Path) -> tuple[dict[str, list[datetime]], dict[str, dict[str, object]]]:
    events: dict[str, list[datetime]] = defaultdict(list)
    recording_info: dict[str, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"recording", "stream", "timestamp", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Physiology CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row["stream"].casefold() != "respiration":
                continue
            recording = row["recording"]
            subject = recording.rsplit("-", 1)[-1].casefold()
            recording_info.setdefault(subject, {"recording": recording, "timestamps": []})
            stamp = datetime.fromisoformat(row["timestamp"])
            recording_info[subject]["timestamps"].append(stamp)  # type: ignore[index]
            if row["value"].strip() == "1":
                events[subject].append(stamp)

    for subject in events:
        events[subject].sort()
    return events, recording_info


def load_baseline(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "duration_seconds", "estimated_rate_bpm"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Baseline CSV is missing columns: {sorted(missing)}")
        return {row["subject"].casefold(): row for row in reader}


def compare(physiology_path: Path, baseline_path: Path, output_dir: Path, overwrite: bool) -> None:
    output_csv = output_dir / CSV_NAME
    output_json = output_dir / JSON_NAME
    existing = [str(path) for path in (output_csv, output_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Comparison outputs already exist: {existing}; pass --overwrite")

    events, recording_info = load_inhalation_events(physiology_path)
    baseline = load_baseline(baseline_path)
    if set(events) != set(baseline):
        raise ValueError(f"Subjects differ: events={sorted(events)} baseline={sorted(baseline)}")

    rows: list[dict[str, object]] = []
    for subject in sorted(events):
        event_times = events[subject]
        if len(event_times) < 2:
            raise ValueError(f"Need at least two mapped inhalation events for {subject}")
        all_times = recording_info[subject]["timestamps"]
        assert isinstance(all_times, list)
        recording_start = min(all_times)
        intervals = [(right - left).total_seconds() for left, right in zip(event_times, event_times[1:])]
        event_span = (event_times[-1] - event_times[0]).total_seconds()
        event_rate = 60.0 * (len(event_times) - 1) / event_span
        median_interval_rate = 60.0 / statistics.median(intervals)
        baseline_row = baseline[subject]
        baseline_rate = float(baseline_row["estimated_rate_bpm"])
        rows.append(
            {
                "subject": subject,
                "code_mapping": "0=no event, 1=inhalation, 2=exhalation",
                "code_mapping_basis": "project-owner-confirmed working interpretation; not an official dataset codebook",
                "inhalation_event_count": len(event_times),
                "inter_inhalation_interval_count": len(intervals),
                "mean_inter_event_interval_seconds": statistics.mean(intervals),
                "median_inter_event_interval_seconds": statistics.median(intervals),
                "inhalation_event_span_seconds": event_span,
                "inhalation_interval_rate_from_event_span_bpm": event_rate,
                "inhalation_interval_rate_from_median_interval_bpm": median_interval_rate,
                "thermal_baseline_rate_bpm": baseline_rate,
                "thermal_baseline_minus_inhalation_interval_rate_bpm": baseline_rate - event_rate,
                "absolute_difference_bpm": abs(baseline_rate - event_rate),
                "annotation_recording_duration_seconds": (max(all_times) - min(all_times)).total_seconds(),
                "thermal_recording_duration_seconds": float(baseline_row["duration_seconds"]),
                "project_owner_confirmed_mapping": True,
                "official_codebook_confirmed": False,
                "official_mapping_verified": False,
                "formal_accuracy_evaluated": False,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    report = {
        "purpose": "Descriptive comparison of full-recording thermal spectral rates with the inhalation-event interval rate under the project working mapping.",
        "code_semantics": {
            "0": "no_event",
            "1": "inhalation",
            "2": "exhalation",
        },
        "code_mapping_basis": "Project-owner-confirmed working interpretation; the dataset authors' numeric codebook was not located.",
        "official_paper_description": "The paper describes manually annotated inhalation instants but does not define numeric CSV code meanings.",
        "event_rate_formula": "60 * (number_of_inhalation_events - 1) / (last_inhalation_timestamp - first_inhalation_timestamp)",
        "thermal_baseline": str(baseline_path),
        "physiology_input": str(physiology_path),
        "project_owner_confirmed_mapping": True,
        "official_codebook_confirmed": False,
        "official_mapping_verified": False,
        "formal_accuracy_evaluated": False,
        "limitations": [
            "The working mapping is project-owner-confirmed but not author-documented; interpret rates under that convention.",
            "Code 2 occurs only in Joao, so exhalation events cannot be compared across all five recordings.",
            "This is a whole-recording descriptive rate comparison, not breath-event matching or a formal accuracy metric.",
            "The thermal face tracker and geometric ROIs have not been validated against landmarks or manual masks.",
            "No motion or ROI-quality rejection is applied to the thermal baseline.",
        ],
        "recordings": rows,
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote mapped inhalation-event comparison for {len(rows)} subjects to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physiology", type=Path, default=DEFAULT_PHYSIOLOGY)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compare(args.physiology.resolve(), args.baseline.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

