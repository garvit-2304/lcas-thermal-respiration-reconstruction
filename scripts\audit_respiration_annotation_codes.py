#!/usr/bin/env python3
"""Audit raw respiration codes and apply the project-owner-confirmed working map."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "original_dataset"
DEFAULT_OUTPUT_CSV = ROOT / "data" / "processed" / "respiration_annotation_code_audit.csv"
DEFAULT_OUTPUT_JSON = ROOT / "data" / "processed" / "respiration_annotation_code_audit.json"
CODE_LABELS = {
    "0": "no_event",
    "1": "inhalation",
    "2": "exhalation",
}
MAPPING_BASIS = (
    "Project-owner-confirmed working interpretation; the official dataset codebook was not located."
)


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d-%H:%M:%S.%f")


def audit(dataset_dir: Path, output_csv: Path, output_json: Path, overwrite: bool) -> None:
    source_files = sorted(dataset_dir.glob("*_respiration.csv"))
    if not source_files:
        raise FileNotFoundError(f"No respiration CSVs found under {dataset_dir}")
    outputs = [output_csv, output_json]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Audit outputs already exist: {[str(path) for path in existing]}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for source in source_files:
        subject = source.stem.removesuffix("_respiration").split("-")[-1].casefold()
        timestamps: list[datetime] = []
        codes: list[str] = []
        with source.open("r", newline="", encoding="utf-8") as file_obj:
            for row in csv.reader(file_obj):
                if len(row) < 2:
                    continue
                timestamps.append(parse_timestamp(row[0]))
                codes.append(row[1].strip())
        if not timestamps:
            raise ValueError(f"No annotation rows found in {source}")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError(f"Timestamps are not strictly increasing in {source.name}")

        code1_times = [time for time, code in zip(timestamps, codes) if code == "1"]
        code2_times = [time for time, code in zip(timestamps, codes) if code == "2"]
        code1_intervals = [
            (right - left).total_seconds() for left, right in zip(code1_times, code1_times[1:])
        ]
        observed_events = [(time, code) for time, code in zip(timestamps, codes) if code not in {"0", ""}]
        alternating_transitions = sum(
            left_code != right_code for (_, left_code), (_, right_code) in zip(observed_events, observed_events[1:])
        )
        transition_count = max(len(observed_events) - 1, 0)
        duration_s = (timestamps[-1] - timestamps[0]).total_seconds()
        summaries.append(
            {
                "subject": subject,
                "source_file": source.name,
                "sample_count": len(codes),
                "duration_seconds": duration_s,
                "code_counts": dict(Counter(codes)),
                "inhalation_event_count": len(code1_times),
                "mean_inter_inhalation_interval_seconds": (
                    sum(code1_intervals) / len(code1_intervals) if code1_intervals else None
                ),
                "inhalation_interval_rate_bpm": (
                    60.0 / (sum(code1_intervals) / len(code1_intervals)) if code1_intervals else None
                ),
                "exhalation_event_count": len(code2_times),
                "nonzero_transition_count": transition_count,
                "nonzero_code_alternation_fraction": (
                    alternating_transitions / transition_count if transition_count else None
                ),
                "interpretation": (
                    "project-owner-confirmed working map: 0=no event, 1=inhalation, "
                    "2=exhalation; not an author-published codebook"
                ),
            }
        )

    fields = [
        "subject",
        "source_file",
        "sample_count",
        "duration_seconds",
        "code_counts_json",
        "inhalation_event_count",
        "mean_inter_inhalation_interval_seconds",
        "inhalation_interval_rate_bpm",
        "exhalation_event_count",
        "nonzero_transition_count",
        "nonzero_code_alternation_fraction",
        "interpretation",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    **{key: summary[key] for key in fields if key in summary},
                    "code_counts_json": json.dumps(summary["code_counts"], sort_keys=True),
                    "duration_seconds": f"{summary['duration_seconds']:.6f}",
                    "mean_inter_inhalation_interval_seconds": (
                        f"{summary['mean_inter_inhalation_interval_seconds']:.6f}"
                        if summary["mean_inter_inhalation_interval_seconds"] is not None
                        else ""
                    ),
                    "inhalation_interval_rate_bpm": (
                        f"{summary['inhalation_interval_rate_bpm']:.6f}"
                        if summary["inhalation_interval_rate_bpm"] is not None
                        else ""
                    ),
                    "nonzero_code_alternation_fraction": (
                        f"{summary['nonzero_code_alternation_fraction']:.6f}"
                        if summary["nonzero_code_alternation_fraction"] is not None
                        else ""
                    ),
                }
            )

    report = {
        "source_directory": str(dataset_dir),
        "source_count": len(source_files),
        "official_annotation_description": "Original L-CAS paper describes manually annotated inhalation time instants.",
        "code_mapping_status": "project_owner_confirmed_working_mapping",
        "code_mapping": {
            code: label for code, label in CODE_LABELS.items()
        },
        "code_mapping_basis": MAPPING_BASIS,
        "code_mapping_confirmation_date": "2026-10-31",
        "official_codebook_status": "not found in the official dataset page, paper, or ROS source reviewed",
        "sources_reviewed": [
            {
                "title": "Official L-CAS Thermal Physiological Monitoring Dataset page",
                "url": "https://lcas.lincoln.ac.uk/wp/research/data-sets-software/lcas-thermal-physiological-monitoring-dataset/",
                "finding": "Confirms respiration ground truth is provided but does not define numeric CSV code meanings.",
                "ground_truth_download_status": "The linked Nextcloud ground-truth download returned HTTP 404 during the source check; the archived official ZIP already in the project was used for the local source-file comparison.",
            },
            {
                "title": "Cosar et al. 2018, Thermal Camera Based Physiological Monitoring",
                "url": "https://cpb-eu-w2.wpmucdn.com/blogs.lincoln.ac.uk/dist/2/1110/files/2020/06/Cosar2018.pdf",
                "finding": "States respiration references were derived from manually annotated inhalation instants; does not map CSV codes 0, 1, or 2.",
            },
            {
                "title": "Official LCAS ENRICHME physiological monitoring ROS source",
                "url": "https://github.com/LCAS/ENRICHME_Releases/blob/master/physiological_monitoring/src/physiological_monitoring.cpp",
                "finding": "Implements physiological rate estimation; no numeric ground-truth CSV annotation mapping was found.",
            },
        ],
        "evidence_and_scope": [
            "The paper says respiration references were obtained by manually marking inhalation instants, which supports the working interpretation of code 1.",
            "Code 2 occurs only in Joao (37 rows) and alternates with code 1; under the project-owner-confirmed working map it is treated as an exhalation marker.",
            "Codes are still preserved unchanged in source and synchronized data; the mapping is attached as interpretation metadata.",
        ],
        "limitations": [
            "The project owner confirmed this working interpretation, but the numeric map is not documented by the dataset authors.",
            "Code 2 occurs only in Joao, so exhalation-event comparisons are not available across all five recordings.",
            "Inhalation interval rates are descriptive and are not a formal event-level accuracy evaluation.",
            "The annotation stream is event-coded; code 0 is treated as no event and is not a continuous respiration waveform.",
        ],
        "recordings": summaries,
    }
    with output_json.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
        file_obj.write("\n")
    print(f"Wrote code audit for {len(summaries)} recordings to {output_csv} and {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit(args.dataset_dir.resolve(), args.output_csv.resolve(), args.output_json.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

