#!/usr/bin/env python3
"""Validate and prepare the official L-CAS physiological reference streams.

The source ZIP contains heartbeat and respiration CSVs, not thermal frames.
This script keeps each stream at its native timestamps and preserves raw
respiration codes alongside the project-owner-confirmed working interpretation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "source" / "lcas_ground_truth_official_2020-08-03.zip"
DEFAULT_LOCAL_RAW = ROOT / "original_dataset"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
TIMESTAMP_FORMAT = "%Y%m%d-%H:%M:%S.%f"
OFFICIAL_ARCHIVE_URL = (
    "https://web.archive.org/web/20200803203427id_/"
    "https://lcas.lincoln.ac.uk/nextcloud/index.php/s/"
    "t5BiQPeLpYnnEwJ/download"
)
RESPIRATION_CODE_LABELS = {
    "0": "no_event",
    "1": "inhalation",
    "2": "exhalation",
}


def respiration_label(value: str) -> str:
    try:
        numeric = float(value)
    except ValueError:
        return "unmapped_code"
    code = str(int(numeric)) if numeric.is_integer() else value
    return RESPIRATION_CODE_LABELS.get(code, "unmapped_code")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def parse_member(content: bytes, member_name: str) -> list[tuple[datetime, str]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{member_name}: expected UTF-8 CSV") from error

    parsed: list[tuple[datetime, str]] = []
    reader = csv.reader(io.StringIO(text, newline=""))
    for line_number, row in enumerate(reader, start=1):
        if len(row) != 2:
            raise ValueError(
                f"{member_name}:{line_number}: expected two columns, found {len(row)}"
            )
        timestamp_text, value_text = (field.strip() for field in row)
        if not timestamp_text or not value_text:
            raise ValueError(f"{member_name}:{line_number}: empty timestamp or value")
        try:
            timestamp = datetime.strptime(timestamp_text, TIMESTAMP_FORMAT)
            numeric_value = float(value_text)
        except ValueError as error:
            raise ValueError(f"{member_name}:{line_number}: invalid timestamp or value") from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"{member_name}:{line_number}: value must be finite")
        parsed.append((timestamp, value_text))

    if not parsed:
        raise ValueError(f"{member_name}: no data rows")

    timestamps = [item[0] for item in parsed]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{member_name}: timestamps must be strictly increasing")
    return parsed


def stream_summary(
    member_name: str,
    stream_name: str,
    parsed: list[tuple[datetime, str]],
    content_hash: str,
    local_raw_dir: Path,
) -> dict[str, Any]:
    timestamps = [item[0] for item in parsed]
    values = [float(item[1]) for item in parsed]
    intervals = [
        (right - left).total_seconds()
        for left, right in zip(timestamps, timestamps[1:])
    ]
    median_interval = statistics.median(intervals) if intervals else None
    gap_threshold = median_interval * 1.5 if median_interval is not None else None
    subject = member_name.rsplit("_", 1)[0]
    local_file = local_raw_dir / member_name
    local_match = local_file.is_file() and sha256_bytes(local_file.read_bytes()) == content_hash

    summary: dict[str, Any] = {
        "source_file": member_name,
        "source_sha256": content_hash,
        "local_original_exists": local_file.is_file(),
        "local_original_matches_source": local_match,
        "stream": stream_name,
        "row_count": len(parsed),
        "start_time": timestamps[0].isoformat(timespec="milliseconds"),
        "end_time": timestamps[-1].isoformat(timespec="milliseconds"),
        "duration_seconds": round((timestamps[-1] - timestamps[0]).total_seconds(), 6),
        "median_interval_seconds": round(median_interval, 6) if median_interval is not None else None,
        "mean_interval_seconds": round(statistics.mean(intervals), 6) if intervals else None,
        "p95_interval_seconds": round(percentile(intervals, 0.95), 6)
        if intervals
        else None,
        "irregular_gap_threshold_seconds": round(gap_threshold, 6)
        if gap_threshold is not None
        else None,
        "irregular_gap_count": sum(interval > gap_threshold for interval in intervals)
        if gap_threshold is not None
        else 0,
        "duplicate_timestamps": 0,
        "non_monotonic_timestamps": 0,
        "numeric_value_min": min(values),
        "numeric_value_max": max(values),
        "unique_value_count": len(set(values)),
    }
    if len(set(values)) <= 20:
        summary["unique_values"] = sorted(set(values))
    return summary


def load_archive(archive_path: Path) -> tuple[dict[str, bytes], str]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Official ground-truth archive not found: {archive_path}")
    archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
        )
        if not csv_names:
            raise ValueError("The source archive contains no CSV files")
        members = {name: archive.read(name) for name in csv_names}
    return members, sha256_bytes(archive_bytes)


def preprocess(
    archive_path: Path,
    local_raw_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    output_csv = output_dir / "physiology_long.csv"
    output_report = output_dir / "preprocessing_report.json"
    if (output_csv.exists() or output_report.exists()) and not overwrite:
        raise FileExistsError(
            f"Processed outputs already exist in {output_dir}; remove or move them, "
            "or pass --overwrite to replace only the managed physiology outputs."
        )

    members, archive_hash = load_archive(archive_path)
    subjects: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    summaries: list[dict[str, Any]] = []

    for member_name, content in members.items():
        basename = Path(member_name).name
        if basename.endswith("_respiration.csv"):
            stream_name = "respiration"
            unit = ""
            semantics = (
                "raw respiration code retained; value_label applies the project-owner-confirmed "
                "working map, which is not an official dataset codebook"
            )
        elif basename.endswith("_heartbeat.csv"):
            stream_name = "heartbeat"
            unit = "bpm"
            semantics = "heart-rate reference in beats per minute"
        else:
            continue

        subject = basename.rsplit("_", 1)[0]
        content_hash = sha256_bytes(content)
        parsed = parse_member(content, basename)
        summary = stream_summary(
            basename, stream_name, parsed, content_hash, local_raw_dir
        )
        summaries.append(summary)
        subjects[subject][stream_name] = {
            "rows": parsed,
            "unit": unit,
            "semantics": semantics,
        }

    if not subjects:
        raise ValueError("No recognized respiration or heartbeat CSVs were found")
    incomplete = {
        subject: sorted({"respiration", "heartbeat"} - set(streams))
        for subject, streams in subjects.items()
        if set(streams) != {"respiration", "heartbeat"}
    }
    if incomplete:
        raise ValueError(f"Each recording must have both streams; missing: {incomplete}")

    output_dir.mkdir(parents=True, exist_ok=True)
    recording_summaries: list[dict[str, Any]] = []
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "recording",
                "stream",
                "sample_index",
                "timestamp",
                "relative_time_s",
                "value",
                "value_label",
                "unit",
                "value_semantics",
            ],
        )
        writer.writeheader()

        for subject in sorted(subjects):
            streams = subjects[subject]
            all_timestamps = [
                timestamp
                for stream in streams.values()
                for timestamp, _ in stream["rows"]
            ]
            recording_start = min(all_timestamps)
            recording_end = max(all_timestamps)
            stream_starts = {
                name: rows[0][0]
                for name, rows in ((name, data["rows"]) for name, data in streams.items())
            }
            stream_ends = {
                name: rows[-1][0]
                for name, rows in ((name, data["rows"]) for name, data in streams.items())
            }
            overlap_start = max(stream_starts.values())
            overlap_end = min(stream_ends.values())

            rows_to_write: list[tuple[datetime, str, int, str, str, str, str]] = []
            for stream_name, stream_data in streams.items():
                for sample_index, (timestamp, value) in enumerate(stream_data["rows"]):
                    elapsed_s = (timestamp - recording_start).total_seconds()
                    label = respiration_label(value) if stream_name == "respiration" else ""
                    rows_to_write.append(
                        (
                            timestamp,
                            stream_name,
                            sample_index,
                            f"{elapsed_s:.3f}",
                            value,
                            label,
                            stream_data["unit"],
                        )
                    )

            for timestamp, stream_name, sample_index, elapsed_s, value, label, unit in sorted(
                rows_to_write, key=lambda row: (row[0], row[1], row[2])
            ):
                writer.writerow(
                    {
                        "recording": subject,
                        "stream": stream_name,
                        "sample_index": sample_index,
                        "timestamp": timestamp.isoformat(timespec="milliseconds"),
                        "relative_time_s": elapsed_s,
                        "value": value,
                        "value_label": label,
                        "unit": unit,
                        "value_semantics": streams[stream_name]["semantics"],
                    }
                )

            recording_summaries.append(
                {
                    "recording": subject,
                    "recording_start": recording_start.isoformat(timespec="milliseconds"),
                    "recording_end": recording_end.isoformat(timespec="milliseconds"),
                    "duration_seconds": round(
                        (recording_end - recording_start).total_seconds(), 6
                    ),
                    "stream_overlap_start": overlap_start.isoformat(timespec="milliseconds"),
                    "stream_overlap_end": overlap_end.isoformat(timespec="milliseconds"),
                    "stream_overlap_seconds": round(
                        max(0.0, (overlap_end - overlap_start).total_seconds()), 6
                    ),
                    "rows": {
                        name: len(data["rows"]) for name, data in sorted(streams.items())
                    },
                }
            )

    report = {
        "dataset": "L-CAS Thermal Physiological Monitoring Dataset - ground-truth streams",
        "source_archive": archive_path.relative_to(ROOT).as_posix()
        if archive_path.is_relative_to(ROOT)
        else str(archive_path),
        "source_archive_sha256": archive_hash,
        "source_archive_url": OFFICIAL_ARCHIVE_URL,
        "source_archive_snapshot_utc": "2020-08-03T20:34:27Z",
        "official_dataset_page": (
            "https://lcas.lincoln.ac.uk/wp/research/data-sets-software/"
            "lcas-thermal-physiological-monitoring-dataset/"
        ),
        "time_handling": {
            "source_timezone": "not specified in CSV files; timestamps retained as naive local values",
            "relative_time_origin": "earliest timestamp across the two streams per recording",
            "resampling": "none; original irregular sample times are preserved",
            "interpolation": "none",
        },
        "respiration_label_handling": {
            "raw_numeric_codes_preserved": True,
            "working_mapping": RESPIRATION_CODE_LABELS,
            "mapping_basis": "Project owner confirmed this working interpretation on 2026-10-31; the dataset authors' numeric codebook was not found.",
            "unknown_codes": "value_label is unmapped_code; raw value is preserved",
        },
        "recordings": recording_summaries,
        "streams": summaries,
        "output_csv": output_csv.relative_to(ROOT).as_posix()
        if output_csv.is_relative_to(ROOT)
        else str(output_csv),
        "total_rows": sum(sum(item["rows"].values()) for item in recording_summaries),
    }
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    matches = sum(summary["local_original_matches_source"] for summary in summaries)
    print(f"Processed {len(summaries)} streams across {len(subjects)} recordings.")
    print(f"Verified local originals against official archive: {matches}/{len(summaries)} match.")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--local-raw-dir", type=Path, default=DEFAULT_LOCAL_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only physiology_long.csv and preprocessing_report.json in the output directory.",
    )
    args = parser.parse_args()
    preprocess(
        args.archive.resolve(),
        args.local_raw_dir.resolve(),
        args.output_dir.resolve(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

