#!/usr/bin/env python3
"""Compare held-out event predictions with reference breath interval rates.

Reads predictions from the nested subject-wise TCN evaluation and computes
event-span, median-interval, and count-per-minute rates for both the TCN and
its peak/valley comparator. No model is retrained and no held-out labels are
used to select predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "data" / "processed" / "tcn_inhalation_event_detector" / "tcn_loso_report.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "tcn_inhalation_event_detector"
RECORDING_HORIZON_SECONDS = 120.0
CSV_NAME = "tcn_event_interval_rate_comparison.csv"
JSON_NAME = "tcn_event_interval_rate_comparison.json"


def interval_rates(events: list[float], horizon_seconds: float) -> dict[str, float | int | None]:
    selected = sorted(float(value) for value in events if 0.0 <= float(value) < horizon_seconds)
    intervals = [right - left for left, right in zip(selected, selected[1:])]
    return {
        "event_count": len(selected),
        "count_rate_bpm": 60.0 * len(selected) / horizon_seconds,
        "event_span_rate_bpm": 60.0 * (len(selected) - 1) / (selected[-1] - selected[0]) if len(selected) >= 2 else None,
        "median_interval_rate_bpm": 60.0 / statistics.median(intervals) if intervals else None,
    }


def run(report_path: Path, output_dir: Path, overwrite: bool) -> None:
    csv_path, json_path = output_dir / CSV_NAME, output_dir / JSON_NAME
    if not overwrite and (csv_path.exists() or json_path.exists()):
        raise FileExistsError(f"Outputs already exist in {output_dir}; pass --overwrite")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    predictions = report.get("heldout_predictions")
    if not predictions:
        raise ValueError(f"No held-out predictions were found in {report_path}")

    rows: list[dict[str, Any]] = []
    for fold in predictions:
        subject = fold["heldout_subject"]
        reference = interval_rates(fold["reference_events_seconds"], RECORDING_HORIZON_SECONDS)
        models = {
            "tcn": fold["predicted_events_seconds"],
            "bandpass_peak_valley": fold["peak_baseline_predicted_events_seconds"],
        }
        for method, events in models.items():
            estimate = interval_rates(events, RECORDING_HORIZON_SECONDS)
            row: dict[str, Any] = {
                "heldout_subject": subject,
                "method": method,
                "recording_horizon_seconds": RECORDING_HORIZON_SECONDS,
                "reference_event_count": reference["event_count"],
                "predicted_event_count": estimate["event_count"],
                "reference_count_rate_bpm": reference["count_rate_bpm"],
                "predicted_count_rate_bpm": estimate["count_rate_bpm"],
                "reference_event_span_rate_bpm": reference["event_span_rate_bpm"],
                "predicted_event_span_rate_bpm": estimate["event_span_rate_bpm"],
                "reference_median_interval_rate_bpm": reference["median_interval_rate_bpm"],
                "predicted_median_interval_rate_bpm": estimate["median_interval_rate_bpm"],
                "event_precision": fold["overall_metrics"]["precision"] if method == "tcn" else fold["peak_baseline_metrics"]["precision"],
                "event_recall": fold["overall_metrics"]["recall"] if method == "tcn" else fold["peak_baseline_metrics"]["recall"],
                "event_f1": fold["overall_metrics"]["f1"] if method == "tcn" else fold["peak_baseline_metrics"]["f1"],
            }
            for rate_name in ("count_rate_bpm", "event_span_rate_bpm", "median_interval_rate_bpm"):
                predicted = estimate[rate_name]
                reference_value = reference[rate_name]
                row[f"absolute_error_{rate_name}"] = (
                    abs(float(predicted) - float(reference_value))
                    if predicted is not None and reference_value is not None else None
                )
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict[str, Any]] = {}
    for method in ("tcn", "bandpass_peak_valley"):
        selected_rows = [row for row in rows if row["method"] == method]
        method_summary: dict[str, Any] = {"subject_count": len(selected_rows)}
        for rate_name in ("count_rate_bpm", "event_span_rate_bpm", "median_interval_rate_bpm"):
            errors = [float(row[f"absolute_error_{rate_name}"]) for row in selected_rows if row[f"absolute_error_{rate_name}"] is not None]
            method_summary[f"mean_absolute_error_{rate_name}"] = statistics.mean(errors) if errors else None
            method_summary[f"median_absolute_error_{rate_name}"] = statistics.median(errors) if errors else None
        method_summary["macro_event_f1"] = statistics.mean(float(row["event_f1"]) for row in selected_rows)
        summary[method] = method_summary

    result = {
        "purpose": "Compute held-out event-derived respiration-rate errors in addition to event precision/recall/F1.",
        "fold_protocol": "Uses only outer leave-one-subject-out predictions already generated by the TCN evaluation; each subject's predictions were produced without training or threshold selection on that subject's labels.",
        "rate_definitions": {
            "count_rate_bpm": "number of events / 120-second evaluation horizon * 60",
            "event_span_rate_bpm": "60 * (number of events - 1) / (last event time - first event time)",
            "median_interval_rate_bpm": "60 / median consecutive event interval",
        },
        "reference_mapping_basis": report.get("mapping_basis"),
        "recording_horizon_seconds": RECORDING_HORIZON_SECONDS,
        "summary": summary,
        "rows": rows,
        "limitations": [
            "Rate errors inherit the project-owner-confirmed but not author-verified respiration-code mapping.",
            "A high rate agreement can coexist with poor event timing; event precision, recall, and F1 are included separately.",
            "Only five held-out subjects are available, and thermal inputs use tracker-derived nose ROIs that are not independently validated across full recordings.",
            "Count-rate uses the fixed 120-second protocol horizon; event-span and median-interval estimates are undefined when fewer than two predictions exist.",
        ],
        "inputs": {"tcn_report": str(report_path)},
        "outputs": {"csv": csv_path.name, "json": json_path.name},
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote event-derived rate comparisons for {len(predictions)} held-out subjects to {csv_path}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.report.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

