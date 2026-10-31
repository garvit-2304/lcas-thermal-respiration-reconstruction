#!/usr/bin/env python3
"""Summarize how raw graph-filter outputs change across exploratory strengths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from graph_denoise_timestamped_raw import (
    DEFAULT_SIGNALS,
    DEFAULT_TRACKS,
    denoise_subject,
    estimate_rate,
    prepare_subjects,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "graph_denoising_rosbag_raw"
    / "parameter_sensitivity.csv"
)
PWS_STRENGTHS = (0.05, 0.2, 0.5)
GTV_STRENGTHS = (0.002, 0.01, 0.05)


def run(signals_path: Path, tracks_path: Path, output_path: Path, overwrite: bool) -> None:
    report_path = output_path.with_name("parameter_sensitivity_summary.json")
    existing = [path for path in (output_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Sensitivity outputs already exist: {existing}; pass --overwrite")
    subjects = prepare_subjects(signals_path, tracks_path, 382, 288)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for subject, data in sorted(subjects.items()):
        raw_nose = data["values"][:, 1]
        timestamps = data["timestamps"]
        raw_rate = estimate_rate(raw_nose, timestamps)["estimated_rate_bpm"]
        std = float(np.std(raw_nose))
        pws_rates: list[float] = []
        gtv_rates: list[float] = []
        gtv_convergence: list[bool] = []
        for pws_strength in PWS_STRENGTHS:
            for gtv_strength in GTV_STRENGTHS:
                pws, gtv, graph_report, _ = denoise_subject(
                    data["values"],
                    data["positions"],
                    timestamps,
                    spatial_scale=0.15,
                    intensity_scale=0.20,
                    temporal_scale_factor=2.0,
                    pws_strength=pws_strength,
                    gtv_strength=gtv_strength,
                    gtv_epsilon=0.01,
                    gtv_iterations=10,
                    convergence_tolerance=1e-5,
                )
                pws_nose = pws[:, 1]
                gtv_nose = gtv[:, 1]
                pws_rate = estimate_rate(pws_nose, timestamps)["estimated_rate_bpm"]
                gtv_rate = estimate_rate(gtv_nose, timestamps)["estimated_rate_bpm"]
                pws_rates.append(pws_rate)
                gtv_rates.append(gtv_rate)
                gtv_convergence.append(bool(graph_report["gtv_converged"]))
                rows.append(
                    {
                        "subject": subject,
                        "pws_strength": pws_strength,
                        "gtv_strength": gtv_strength,
                        "raw_nose_rate_bpm": raw_rate,
                        "pws_nose_rate_bpm": pws_rate,
                        "gtv_nose_rate_bpm": gtv_rate,
                        "pws_rms_change_over_raw_std": float(np.sqrt(np.mean((pws_nose - raw_nose) ** 2)) / std),
                        "gtv_rms_change_over_raw_std": float(np.sqrt(np.mean((gtv_nose - raw_nose) ** 2)) / std),
                        "gtv_iterations_run": graph_report["gtv_iterations_run"],
                        "gtv_converged": graph_report["gtv_converged"],
                        "parameters_tuned_on_labels": False,
                    }
                )
        summaries.append(
            {
                "subject": subject,
                "raw_nose_rate_bpm": raw_rate,
                "pws_rate_min_bpm": min(pws_rates),
                "pws_rate_max_bpm": max(pws_rates),
                "gtv_rate_min_bpm": min(gtv_rates),
                "gtv_rate_max_bpm": max(gtv_rates),
                "gtv_converged_runs": sum(gtv_convergence),
                "gtv_total_runs": len(gtv_convergence),
                "strength_grid": {
                    "pws": list(PWS_STRENGTHS),
                    "gtv": list(GTV_STRENGTHS),
                },
                "label_based_selection": False,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject",
        "pws_strength",
        "gtv_strength",
        "raw_nose_rate_bpm",
        "pws_nose_rate_bpm",
        "gtv_nose_rate_bpm",
        "pws_rms_change_over_raw_std",
        "gtv_rms_change_over_raw_std",
        "gtv_iterations_run",
        "gtv_converged",
        "parameters_tuned_on_labels",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "raw_nose_rate_bpm": f"{row['raw_nose_rate_bpm']:.6f}",
                    "pws_nose_rate_bpm": f"{row['pws_nose_rate_bpm']:.6f}",
                    "gtv_nose_rate_bpm": f"{row['gtv_nose_rate_bpm']:.6f}",
                    "pws_rms_change_over_raw_std": f"{row['pws_rms_change_over_raw_std']:.8g}",
                    "gtv_rms_change_over_raw_std": f"{row['gtv_rms_change_over_raw_std']:.8g}",
                }
            )
    report_path.write_text(json.dumps({"subjects": summaries}, indent=2) + "\n", encoding="utf-8")
    for summary in summaries:
        print(
            f"{summary['subject']}: raw {summary['raw_nose_rate_bpm']:.2f} BPM; "
            f"PWS {summary['pws_rate_min_bpm']:.2f}–{summary['pws_rate_max_bpm']:.2f}; "
            f"GTV {summary['gtv_rate_min_bpm']:.2f}–{summary['gtv_rate_max_bpm']:.2f}"
        )
    print(f"Wrote parameter grid to {output_path} and {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.signals.resolve(), args.tracks.resolve(), args.output.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

