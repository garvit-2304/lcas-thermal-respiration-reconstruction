#!/usr/bin/env python3
"""Tune graph strengths on four subjects and report the held-out fifth subject.

This is an exploratory subject-wise cross-validation using full-recording
inhalation interval rates as the training objective. It does not replace
independent face/ROI validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = ROOT / "data" / "processed" / "graph_denoising_rosbag_raw" / "parameter_sensitivity.csv"
DEFAULT_AUDIT = ROOT / "data" / "processed" / "respiration_annotation_code_audit.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "graph_denoising_rosbag_raw"
PWS_PARAMETER = "pws_strength"
GTV_PARAMETER = "gtv_strength"
PARAMETERS = {
    "pws": (PWS_PARAMETER, "pws_nose_rate_bpm"),
    "gtv": (GTV_PARAMETER, "gtv_nose_rate_bpm"),
}


def load_grid(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {
            "subject", "pws_strength", "gtv_strength", "raw_nose_rate_bpm",
            "pws_nose_rate_bpm", "gtv_nose_rate_bpm", "gtv_converged",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Parameter grid is missing fields: {sorted(missing)}")
        for row in reader:
            groups.setdefault(row["subject"].casefold(), []).append(row)
    if not groups:
        raise ValueError(f"No parameter grid rows found in {path}")
    return groups


def load_reference_rates(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rates = {
        str(item["subject"]).casefold(): float(item["inhalation_interval_rate_bpm"])
        for item in report["recordings"]
        if item.get("inhalation_interval_rate_bpm") is not None
    }
    if not rates:
        raise ValueError(f"No mapped inhalation rates found in {path}")
    return rates


def candidate_score(
    grid: dict[str, list[dict[str, str]]],
    references: dict[str, float],
    training_subjects: list[str],
    method: str,
    strength: float,
) -> tuple[float | None, bool]:
    parameter, rate_field = PARAMETERS[method]
    errors: list[float] = []
    all_converged = True
    for subject in training_subjects:
        matching = [row for row in grid[subject] if float(row[parameter]) == strength]
        if not matching:
            return None, False
        rates = [float(row[rate_field]) for row in matching]
        subject_rate = statistics.mean(rates)
        errors.append(abs(subject_rate - references[subject]))
        if method == "gtv" and not all(row["gtv_converged"].casefold() == "true" for row in matching):
            all_converged = False
    return statistics.mean(errors), all_converged


def heldout_rate(
    grid: dict[str, list[dict[str, str]]],
    subject: str,
    method: str,
    strength: float,
) -> tuple[float, bool]:
    parameter, rate_field = PARAMETERS[method]
    matching = [row for row in grid[subject] if float(row[parameter]) == strength]
    if not matching:
        raise ValueError(f"No {method} rows for {subject} at strength {strength:g}")
    rate = statistics.mean(float(row[rate_field]) for row in matching)
    converged = all(row["gtv_converged"].casefold() == "true" for row in matching)
    return rate, converged


def run(grid_path: Path, audit_path: Path, output_dir: Path, overwrite: bool) -> None:
    output_csv = output_dir / "leave_one_subject_out_tuning.csv"
    output_json = output_dir / "leave_one_subject_out_tuning.json"
    existing = [path for path in (output_csv, output_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"LOSO graph-tuning outputs already exist: {existing}; pass --overwrite")

    grid = load_grid(grid_path)
    references = load_reference_rates(audit_path)
    if set(grid) != set(references):
        raise ValueError(f"Grid/reference subjects differ: {sorted(grid)} vs {sorted(references)}")
    strengths = {
        method: sorted({float(row[parameter]) for subject_rows in grid.values() for row in subject_rows})
        for method, (parameter, _) in PARAMETERS.items()
    }

    folds: list[dict[str, Any]] = []
    selection_details: list[dict[str, Any]] = []
    for heldout_subject in sorted(grid):
        training_subjects = [subject for subject in sorted(grid) if subject != heldout_subject]
        selected: dict[str, float] = {}
        scores_by_method: dict[str, list[dict[str, Any]]] = {}
        for method in PARAMETERS:
            scores: list[dict[str, Any]] = []
            for strength in strengths[method]:
                score, converged = candidate_score(
                    grid, references, training_subjects, method, strength
                )
                scores.append({
                    "strength": strength,
                    "training_mean_absolute_error_bpm": score,
                    "converged_on_all_training_subjects": converged if method == "gtv" else None,
                })
            eligible = [
                item for item in scores
                if item["training_mean_absolute_error_bpm"] is not None
                and (method != "gtv" or item["converged_on_all_training_subjects"])
            ]
            if not eligible:
                raise ValueError(f"No eligible {method} strengths for held-out fold {heldout_subject}")
            # Prefer lower regularization when training errors tie.
            best = min(
                eligible,
                key=lambda item: (item["training_mean_absolute_error_bpm"], item["strength"]),
            )
            selected[method] = float(best["strength"])
            scores_by_method[method] = scores

        pws_rate, _ = heldout_rate(grid, heldout_subject, "pws", selected["pws"])
        gtv_rate, gtv_converged = heldout_rate(grid, heldout_subject, "gtv", selected["gtv"])
        raw_rate = statistics.mean(float(row["raw_nose_rate_bpm"]) for row in grid[heldout_subject])
        reference_rate = references[heldout_subject]
        fold = {
            "heldout_subject": heldout_subject,
            "training_subjects": ";".join(training_subjects),
            "training_subject_count": len(training_subjects),
            "reference_inhalation_rate_bpm": reference_rate,
            "raw_rate_bpm": raw_rate,
            "raw_absolute_error_bpm": abs(raw_rate - reference_rate),
            "selected_pws_strength": selected["pws"],
            "pws_training_mae_bpm": next(
                item["training_mean_absolute_error_bpm"]
                for item in scores_by_method["pws"] if item["strength"] == selected["pws"]
            ),
            "pws_heldout_rate_bpm": pws_rate,
            "pws_heldout_absolute_error_bpm": abs(pws_rate - reference_rate),
            "selected_gtv_strength": selected["gtv"],
            "gtv_training_mae_bpm": next(
                item["training_mean_absolute_error_bpm"]
                for item in scores_by_method["gtv"] if item["strength"] == selected["gtv"]
            ),
            "gtv_heldout_rate_bpm": gtv_rate,
            "gtv_heldout_absolute_error_bpm": abs(gtv_rate - reference_rate),
            "gtv_converged_on_heldout_subject": gtv_converged,
        }
        folds.append(fold)
        selection_details.append({
            "heldout_subject": heldout_subject,
            "training_subjects": training_subjects,
            "selected_strengths": selected,
            "candidate_scores": scores_by_method,
        })

    summary: dict[str, Any] = {"heldout_subject_count": len(folds)}
    for name, error_field in (
        ("raw", "raw_absolute_error_bpm"),
        ("pws", "pws_heldout_absolute_error_bpm"),
        ("gtv", "gtv_heldout_absolute_error_bpm"),
    ):
        errors = [float(row[error_field]) for row in folds]
        summary[f"{name}_mean_absolute_error_bpm"] = statistics.mean(errors)
        summary[f"{name}_median_absolute_error_bpm"] = statistics.median(errors)
    summary["gtv_heldout_converged_count"] = sum(
        bool(row["gtv_converged_on_heldout_subject"]) for row in folds
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(folds[0]))
        writer.writeheader()
        writer.writerows(folds)
    report = {
        "purpose": "Exploratory leave-one-subject-out tuning of PWS and GTV strengths using inhalation-interval rate error on training subjects.",
        "code_mapping": {"0": "no_event", "1": "inhalation", "2": "exhalation"},
        "mapping_basis": "Project-owner-confirmed working interpretation; not an official dataset codebook.",
        "fold_protocol": "For each fold, select PWS and converged GTV strengths using only the other four subjects; report the held-out subject once.",
        "raw_output_grid": str(grid_path),
        "reference_rate_source": str(audit_path),
        "candidate_strengths": strengths,
        "summary": summary,
        "folds": folds,
        "selection_details": selection_details,
        "limitations": [
            "Only five subjects are available, so each training fold has four subjects and held-out estimates have high uncertainty.",
            "The reference mapping is project-owner-confirmed but is not author-documented.",
            "The face tracker and geometric ROIs are not validated against complete manual labels; known frame failures may influence signal rates.",
            "Full-recording spectral rates are compared with inhalation interval rates; this is not event-by-event validation.",
            "This exploratory tuning does not establish a final graph setting for deployment or publication.",
        ],
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "LOSO MAE: "
        f"raw {summary['raw_mean_absolute_error_bpm']:.2f}, "
        f"PWS {summary['pws_mean_absolute_error_bpm']:.2f}, "
        f"GTV {summary['gtv_mean_absolute_error_bpm']:.2f} BPM"
    )
    print(f"Wrote held-out graph-tuning results to {output_csv} and {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.grid.resolve(), args.audit.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

