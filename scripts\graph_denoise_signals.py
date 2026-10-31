#!/usr/bin/env python3
"""Apply sparse fixed-graph PWS and reweighted-GTV filters to L-CAS ROI traces.

This is an image-intensity adaptation, not a full reproduction of the learned
Mahalanobis-metric MPT/MGT formulation from the reference paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNALS = ROOT / "data" / "processed" / "roi_signals.csv"
DEFAULT_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "graph_denoising"
REGIONS = ("forehead", "nose", "mouth", "left_cheek", "right_cheek")
MANAGED_OUTPUTS = {"graph_denoised_signals.csv", "graph_denoising_report.json"}


def load_subject_data(
    signals_csv: Path,
    tracks_csv: Path,
    image_width: int,
    image_height: int,
) -> dict[str, dict[str, np.ndarray]]:
    signal_rows: dict[tuple[str, int], dict[str, str]] = {}
    signal_columns = {f"{region}_mean_intensity" for region in REGIONS}
    with signals_csv.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "frame_index", *signal_columns}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ROI CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            key = (row["subject"], int(row["frame_index"]))
            if key in signal_rows:
                raise ValueError(f"Duplicate ROI signal row for {key}")
            signal_rows[key] = row

    track_rows: dict[tuple[str, int], dict[str, str]] = {}
    coordinate_columns = {
        f"{region}_{suffix}"
        for region in REGIONS
        for suffix in ("x", "y", "width", "height")
    }
    with tracks_csv.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "frame_index", "valid_roi", *coordinate_columns}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Face-track CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            key = (row["subject"], int(row["frame_index"]))
            if key in track_rows:
                raise ValueError(f"Duplicate face-track row for {key}")
            track_rows[key] = row

    if set(signal_rows) != set(track_rows):
        signal_only = len(set(signal_rows) - set(track_rows))
        track_only = len(set(track_rows) - set(signal_rows))
        raise ValueError(
            "ROI signals and face tracks must contain the same subject/frame keys "
            f"(signals-only={signal_only}, tracks-only={track_only})."
        )

    grouped: dict[str, list[tuple[int, dict[str, str], dict[str, str]]]] = defaultdict(list)
    for (subject, frame_index), signal_row in signal_rows.items():
        grouped[subject].append((frame_index, signal_row, track_rows[(subject, frame_index)]))

    result: dict[str, dict[str, np.ndarray]] = {}
    for subject, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        indices = np.array([row[0] for row in rows], dtype=np.int64)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Duplicate frame indices found for {subject}")
        if np.any(np.diff(indices) != 1):
            raise ValueError(f"Non-contiguous frame indices found for {subject}")

        values = np.empty((len(rows), len(REGIONS)), dtype=np.float64)
        positions = np.empty((len(rows), len(REGIONS), 2), dtype=np.float64)
        for time_index, (_, signal_row, track_row) in enumerate(rows):
            if track_row["valid_roi"].strip().lower() != "true":
                raise ValueError(f"Invalid candidate ROI for {subject} frame {indices[time_index]}")
            for region_index, region in enumerate(REGIONS):
                try:
                    value = float(signal_row[f"{region}_mean_intensity"])
                    x = float(track_row[f"{region}_x"])
                    y = float(track_row[f"{region}_y"])
                    width = float(track_row[f"{region}_width"])
                    height = float(track_row[f"{region}_height"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {region} signal or box for {subject} frame {indices[time_index]}"
                    ) from exc
                if not np.isfinite([value, x, y, width, height]).all() or width <= 0 or height <= 0:
                    raise ValueError(
                        f"Non-finite value or non-positive box for {subject} frame {indices[time_index]}"
                    )
                values[time_index, region_index] = value
                positions[time_index, region_index] = (
                    (x + width / 2.0) / image_width,
                    (y + height / 2.0) / image_height,
                )
        result[subject] = {"frame_indices": indices, "values": values, "positions": positions}
    if not result:
        raise ValueError("No subject rows found in ROI and face-track inputs")
    return result


def make_edges(
    values: np.ndarray,
    positions: np.ndarray,
    spatial_scale: float,
    intensity_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    frame_count, region_count = values.shape
    left: list[int] = []
    right: list[int] = []
    weights: list[float] = []
    spatial_edges = 0
    temporal_edges = 0

    def add_edge(time_a: int, region_a: int, time_b: int, region_b: int) -> None:
        nonlocal spatial_edges, temporal_edges
        node_a = time_a * region_count + region_a
        node_b = time_b * region_count + region_b
        position_delta = positions[time_a, region_a] - positions[time_b, region_b]
        intensity_delta = (values[time_a, region_a] - values[time_b, region_b]) / 255.0
        distance = (
            float(np.dot(position_delta, position_delta)) / (spatial_scale**2)
            + float(intensity_delta**2) / (intensity_scale**2)
        )
        weight = float(np.exp(-min(distance, 50.0)))
        if weight > 1e-12:
            left.append(node_a)
            right.append(node_b)
            weights.append(weight)
        if time_a == time_b:
            spatial_edges += 1
        else:
            temporal_edges += 1

    # Within each frame, connect all candidate regions using a Gaussian feature kernel.
    for time_index in range(frame_count):
        for region_a in range(region_count):
            for region_b in range(region_a + 1, region_count):
                add_edge(time_index, region_a, time_index, region_b)

    # Across time, connect each region to its immediately adjacent frame.
    for time_index in range(frame_count - 1):
        for region_index in range(region_count):
            add_edge(time_index, region_index, time_index + 1, region_index)

    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        spatial_edges,
        temporal_edges,
    )


def solve_graph_filter(
    observations: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    edge_weights: np.ndarray,
    strength: float,
) -> np.ndarray:
    node_count = observations.size
    degree = np.bincount(
        np.concatenate((left, right)),
        weights=np.concatenate((edge_weights, edge_weights)),
        minlength=node_count,
    )
    adjacency = sparse.coo_matrix(
        (
            np.concatenate((edge_weights, edge_weights)),
            (np.concatenate((left, right)), np.concatenate((right, left))),
        ),
        shape=(node_count, node_count),
    ).tocsr()
    laplacian = sparse.diags(degree, format="csr") - adjacency
    system = sparse.eye(node_count, format="csr") + strength * laplacian
    filtered = spsolve(system, observations.reshape(-1))
    if not np.isfinite(filtered).all():
        raise ArithmeticError("Sparse graph solve returned non-finite values")
    return np.asarray(filtered).reshape(observations.shape)


def run_subject(
    values: np.ndarray,
    positions: np.ndarray,
    spatial_scale: float,
    intensity_scale: float,
    mpt_strength: float,
    mgt_strength: float,
    mgt_epsilon: float,
    mgt_iterations: int,
    convergence_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    left, right, base_weights, spatial_edges, temporal_edges = make_edges(
        values, positions, spatial_scale, intensity_scale
    )
    if len(base_weights) == 0:
        raise ValueError("Graph construction produced no positive-weight edges")

    normalized = values / 255.0
    mpt = solve_graph_filter(normalized, left, right, base_weights, mpt_strength)

    # Iteratively reweighted quadratic surrogate of graph total variation (GTV).
    mgt = normalized.copy()
    iteration_count = 0
    converged = False
    for iteration in range(mgt_iterations):
        edge_difference = np.abs(mgt.reshape(-1)[left] - mgt.reshape(-1)[right])
        reweighted = base_weights / np.maximum(edge_difference, mgt_epsilon)
        updated = solve_graph_filter(normalized, left, right, reweighted, mgt_strength)
        relative_change = float(
            np.linalg.norm(updated - mgt) / max(np.linalg.norm(mgt), np.finfo(float).eps)
        )
        mgt = updated
        iteration_count = iteration + 1
        if relative_change <= convergence_tolerance:
            converged = True
            break

    summary = {
        "frame_count": int(values.shape[0]),
        "region_count": int(values.shape[1]),
        "graph_node_count": int(values.size),
        "candidate_spatial_edges": int(spatial_edges),
        "candidate_temporal_edges": int(temporal_edges),
        "positive_weight_edge_count": int(len(base_weights)),
        "base_edge_weight_min": float(np.min(base_weights)),
        "base_edge_weight_median": float(np.median(base_weights)),
        "base_edge_weight_max": float(np.max(base_weights)),
        "mgt_iterations_run": iteration_count,
        "mgt_converged": converged,
        "mgt_relative_change_last": relative_change if iteration_count else None,
    }
    return mpt * 255.0, mgt * 255.0, summary


def write_outputs(
    subject_data: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    parameters: dict[str, object],
    overwrite: bool,
) -> None:
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir() if path.is_file()}
        unexpected = existing - MANAGED_OUTPUTS
        if unexpected:
            raise FileExistsError(
                f"Output directory contains files not managed by this script: {sorted(unexpected)}"
            )
        if existing and not overwrite:
            raise FileExistsError(
                f"Graph outputs already exist under {output_dir}; pass --overwrite to refresh them."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "method": "sparse fixed-feature graph Laplacian and IRLS-GTV adaptation",
        "reference": "Yang et al., IEEE Internet of Things Journal, 2022, DOI 10.1109/JIOT.2022.3150147",
        "graph": {
            "nodes": "one node per frame and each of five candidate facial regions",
            "candidate_edges": [
                "all unordered pairs of regions within the same frame",
                "each region to itself in the immediately adjacent frame",
            ],
            "weight": "exp(-normalized_position_squared/spatial_scale^2 - normalized_intensity_squared/intensity_scale^2)",
            "feature_metric": "fixed isotropic Gaussian scales; paper's learned Mahalanobis metric is not implemented",
            "values": "8-bit ROI mean image intensity normalized to [0, 1] for solving",
        },
        "filters": {
            "mpt_fixed_graph": "solve (I + strength*L) v = y with the combinatorial graph Laplacian",
            "mgt_irls_fixed_graph": "iteratively reweighted quadratic GTV surrogate with edge weight w/max(abs(delta v), epsilon)",
            "paper_trace_constraint_and_metric_learning": "not implemented",
        },
        "parameters": parameters,
        "limitations": [
            "This is a sparse fixed-graph adaptation, not an exact MPT/MGT reproduction; the paper's learned Mahalanobis metric and trace constraint are omitted.",
            "The five face regions are geometric candidates and have not been validated against landmarks or manual masks.",
            "Inputs are 8-bit image intensities, not calibrated thermal temperatures.",
            "Temporal edges join adjacent frame indices; this PNG-based graph does not weight edges by the recovered irregular image-timestamp intervals.",
            "Graph smoothing output is not evaluated against respiration annotations and is not evidence of improved respiration estimation.",
        ],
        "subjects": {},
    }

    output_path = output_dir / "graph_denoised_signals.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        fields = [
            "subject",
            "frame_index",
            "region",
            "raw_mean_intensity",
            "mpt_fixed_graph_intensity",
            "mgt_irls_fixed_graph_intensity",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for subject, data in sorted(subject_data.items()):
            values = data["values"]
            positions = data["positions"]
            mpt, mgt, summary = run_subject(
                values,
                positions,
                float(parameters["spatial_scale"]),
                float(parameters["intensity_scale"]),
                float(parameters["mpt_strength"]),
                float(parameters["mgt_strength"]),
                float(parameters["mgt_epsilon"]),
                int(parameters["mgt_iterations"]),
                float(parameters["convergence_tolerance"]),
            )
            report["subjects"][subject] = summary
            for time_index, frame_index in enumerate(data["frame_indices"]):
                for region_index, region in enumerate(REGIONS):
                    writer.writerow(
                        {
                            "subject": subject,
                            "frame_index": int(frame_index),
                            "region": region,
                            "raw_mean_intensity": f"{values[time_index, region_index]:.8f}",
                            "mpt_fixed_graph_intensity": f"{mpt[time_index, region_index]:.8f}",
                            "mgt_irls_fixed_graph_intensity": f"{mgt[time_index, region_index]:.8f}",
                        }
                    )

    with (output_dir / "graph_denoising_report.json").open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
        file_obj.write("\n")
    print(f"Wrote sparse graph outputs for {len(subject_data)} subjects under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-csv", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--tracks-csv", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-width", type=int, default=382)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--spatial-scale", type=float, default=0.15)
    parser.add_argument("--intensity-scale", type=float, default=0.20)
    parser.add_argument("--mpt-strength", type=float, default=0.05)
    parser.add_argument("--mgt-strength", type=float, default=0.002)
    parser.add_argument("--mgt-epsilon", type=float, default=1.0 / 255.0)
    parser.add_argument("--mgt-iterations", type=int, default=10)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    positive_parameters = {
        "image-width": args.image_width,
        "image-height": args.image_height,
        "spatial-scale": args.spatial_scale,
        "intensity-scale": args.intensity_scale,
        "mpt-strength": args.mpt_strength,
        "mgt-strength": args.mgt_strength,
        "mgt-epsilon": args.mgt_epsilon,
        "convergence-tolerance": args.convergence_tolerance,
    }
    for name, value in positive_parameters.items():
        if value <= 0:
            parser.error(f"--{name} must be greater than zero")
    if args.mgt_iterations < 1:
        parser.error("--mgt-iterations must be at least 1")

    params = {
        "image_width_pixels": args.image_width,
        "image_height_pixels": args.image_height,
        "spatial_scale": args.spatial_scale,
        "intensity_scale": args.intensity_scale,
        "mpt_strength": args.mpt_strength,
        "mgt_strength": args.mgt_strength,
        "mgt_epsilon": args.mgt_epsilon,
        "mgt_iterations": args.mgt_iterations,
        "convergence_tolerance": args.convergence_tolerance,
    }
    subjects = load_subject_data(
        args.signals_csv.resolve(),
        args.tracks_csv.resolve(),
        args.image_width,
        args.image_height,
    )
    write_outputs(subjects, args.output_dir.resolve(), params, args.overwrite)


if __name__ == "__main__":
    main()

