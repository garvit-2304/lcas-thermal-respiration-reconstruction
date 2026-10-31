#!/usr/bin/env python3
"""Train/evaluate a TCN for sparse inhalation-event detection with subject-wise CV.

Because L-CAS supplies inhalation instants rather than a clean continuous
respiration waveform, this is an event-classification adaptation, not waveform
denoising. Results remain exploratory because the thermal ROIs are not manually
validated and the code map is a project-owner-confirmed working convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_SIGNALS = ROOT / "data" / "processed" / "graph_denoising_rosbag_raw" / "timestamped_raw_graph_signals.csv"
DEFAULT_ANNOTATIONS = ROOT / "data" / "processed" / "physiology_long.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "tcn_inhalation_event_detector"
SAMPLE_RATE_HZ = 20.0
RECORDING_HORIZON_S = 120.0
FEATURE_COLUMNS = ("raw_temperature_c", "pws_temperature_c", "gtv_temperature_c")
EVENT_TOLERANCE_S = 0.5
MIN_EVENT_DISTANCE_S = 1.2
THRESHOLD_CANDIDATES = tuple(round(float(value), 2) for value in np.arange(0.20, 0.81, 0.05))
PEAK_PROMINENCE_CANDIDATES = (0.10, 0.20, 0.30, 0.40, 0.50)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.activation(self.conv1(x)))
        y = self.dropout(self.activation(self.conv2(y)))
        return self.activation(y + self.skip(x))


class InhalationTCN(nn.Module):
    def __init__(self, in_channels: int = 3, channels: int = 16, dropout: float = 0.10):
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = in_channels
        for dilation in (1, 2, 4, 8, 16):
            blocks.append(ResidualTemporalBlock(current_channels, channels, dilation, dropout))
            current_channels = channels
        self.network = nn.Sequential(*blocks)
        self.output = nn.Conv1d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.network(x))


def parse_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def load_graph_signals(path: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"subject", "frame_index", "image_header_stamp_unix", "region", *FEATURE_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Graph signal file is missing columns: {sorted(missing)}")
        for row in reader:
            if row["region"].casefold() == "nose":
                groups[row["subject"].casefold()].append(row)

    output: dict[str, dict[str, Any]] = {}
    grid = np.arange(0.0, RECORDING_HORIZON_S, 1.0 / SAMPLE_RATE_HZ, dtype=np.float64)
    for subject, rows in groups.items():
        rows.sort(key=lambda row: int(row["frame_index"]))
        timestamps = np.array([float(row["image_header_stamp_unix"]) for row in rows])
        if len(timestamps) < 100 or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Insufficient or nonmonotonic nose samples for {subject}")
        origin = float(timestamps[0])
        elapsed = timestamps - origin
        features = []
        for column in FEATURE_COLUMNS:
            values = np.array([float(row[column]) for row in rows], dtype=np.float64)
            uniform = np.interp(grid, elapsed, values)
            center = float(np.median(uniform))
            scale = float(1.4826 * np.median(np.abs(uniform - center)))
            if scale <= 1e-8:
                scale = float(np.std(uniform)) or 1.0
            features.append(np.clip((uniform - center) / scale, -8.0, 8.0))
        output[subject] = {
            "origin_unix": origin,
            "time_s": grid,
            "features": np.stack(features).astype(np.float32),
            "duration_seconds": float(elapsed[-1]),
        }
    if not output:
        raise ValueError(f"No nose-region graph signals found in {path}")
    return output


def load_event_times(path: Path) -> dict[str, list[float]]:
    output: dict[str, list[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"recording", "stream", "timestamp", "value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Physiology table is missing columns: {sorted(missing)}")
        for row in reader:
            if row["stream"].casefold() != "respiration":
                continue
            raw_value = row["value"].strip()
            label = row.get("value_label", "").strip()
            if not label:
                label = {"0": "no_event", "1": "inhalation", "2": "exhalation"}.get(raw_value, "unmapped_code")
            if label == "inhalation":
                subject = row["recording"].rsplit("-", 1)[-1].casefold()
                output[subject].append(parse_timestamp(row["timestamp"]))
    return {subject: sorted(values) for subject, values in output.items()}


def make_targets(time_s: np.ndarray, event_times: list[float], origin_unix: float) -> np.ndarray:
    target = np.zeros_like(time_s, dtype=np.float32)
    event_offsets = [stamp - origin_unix for stamp in event_times]
    sigma_s, support_s = 0.10, 0.25
    for event in event_offsets:
        if not 0 <= event < RECORDING_HORIZON_S:
            continue
        distances = np.abs(time_s - event)
        local = np.exp(-0.5 * (distances / sigma_s) ** 2)
        local[distances > support_s] = 0.0
        target = np.maximum(target, local.astype(np.float32))
    return target


def match_events(predicted_s: list[float], reference_s: list[float]) -> dict[str, Any]:
    unmatched = set(range(len(reference_s)))
    matches: list[tuple[float, float]] = []
    for prediction in predicted_s:
        candidates = [index for index in unmatched if abs(reference_s[index] - prediction) <= EVENT_TOLERANCE_S]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda index: (abs(reference_s[index] - prediction), index))
        unmatched.remove(chosen)
        matches.append((prediction, reference_s[chosen]))
    tp = len(matches)
    fp = len(predicted_s) - tp
    fn = len(reference_s) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    timing_errors = [abs(prediction - reference) for prediction, reference in matches]
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_absolute_timing_error_seconds": statistics.median(timing_errors) if timing_errors else None,
        "matched_event_timing_errors_seconds": timing_errors,
    }


def train_fold(
    features: dict[str, dict[str, Any]],
    targets: dict[str, np.ndarray],
    train_subjects: list[str],
    epochs: int,
    seed: int,
) -> tuple[InhalationTCN, float, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    x = torch.tensor(np.stack([features[s]["features"] for s in train_subjects]), dtype=torch.float32)
    y = torch.tensor(np.stack([targets[s] for s in train_subjects])[:, None, :], dtype=torch.float32)
    positive_count = int(torch.count_nonzero(y >= 0.5).item())
    negative_count = int(torch.count_nonzero(y < 0.5).item())
    if positive_count == 0:
        raise ValueError("Training fold contains no positive inhalation labels")
    pos_weight = min(20.0, negative_count / positive_count)

    model = InhalationTCN().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("TCN training loss became non-finite")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        last_loss = float(loss.detach().item())
    return model, last_loss, pos_weight


def detect_events(
    probabilities: np.ndarray,
    probability_threshold: float,
) -> tuple[list[float], list[float]]:
    peaks, properties = find_peaks(
        probabilities,
        height=probability_threshold,
        distance=max(1, round(MIN_EVENT_DISTANCE_S * SAMPLE_RATE_HZ)),
        prominence=0.05,
    )
    return (peaks / SAMPLE_RATE_HZ).tolist(), properties["peak_heights"].astype(float).tolist()


def peak_baseline_events(
    features: dict[str, dict[str, Any]],
    subject: str,
    polarity: int,
    prominence: float,
) -> list[float]:
    sos = butter(
        4,
        [10.0 / 60.0, 40.0 / 60.0],
        btype="bandpass",
        fs=SAMPLE_RATE_HZ,
        output="sos",
    )
    filtered = sosfiltfilt(sos, features[subject]["features"][0])
    peaks, _ = find_peaks(
        filtered * polarity,
        distance=max(1, round(MIN_EVENT_DISTANCE_S * SAMPLE_RATE_HZ)),
        prominence=prominence,
    )
    return (peaks / SAMPLE_RATE_HZ).tolist()


def select_peak_baseline_parameters(
    features: dict[str, dict[str, Any]],
    reference_events: dict[str, list[float]],
    outer_training_subjects: list[str],
) -> tuple[int, float, list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    for polarity, name in ((1, "maxima"), (-1, "minima")):
        for prominence in PEAK_PROMINENCE_CANDIDATES:
            subject_f1: list[float] = []
            for subject in outer_training_subjects:
                origin = features[subject]["origin_unix"]
                reference = [
                    stamp - origin for stamp in reference_events[subject]
                    if 0 <= stamp - origin < RECORDING_HORIZON_S
                ]
                predicted = peak_baseline_events(features, subject, polarity, prominence)
                subject_f1.append(match_events(predicted, reference)["f1"])
            scores.append({
                "polarity": polarity,
                "polarity_name": name,
                "prominence": prominence,
                "inner_validation_macro_f1": statistics.mean(subject_f1),
            })
    best = min(
        scores,
        key=lambda item: (-item["inner_validation_macro_f1"], -item["prominence"], -item["polarity"]),
    )
    return int(best["polarity"]), float(best["prominence"]), scores


def select_threshold(
    features: dict[str, dict[str, Any]],
    targets: dict[str, np.ndarray],
    reference_events: dict[str, list[float]],
    outer_training_subjects: list[str],
    calibration_epochs: int,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    validation_probabilities: list[tuple[str, np.ndarray]] = []
    for inner_index, validation_subject in enumerate(outer_training_subjects):
        inner_training = [subject for subject in outer_training_subjects if subject != validation_subject]
        model, _, _ = train_fold(
            features, targets, inner_training, calibration_epochs, seed + inner_index + 100
        )
        model.eval()
        with torch.no_grad():
            values = torch.tensor(
                features[validation_subject]["features"][None, :, :], dtype=torch.float32
            )
            probabilities = torch.sigmoid(model(values))[0, 0].cpu().numpy()
        validation_probabilities.append((validation_subject, probabilities))

    scores: list[dict[str, Any]] = []
    for threshold in THRESHOLD_CANDIDATES:
        subject_f1: list[float] = []
        for subject, probabilities in validation_probabilities:
            predicted, _ = detect_events(probabilities, threshold)
            origin = features[subject]["origin_unix"]
            reference = [
                stamp - origin for stamp in reference_events[subject]
                if 0 <= stamp - origin < RECORDING_HORIZON_S
            ]
            subject_f1.append(match_events(predicted, reference)["f1"])
        scores.append({
            "threshold": threshold,
            "inner_validation_macro_f1": statistics.mean(subject_f1),
            "inner_validation_subject_count": len(subject_f1),
        })
    best = min(scores, key=lambda item: (-item["inner_validation_macro_f1"], -item["threshold"]))
    return float(best["threshold"]), scores


def run(
    graph_signals_path: Path,
    annotations_path: Path,
    output_dir: Path,
    epochs: int,
    calibration_epochs: int,
    overwrite: bool,
) -> None:
    if epochs < 1 or calibration_epochs < 1:
        raise ValueError("training and calibration epochs must be positive")
    output_csv = output_dir / "tcn_loso_metrics.csv"
    output_json = output_dir / "tcn_loso_report.json"
    existing = [path for path in (output_csv, output_json) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"TCN outputs already exist under {output_dir}; pass --overwrite")

    features = load_graph_signals(graph_signals_path)
    reference_events = load_event_times(annotations_path)
    if set(features) != set(reference_events):
        raise ValueError(f"Signal and label subjects differ: {sorted(features)} vs {sorted(reference_events)}")
    targets = {
        subject: make_targets(data["time_s"], reference_events[subject], data["origin_unix"])
        for subject, data in features.items()
    }

    metric_rows: list[dict[str, Any]] = []
    prediction_report: list[dict[str, Any]] = []
    sorted_subjects = sorted(features)
    for fold_index, heldout in enumerate(sorted_subjects):
        training = [subject for subject in sorted_subjects if subject != heldout]
        selected_threshold, threshold_scores = select_threshold(
            features,
            targets,
            reference_events,
            training,
            calibration_epochs,
            20261031 + fold_index * 1000,
        )
        peak_polarity, peak_prominence, peak_scores = select_peak_baseline_parameters(
            features, reference_events, training
        )
        model, final_loss, pos_weight = train_fold(
            features, targets, training, epochs, 20261031 + fold_index
        )
        model.eval()
        with torch.no_grad():
            heldout_x = torch.tensor(features[heldout]["features"][None, :, :], dtype=torch.float32)
            probabilities = torch.sigmoid(model(heldout_x))[0, 0].cpu().numpy()
        predicted_s, confidences = detect_events(probabilities, selected_threshold)
        origin = features[heldout]["origin_unix"]
        reference_s = [stamp - origin for stamp in reference_events[heldout] if 0 <= stamp - origin < RECORDING_HORIZON_S]
        overall = match_events(predicted_s, reference_s)
        peak_predictions = peak_baseline_events(
            features, heldout, peak_polarity, peak_prominence
        )
        peak_metrics = match_events(peak_predictions, reference_s)
        phase_metrics = {}
        for phase, start_s, end_s in (
            ("static_first_minute", 0.0, 60.0),
            ("head_movement_second_minute", 60.0, 120.0),
        ):
            phase_pred = [event for event in predicted_s if start_s <= event < end_s]
            phase_ref = [event for event in reference_s if start_s <= event < end_s]
            phase_metrics[phase] = match_events(phase_pred, phase_ref)

        metric_rows.append({
            "heldout_subject": heldout,
            "training_subjects": ";".join(training),
            "training_subject_count": len(training),
            "training_epochs": epochs,
            "training_final_loss": final_loss,
            "training_positive_weight": pos_weight,
            "selected_probability_threshold": selected_threshold,
            "reference_event_count": len(reference_s),
            "predicted_event_count": len(predicted_s),
            "true_positives": overall["true_positives"],
            "false_positives": overall["false_positives"],
            "false_negatives": overall["false_negatives"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "median_absolute_timing_error_seconds": overall["median_absolute_timing_error_seconds"],
            "static_f1": phase_metrics["static_first_minute"]["f1"],
            "movement_f1": phase_metrics["head_movement_second_minute"]["f1"],
            "peak_baseline_polarity": "maxima" if peak_polarity > 0 else "minima",
            "peak_baseline_prominence": peak_prominence,
            "peak_baseline_predicted_count": len(peak_predictions),
            "peak_baseline_true_positives": peak_metrics["true_positives"],
            "peak_baseline_false_positives": peak_metrics["false_positives"],
            "peak_baseline_false_negatives": peak_metrics["false_negatives"],
            "peak_baseline_precision": peak_metrics["precision"],
            "peak_baseline_recall": peak_metrics["recall"],
            "peak_baseline_f1": peak_metrics["f1"],
        })
        prediction_report.append({
            "heldout_subject": heldout,
            "selected_probability_threshold": selected_threshold,
            "inner_threshold_scores": threshold_scores,
            "selected_peak_baseline": {
                "polarity": peak_polarity,
                "polarity_name": "maxima" if peak_polarity > 0 else "minima",
                "prominence": peak_prominence,
                "training_selection_scores": peak_scores,
            },
            "reference_events_seconds": reference_s,
            "predicted_events_seconds": predicted_s,
            "predicted_event_confidences": confidences,
            "overall_metrics": overall,
            "peak_baseline_predicted_events_seconds": peak_predictions,
            "peak_baseline_metrics": peak_metrics,
            "phase_metrics": phase_metrics,
        })
        print(
            f"{heldout}: {overall['true_positives']} TP, {overall['false_positives']} FP, "
            f"{overall['false_negatives']} FN; TCN F1={overall['f1']:.3f}, "
            f"peak baseline F1={peak_metrics['f1']:.3f}"
        )

    total_tp = sum(int(row["true_positives"]) for row in metric_rows)
    total_fp = sum(int(row["false_positives"]) for row in metric_rows)
    total_fn = sum(int(row["false_negatives"]) for row in metric_rows)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    summary = {
        "subject_count": len(metric_rows),
        "macro_precision": statistics.mean(float(row["precision"]) for row in metric_rows),
        "macro_recall": statistics.mean(float(row["recall"]) for row in metric_rows),
        "macro_f1": statistics.mean(float(row["f1"]) for row in metric_rows),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
    }
    peak_tp = sum(int(row["peak_baseline_true_positives"]) for row in metric_rows)
    peak_fp = sum(int(row["peak_baseline_false_positives"]) for row in metric_rows)
    peak_fn = sum(int(row["peak_baseline_false_negatives"]) for row in metric_rows)
    peak_precision = peak_tp / (peak_tp + peak_fp) if peak_tp + peak_fp else 0.0
    peak_recall = peak_tp / (peak_tp + peak_fn) if peak_tp + peak_fn else 0.0
    summary.update({
        "peak_baseline_macro_precision": statistics.mean(
            float(row["peak_baseline_precision"]) for row in metric_rows
        ),
        "peak_baseline_macro_recall": statistics.mean(
            float(row["peak_baseline_recall"]) for row in metric_rows
        ),
        "peak_baseline_macro_f1": statistics.mean(
            float(row["peak_baseline_f1"]) for row in metric_rows
        ),
        "peak_baseline_micro_precision": peak_precision,
        "peak_baseline_micro_recall": peak_recall,
        "peak_baseline_micro_f1": (
            2 * peak_precision * peak_recall / (peak_precision + peak_recall)
            if peak_precision + peak_recall else 0.0
        ),
        "peak_baseline_true_positives": peak_tp,
        "peak_baseline_false_positives": peak_fp,
        "peak_baseline_false_negatives": peak_fn,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    report = {
        "purpose": "Subject-wise evaluation of a TCN inhalation-event detector trained from the sparse event labels supplied with L-CAS.",
        "target_definition": "Gaussian-shaped positive target centered on each mapped inhalation timestamp (sigma 0.10 s, support +/-0.25 s); all other samples are negative.",
        "annotation_mapping": {"0": "no_event", "1": "inhalation", "2": "exhalation"},
        "mapping_basis": "Project-owner-confirmed working interpretation; not an author-published numeric codebook.",
        "model": {
            "architecture": "Five residual temporal blocks; 16 channels; kernel size 3; dilations 1, 2, 4, 8, 16; GELU; dropout 0.10; sigmoid event probability head.",
            "input_features": list(FEATURE_COLUMNS),
            "feature_region": "nose",
            "graph_feature_generation": {
                "pws_strength": 0.05,
                "gtv_strength": 0.002,
                "source_report": "data/processed/graph_denoising_rosbag_raw/timestamped_raw_graph_report.json",
            },
            "resampling_rate_hz": SAMPLE_RATE_HZ,
            "input_normalization": "Per-recording robust centering and scaling from that recording's signal values; no label information used.",
            "training": {"loss": "positive-weighted binary cross entropy", "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "outer_epochs": epochs, "inner_calibration_epochs": calibration_epochs, "fold_split": "outer leave one subject out; held-out labels are not used for training or threshold selection"},
            "event_postprocessing": {"probability_threshold": "selected only within each outer training fold using inner leave-one-subject-out validation", "candidate_thresholds": list(THRESHOLD_CANDIDATES), "minimum_peak_distance_seconds": MIN_EVENT_DISTANCE_S, "prominence": 0.05, "matching_tolerance_seconds": EVENT_TOLERANCE_S},
        },
        "nonlearned_comparator": {
            "method": "Fourth-order 10-40 BPM bandpass on the raw nose signal followed by peak or valley detection.",
            "minimum_peak_distance_seconds": MIN_EVENT_DISTANCE_S,
            "prominence_candidates": list(PEAK_PROMINENCE_CANDIDATES),
            "polarity_and_prominence_selection": "selected using only outer-fold training subjects; held-out subject is excluded",
        },
        "summary": summary,
        "heldout_subject_metrics": metric_rows,
        "heldout_predictions": prediction_report,
        "limitations": [
            "This event detector is an adaptation to available point annotations; it is not the continuous-signal TCN denoiser described by the workflow papers.",
            "Only five subjects and about two hundred inhalation events are available, so subject-wise estimates have high uncertainty.",
            "The project code mapping is user-confirmed but not documented by the dataset authors.",
            "The input nose ROIs come from a tracker with known sample failures and have not been validated against full manual face/region labels.",
            "Probability threshold is selected by inner subject-wise validation; the 1.2-second minimum peak distance is fixed before evaluation.",
            "The model detects inhalation events; it does not output a continuous respiration waveform or a calibrated confidence score.",
        ],
        "inputs": {"graph_signals": str(graph_signals_path), "annotations": str(annotations_path)},
        "outputs": {"metrics_csv": output_csv.name, "report_json": output_json.name},
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Macro F1={summary['macro_f1']:.3f}; micro F1={summary['micro_f1']:.3f}")
    print(f"Wrote TCN LOSO metrics to {output_csv} and {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-signals", type=Path, default=DEFAULT_GRAPH_SIGNALS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--calibration-epochs", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.graph_signals.resolve(),
        args.annotations.resolve(),
        args.output_dir.resolve(),
        args.epochs,
        args.calibration_epochs,
        args.overwrite,
    )


if __name__ == "__main__":
    main()

