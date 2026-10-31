#!/usr/bin/env python3
"""Create a face-box and geometric ROI baseline for the extracted L-CAS PNGs.

The detector uses OpenCV Haar cascades and pyramidal Lucas-Kanade optical flow.
The resulting boxes and facial regions are candidates for inspection, not
validated anatomical segmentations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "original_dataset"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
CASCADE_DIR = Path(cv2.data.haarcascades)
CASCADE_NAMES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
)
ROI_RATIOS: dict[str, tuple[float, float, float, float]] = {
    "forehead": (0.20, 0.05, 0.80, 0.32),
    "nose": (0.34, 0.35, 0.66, 0.68),
    "mouth": (0.25, 0.66, 0.75, 0.89),
    "left_cheek": (0.04, 0.44, 0.39, 0.78),
    "right_cheek": (0.61, 0.44, 0.96, 0.78),
}


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, w, h = box
    left = max(0, min(width - 1, int(round(x))))
    top = max(0, min(height - 1, int(round(y))))
    right = max(left + 1, min(width, int(round(x + w))))
    bottom = max(top + 1, min(height, int(round(y + h))))
    result = (left, top, right - left, bottom - top)
    if result[2] < 1 or result[3] < 1:
        return None
    return result


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def valid_box(box: tuple[int, int, int, int], width: int, height: int) -> bool:
    x, y, w, h = box
    center_x = x + w / 2
    center_y = y + h / 2
    aspect = w / h if h else 0.0
    return (
        w >= 35
        and h >= 35
        and w <= width * 0.75
        and h <= height * 0.90
        and 0.62 <= aspect <= 1.55
        and width * 0.10 <= center_x <= width * 0.90
        and height * 0.06 <= center_y <= height * 0.94
    )


def detector_candidates(
    gray: np.ndarray,
    classifiers: list[tuple[str, cv2.CascadeClassifier]],
    min_neighbors: int = 4,
    scale_factor: float = 1.10,
) -> list[tuple[tuple[int, int, int, int], str]]:
    height, width = gray.shape[:2]
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    for name, classifier in classifiers:
        boxes = classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(35, 35),
            maxSize=(int(width * 0.70), int(height * 0.90)),
        )
        for raw_box in boxes:
            box = tuple(int(value) for value in raw_box)
            if valid_box(box, width, height):
                candidates.append((box, name))

    # The two frontal cascades often return overlapping boxes for one face.
    unique: list[tuple[tuple[int, int, int, int], str]] = []
    for candidate in sorted(candidates, key=lambda item: item[0][2] * item[0][3], reverse=True):
        if all(box_iou(candidate[0], present[0]) < 0.72 for present in unique):
            unique.append(candidate)
    return unique


def choose_detection(
    candidates: list[tuple[tuple[int, int, int, int], str]],
    prediction: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], str] | None:
    if not candidates:
        return None
    if prediction is not None:
        best = max(candidates, key=lambda item: box_iou(item[0], prediction))
        if box_iou(best[0], prediction) >= 0.08:
            return best

    def centrality(item: tuple[tuple[int, int, int, int], str]) -> float:
        x, y, w, h = item[0]
        dx = (x + w / 2 - width / 2) / width
        dy = (y + h / 2 - height / 2) / height
        return (w * h) / (width * height) - 0.35 * math.hypot(dx, dy)

    return max(candidates, key=centrality)


def optical_flow_box(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int] | None, int, int, float]:
    height, width = previous_gray.shape[:2]
    x, y, w, h = box
    mask = np.zeros_like(previous_gray)
    inner_x1 = max(0, x + int(w * 0.12))
    inner_x2 = min(width, x + int(w * 0.88))
    inner_y1 = max(0, y + int(h * 0.12))
    inner_y2 = min(height, y + int(h * 0.90))
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        return None, 0, 0, 0.0
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = 255
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=4,
        mask=mask,
        blockSize=5,
    )
    if points is None or len(points) < 6:
        return None, 0 if points is None else len(points), 0, 0.0

    moved, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if moved is None or status is None:
        return None, len(points), 0, 0.0

    status_mask = status.reshape(-1).astype(bool)
    error_mask = errors.reshape(-1) < 30 if errors is not None else status_mask
    good = status_mask & error_mask
    source_points = points.reshape(-1, 2)[good]
    target_points = moved.reshape(-1, 2)[good]
    if len(source_points) < 6:
        return None, len(points), len(source_points), 0.0

    transform, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=1000,
        confidence=0.99,
    )
    if transform is None or inlier_mask is None:
        return None, len(points), len(source_points), 0.0

    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / len(source_points)
    flow_quality = inlier_ratio * min(1.0, inlier_count / 15.0)
    corners = np.array(
        [[x, y, 1.0], [x + w, y, 1.0], [x, y + h, 1.0], [x + w, y + h, 1.0]],
        dtype=np.float64,
    )
    transformed = corners @ transform.T
    min_x, min_y = transformed.min(axis=0)
    max_x, max_y = transformed.max(axis=0)
    prediction = clamp_box((min_x, min_y, max_x - min_x, max_y - min_y), width, height)
    if prediction is None or not valid_box(prediction, width, height):
        prediction = None
    return prediction, len(points), inlier_count, flow_quality


def region_boxes(face_box: tuple[int, int, int, int], width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    x, y, w, h = face_box
    regions: dict[str, tuple[int, int, int, int]] = {}
    for name, (x1, y1, x2, y2) in ROI_RATIOS.items():
        region = clamp_box(
            (x + x1 * w, y + y1 * h, (x2 - x1) * w, (y2 - y1) * h),
            width,
            height,
        )
        if region is not None:
            regions[name] = region
    return regions


def draw_preview(
    gray: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    status: str,
    subject: str,
    frame_index: int,
) -> np.ndarray:
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = {
        "forehead": (255, 255, 0),
        "nose": (0, 0, 255),
        "mouth": (0, 255, 255),
        "left_cheek": (255, 0, 0),
        "right_cheek": (255, 0, 255),
    }
    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for name, region in region_boxes(face_box, gray.shape[1], gray.shape[0]).items():
            rx, ry, rw, rh = region
            cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), colors[name], 1)
    cv2.putText(
        preview,
        f"{subject} frame {frame_index} | {status}",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    legend = np.full((48, gray.shape[1], 3), 18, dtype=np.uint8)
    for index, (name, color) in enumerate(colors.items()):
        column = index % 3
        row = index // 3
        origin_x = 6 + column * 126
        origin_y = 16 + row * 19
        cv2.rectangle(legend, (origin_x, origin_y - 9), (origin_x + 8, origin_y - 1), color, -1)
        cv2.putText(
            legend,
            name,
            (origin_x + 12, origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((preview, legend))


def subject_frames(subject_dir: Path) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    for path in subject_dir.glob("frame_*.png"):
        try:
            frame_index = int(path.stem.split("_")[-1])
        except ValueError as error:
            raise ValueError(f"Unexpected frame filename: {path.name}") from error
        frames.append((frame_index, path))
    return sorted(frames)


def refresh_previews(input_dir: Path, output_dir: Path) -> None:
    tracks_path = output_dir / "face_tracks.csv"
    report_path = output_dir / "face_tracking_report.json"
    if not tracks_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Existing face_tracks.csv and face_tracking_report.json are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = {
        item["subject"]: set(item["preview_frames"])
        for item in report["subjects"]
    }
    preview_root = output_dir / "face_previews"
    with tracks_path.open("r", newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            subject = row["subject"]
            frame_index = int(row["frame_index"])
            if frame_index not in selected.get(subject, set()):
                continue
            image_path = input_dir / subject / f"frame_{frame_index:06d}.png"
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Could not decode preview frame: {image_path}")
            face_box = None
            if row["valid_roi"].lower() == "true":
                face_box = tuple(
                    int(row[field])
                    for field in ("face_x", "face_y", "face_width", "face_height")
                )
            preview = draw_preview(
                gray,
                face_box,
                row["tracking_status"],
                subject,
                frame_index,
            )
            preview_path = preview_root / subject / f"frame_{frame_index:06d}_regions.png"
            if not cv2.imwrite(str(preview_path), preview):
                raise OSError(f"Could not write preview image: {preview_path}")
    print(f"Refreshed preview overlays under {preview_root}")


def detect_and_track(
    input_dir: Path,
    output_dir: Path,
    detect_stride: int,
    min_neighbors: int,
    scale_factor: float,
) -> None:
    if detect_stride < 1:
        raise ValueError("--detect-stride must be at least 1")
    if min_neighbors < 1:
        raise ValueError("--min-neighbors must be at least 1")
    if scale_factor <= 1.0:
        raise ValueError("--scale-factor must be greater than 1.0")

    tracks_path = output_dir / "face_tracks.csv"
    report_path = output_dir / "face_tracking_report.json"
    preview_root = output_dir / "face_previews"
    if tracks_path.exists() or report_path.exists() or preview_root.exists():
        raise FileExistsError(
            "Face-tracking outputs already exist under "
            f"{output_dir}; choose another --output-dir to preserve them."
        )
    classifiers: list[tuple[str, cv2.CascadeClassifier]] = []
    for name in CASCADE_NAMES:
        cascade_path = CASCADE_DIR / name
        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            raise RuntimeError(f"Could not load face cascade: {cascade_path}")
        classifiers.append((name.removeprefix("haarcascade_").removesuffix(".xml"), classifier))

    subject_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(f"No subject frame folders found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True)

    fields = [
        "subject",
        "frame_index",
        "image_path",
        "face_x",
        "face_y",
        "face_width",
        "face_height",
        "tracking_status",
        "valid_roi",
        "cascade_name",
        "flow_points",
        "flow_inliers",
        "flow_quality",
    ]
    for region_name in ROI_RATIOS:
        fields.extend(
            [f"{region_name}_x", f"{region_name}_y", f"{region_name}_width", f"{region_name}_height"]
        )

    subject_reports: list[dict[str, Any]] = []
    total_frames = 0
    total_valid = 0
    with tracks_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()

        for subject_dir in subject_dirs:
            subject = subject_dir.name
            frames = subject_frames(subject_dir)
            if not frames:
                continue
            if [index for index, _ in frames] != list(range(1, len(frames) + 1)):
                raise ValueError(f"{subject}: frame numbers must be contiguous from 1")

            preview_indices = {
                1,
                max(1, round(len(frames) * 0.50)),
                max(1, round(len(frames) * 0.75)),
                len(frames),
            }
            subject_preview_dir = preview_root / subject
            subject_preview_dir.mkdir(parents=True)
            previous_gray: np.ndarray | None = None
            previous_box: tuple[int, int, int, int] | None = None
            status_counts: Counter[str] = Counter()
            flow_qualities: list[float] = []
            valid_count = 0
            detector_count = 0

            for frame_index, image_path in frames:
                gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError(f"Could not decode frame: {image_path}")
                height, width = gray.shape[:2]
                prediction: tuple[int, int, int, int] | None = None
                flow_points = 0
                flow_inliers = 0
                flow_quality = 0.0
                if previous_gray is not None and previous_box is not None:
                    prediction, flow_points, flow_inliers, flow_quality = optical_flow_box(
                        previous_gray, gray, previous_box
                    )

                detector_due = (
                    previous_box is None
                    or (frame_index - 1) % detect_stride == 0
                    or prediction is None
                    or flow_quality < 0.15
                )
                selected_box: tuple[int, int, int, int] | None = None
                cascade_name = ""
                if detector_due:
                    candidates = detector_candidates(gray, classifiers, min_neighbors, scale_factor)
                    detection = choose_detection(candidates, prediction, width, height)
                    if detection is not None:
                        detection_box, cascade_name = detection
                        if prediction is None or box_iou(detection_box, prediction) >= 0.08:
                            selected_box = detection_box
                            tracking_status = "detected"
                            detector_count += 1
                        elif flow_quality >= 0.15:
                            selected_box = prediction
                            tracking_status = "tracked_detector_disagreed"
                            cascade_name = ""
                        else:
                            selected_box = detection_box
                            tracking_status = "detected_reacquisition"
                            detector_count += 1
                    elif prediction is not None and flow_quality >= 0.15:
                        selected_box = prediction
                        tracking_status = "tracked_no_detection"
                    else:
                        tracking_status = "missing"
                elif prediction is not None and flow_quality >= 0.15:
                    selected_box = prediction
                    tracking_status = "tracked"
                else:
                    tracking_status = "missing"

                if selected_box is not None and not valid_box(selected_box, width, height):
                    selected_box = None
                    tracking_status = "missing_invalid_box"

                valid_roi = selected_box is not None
                if valid_roi:
                    valid_count += 1
                    status_counts[tracking_status] += 1
                    if tracking_status.startswith("tracked"):
                        flow_qualities.append(flow_quality)
                    region_data = region_boxes(selected_box, width, height)
                    x, y, w, h = selected_box
                else:
                    status_counts[tracking_status] += 1
                    region_data = {}
                    x = y = w = h = ""

                row: dict[str, Any] = {
                    "subject": subject,
                    "frame_index": frame_index,
                    "image_path": image_path.relative_to(ROOT).as_posix(),
                    "face_x": x,
                    "face_y": y,
                    "face_width": w,
                    "face_height": h,
                    "tracking_status": tracking_status,
                    "valid_roi": valid_roi,
                    "cascade_name": cascade_name,
                    "flow_points": flow_points,
                    "flow_inliers": flow_inliers,
                    "flow_quality": f"{flow_quality:.6f}",
                }
                for region_name in ROI_RATIOS:
                    region = region_data.get(region_name)
                    if region is None:
                        row.update(
                            {
                                f"{region_name}_x": "",
                                f"{region_name}_y": "",
                                f"{region_name}_width": "",
                                f"{region_name}_height": "",
                            }
                        )
                    else:
                        rx, ry, rw, rh = region
                        row.update(
                            {
                                f"{region_name}_x": rx,
                                f"{region_name}_y": ry,
                                f"{region_name}_width": rw,
                                f"{region_name}_height": rh,
                            }
                        )
                writer.writerow(row)

                if frame_index in preview_indices:
                    preview = draw_preview(gray, selected_box, tracking_status, subject, frame_index)
                    preview_path = subject_preview_dir / f"frame_{frame_index:06d}_regions.png"
                    if not cv2.imwrite(str(preview_path), preview):
                        raise OSError(f"Could not write preview image: {preview_path}")

                previous_gray = gray
                previous_box = selected_box

                if frame_index % 1000 == 0:
                    print(f"{subject}: processed {frame_index}/{len(frames)} frames")

            total_frames += len(frames)
            total_valid += valid_count
            subject_reports.append(
                {
                    "subject": subject,
                    "frame_count": len(frames),
                    "valid_roi_frames": valid_count,
                    "valid_roi_fraction": round(valid_count / len(frames), 6),
                    "detector_reacquisitions": detector_count,
                    "tracking_status_counts": dict(status_counts),
                    "median_optical_flow_quality_on_tracked_frames": round(
                        float(np.median(flow_qualities)), 6
                    )
                    if flow_qualities
                    else None,
                    "preview_frames": sorted(preview_indices),
                }
            )

    report = {
        "method": "OpenCV Haar frontal-face cascades with Lucas-Kanade affine optical-flow tracking",
        "cascade_files": list(CASCADE_NAMES),
        "redetection_interval_frames": detect_stride,
        "cascade_min_neighbors": min_neighbors,
        "cascade_scale_factor": scale_factor,
        "total_frames": total_frames,
        "valid_roi_frames": total_valid,
        "valid_roi_fraction": round(total_valid / total_frames, 6) if total_frames else 0.0,
        "roi_definition": ROI_RATIOS,
        "limitations": [
            "Haar detections and optical-flow tracks are a practical baseline, not a validated face tracker.",
            "The flow quality score is a heuristic consistency value, not a calibrated probability.",
            "Geometric ROIs are candidate masks without landmark or manual-mask validation.",
            "Frame timestamps are not available; output is indexed by image sequence only.",
            "Image values are 8-bit grayscale intensities, not calibrated temperatures.",
        ],
        "subjects": subject_reports,
        "track_csv": tracks_path.relative_to(ROOT).as_posix()
        if tracks_path.is_relative_to(ROOT)
        else str(tracks_path),
        "preview_directory": preview_root.relative_to(ROOT).as_posix()
        if preview_root.is_relative_to(ROOT)
        else str(preview_root),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {tracks_path}")
    print(f"Wrote {report_path}")
    print(f"Valid face ROI boxes: {total_valid}/{total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detect-stride", type=int, default=15)
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=4,
        help="OpenCV Haar cascade threshold; lower values return more candidate detections.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.10,
        help="OpenCV Haar scale step; values closer to 1.0 search more scales and take longer.",
    )
    parser.add_argument(
        "--refresh-previews",
        action="store_true",
        help="Redraw existing preview overlays without rerunning detection/tracking.",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.refresh_previews:
        refresh_previews(input_dir, output_dir)
    else:
        detect_and_track(
            input_dir,
            output_dir,
            args.detect_stride,
            args.min_neighbors,
            args.scale_factor,
        )


if __name__ == "__main__":
    main()

