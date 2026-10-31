#!/usr/bin/env python3
"""Create a human-reviewable face/ROI reference set from the L-CAS frames.

The script intentionally does not prefill detections. Boxes are drawn from the
thermal image by the reviewer, and each edit is saved immediately to JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS = ROOT / "data" / "processed" / "face_tracks.csv"
DEFAULT_OUTPUT = ROOT / "data" / "annotations" / "roi_reference_review.json"
REGIONS = (
    "face",
    "forehead",
    "nose",
    "mouth",
    "left_cheek",
    "right_cheek",
)
COLORS = {
    "face": (0, 255, 0),
    "forehead": (255, 255, 0),
    "nose": (0, 0, 255),
    "mouth": (0, 255, 255),
    "left_cheek": (255, 0, 0),
    "right_cheek": (255, 0, 255),
}
MAX_DISPLAY = (1120, 800)


def selected_frames(tracks_path: Path) -> list[dict[str, Any]]:
    report_path = tracks_path.parent / "face_tracking_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Tracker report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selections = {item["subject"] for item in report.get("subjects", [])}
    preview_frames = {
        item["subject"]: set(int(value) for value in item.get("preview_frames", []))
        for item in report.get("subjects", [])
    }
    records: list[dict[str, Any]] = []
    with tracks_path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        for row in csv.DictReader(file_obj):
            subject = row["subject"]
            frame_index = int(row["frame_index"])
            if subject in selections and frame_index in preview_frames.get(subject, set()):
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = ROOT / image_path
                records.append(
                    {
                        "subject": subject,
                        "frame_index": frame_index,
                        "image_path": image_path.resolve(),
                    }
                )
    records.sort(key=lambda item: (str(item["subject"]).casefold(), item["frame_index"]))
    if not records:
        raise ValueError("No preview frames found in the track report")
    return records


def empty_annotation(record: dict[str, Any], annotator: str) -> dict[str, Any]:
    return {
        "subject": record["subject"],
        "frame_index": record["frame_index"],
        "image_path": Path(record["image_path"]).relative_to(ROOT).as_posix(),
        "annotator": annotator,
        "review_status": "unreviewed",
        "reviewed_regions": [],
        "not_visible_regions": [],
        "boxes_xywh": {},
        "notes": "",
    }


class ReferenceAnnotator:
    def __init__(self, records: list[dict[str, Any]], output_path: Path, annotator: str):
        self.records = records
        self.output_path = output_path
        self.annotator = annotator
        self.position = 0
        self.selected_region = 0
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.scale = 1.0
        self.display_size = (0, 0)
        self.image_size = (0, 0)
        self.annotations: dict[tuple[str, int], dict[str, Any]] = {}
        self.created_utc = datetime.now(timezone.utc).isoformat()

        if output_path.exists():
            data = json.loads(output_path.read_text(encoding="utf-8"))
            existing_annotator = data.get("metadata", {}).get("annotator")
            self.created_utc = data.get("metadata", {}).get("created_utc", self.created_utc)
            if existing_annotator and existing_annotator != annotator:
                raise ValueError(
                    f"Existing file belongs to {existing_annotator!r}; use another --output path "
                    "to keep annotators separate."
                )
            for item in data.get("annotations", []):
                key = (str(item["subject"]).casefold(), int(item["frame_index"]))
                self.annotations[key] = item

        self.window_name = "L-CAS visual ROI reference annotation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse)

    def _record(self) -> dict[str, Any]:
        return self.records[self.position]

    def _key(self) -> tuple[str, int]:
        record = self._record()
        return (str(record["subject"]).casefold(), int(record["frame_index"]))

    def _annotation(self) -> dict[str, Any]:
        key = self._key()
        if key not in self.annotations:
            self.annotations[key] = empty_annotation(self._record(), self.annotator)
        return self.annotations[key]

    def _image_point(self, point: tuple[int, int]) -> tuple[int, int]:
        x = round(point[0] / self.scale)
        y = round(point[1] / self.scale)
        return (max(0, min(self.image_size[0] - 1, x)), max(0, min(self.image_size[1] - 1, y)))

    def _mouse(self, event: int, x: int, y: int, flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = (x, y)
            x1, y1 = self._image_point(self.drag_start)
            x2, y2 = self._image_point(self.drag_end)
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            if right > left and bottom > top:
                region = REGIONS[self.selected_region]
                self._annotation()["boxes_xywh"][region] = {
                    "x": left,
                    "y": top,
                    "width": right - left,
                    "height": bottom - top,
                }
                self._mark_reviewed(region, not_visible=False)
                self._save()
            self.drag_start = None
            self.drag_end = None

    def _mark_reviewed(self, region: str, not_visible: bool) -> None:
        annotation = self._annotation()
        reviewed = set(annotation.get("reviewed_regions", []))
        unavailable = set(annotation.get("not_visible_regions", []))
        reviewed.add(region)
        unavailable.discard(region)
        if not_visible:
            unavailable.add(region)
            annotation.get("boxes_xywh", {}).pop(region, None)
        annotation["reviewed_regions"] = [name for name in REGIONS if name in reviewed]
        annotation["not_visible_regions"] = [name for name in REGIONS if name in unavailable]
        annotation["review_status"] = (
            "complete" if set(REGIONS).issubset(reviewed) else "partial"
        )

    def _save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": {
                "dataset": "L-CAS Thermal Physiological Monitoring Dataset",
                "annotator": self.annotator,
                "annotation_role": "single-annotator visual reference; requires independent review before being treated as ground truth",
                "created_utc": self.created_utc,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "coordinate_format": "pixel x,y,width,height in original 382x288 frame coordinates",
                "region_convention": {
                    "face": "visible facial skin from forehead to chin; exclude hair, ears, and neck",
                    "forehead": "visible forehead skin above the brows and below the hairline",
                    "nose": "nose bridge and nostril area; exclude eyes and cheeks",
                    "mouth": "lips and immediate perioral area",
                    "left_cheek": "cheek on the left side of the displayed image",
                    "right_cheek": "cheek on the right side of the displayed image",
                    "not_visible": "mark a region unavailable when occluded or outside the frame",
                },
            },
            "annotations": sorted(
                self.annotations.values(),
                key=lambda item: (str(item["subject"]).casefold(), int(item["frame_index"])),
            ),
        }
        temp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self.output_path)

    def _render(self) -> np.ndarray:
        record = self._record()
        image = cv2.imread(str(record["image_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {record['image_path']}")
        self.image_size = (image.shape[1], image.shape[0])
        self.scale = min(MAX_DISPLAY[0] / image.shape[1], MAX_DISPLAY[1] / image.shape[0])
        display = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            (round(image.shape[1] * self.scale), round(image.shape[0] * self.scale)),
            interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_NEAREST,
        )
        self.display_size = (display.shape[1], display.shape[0])
        annotation = self._annotation()
        for region, box in annotation.get("boxes_xywh", {}).items():
            if region not in COLORS:
                continue
            x, y = round(box["x"] * self.scale), round(box["y"] * self.scale)
            x2 = round((box["x"] + box["width"]) * self.scale)
            y2 = round((box["y"] + box["height"]) * self.scale)
            cv2.rectangle(display, (x, y), (x2, y2), COLORS[region], 2)
            cv2.putText(display, region, (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[region], 1)

        if self.drag_start is not None and self.drag_end is not None:
            color = COLORS[REGIONS[self.selected_region]]
            cv2.rectangle(display, self.drag_start, self.drag_end, color, 1)

        key = (str(record["subject"]).casefold(), int(record["frame_index"]))
        saved = self.annotations.get(key)
        status = saved.get("review_status", "unreviewed") if saved else "unreviewed"
        header = (
            f"{record['subject']} frame {record['frame_index']} | "
            f"region: {REGIONS[self.selected_region]} | {status} | "
            f"{self.position + 1}/{len(self.records)}"
        )
        cv2.putText(display, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "0-5 select region | drag box | x not visible | u clear | Enter save/next | Backspace prev | s skip | q quit", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        return display

    def run(self) -> None:
        try:
            while 0 <= self.position < len(self.records):
                cv2.imshow(self.window_name, self._render())
                key = cv2.waitKey(30) & 0xFF
                if key in [ord(str(index)) for index in range(len(REGIONS))]:
                    self.selected_region = key - ord("0")
                elif key in (13, 10):
                    self._save()
                    self.position = min(self.position + 1, len(self.records) - 1)
                elif key in (8, 127):
                    self._save()
                    self.position = max(0, self.position - 1)
                elif key == ord("s"):
                    self.position = min(self.position + 1, len(self.records) - 1)
                elif key == ord("x"):
                    self._mark_reviewed(REGIONS[self.selected_region], not_visible=True)
                    self._save()
                elif key == ord("u"):
                    region = REGIONS[self.selected_region]
                    annotation = self._annotation()
                    annotation.get("boxes_xywh", {}).pop(region, None)
                    annotation["reviewed_regions"] = [
                        name for name in annotation.get("reviewed_regions", []) if name != region
                    ]
                    annotation["not_visible_regions"] = [
                        name for name in annotation.get("not_visible_regions", []) if name != region
                    ]
                    annotation["review_status"] = "partial" if annotation["reviewed_regions"] else "unreviewed"
                    self._save()
                elif key == ord("q"):
                    self._save()
                    break
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self._save()
                    break
        finally:
            self._save()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotator", required=True, help="Name recorded in annotation provenance.")
    args = parser.parse_args()
    tracks_path = args.tracks.resolve()
    output_path = args.output.resolve()
    annotator = args.annotator.strip()
    if not annotator:
        raise ValueError("--annotator must not be blank")
    records = selected_frames(tracks_path)
    ReferenceAnnotator(records, output_path, annotator).run()
    print(f"Saved visual reference annotations to {output_path}")


if __name__ == "__main__":
    main()

