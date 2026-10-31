#!/usr/bin/env python3
"""Extract original mono16 thermal frames and timestamps from ROS1 bags."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from audit_rosbag_timestamps import (
    DEFAULT_BAG_DIR,
    audit_bag,
    decompress_chunk,
    op_code,
    parse_header,
    read_bag_header,
    read_record_header,
    text_field,
    uint_field,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_OUTPUT = ROOT / "data" / "processed" / "raw_thermal"
EXPECTED_ENCODING = "mono16"
TEMPERATURE_CONVERSION = "(mono16_count - 1000) / 10.0 degrees Celsius"
TEMPERATURE_CONVERSION_SOURCE = "https://lcas.lincoln.ac.uk/wp/research/data-sets-software/lcas-thermal-physiological-monitoring-dataset/"


def iter_chunk_records(raw_chunk: bytes) -> Iterator[tuple[dict[str, bytes], memoryview]]:
    view = memoryview(raw_chunk)
    offset = 0
    while offset < len(view):
        if offset + 4 > len(view):
            raise ValueError("Truncated record header inside ROS bag chunk")
        header_length = struct.unpack_from("<I", view, offset)[0]
        offset += 4
        header_end = offset + header_length
        if header_end + 4 > len(view):
            raise ValueError("Invalid record header length inside ROS bag chunk")
        header = parse_header(bytes(view[offset:header_end]))
        data_length = struct.unpack_from("<I", view, header_end)[0]
        data_start = header_end + 4
        data_end = data_start + data_length
        if data_end > len(view):
            raise ValueError("Invalid record data length inside ROS bag chunk")
        yield header, view[data_start:data_end]
        offset = data_end


def parse_image_payload(payload: memoryview) -> tuple[int, int, str, bool, int, memoryview]:
    """Return dimensions, encoding, byte order, row step, and pixel bytes."""
    if len(payload) < 16:
        raise ValueError("Truncated sensor_msgs/Image header")
    frame_id_length = struct.unpack_from("<I", payload, 12)[0]
    offset = 16 + frame_id_length
    if offset + 8 > len(payload):
        raise ValueError("Truncated sensor_msgs/Image dimensions")
    height, width = struct.unpack_from("<II", payload, offset)
    offset += 8
    if offset + 4 > len(payload):
        raise ValueError("Truncated sensor_msgs/Image encoding length")
    encoding_length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    if offset + encoding_length + 9 > len(payload):
        raise ValueError("Truncated sensor_msgs/Image encoding or layout")
    encoding = bytes(payload[offset : offset + encoding_length]).decode("ascii")
    offset += encoding_length
    big_endian = bool(payload[offset])
    step = struct.unpack_from("<I", payload, offset + 1)[0]
    pixel_length = struct.unpack_from("<I", payload, offset + 5)[0]
    pixel_start = offset + 9
    pixel_end = pixel_start + pixel_length
    if pixel_end != len(payload):
        raise ValueError(
            f"sensor_msgs/Image data length is inconsistent: {pixel_length} declared, "
            f"{len(payload) - pixel_start} available"
        )
    if height == 0 or width == 0 or step == 0:
        raise ValueError("sensor_msgs/Image contains invalid dimensions or row step")
    return height, width, encoding, big_endian, step, payload[pixel_start:pixel_end]


def extract_one(path: Path, output_dir: Path) -> dict[str, object]:
    subject = path.stem.replace("_onlythermal", "").split("-")[-1].casefold()
    _, audit_report, _ = audit_bag(path)
    image_topics = [
        topic
        for topic in audit_report["topics"]
        if topic["type"] == "sensor_msgs/Image"
    ]
    if len(image_topics) != 1:
        raise ValueError(
            f"Expected one sensor_msgs/Image connection in {path.name}, found {len(image_topics)}"
        )
    topic_info = image_topics[0]
    expected_count = int(topic_info["message_count"])
    output_path = output_dir / f"{subject}_thermal_raw.h5"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    connection_metadata: dict[int, dict[str, str]] = {}
    frame_index = 0
    first_header_stamp: float | None = None
    last_header_stamp: float | None = None
    with h5py.File(output_path, "w") as output:
        frames = output.create_dataset(
            "frames",
            shape=(expected_count, 288, 382),
            dtype=np.dtype("<u2"),
            chunks=(1, 288, 382),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )
        bag_times = output.create_dataset("bag_timestamp_unix", shape=(expected_count,), dtype="<f8")
        header_times = output.create_dataset(
            "image_header_stamp_unix", shape=(expected_count,), dtype="<f8"
        )

        with path.open("rb") as bag_file:
            read_bag_header(bag_file)
            while True:
                try:
                    record_header, data_offset, data_length = read_record_header(bag_file)
                except EOFError:
                    break
                operation = op_code(record_header)
                if operation == 0x05:
                    compression = text_field(record_header, "compression")
                    compressed = bag_file.read(data_length)
                    if len(compressed) != data_length:
                        raise ValueError(f"Truncated chunk payload in {path.name}")
                    chunk = decompress_chunk(compression, compressed)
                    for inner_header, payload in iter_chunk_records(chunk):
                        inner_operation = op_code(inner_header)
                        if inner_operation == 0x07:
                            connection_id = uint_field(inner_header, "conn", "<I")
                            connection_header = parse_header(bytes(payload))
                            connection_metadata[connection_id] = {
                                "topic": text_field(inner_header, "topic", "unknown"),
                                "type": text_field(connection_header, "type", "unknown"),
                            }
                        elif inner_operation == 0x02:
                            connection_id = uint_field(inner_header, "conn", "<I")
                            metadata = connection_metadata.get(connection_id)
                            if metadata is None:
                                raise ValueError(
                                    f"Message uses unknown connection {connection_id} in {path.name}"
                                )
                            if metadata["topic"] != topic_info["topic"]:
                                continue
                            if metadata["type"] != "sensor_msgs/Image":
                                raise ValueError(f"Unexpected message type: {metadata['type']}")
                            if frame_index >= expected_count:
                                raise ValueError(f"More thermal frames than indexed in {path.name}")

                            height, width, encoding, big_endian, step, pixel_view = parse_image_payload(payload)
                            if (height, width) != (288, 382):
                                raise ValueError(
                                    f"Unexpected frame dimensions in {path.name}: {width}x{height}"
                                )
                            if encoding != EXPECTED_ENCODING:
                                raise ValueError(
                                    f"Expected {EXPECTED_ENCODING} in {path.name}, found {encoding}"
                                )
                            if step != width * 2 or len(pixel_view) != height * step:
                                raise ValueError("mono16 image has unexpected row stride or size")

                            byte_order = ">u2" if big_endian else "<u2"
                            matrix = np.frombuffer(pixel_view, dtype=np.dtype(byte_order)).reshape(height, width)
                            if big_endian:
                                matrix = matrix.astype(np.uint16, copy=False)
                            frames[frame_index] = matrix
                            bag_times[frame_index] = struct.unpack_from(
                                "<II", inner_header["time"]
                            )[0] + struct.unpack_from("<II", inner_header["time"])[1] / 1e9
                            image_stamp = struct.unpack_from("<III", payload, 0)
                            header_stamp = image_stamp[1] + image_stamp[2] / 1e9
                            header_times[frame_index] = header_stamp
                            first_header_stamp = header_stamp if first_header_stamp is None else first_header_stamp
                            last_header_stamp = header_stamp
                            frame_index += 1
                elif operation == 0x07:
                    connection_id = uint_field(record_header, "conn", "<I")
                    bag_file.seek(data_offset)
                    connection_header = parse_header(bag_file.read(data_length))
                    connection_metadata[connection_id] = {
                        "topic": text_field(record_header, "topic", "unknown"),
                        "type": text_field(connection_header, "type", "unknown"),
                    }
                bag_file.seek(data_offset + data_length)

        if frame_index != expected_count:
            raise ValueError(
                f"Extracted {frame_index} frames from {path.name}, expected {expected_count}"
            )
        output.attrs["subject"] = subject
        output.attrs["source_bag"] = path.name
        output.attrs["topic"] = topic_info["topic"]
        output.attrs["message_type"] = "sensor_msgs/Image"
        output.attrs["encoding"] = EXPECTED_ENCODING
        output.attrs["frame_shape"] = [288, 382]
        output.attrs["frame_count"] = frame_index
        output.attrs["dtype"] = "uint16"
        output.attrs["byte_order"] = "little-endian"
        output.attrs["pixel_semantics"] = "raw mono16 sensor counts; frames dataset remains unmodified"
        output.attrs["temperature_conversion"] = TEMPERATURE_CONVERSION
        output.attrs["temperature_conversion_source"] = TEMPERATURE_CONVERSION_SOURCE
        output.attrs["frame_index_origin"] = 1
        output.attrs["timestamp_note"] = "ROS image header stamp and bag record time stored per frame"

    file_size = output_path.stat().st_size
    return {
        "subject": subject,
        "source_bag": str(path),
        "output_hdf5": str(output_path),
        "topic": topic_info["topic"],
        "encoding": EXPECTED_ENCODING,
        "shape": [expected_count, 288, 382],
        "dtype": "uint16",
        "temperature_conversion": TEMPERATURE_CONVERSION,
        "temperature_conversion_source": TEMPERATURE_CONVERSION_SOURCE,
        "stored_bytes": file_size,
        "raw_uncompressed_bytes": expected_count * 288 * 382 * 2,
        "first_image_header_stamp_unix": first_header_stamp,
        "last_image_header_stamp_unix": last_header_stamp,
        "compression": "HDF5 gzip level 4 with byte shuffle and Fletcher32 checksum",
    }


def run(bag_dir: Path, output_dir: Path) -> None:
    bags = sorted(bag_dir.glob("*.bag"))
    if not bags:
        raise FileNotFoundError(f"No .bag files found under {bag_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"{p.stem.replace('_onlythermal', '').split('-')[-1].casefold()}_thermal_raw.h5" for p in bags]
    collisions = [str(path) for path in existing if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite existing raw-frame outputs: {collisions}")

    reports = []
    for bag in bags:
        report = extract_one(bag, output_dir)
        reports.append(report)
        print(
            f"Extracted {report['shape'][0]} mono16 frames for {report['subject']}: "
            f"{report['stored_bytes']:,} bytes"
        )

    report_path = output_dir / "raw_thermal_extraction_report.json"
    report_path.write_text(
        json.dumps(
            {
                "pixel_semantics": "Original mono16 count arrays are preserved without modification.",
                "temperature_conversion": TEMPERATURE_CONVERSION,
                "temperature_conversion_source": TEMPERATURE_CONVERSION_SOURCE,
                "recordings": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote extraction report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, default=DEFAULT_BAG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FRAME_OUTPUT)
    args = parser.parse_args()
    run(args.bag_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

