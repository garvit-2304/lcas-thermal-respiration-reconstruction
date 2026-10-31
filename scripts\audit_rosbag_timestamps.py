#!/usr/bin/env python3
"""Export ROS1 bag image timestamps without decoding or saving image payloads."""

from __future__ import annotations

import argparse
import bz2
import csv
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAG_DIR = ROOT / "ros"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "rosbag_timestamps"
MAGIC = b"#ROSBAG V2.0\n"


def parse_header(raw: bytes) -> dict[str, bytes]:
    fields: dict[str, bytes] = {}
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            raise ValueError("Truncated ROS bag header field length")
        field_length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        end = offset + field_length
        if field_length == 0 or end > len(raw):
            raise ValueError("Invalid ROS bag header field length")
        field = raw[offset:end]
        offset = end
        key, separator, value = field.partition(b"=")
        if not separator:
            raise ValueError(f"Malformed ROS bag header field: {field!r}")
        fields[key.decode("ascii")] = value
    return fields


def read_record_header(file_obj) -> tuple[dict[str, bytes], int, int]:
    length_bytes = file_obj.read(4)
    if not length_bytes:
        raise EOFError
    if len(length_bytes) != 4:
        raise ValueError("Truncated ROS bag record header length")
    header_length = struct.unpack("<I", length_bytes)[0]
    raw_header = file_obj.read(header_length)
    data_length_bytes = file_obj.read(4)
    if len(raw_header) != header_length or len(data_length_bytes) != 4:
        raise ValueError("Truncated ROS bag record")
    data_length = struct.unpack("<I", data_length_bytes)[0]
    return parse_header(raw_header), file_obj.tell(), data_length


def uint_field(header: dict[str, bytes], key: str, fmt: str) -> int:
    raw = header.get(key)
    size = struct.calcsize(fmt)
    if raw is None or len(raw) != size:
        raise ValueError(f"Missing or malformed {key!r} field")
    return int(struct.unpack(fmt, raw)[0])


def text_field(header: dict[str, bytes], key: str, default: str = "") -> str:
    return header.get(key, b"").decode("utf-8", errors="replace") or default


def time_field(header: dict[str, bytes], key: str) -> float:
    raw = header.get(key)
    if raw is None or len(raw) != 8:
        raise ValueError(f"Missing or malformed time field {key!r}")
    seconds, nanoseconds = struct.unpack("<II", raw)
    return seconds + nanoseconds / 1_000_000_000.0


def op_code(header: dict[str, bytes]) -> int:
    raw = header.get("op")
    if raw is None or len(raw) != 1:
        raise ValueError("Missing or malformed ROS bag operation code")
    return raw[0]


def read_bag_header(file_obj) -> dict[str, bytes]:
    if file_obj.read(len(MAGIC)) != MAGIC:
        raise ValueError("File does not have ROS bag v2 magic")
    header, data_offset, data_length = read_record_header(file_obj)
    if op_code(header) != 0x03:
        raise ValueError("First ROS bag record is not a bag header")
    # ROS1 reserves a padding area here so index_pos can be rewritten in place.
    file_obj.seek(data_offset + data_length)
    return header


def image_header_stamp(payload: bytes, message_type: str) -> float | None:
    if message_type not in {"sensor_msgs/Image", "sensor_msgs/CompressedImage"}:
        return None
    if len(payload) < 12:
        return None
    _sequence, seconds, nanoseconds = struct.unpack_from("<III", payload, 0)
    if seconds == 0 and nanoseconds == 0:
        return None
    return seconds + nanoseconds / 1_000_000_000.0


def parse_chunk_records(
    raw_chunk: bytes,
    connections: dict[int, dict[str, str]],
    records: list[dict[str, object]],
) -> None:
    offset = 0
    while offset < len(raw_chunk):
        if offset + 4 > len(raw_chunk):
            raise ValueError("Truncated record header inside ROS bag chunk")
        header_length = struct.unpack_from("<I", raw_chunk, offset)[0]
        offset += 4
        header_end = offset + header_length
        if header_end + 4 > len(raw_chunk):
            raise ValueError("Invalid record header length inside ROS bag chunk")
        header = parse_header(raw_chunk[offset:header_end])
        data_length = struct.unpack_from("<I", raw_chunk, header_end)[0]
        data_start = header_end + 4
        data_end = data_start + data_length
        if data_end > len(raw_chunk):
            raise ValueError("Invalid record data length inside ROS bag chunk")
        operation = op_code(header)

        if operation == 0x07:
            connection_id = uint_field(header, "conn", "<I")
            connection_header = parse_header(raw_chunk[data_start:data_end])
            connections[connection_id] = {
                "topic": text_field(header, "topic", "unknown"),
                "type": text_field(connection_header, "type", "unknown"),
                "md5sum": text_field(connection_header, "md5sum"),
                "callerid": text_field(connection_header, "callerid"),
            }
        elif operation == 0x02:
            connection_id = uint_field(header, "conn", "<I")
            records.append(
                {
                    "connection_id": connection_id,
                    "bag_timestamp_unix": time_field(header, "time"),
                    # Only the first ROS Header fields are needed for timestamps.
                    # Retaining the full chunk here would keep the entire bag in RAM.
                    "payload_prefix": raw_chunk[data_start:min(data_start + 12, data_end)],
                }
            )
        offset = data_end


def decompress_chunk(compression: str, payload: bytes) -> bytes:
    if compression == "none":
        return payload
    if compression == "bz2":
        return bz2.decompress(payload)
    if compression == "lz4":
        try:
            import lz4.frame  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("This bag uses lz4 compression; install the Python lz4 package to read it") from exc
        return lz4.frame.decompress(payload)
    raise ValueError(f"Unsupported ROS bag chunk compression: {compression!r}")


def audit_bag(path: Path) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    subject = path.stem.replace("_onlythermal", "").split("-")[-1].casefold()
    connections: dict[int, dict[str, str]] = {}
    records: list[dict[str, object]] = []
    compression_counts: Counter[str] = Counter()
    bag_header: dict[str, bytes]

    with path.open("rb") as file_obj:
        bag_header = read_bag_header(file_obj)
        while True:
            try:
                header, data_offset, data_length = read_record_header(file_obj)
            except EOFError:
                break
            operation = op_code(header)
            if operation == 0x05:
                compression = text_field(header, "compression")
                compression_counts[compression] += 1
                raw_payload = file_obj.read(data_length)
                if len(raw_payload) != data_length:
                    raise ValueError(f"Truncated chunk payload in {path.name}")
                chunk_data = decompress_chunk(compression, raw_payload)
                parse_chunk_records(chunk_data, connections, records)
            elif operation == 0x07:
                connection_id = uint_field(header, "conn", "<I")
                file_obj.seek(data_offset)
                connection_header = parse_header(file_obj.read(data_length))
                connections[connection_id] = {
                    "topic": text_field(header, "topic", "unknown"),
                    "type": text_field(connection_header, "type", "unknown"),
                    "md5sum": text_field(connection_header, "md5sum"),
                    "callerid": text_field(connection_header, "callerid"),
                }
            elif operation == 0x02:
                connection_id = uint_field(header, "conn", "<I")
                payload_prefix = file_obj.read(min(12, data_length))
                records.append(
                    {
                        "connection_id": connection_id,
                        "bag_timestamp_unix": time_field(header, "time"),
                        "payload_prefix": payload_prefix,
                    }
                )
            file_obj.seek(data_offset + data_length)

    topic_times: dict[int, list[tuple[float, float | None]]] = defaultdict(list)
    for record in records:
        connection_id = int(record["connection_id"])
        metadata = connections.get(connection_id)
        if metadata is None:
            raise ValueError(f"Message refers to missing connection {connection_id} in {path.name}")
        payload = record["payload_prefix"]
        stamp = image_header_stamp(payload, metadata["type"])
        topic_times[connection_id].append((float(record["bag_timestamp_unix"]), stamp))

    topic_summaries: list[dict[str, object]] = []
    timestamp_rows: list[dict[str, object]] = []
    for connection_id, metadata in sorted(connections.items()):
        messages = topic_times.get(connection_id, [])
        bag_times = [message[0] for message in messages]
        deltas = [right - left for left, right in zip(bag_times, bag_times[1:])]
        topic_summaries.append(
            {
                "connection_id": connection_id,
                **metadata,
                "message_count": len(messages),
                "first_bag_time_unix": bag_times[0] if bag_times else None,
                "last_bag_time_unix": bag_times[-1] if bag_times else None,
                "duration_seconds": bag_times[-1] - bag_times[0] if len(bag_times) > 1 else None,
                "median_interval_seconds": sorted(deltas)[len(deltas) // 2] if deltas else None,
                "non_increasing_time_count": sum(delta <= 0 for delta in deltas),
                "message_header_stamp_count": sum(stamp is not None for _, stamp in messages),
            }
        )
        for message_index, (bag_time, header_stamp) in enumerate(messages):
            timestamp_rows.append(
                {
                    "subject": subject,
                    "connection_id": connection_id,
                    "topic": metadata["topic"],
                    "message_type": metadata["type"],
                    "message_index": message_index,
                    "bag_timestamp_unix": f"{bag_time:.9f}",
                    "image_header_stamp_unix": "" if header_stamp is None else f"{header_stamp:.9f}",
                    "header_minus_bag_seconds": "" if header_stamp is None else f"{header_stamp - bag_time:.9f}",
                    "delta_seconds": "" if message_index == 0 else f"{bag_time - bag_times[message_index - 1]:.9f}",
                }
            )

    report = {
        "bag_file": str(path),
        "file_size_bytes": path.stat().st_size,
        "declared_chunk_count": uint_field(bag_header, "chunk_count", "<I"),
        "declared_connection_count": uint_field(bag_header, "conn_count", "<I"),
        "chunk_compression_counts": dict(compression_counts),
        "timestamp_source": "ROS bag message record times; image payloads were not decoded or written",
        "topics": topic_summaries,
    }
    return subject, report, timestamp_rows


def run(bag_dir: Path, output_dir: Path, overwrite: bool) -> None:
    bags = sorted(bag_dir.glob("*.bag"))
    if not bags:
        raise FileNotFoundError(f"No .bag files found under {bag_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    all_rows: list[dict[str, object]] = []
    for bag in bags:
        subject, report, rows = audit_bag(bag)
        report["subject"] = subject
        reports.append(report)
        all_rows.extend(rows)
        print(f"Read {sum(int(topic['message_count']) for topic in report['topics'])} messages: {bag.name}")

    timestamps_path = output_dir / "rosbag_message_timestamps.csv"
    with timestamps_path.open("w", newline="", encoding="utf-8") as file_obj:
        fields = [
            "subject",
            "connection_id",
            "topic",
            "message_type",
            "message_index",
            "bag_timestamp_unix",
            "image_header_stamp_unix",
            "header_minus_bag_seconds",
            "delta_seconds",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    report_path = output_dir / "rosbag_timestamp_audit.json"
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump({"bag_count": len(reports), "bags": reports}, file_obj, indent=2)
        file_obj.write("\n")
    print(f"Wrote bag timestamps to {timestamps_path} and {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", type=Path, default=DEFAULT_BAG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.bag_dir.resolve(), args.output_dir.resolve(), args.overwrite)


if __name__ == "__main__":
    main()

