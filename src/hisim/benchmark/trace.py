"""Load token traces without decoding or applying chat templates."""

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TraceRequest:
    input_ids: list[int]
    output_length: int
    timestamp: float
    request_id: str


def load_trace(path, limit, timestamp_unit="s"):
    units = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if limit <= 0:
        raise ValueError("num-prompts must be positive")
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            ids = row["input_ids"]
            output = row["output_length"]
            timestamp = float(row.get("created_time", row.get("timestamp")))
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(token) is not int or token < 0 for token in ids)
            ):
                raise ValueError(f"line {line_number}: invalid input_ids")
            if row.get("input_length", len(ids)) != len(ids):
                raise ValueError(f"line {line_number}: input_length mismatch")
            if type(output) is not int or output <= 0 or not math.isfinite(timestamp):
                raise ValueError(f"line {line_number}: invalid length/timestamp")
            rows.append(
                TraceRequest(
                    ids,
                    output,
                    timestamp * units[timestamp_unit],
                    str(row.get("request_id", line_number)),
                )
            )
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(f"requested {limit} rows, found {len(rows)}")
    return sorted(rows, key=lambda row: row.timestamp)


async def replay_requests(
    requests, scale, *, clock=time.monotonic, sleep=asyncio.sleep
):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("timestamp-scale must be finite and positive")
    if not requests:
        return
    origin = requests[0].hisim_timestamp
    started = clock()
    span = (requests[-1].hisim_timestamp - origin) * scale
    rate = (len(requests) - 1) / span if span else float("inf")
    for request in requests:
        target = started + (request.hisim_timestamp - origin) * scale
        delay = target - clock()
        if delay > 0:
            await sleep(delay)
        request.hisim_send_lag_seconds = max(0.0, clock() - target)
        yield request, rate
