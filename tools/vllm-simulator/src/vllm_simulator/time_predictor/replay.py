# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle lookup predictor — replays real GPU iter_latency from a pre-built table.

Replay is a diagnostic predictor for separating latency-prediction error from
scheduler, cache, and simulator behavior. A replay table maps a JSON-encoded,
sorted list of `[extend_input_length, prefix_length]` pairs to measured
forward and sample_tokens latencies in seconds.

Example predictor settings under `additional_config.simulator`:
    "predictor": {
        "name": "replay",
        "database_path": "/path/to/replay_table.json",
        "miss_strategy": "knn_work",  # "zero", "knn", or "knn_work"
        "miss_knn_k": 3,              # KNN k for "knn" strategy
        "miss_fallback_seconds": 0.0  # used only when strategy=="zero"
    }

Build the table from collector exports with
`ReplayTimePredictor.build_table_from_schedule_batch`.
"""

import gzip
import json
import math
import os
from logging import getLogger

import numpy as np

from vllm_simulator.time_predictor.base import PerRankTimePredictor, ScheduleBatch

logger = getLogger(__name__)


def _encode_key(pairs) -> str:
    """Canonical lookup key: JSON of the sorted [extend, past] pairs."""
    return json.dumps(sorted(pairs))


def _decode_key(key: str):
    """Parse a sorted-tuple lookup key back to list of (extend, past) pairs."""
    return [tuple(pair) for pair in json.loads(key)]


def _shape_feat(extends, pasts):
    """3-D feature for KNN: (batch_size, sum_extend, sum_past).

    Low-dim and aligned with the dominant latency drivers; avoids the curse of
    dimensionality on the typically small (~1-3K) entries per replay table.
    """
    return (len(extends), sum(extends), sum(pasts))


def _attention_feat(extends, pasts):
    # Causal attention visits a different number of keys for each query token.
    attention_work = sum(
        extend * (past + (extend + 1) / 2) for extend, past in zip(extends, pasts)
    )
    return len(extends), sum(extends), attention_work


class ReplayTimePredictor(PerRankTimePredictor):
    """Look up measured GPU latency by batch composition.

    Compositions are matched exactly by sorted (extend_len, past_kv_len) tuples.
    On lookup miss the behavior is controlled by `miss_strategy`:

      - "zero" (default, legacy): returns `miss_fallback_seconds` (default 0.0).
        Use for "is the gap NOT from the predictor?" diagnostic. Only meaningful
        when miss rate is small AND you accept the bias of dropping miss work.

      - "knn": KNN-interpolated latency from the k batches in the table with
        the closest (batch_size, sum_extend, sum_past) shape (per-dim z-scored
        Euclidean distance). Use when miss rate matters or you want a logically
        complete oracle. Typical k=3.

      - "knn_work": KNN with causal attention work as the third feature and
        separate indexes for decode-only batches and batches with prefill.
        Decode misses use the median sampling time by batch size, interpolated
        between observed sizes.

    For workloads where sim batch composition diverges from real (e.g., bursty
    max-tps), the miss rate can be high; prefer "knn" there.
    """

    @staticmethod
    def build_table_from_schedule_batch(
        jsonl_paths: str | list[str],
        output_path: str | None = None,
    ) -> dict:
        """Build a replay lookup table from collector schedule_batch export(s).

        Each input line is one scheduled engine step from a
        `{rank}.schedule_batch.jsonl[.gz]` collector export; its
        `requests` field holds `(req_id, extend_input_len, past_kv_len,
        output_len)` tuples. Steps with the same sorted (extend, past)
        composition are collapsed by averaging `iter_latency` and
        `sample_tokens_latency` separately.

        Args:
            jsonl_paths: One schedule_batch JSONL path, or several to merge
                (e.g. multiple rounds/ranks). `.gz` is auto-detected.
            output_path: If set, dump the table JSON here; use it as the
                predictor's `database_path`.

        Returns:
            Replay table mapping `json.dumps(sorted [[extend, past], ...])`
            to mean forward and sample_tokens latencies in seconds.

        """
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        forward_sums: dict[str, float] = {}
        sample_sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        n_rows = 0
        for path in jsonl_paths:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    requests = row.get("requests") or []
                    if not requests:
                        continue
                    key = _encode_key([r[1], r[2]] for r in requests)
                    forward = float(row["iter_latency"])
                    sample = float(row.get("sample_tokens_latency", 0.0))
                    if not all(
                        math.isfinite(value) and value >= 0
                        for value in (forward, sample)
                    ):
                        raise ValueError(f"Invalid replay latency in {path}: {row}")
                    forward_sums[key] = forward_sums.get(key, 0.0) + forward
                    sample_sums[key] = sample_sums.get(key, 0.0) + sample
                    counts[key] = counts.get(key, 0) + 1
                    n_rows += 1
        table = {
            key: {
                "iter_latency": forward_sums[key] / counts[key],
                "sample_tokens_latency": sample_sums[key] / counts[key],
            }
            for key in forward_sums
        }
        if not table:
            raise ValueError("Cannot build a replay table without scheduled steps")
        logger.info(
            "Built replay table from %d schedule_batch steps (%s): "
            "%d unique compositions, %d duplicates averaged",
            n_rows,
            ", ".join(jsonl_paths),
            len(table),
            n_rows - len(table),
        )
        if output_path:
            with open(output_path, "w") as f:
                json.dump(table, f)
            logger.info("Replay table written to %s", output_path)
        return table

    def __init__(
        self,
        database_path: str,
        miss_fallback_seconds: float = 0.0,
        miss_strategy: str = "zero",
        miss_knn_k: int = 3,
    ):
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"ReplayTimePredictor database_path not found: {database_path}"
            )
        with open(database_path) as f:
            raw_table = json.load(f)
        if not raw_table:
            raise ValueError("Replay table must not be empty")
        self._table = {}
        legacy_entries = 0
        for key, value in raw_table.items():
            if isinstance(value, (int, float)):
                forward, sample = float(value), 0.0
                legacy_entries += 1
            elif isinstance(value, dict):
                forward = float(value["iter_latency"])
                sample = float(value.get("sample_tokens_latency", 0.0))
            else:
                raise ValueError(f"Invalid replay table entry for {key}: {value!r}")
            if not all(
                math.isfinite(latency) and latency >= 0 for latency in (forward, sample)
            ):
                raise ValueError(f"Invalid replay latency for {key}: {value!r}")
            self._table[key] = (forward, sample)
        if legacy_entries:
            logger.warning(
                "Loaded %d legacy replay entries without sample_tokens latency",
                legacy_entries,
            )
        if miss_strategy not in ("zero", "knn", "knn_work"):
            raise ValueError(
                "miss_strategy must be 'zero', 'knn' or 'knn_work', "
                f"got {miss_strategy!r}"
            )
        self._miss_strategy = miss_strategy
        self._miss_fallback = float(miss_fallback_seconds)
        if not math.isfinite(self._miss_fallback) or self._miss_fallback < 0:
            raise ValueError("miss_fallback_seconds must be finite and non-negative")
        self._miss_knn_k = int(miss_knn_k)
        if self._miss_knn_k < 1:
            raise ValueError("miss_knn_k must be at least 1")

        if self._miss_strategy != "zero":
            self._prep_knn_index()
            logger.info(
                "ReplayTimePredictor loaded %d unique compositions from %s "
                "(miss_strategy=%s, k=%d)",
                len(self._table),
                database_path,
                self._miss_strategy,
                self._miss_knn_k,
            )
        else:
            logger.info(
                "ReplayTimePredictor loaded %d unique compositions from %s "
                "(miss_strategy=zero, fallback=%.4fs)",
                len(self._table),
                database_path,
                self._miss_fallback,
            )

    def _prep_knn_index(self):
        """Standardize features once per index."""
        uses_work = self._miss_strategy == "knn_work"
        self._knn_feature = _attention_feat if uses_work else _shape_feat
        groups = {}
        decode_samples = {}
        for key, latencies in self._table.items():
            extends_pasts = _decode_key(key)
            extends = [e for e, _ in extends_pasts]
            pasts = [p for _, p in extends_pasts]
            group = uses_work and any(e > 1 for e in extends)
            features, timings = groups.setdefault(group, ([], []))
            features.append(self._knn_feature(extends, pasts))
            timings.append(latencies)
            if uses_work and extends and not group:
                decode_samples.setdefault(len(extends), []).append(latencies[1])
        self._decode_sample_sizes = sorted(decode_samples)
        self._decode_sample_medians = [
            float(np.median(decode_samples[size])) for size in self._decode_sample_sizes
        ]
        self._knn_indexes = {}
        for group, (features, timings) in groups.items():
            features = np.asarray(features, dtype=float)
            mean = features.mean(axis=0)
            var = features.var(axis=0)
            scale = np.sqrt(np.where(var > 1e-12, var, 1.0))
            self._knn_indexes[group] = (
                (features - mean) / scale,
                np.asarray(timings),
                mean,
                scale,
            )

    def _knn_predict(self, extends, pasts):
        group = self._miss_strategy == "knn_work" and any(e > 1 for e in extends)
        if group not in self._knn_indexes:
            group = not group
        features, timings, mean, scale = self._knn_indexes[group]
        query = (self._knn_feature(extends, pasts) - mean) / scale
        delta = features - query
        distances = np.einsum("ij,ij->i", delta, delta)
        k = min(self._miss_knn_k, len(distances))
        boundary = np.partition(distances, k - 1)[k - 1]
        closer = np.flatnonzero(distances < boundary)
        tied = np.flatnonzero(distances == boundary)[: k - len(closer)]
        forward, sample = timings[np.concatenate((closer, tied))].mean(axis=0)
        if self._decode_sample_sizes and not any(e > 1 for e in extends):
            # Sampling depends on the token batch, while forward attention
            # depends on the KV lengths. Pool transient timing spikes across
            # shapes with the same sampling work.
            sample = np.interp(
                len(extends), self._decode_sample_sizes, self._decode_sample_medians
            )
        return float(forward), float(sample)

    def predict_step_time(self, batch: ScheduleBatch) -> tuple[float, float]:
        if batch.is_empty():
            return 0.0, 0.0
        extends = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]
        key = _encode_key([req.extend_length, req.past_kv_length] for req in batch.reqs)
        latencies = self._table.get(key)
        if latencies is None:
            if self._miss_strategy != "zero":
                return self._knn_predict(extends, pasts)
            return self._miss_fallback, 0.0
        return latencies

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        return self.predict_step_time(batch)[0]
