"""Model v6d transfer deadlines without moving KV payloads.

SimulatedKVController uses load/store latencies for native acknowledgements
and block release, independently of inference timing. HISIM_BW_PROFILE supplies
layout and calibration data; local transfer modeling is disabled without it.
"""

from __future__ import annotations

import json
import logging

from hisim.env import Envs

logger = logging.getLogger(__name__)


class SegmentModel:
    """Segment latency: max(floor, intercept + bytes / bandwidth)."""

    def __init__(self, name: str, floor_s: float, t0_s: float, bw_bytes_per_s: float):
        self.name = name
        self.floor_s = max(0.0, floor_s)
        self.t0_s = max(0.0, t0_s)
        self.bw = bw_bytes_per_s if bw_bytes_per_s > 0 else float("inf")

    def latency(self, nbytes: int) -> float:
        if nbytes <= 0:
            return 0.0
        return max(self.floor_s, self.t0_s + nbytes / self.bw)

    @classmethod
    def from_samples(cls, name: str, seg: dict) -> "SegmentModel":
        """Fit the two largest samples; use profile coefficients if no rising pair."""
        rows = sorted(seg.get("samples", []), key=lambda r: r["bytes"])
        if len(rows) >= 2:
            (b0, t0), (b1, t1) = (
                (rows[-2]["bytes"], rows[-2]["median_s"]),
                (rows[-1]["bytes"], rows[-1]["median_s"]),
            )
            if t1 > t0 and b1 > b0:
                bw = (b1 - b0) / (t1 - t0)
                floor = min(r["median_s"] for r in rows)
                # Anchor the intercept on the largest sample, clamped at zero.
                intercept = max(0.0, t1 - b1 / bw)
                return cls(name, floor, intercept, bw)
        return cls(
            name,
            0.0,
            seg.get("fixed_overhead_s", 0.0),
            seg.get("bandwidth_bytes_per_s", 0.0),
        )

    def __repr__(self):
        return (
            f"SegmentModel({self.name}: floor={self.floor_s * 1e3:.3f}ms, "
            f"t0={self.t0_s * 1e6:.1f}us, bw={self.bw / (1 << 30):.2f}GiB/s)"
        )


class BandwidthModel:
    """Singleton holding the calibrated segments and the byte-count layout."""

    _instance: "BandwidthModel | None" = None

    def __init__(self, profile: dict | None):
        self.enabled = False
        self.local_load: SegmentModel | None = None
        self.local_store: SegmentModel | None = None
        self.page_size_bytes = 0
        self.num_layers = 0
        self.peer_topology = None
        self._seg1_floor_ms = 0.0
        self._seg1_per_blk_ms = 0.0
        if not profile:
            return
        layout = profile.get("layout", {})
        self.page_size_bytes = int(layout.get("page_size_bytes", 0))
        self.num_layers = int(layout.get("num_layers", 0))
        self.peer_topology = profile.get("peer_topology")
        cp = profile.get("control_plane", {})
        _s1 = cp.get("seg1_cross_node", {})
        self._seg1_floor_ms = float(_s1.get("floor_ms", 0) or 0)
        self._seg1_per_blk_ms = float(_s1.get("per_block_ms", 0) or 0)
        # Store readiness includes a polling floor and rank-acknowledgement cost.
        _st = cp.get("store_completion", {})
        self._poll_granularity_ms = float(_st.get("poll_granularity_ms", 10.0))
        self._rank_sync_ms = float(_st.get("rank_sync_ms", 1.0))
        segs = profile.get("segments", {})
        if "local_load" in segs and "local_store" in segs:
            self.local_load = SegmentModel.from_samples(
                "local_load", segs["local_load"]
            )
            self.local_store = SegmentModel.from_samples(
                "local_store", segs["local_store"]
            )
        if self.page_size_bytes > 0 and self.num_layers > 0 and self.local_load:
            self.enabled = True

    # Byte counts and local transfer deadlines.

    def seg2_bytes(self, num_blocks: int) -> int:
        """Per-rank bytes: block count × layers × post-sharding page size."""
        return max(0, num_blocks) * self.num_layers * self.page_size_bytes

    def latency_for(self, num_blocks: int, swap_in: bool) -> float:
        if not self.enabled or num_blocks <= 0:
            return 0.0
        seg = self.local_load if swap_in else self.local_store
        return seg.latency(self.seg2_bytes(num_blocks)) if seg else 0.0

    def store_completion_latency(self, num_blocks: int) -> float:
        """Store delay: max(DMA, polling floor) + rank synchronization.

        Environment settings override the profile's completion constants.
        """
        if not self.enabled or num_blocks <= 0:
            return 0.0
        dma_s = self.latency_for(num_blocks, False)
        poll_s = Envs.store_poll_ms(self._poll_granularity_ms) / 1000.0
        rank_s = Envs.store_rank_sync_ms(self._rank_sync_ms) / 1000.0
        return max(dma_s, poll_s) + rank_s

    # Cross-node latency; environment parameters override profile values.

    @staticmethod
    def _cp_latency(nblocks, floor_ms, per_ms):
        if floor_ms <= 0 and per_ms <= 0:
            return 0.0
        return (floor_ms + per_ms * max(0, nblocks)) / 1000.0

    def seg1_latency(self, nblocks: int) -> float:
        """Cross-node delay from configured floor/per-block costs, zero by default."""
        floor_ms, per_ms = Envs.seg1_latency_parameters(
            self._seg1_floor_ms, self._seg1_per_blk_ms
        )
        return self._cp_latency(nblocks, floor_ms, per_ms)

    # Load the profile once per process; reset permits reconfiguration.

    @classmethod
    def get(cls) -> "BandwidthModel":
        if cls._instance is None:
            cls._instance = cls._load()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    @classmethod
    def _load(cls) -> "BandwidthModel":
        path = Envs.bandwidth_profile_path()
        if not path:
            logger.info(
                "[v6d-bw] %s unset: transfer-latency modelling disabled",
                Envs.BW_PROFILE,
            )
            return cls(None)
        try:
            with open(path) as f:
                profile = json.load(f)
        except Exception as exc:
            logger.warning(
                "[v6d-bw] cannot read profile %s (%s): "
                "transfer-latency modelling disabled",
                path,
                exc,
            )
            return cls(None)
        model = cls(profile)
        if not model.enabled:
            logger.warning("[v6d-bw] profile %s incomplete: modelling disabled", path)
            return model
        logger.info(
            "[v6d-bw] enabled from %s | layout %d layers x %d B "
            "(= %d B/block per rank) | topology=%s | %s | %s",
            path,
            model.num_layers,
            model.page_size_bytes,
            model.seg2_bytes(1),
            model.peer_topology,
            model.local_load,
            model.local_store,
        )
        return model
