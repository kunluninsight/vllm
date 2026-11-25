"""Provider side of the PACE mock: one virtual tair_mempool_server.

GAManager (GA ledger per segment), TieringManager (transparent DRAM->SSD demotion),
the GETSTATUS report and the provider-side free / copy_ga checks. A Provider knows
nothing about other Providers or about placement; the meta service reaches it only
through these public methods (the "RPC" boundary).

This module also owns the primitives the meta service borrows: media codes, the GA
bit layout, size parsing, the shared logger and ApiError.

Provider tiering (transparent nodes; tiering_manager.cpp / dram_slot_pool.cpp / ga_manager.cpp)
  - Watermarks: DramSlotPool hi=85 / lo=70 (--demote-high/low-watermark-pct, hi clamped to <=85);
    usage_pct = DRAM used * 100 / DRAM pool (integer); demote when pct >= hi, stop when pct < lo.
  - WaterMarkLoop every watermark_check_ms (100): at most min(worker*batch, 256) GAs and
    worker*batch_mb bytes per tick; each victim is the min last_access among 32 random samples
    (64, then a full scan, when sampling finds nothing).
  - Demote: SSD write ok -> SSD_ONLY, DRAM goes back to BFC at once, GA unchanged; write failure ->
    back to DRAM_ONLY, demote_fail_count++; 10 consecutive ticks with failures while SSD >= 95%
    -> back off 300 ticks; freeing a GA that lives on SSD (NotifySsdFreed) clears the back-off.
  - Alloc: AdmitByWatermark (DRAM within 95%, the overflow within the SSD watermark) rejects
    without demoting; when DRAM cannot fit, SyncDemoteForAlloc(fail_fast) up to 3 times: no-op below
    hi; ERR_NO_MEM during back-off or with SSD >= 95%; otherwise demotes up to demote_batch_size GAs
    synchronously. Meta placement never triggers demotion.
  - Report: used_bytes / alloced_ga_num cover DRAM residents only, ssd_used_bytes covers SSD.
"""

import collections
import logging
import random
import re
import threading
import time
import uuid


LOG = logging.getLogger("mock_pace")
MEDIA_AUTO = 0x00
MEDIA_DRAM = 0x02
MEDIA_SSD = 0x05
TIERS = (MEDIA_DRAM, MEDIA_SSD)
TIERING_MODES = ("transparent", "explicit", "off")
GA_SEQ_BITS = 36  # common.h: low 36 bits ga_seq, high 28 bits segment_id
GA_SEQ_MASK = (1 << GA_SEQ_BITS) - 1
NODE_UUID_NS = uuid.UUID("6f1d2a4e-3c1b-4f5e-9a7d-2b8c0e4d1a90")
PROVIDER_DEFAULT_WATERMARK_PCT = 95
_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?)(I?B)?\s*$", re.IGNORECASE)
_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40, "P": 1 << 50}


def parse_size(value):
    """'64GB' / '1.5TiB' / '1024' / 1024 -> bytes (binary units); mock options only."""
    if isinstance(value, bool):
        raise ValueError("invalid size: %r" % (value,))
    if isinstance(value, (int, float)):
        return int(value)
    m = _SIZE_RE.match(str(value))
    if not m:
        raise ValueError("invalid size: %r" % (value,))
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()])


# GA layout: [segment_id:28][ga_seq:36]; segment_id = (pace_node_id << 4) | segment media
def make_segment_id(pace_id, media):
    return (pace_id << 4) | media


def ga_segment(ga):
    return ga >> GA_SEQ_BITS


def ga_media(ga):
    """Media of the segment a GA was allocated from; unchanged by transparent demotion."""
    return ga_segment(ga) & 0xF


class ApiError(Exception):
    """SetErrorResponse: {"status":"failed","data":msg}."""

    def __init__(self, http_code, message):
        super().__init__(message)
        self.http_code = http_code
        self.message = message


# ============================================================ provider: media pools
class Tier(object):
    """One media pool of a Provider (DRAM segment or LocalSSD backend)."""

    __slots__ = ("media", "total", "used", "ga_num")

    def __init__(self, media, total):
        self.media = media
        self.total = int(total)
        self.used = 0
        self.ga_num = 0

    @property
    def free(self):
        return max(self.total - self.used, 0)


class TieringParams(object):
    """Provider-side tiering knobs, identical for every virtual Provider (start-up flags / env)."""

    def __init__(self, cfg, env):
        # DramSlotPool constructor clamp: out-of-range hi -> 85, invalid lo -> hi - 15
        hi, lo = cfg.demote_high_watermark_pct, cfg.demote_low_watermark_pct
        if hi <= 0 or hi > 85:
            LOG.warning("DramSlotPool: hi_pct=%d clamped to [1, 85]", hi)
            hi = 85
        if lo < 0 or lo >= hi:
            LOG.warning(
                "DramSlotPool: lo_pct=%d invalid, set to hi_pct-15=%d", lo, hi - 15
            )
            lo = hi - 15 if hi > 15 else 0
        self.hi_pct, self.lo_pct = hi, lo
        # GlobalAddressManager::ParseWatermarkPct: Provider admission watermark, default 95
        raw = env.get("TAIR_MEMPOOL_TIERING_CAPACITY_WATERMARK_PCT", "")
        self.admit_watermark_pct = (
            int(raw)
            if raw.isdigit() and 0 < int(raw) <= 100
            else PROVIDER_DEFAULT_WATERMARK_PCT
        )
        self.alloc_ssd_full_pct = cfg.alloc_ssd_full_pct
        self.worker_count = max(cfg.demote_worker_count, 1)
        self.batch_size = cfg.demote_batch_size
        self.batch_mb = cfg.demote_batch_mb

    def to_json(self):
        return dict(vars(self))


class Provider(object):
    """One virtual tair_mempool_server.

    Owns its segments and GA ledger (GAManager), the transparent DRAM->SSD tiering state
    (ResidencyTable + TieringManager) and produces the GETSTATUS report Meta consumes. Public
    methods are the Provider's "RPC surface" and take self.lock; private helpers expect it held.
    `healthy` models reachability from Meta and is flipped by the harness only.
    """

    def __init__(
        self, pace_id, dram, ssd, ip, port, tiering_mode, params, uid=None, healthy=True
    ):
        if tiering_mode not in TIERING_MODES:
            raise ApiError(400, "tiering_mode must be one of %s" % (TIERING_MODES,))
        self.pace_id = int(pace_id)
        self.uid = uid or str(
            uuid.uuid5(NODE_UUID_NS, "mock-pace-node-%d" % self.pace_id)
        )
        self.ip = ip
        self.port = int(port)
        self.healthy = bool(healthy)
        self.tiering_mode = tiering_mode
        self.params = params
        self.lock = threading.RLock()
        self.tiers = {
            MEDIA_DRAM: Tier(MEDIA_DRAM, dram),
            MEDIA_SSD: Tier(MEDIA_SSD, ssd),
        }
        self.seq = 0
        self.gas = {}  # ga -> [resident media, size]
        self.sim_written = set()  # Simulated payload state; no bytes are stored.
        # ResidencyTable DRAM residents (RESIDENT_DRAM_ONLY): ga -> [size, last_access, list index];
        # the parallel list gives O(1) random sampling for PickColdestForDemote
        self.dram_resident = {}
        self.dram_list = []
        # TieringManager WaterMarkLoop failure / back-off counters
        self.demote_fail_count = 0
        self.last_seen_fail_count = 0
        self.consecutive_demote_fail = 0
        self.demote_backoff_skips = 0
        self.stats = collections.Counter()

    # ----------------------------------------------------------------- identity
    @property
    def tiering_enabled(self):
        return self.tiering_mode != "off" and self.tiers[MEDIA_SSD].total > 0

    @property
    def transparent(self):
        return self.tiering_mode == "transparent" and self.tiering_enabled

    def segment_id(self, media):
        """virt_segment_id: one segment per node per media; the high 28 bits of its GAs."""
        return make_segment_id(self.pace_id, media)

    def to_json(self):
        return {
            "pace_node_id": self.pace_id,
            "id": self.uid,
            "ip": self.ip,
            "port": self.port,
            "healthy": self.healthy,
            "tiering_mode": self.tiering_mode,
            "dram": self.tiers[MEDIA_DRAM].total,
            "ssd": self.tiers[MEDIA_SSD].total,
        }

    def reconfigure(self, dram=None, ssd=None, tiering_mode=None):
        """Harness only: resize pools or switch tiering mode in place (existing GAs are kept)."""
        with self.lock:
            if dram is not None:
                self.tiers[MEDIA_DRAM].total = parse_size(dram)
            if ssd is not None:
                self.tiers[MEDIA_SSD].total = parse_size(ssd)
            if tiering_mode is not None:
                if tiering_mode not in TIERING_MODES:
                    raise ApiError(
                        400, "tiering_mode must be one of %s" % (TIERING_MODES,)
                    )
                self.tiering_mode = tiering_mode

    # ------------------------------------------------------- GAManager (RPC surface)
    def alloc(self, layer, size):
        """GlobalAddressManager::Alloc for a Meta-forwarded AllocGA(media=layer): GA or None.

        Transparent: AdmitByWatermark first (a rejection fails without demoting), then BFC; when DRAM
        cannot fit, SyncDemoteForAlloc(fail_fast=true) up to 3 times (ga_manager.cpp:1180-1215)."""
        with self.lock:
            media = MEDIA_DRAM if layer == MEDIA_AUTO else layer
            tier = self.tiers[media]
            if self.transparent:
                if not self._admit_by_watermark(size):
                    self.stats["provider_admit_reject"] += 1
                    return None
                for _ in range(3):
                    if tier.free >= size:
                        break
                    if self._sync_demote_for_alloc(size, fail_fast=True) != "SUCCESS":
                        break
            if tier.free < size:
                return None
            return self._mint(media, size)

    def free(self, ga):
        """Free one GA: its size, or None when unknown (Meta treats an unknown GA as success)."""
        with self.lock:
            rec = self._release(ga)
            return None if rec is None else rec[1]

    def free_all(self):
        with self.lock:
            victims = list(self.gas)
            for ga in victims:
                self._release(ga)
            return len(victims)

    def ga_size(self, ga):
        with self.lock:
            rec = self.gas.get(ga)
            return None if rec is None else rec[1]

    def ga_count(self):
        with self.lock:
            return len(self.gas)

    def list_gas(self):
        with self.lock:
            return list(self.gas)

    def describe(self, ga):
        with self.lock:
            rec = self.gas.get(ga)
            if rec is None:
                return None
            return {
                "pace_node_id": self.pace_id,
                "resident_media_type": rec[0],
                "segment_media_type": ga_media(ga),
                "size": rec[1],
                "segment_id": ga_segment(ga),
                "seq": ga & GA_SEQ_MASK,
                "sim_written": ga in self.sim_written,
            }

    def simulate_io(self, ga, size, write):
        """Validate one simulated SDK operation without accessing payload bytes."""
        operation = "save" if write else "load"
        with self.lock:
            self.stats["sim_" + operation + "_attempts"] += 1
            record = self.gas.get(ga)
            reason = ""
            if not self.healthy:
                reason = "provider unavailable"
            elif record is None:
                reason = "GA not allocated"
            elif size != record[1]:
                reason = "GA size mismatch"
            elif not write and ga not in self.sim_written:
                reason = "GA has not been written"
            if reason:
                self.stats["sim_" + operation + "_failures"] += 1
                return {"ga": ga, "success": False, "reason": reason}
            if write:
                self.sim_written.add(ga)
            if ga in self.dram_resident:
                self.dram_resident[ga][1] = time.monotonic()
            self.stats["sim_" + operation + "_objects"] += 1
            self.stats["sim_" + operation + "_bytes"] += size
            return {
                "ga": ga,
                "success": True,
                "reason": "",
                "resident_media_type": record[0],
            }

    def copy_to(self, req, dst):
        """Provider-side CopyGA checks (meta_restful_server.cpp); returns (http_status, error).

        `dst` is the destination Provider; its GA size is read before taking our lock so two
        Provider locks are never held together."""
        size, src_off, dst_off = req["size"], req["src_offset"], req["dst_offset"]
        dst_size = dst.ga_size(req["dst_ga"])
        with self.lock:
            src_size = self.ga_size(req["src_ga"])
            if src_size is not None and src_off + size > src_size:
                return 400, "src_offset + size (%d) exceeds src GA size (%d)" % (
                    src_off + size,
                    src_size,
                )
            if dst_size is not None and dst_off + size > dst_size:
                return 400, "dst_offset + size (%d) exceeds dst GA size (%d)" % (
                    dst_off + size,
                    dst_size,
                )
            for tag, ga, expect in (
                ("src", req["src_ga"], req["src_media_type"]),
                ("dst", req["dst_ga"], req["dst_media_type"]),
            ):
                actual = ga_media(ga)
                if expect and actual != expect:
                    return 400, (
                        "%s: media_type mismatch, expect=%d actual=%d (DRAM=2, LOCALSSD=5)"
                        % (tag, expect, actual)
                    )
            if src_size is None or dst_size is None:
                return 500, "WriteRemote failed (mock: %s GA not allocated)" % (
                    "src" if src_size is None else "dst"
                )
            self.stats["copy_bytes"] += size
            return 200, ""

    def status(self):
        """GETSTATUS report (ga_meta_operator.cpp:1344-1377): total = GetMPTotalSize (DRAM pool),
        used = GetMPUsedSize (DRAM segment; demoted GAs excluded), alloced_ga_num = DRAM segment
        count, ssd_* = SSD backend GetSsdSpaceStats (only when tiering is enabled)."""
        with self.lock:
            dram, ssd = self.tiers[MEDIA_DRAM], self.tiers[MEDIA_SSD]
            enabled = self.tiering_enabled
            return {
                "total_bytes": dram.total,
                "used_bytes": dram.used,
                "alloced_ga_num": dram.ga_num,
                "ssd_total_bytes": ssd.total if enabled else 0,
                "ssd_used_bytes": ssd.used if enabled else 0,
                "tiering_enabled": enabled,
                "tiering_mode": self.tiering_mode,
                "segment": self._segments(),
            }

    def tiering_state(self):
        with self.lock:
            return {
                "usage_pct": self._usage_pct(),
                "dram_residents": len(self.dram_list),
                "demote_fail_count": self.demote_fail_count,
                "consecutive_demote_fail": self.consecutive_demote_fail,
                "demote_backoff_skips": self.demote_backoff_skips,
            }

    # ------------------------------------------------------ TieringManager (RPC surface)
    def tiering_tick(self):
        """One TieringManager::WaterMarkLoop iteration (every watermark_check_ms)."""
        with self.lock:
            if not self.transparent:
                return
            if self.demote_backoff_skips > 0:
                self.demote_backoff_skips -= 1
                return
            if self.demote_fail_count > self.last_seen_fail_count:
                self.consecutive_demote_fail += 1
                if self.consecutive_demote_fail >= 10:
                    ssd = self.tiers[MEDIA_SSD]
                    if ssd.total > 0 and ssd.used >= ssd.total * 95 // 100:
                        self.demote_backoff_skips = 300
                        self.stats["ssd_full_backoff"] += 1
                        LOG.warning(
                            "provider %d: SSD nearly full %d/%d after %d consecutive fail ticks, "
                            "backoff 300 iters",
                            self.pace_id,
                            ssd.used,
                            ssd.total,
                            self.consecutive_demote_fail,
                        )
                    self.consecutive_demote_fail = 0
            else:
                self.consecutive_demote_fail = 0
            self.last_seen_fail_count = self.demote_fail_count
            if self._usage_pct() < self.params.hi_pct:
                return
            p = self.params
            hard_cap = min(p.worker_count * p.batch_size, 256)  # kMaxDemotePicksPerTick
            target_bytes = p.worker_count * p.batch_mb * 1024 * 1024
            self._collect_and_demote(hard_cap, target_bytes, stop_below_lwm=True)

    def sync_demote_for_alloc(self, need, fail_fast):
        with self.lock:
            return self._sync_demote_for_alloc(need, fail_fast)

    # ----------------------------------------------------------- persistence (harness)
    def dump(self):
        with self.lock:
            return dict(
                self.to_json(),
                seq=self.seq,
                sim_written=sorted(self.sim_written),
                gas=[[ga, rec[0], rec[1]] for ga, rec in self.gas.items()],
            )

    def restore(self, seq, gas, sim_written=()):
        """Rebuild the ledger from a snapshot: [[ga, resident media, size], ...]."""
        with self.lock:
            self._clear_locked()
            for ga, media, size in sorted(gas, key=lambda r: r[0] & GA_SEQ_MASK):
                self._account(ga, media, size)
            self.seq = max([int(seq)] + [ga & GA_SEQ_MASK for ga, _m, _s in gas])
            self.sim_written.update(ga for ga in sim_written if ga in self.gas)

    def clear(self):
        with self.lock:
            self._clear_locked()
            self.stats.clear()

    # ------------------------------------------------------------ internals (locked)
    def _clear_locked(self):
        self.gas.clear()
        self.sim_written.clear()
        self.seq = 0
        self.dram_resident.clear()
        self.dram_list.clear()
        self.demote_fail_count = self.last_seen_fail_count = 0
        self.consecutive_demote_fail = self.demote_backoff_skips = 0
        for tier in self.tiers.values():
            tier.used = tier.ga_num = 0

    def _segments(self):
        out = []
        for media in TIERS:
            tier = self.tiers[media]
            if tier.total <= 0 or (media == MEDIA_SSD and not self.tiering_enabled):
                continue
            sid = self.segment_id(media)
            out.append(
                {
                    "segment_id": sid,
                    "virt_segment_id": sid,
                    "total_bytes": tier.total,
                    "used_bytes": tier.used,
                    "alloced_ga_num": tier.ga_num,
                    "is_active": self.healthy,
                    "media_type": media,
                }
            )
        return out

    def _mint(self, media, size):
        self.seq += 1
        if self.seq > GA_SEQ_MASK:
            raise ApiError(500, "ga_seq exhausted")
        ga = (self.segment_id(media) << GA_SEQ_BITS) | self.seq
        self._account(ga, media, size)
        return ga

    def _account(self, ga, media, size):
        self.gas[ga] = [media, size]
        tier = self.tiers[media]
        tier.used += size
        tier.ga_num += 1
        if media == MEDIA_DRAM and self.tiering_mode == "transparent":
            self._resident_add(
                ga, size
            )  # ResidencyTable::CreateForNewAlloc sets last_access_us

    def _release(self, ga):
        rec = self.gas.pop(ga, None)
        self.sim_written.discard(ga)
        if rec is None:
            return None
        tier = self.tiers[rec[0]]
        tier.used = max(0, tier.used - rec[1])
        tier.ga_num = max(0, tier.ga_num - 1)
        self._resident_remove(ga)
        if rec[0] == MEDIA_SSD and self.tiering_mode == "transparent":
            self.demote_backoff_skips = 0  # NotifySsdFreed clears the SSD-full back-off
        return rec

    def _resident_add(self, ga, size):
        self.dram_resident[ga] = [size, time.monotonic(), len(self.dram_list)]
        self.dram_list.append(ga)

    def _resident_remove(self, ga):
        entry = self.dram_resident.pop(ga, None)
        if entry is None:
            return
        idx, last = entry[2], self.dram_list.pop()
        if last != ga:
            self.dram_list[idx] = last
            self.dram_resident[last][2] = idx

    def _admit_by_watermark(self, size):
        """GlobalAddressManager::WatermarkAdmit: DRAM watermark first, the overflow must fit the SSD
        watermark (footprint = size)."""
        dram, ssd = self.tiers[MEDIA_DRAM], self.tiers[MEDIA_SSD]
        pct = self.params.admit_watermark_pct
        demand_x100 = (dram.used + size) * 100
        cap_x100 = pct * dram.total
        if demand_x100 <= cap_x100:
            return True
        return ssd.used * 100 + (demand_x100 - cap_x100) <= pct * ssd.total

    def _usage_pct(self):
        """DramSlotPool::GetUsagePct: used * 100 / data_pool, integer, capped at 100 (no shm_meta /
        cache segment in the mock)."""
        dram = self.tiers[MEDIA_DRAM]
        if dram.total <= 0:
            return 0
        return 100 if dram.used > dram.total else dram.used * 100 // dram.total

    def _ssd_full_for_alloc(self):
        ssd = self.tiers[MEDIA_SSD]
        pct = self.params.alloc_ssd_full_pct
        return self.demote_backoff_skips > 0 or (
            pct > 0 and ssd.total > 0 and ssd.used >= ssd.total * pct // 100
        )

    def _pick_coldest(self, exclude):
        """PickColdestForDemote: min last_access among demote_sample_size (32) random DRAM residents;
        widen to 2x samples, then a full scan, when no candidate is found."""
        n = len(self.dram_list)
        if n == 0:
            return None
        for sample in (32, 64, n):
            picks = (
                self.dram_list if sample >= n else random.sample(self.dram_list, sample)
            )
            best = None
            for ga in picks:
                if ga in exclude:
                    continue
                ts = self.dram_resident[ga][1]
                if best is None or ts < best[1]:
                    best = (ga, ts)
            if best is not None:
                return best[0]
            if sample >= n:
                break
        return None

    def _demote_one(self, ga):
        """DoDemoteBatch for one GA: SSD write ok -> DRAM_ONLY becomes SSD_ONLY and DRAM goes back to
        BFC; write failure (mock: SSD out of capacity) -> stays DRAM_ONLY, demote_fail_count++.
        There are no readers, so no DUAL residency."""
        dram, ssd = self.tiers[MEDIA_DRAM], self.tiers[MEDIA_SSD]
        size = self.dram_resident[ga][0]
        if ssd.used + size > ssd.total:
            self.demote_fail_count += 1
            self.stats["demote_failed"] += 1
            return False
        self._resident_remove(ga)
        self.gas[ga][0] = MEDIA_SSD
        dram.used -= size
        dram.ga_num -= 1
        ssd.used += size
        ssd.ga_num += 1
        self.stats["demoted_gas"] += 1
        self.stats["demoted_bytes"] += size
        return True

    def _collect_and_demote(self, max_count, max_bytes, stop_below_lwm):
        """CollectDemoteBatch + DoDemoteBatch (the SSD write is instant). Returns GAs demoted."""
        picked, picked_bytes, done = set(), 0, 0
        for _ in range(max_count):
            if max_bytes > 0 and picked_bytes >= max_bytes:
                break
            if stop_below_lwm and self._usage_pct() < self.params.lo_pct:
                break
            ga = self._pick_coldest(picked)
            if ga is None:
                break
            picked.add(ga)
            picked_bytes += self.dram_resident[ga][0]
            if self._demote_one(ga):
                done += 1
        return done

    def _sync_demote_for_alloc(self, need, fail_fast):
        """TieringManager::SyncDemoteForAlloc (tiering_manager.cpp)."""
        self.stats["sync_demote_calls"] += 1
        if self._usage_pct() < self.params.hi_pct:
            return "SUCCESS"  # below the high watermark: nothing to do
        if fail_fast and self._ssd_full_for_alloc():
            self.stats["alloc_ssd_full_fastfail"] += 1
            return "ERR_NO_MEM"
        if fail_fast and not self.dram_list:
            self.stats["alloc_no_demotable_fastfail"] += 1
            return "ERR_NO_MEM"
        entry_used = self.tiers[MEDIA_DRAM].used
        self._collect_and_demote(
            min(self.params.batch_size, 256), 0, stop_below_lwm=False
        )
        used = self.tiers[MEDIA_DRAM].used
        if used + need <= entry_used or self._usage_pct() < self.params.hi_pct:
            return "SUCCESS"
        return "ERR_TIMEOUT"  # the real one waits until its timeout (5s default)
