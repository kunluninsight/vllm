"""Keep KVCM control-plane transfers while modeling MemPool DRAM/SSD I/O."""

from __future__ import annotations

import json
import logging
import re
import time
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from hisim.env import Envs
from hisim.hook import BaseHook

logger = logging.getLogger("hisim")

_MEDIA_SSD = 0x05
_GA_QUERY_KEYS = ("ga", "global_address", "addr", "address")
_GA_PATH_RE = re.compile(r"(?:^|/)(?:ga[=/])?(\d+|0x[0-9a-fA-F]+)(?:$|/)")


def _parse_positive_int(raw: str | None) -> int | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        val = int(text, 0)
    except ValueError:
        return None
    return val if val > 0 else None


def extract_ga_from_uri(uri: str) -> int | None:
    """Best-effort GA extraction from a Tair MemPool / PACE location URI."""
    if not isinstance(uri, str) or not uri:
        return None
    parsed = urlparse(uri)
    params = parse_qs(parsed.query)
    for key in _GA_QUERY_KEYS:
        values = params.get(key)
        if values:
            ga = _parse_positive_int(values[0])
            if ga is not None:
                return ga
    match = _GA_PATH_RE.search(parsed.path or "")
    if match:
        return _parse_positive_int(match.group(1))
    return _parse_positive_int(uri)


def _post_mock_transfer(
    meta_url: str, path: str, items: list[dict]
) -> dict[int, int]:
    """Notify mock_meta_service to refresh DRAM LRU and report resident media."""
    if not meta_url or not items:
        return {}
    body = json.dumps({"items": items}).encode("utf-8")
    req = Request(
        f"{meta_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("[v6d-sim] mock MemPool %s skipped: %s", path, exc)
        return {}
    media_by_ga: dict[int, int] = {}
    for entry in payload.get("results") or ():
        if entry.get("success") and "resident_media_type" in entry:
            media_by_ga[int(entry["ga"])] = int(entry["resident_media_type"])
    return media_by_ga


def _compute_tier_delay(
    dram_bytes: int,
    ssd_bytes: int,
    dram_base_s: float,
    dram_bps: float,
    ssd_base_s: float,
    ssd_bps: float,
) -> float:
    delay = 0.0
    if dram_bytes > 0 and (dram_base_s > 0.0 or dram_bps > 0.0):
        delay += dram_base_s + (dram_bytes / dram_bps if dram_bps > 0.0 else 0.0)
    if ssd_bytes > 0:
        eff_base = ssd_base_s if (ssd_base_s > 0.0 or ssd_bps > 0.0) else dram_base_s
        eff_bps = ssd_bps if (ssd_base_s > 0.0 or ssd_bps > 0.0) else dram_bps
        if eff_base > 0.0 or eff_bps > 0.0:
            delay += eff_base + (ssd_bytes / eff_bps if eff_bps > 0.0 else 0.0)
    return delay


def _simulate_kvcm_io(uris, sizes, *, write: bool) -> float:
    """Sync optional mock MemPool residency and sleep for modeled I/O latency."""
    meta_url = Envs.tair_mempool_meta_url()
    items = []
    ga_for_index: list[int | None] = []
    for uri, size in zip(uris, sizes):
        sz = int(size) if size and size > 0 else 0
        ga = extract_ga_from_uri(uri) if meta_url and sz > 0 else None
        ga_for_index.append(ga)
        if ga is not None:
            items.append({"ga": ga, "size": sz})

    media_by_ga: dict[int, int] = {}
    if meta_url and items:
        endpoint = "/admin/transfer/save" if write else "/admin/transfer/load"
        media_by_ga = _post_mock_transfer(meta_url, endpoint, items)

    dram_bytes = 0
    ssd_bytes = 0
    for ga, size in zip(ga_for_index, sizes):
        sz = int(size) if size and size > 0 else 0
        if sz <= 0:
            continue
        if ga is not None and media_by_ga.get(ga) == _MEDIA_SSD:
            ssd_bytes += sz
        else:
            dram_bytes += sz

    if write:
        delay = _compute_tier_delay(
            dram_bytes + ssd_bytes,
            0,
            Envs.tair_kvcm_write_base_s(),
            Envs.tair_kvcm_write_bandwidth_bps(),
            0.0,
            0.0,
        )
    else:
        delay = _compute_tier_delay(
            dram_bytes,
            ssd_bytes,
            Envs.tair_kvcm_read_base_s(),
            Envs.tair_kvcm_read_bandwidth_bps(),
            Envs.tair_kvcm_ssd_read_base_s(),
            Envs.tair_kvcm_ssd_read_bandwidth_bps(),
        )
    if delay > 0.0:
        time.sleep(delay)
    return delay


class _NoopTransferClient:
    """Stands in for kvcm_py_client.TransferClient, whose only uses are stubbed below."""

    def close(self):
        pass


class C_TairKVCMTransferRuntimeHook(BaseHook):
    HOOK_CLASS_NAME = "TairKVCMTransferRuntime"
    HOOK_MODULE_NAME = "v6d.common.tair_kvcm.transfer"

    @classmethod
    def hook(cls, target):
        if target.__dict__.get("_sim_tair_kvcm_transfer_hook", False):
            return

        # Skip the native client: it needs a local tair_mempool_server (TENT on CPU
        # hosts), otherwise init fails with "kvcm_py_client.TransferClient.Create failed"
        # and v6d silently stops reporting BLOCK_ADD / writing to KVCM.
        def create_transfer_client(self):
            return _NoopTransferClient()

        def batch_load(self, uris, pointers, sizes):
            if not (len(uris) == len(pointers) == len(sizes)):
                raise ValueError("uris, pointers, and sizes length mismatch")
            if not uris:
                return True
            from v6d.common.tair_kvcm.transfer import _record_transfer_result

            _simulate_kvcm_io(uris, sizes, write=False)
            _record_transfer_result("read", True)
            return True

        def batch_save(self, uris, pointers, sizes):
            if not (len(uris) == len(pointers) == len(sizes)):
                raise ValueError("uris, pointers, and sizes length mismatch")
            if not uris:
                return []
            from v6d.common.tair_kvcm.transfer import _record_transfer_result

            _simulate_kvcm_io(uris, sizes, write=True)
            _record_transfer_result("write", True)
            return [True] * len(uris)

        target._create_transfer_client = create_transfer_client
        target.batch_load_to_memory = batch_load
        target.batch_save_from_memory = batch_save
        target._sim_tair_kvcm_transfer_hook = True
        logger.info(
            "[v6d-sim] C_TairKVCMTransferRuntimeHook installed: "
            "native KVCM transfer client stubbed; tiered MemPool I/O modeling active"
        )
