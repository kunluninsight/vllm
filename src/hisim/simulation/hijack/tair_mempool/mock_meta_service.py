#!/usr/bin/env python3
"""
mock_meta.py -- PACE (Tair Mempool) meta service mock: bookkeeping only, no data is moved.

Sits behind KVCM (closed-source TairMempoolBackend) in place of a Mempool MetaService plus its
Providers. GA allocation / free / query / sync copy and DRAM->SSD demotion only update in-process
ledgers: no memory is reserved, no RDMA is needed, only the Python 3.8+ standard library is used.

Architecture (mirrors the real split between the two processes)
  Provider   one virtual tair_mempool_server, in mock_provider.py: GAManager (GA ledger per
             segment), TieringManager (transparent DRAM->SSD demotion), GETSTATUS report,
             provider-side copy checks. Knows nothing about other Providers or about placement.
  Meta       tair_mempool_meta_service: node registry, WithAuth, GaAllocConfig, GlobalAddressService
             (placement, batch alloc with rollback, free routing by GA segment), copy dispatch,
             memnode view. Talks to Providers only through their public methods (the "RPC" boundary)
             and keeps no GA ledger of its own, like the real service.
  MockPace   test harness around both: creates / reconfigures Providers, drives their WaterMarkLoop,
             fault injection, counters, ledger snapshots, mock-only /admin routes.
  REST/HTTP  request parsing and response formatting identical to meta_restful_service.cpp.

Protocol sources (each behaviour is checked against these; deviations are listed at the end)
  tair-mempool  src/pace_meta_service.cpp                  command-line options
                docker/start_metaservice_byenv.sh          env based start-up
                src/meta_service/meta_restful_service.cpp  routes, auth, request parsing, responses
                include/meta_service/meta_restful_service.h   SetErrorResponse / SetJsonResultResponse
                src/meta_service/global_address_service.cpp   AllocGA / AllocBatchGAs / FreeGAs / GetNodesToAlloc
                include/meta_service/global_address_service.h ResolveMediaPolicyFromJson
                include/meta_service/ga_alloc_config.h     GaAllocConfig fields / env / patch validation
                src/meta_service/node_management_service.cpp  GetNodeMemStatus / CopyGA / QueryBatchCopyGA
                src/listener/meta_restful_server.cpp       provider-side free / copy_ga checks
                src/address_mgmt/{tiering_manager,dram_slot_pool,ga_manager,ga_meta_operator}.cpp
  KVCacheManager internal_source/kv_cache_manager/data_storage/pace_service_client.cc  (the caller)

Implemented routes (same paths and auth as the real Meta)
  GET    /v1/api/healthy                                no auth
  POST   /v1/api/admin_user, DELETE /v1/api/admin_user  Basic accounts (default admin:123)
  GET    /v1/api/memnode[?id=|nodeid=|ip=]
  POST   /v1/api/nodes                                    consumer registration
  POST   /v1/api/gas, POST /v1/api/gas/batch            allocate
  GET    /v1/api/gas[?ga=&provider_id=&status=]         query
  DELETE /v1/api/gas[?ga=a,b | ?id=|nodeid=|ip=]        free
  POST   /v1/api/copy_ga/sync, POST /v1/api/copy_ga/batch_sync
  GET|POST /v1/api/admin/ga_alloc_config, POST /v1/api/admin/ga_alloc_config/reset
  Other real routes (node deletion/query / segment / async copy_ga / tiering admin ...) are not implemented: they
  return 404 and are recorded in unknown_requests of /admin/stats.

Mock admin routes (not in the real service, no auth)
  GET /admin/stats | /admin/nodes | /admin/fault | /admin/ga?ga=<ga>
  POST /admin/node | /admin/fault | /admin/reset | /admin/snapshot ; DELETE /admin/fault[?endpoint=]
  Fault endpoint keys: healthy memnode gas_batch gas_create gas_get gas_delete copy_batch_sync
  copy_sync ga_alloc_config admin_user, "*" for all; rule
  {delay_ms: [min, max], fail_ratio: 0~1, fail_mode: status|http|drop|hang, http_code: 500, hang_s: 30}

Start-up (launch_server.py owns the command line, env expansion, logging and background threads)
  Same command line as tair_mempool_meta_service (-p defaults to 12348; -o / -L take effect, the
  rest is accepted and ignored), or --byenv to read TAIR_MEMPOOL_META_* like start_metaservice_byenv.sh
  (TAIR_MEMPOOL_META_LOG_PATH defaults to /tmp/meta.log).
  Placement tuning only reads the real env (TAIR_MEMPOOL_GA_* etc., see GA_INT_FIELDS) and can be
  changed at runtime through ga_alloc_config. Virtual Providers are described by mock options
  --nodes/--dram/--ssd/--tiering-mode or node_specs in --config, also as env HISIM_TAIR_MEMPOOL_<OPTION>;
  their shape and tiering knobs are additionally taken from the real start_server_byenv.sh env
  (TAIR_MEMPOOL_SERVER_LOCAL_MEMORY_SIZE -s, SSD_SIZE_MB, ENABLE_TRANSPARENT_TIERING,
  PUSH_ONLY_MODE, TIERED_*), so one deployment env can drive both the real server and the mock
  (precedence: default < --config < TAIR_MEMPOOL_* < HISIM_TAIR_MEMPOOL_* < command line).

  python3 -m hisim.simulation.hijack.tair_mempool.launch_server --nodes 3 --dram 64GB --ssd 512GB
  python3 -m hisim.simulation.hijack.tair_mempool.launch_server -p 12348 -b redis -l redis -o /tmp/meta.log
  HISIM_TAIR_MEMPOOL_NODES=8 python3 -m hisim.simulation.hijack.tair_mempool.launch_server --byenv
  python3 test/test_tair_mempool_mocker.py        # self test (separate script)

KVCM caveat (follows from the real semantics)
  - A request with media_type=2 and no media_policy is treated as AUTO by the real Meta; AUTO on a
    transparent node answers media_type=0, while KVCM requires the answer to match a configured
    media_type. So do not set --media_type on pace storages backed by transparent nodes.
    media_type=5 (pace_ssd) can only land on explicit nodes.

Provider tiering (transparent nodes)
  Watermarks, WaterMarkLoop victim selection, demotion and its back-off, and the admission rules
  that Meta placement never triggers: see mock_provider.py.

Deviations from the real services (cannot or need not be reproduced)
  - No real Provider process: alloc / demote / copy only touch ledgers; Provider RPC failures are
    simulated with /admin/fault. Meta reads live Provider status instead of the periodic GETSTATUS
    cache, so there is no staleness window.
  - Node ordering: capacity_first sorts by order_free descending (as the real one); round_robin /
    random are approximations; weighted_k_stripe / dram_ssd_aware behave as capacity_first. Batch
    alloc places GAs one by one in rounds of per_node_alloc_limit, without DoBatchAllocOnNodes'
    ascending fallback.
  - Demote IO completes instantly (the real worker keeps DRAM until the SSD write finishes); there
    is no read path, so last_access is only set at alloc time and there is no promote / DUAL; no lazy
    materialize, shm_meta / cache segment, SSD slot footprint rounding or BFC fragmentation
    (BFC reports requested bytes anyway); SSD writes only fail for lack of capacity.
  - Copy tasks finish synchronously, without pending/copying/draining; an unhealthy node fails the
    task right away.
  - ga_alloc_config has no shared store: persistence_supported=false (same as -b memory).
  - The 404 body of unregistered routes is mock specific.
"""

import base64
import collections
import ipaddress
import json
import os
import random
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import urllib.request

from hisim.simulation.hijack.tair_mempool.mock_provider import (
    LOG,
    MEDIA_AUTO,
    MEDIA_DRAM,
    MEDIA_SSD,
    PROVIDER_DEFAULT_WATERMARK_PCT,
    TIERS,
    ApiError,
    Provider,
    TieringParams,
    ga_media,
    ga_segment,
    parse_size,
)

VERSION = "mock-pace-3.1 (tair_mempool_meta_service 3.1.21 compatible)"
STATE_FORMAT = 2

# MediaPolicy -> media layers tried in order (global_address_service.cpp MediaLayers)
MEDIA_LAYERS = {
    "auto": [MEDIA_AUTO],
    "prefer_dram": [MEDIA_DRAM, MEDIA_SSD],
    "prefer_ssd": [MEDIA_SSD, MEDIA_DRAM],
    "only_dram": [MEDIA_DRAM],
    "only_ssd": [MEDIA_SSD],
}

NODE_STATUS_HEALTHY = 3  # NodeStatusCode::NODE_STATUS_HEALTHY
NODE_STATUS_CONNECT_FAILED = 5  # NodeStatusCode::NODE_STATUS_CONNECT_FAILED

MAX_REQUEST_LINE = 4 * 1024 * 1024  # DELETE /v1/api/gas?ga=... can be very long
MAX_PACE_NODE_ID = 0xFFFF  # KVCM stores node_id as uint16

# ga_alloc_config.h GaAllocConfigIntFields(): (key, env, min, max, default)
GA_INT_FIELDS = (
    (
        "per_node_alloc_limit",
        "TAIR_MEMPOOL_BATCH_ALLOC_CNT_LIMIT_PER_NODE",
        1,
        1000000,
        10,
    ),
    ("stripe_kmin", "TAIR_MEMPOOL_GA_STRIPE_KMIN", 1, 100000, 2),
    ("stripe_kmax", "TAIR_MEMPOOL_GA_STRIPE_KMAX", 0, 100000, 0),
    ("stripe_kmax_divisor", "TAIR_MEMPOOL_GA_STRIPE_KMAX_DIVISOR", 1, 100000, 16),
    ("stripe_kmax_floor", "TAIR_MEMPOOL_GA_STRIPE_KMAX_FLOOR", 1, 100000, 8),
    ("candidate_shuffle", "TAIR_MEMPOOL_GA_CANDIDATE_SHUFFLE", 0, 1, 1),
    (
        "tiering_capacity_watermark_pct",
        "TAIR_MEMPOOL_TIERING_CAPACITY_WATERMARK_PCT",
        1,
        100,
        95,
    ),
    ("dram_gate", "TAIR_MEMPOOL_GA_DRAM_GATE", 0, 1, 1),
    ("ssd_prefer_ratio_pct", "TAIR_MEMPOOL_GA_SSD_PREFER_RATIO_PCT", 1, 100, 95),
    ("ssd_alloc_max_ratio_pct", "TAIR_MEMPOOL_GA_SSD_ALLOC_MAX_RATIO_PCT", 1, 100, 90),
    ("ssd_prefer_alloc_limit", "TAIR_MEMPOOL_GA_SSD_PREFER_ALLOC_LIMIT", 0, 1000000, 1),
    ("ga_mutation_timeout_s", "TAIR_MEMPOOL_GA_MUTATION_TIMEOUT_S", 1, 120, 5),
    ("server_client_timeout_s", "TAIR_MEMPOOL_META_SERVER_CLIENT_TIMEOUT_S", 1, 120, 1),
    (
        "copy_ga_reconcile_timeout_ms",
        "TAIR_MEMPOOL_COPY_GA_RECONCILE_TIMEOUT_MS",
        1000,
        3600000,
        120000,
    ),
    (
        "copy_ga_sync_timeout_ms",
        "TAIR_MEMPOOL_COPY_GA_SYNC_TIMEOUT_MS",
        1,
        3600000,
        300000,
    ),
)
GA_INT_SPEC = {f[0]: f for f in GA_INT_FIELDS}
POLICY_ALIASES = {
    "capacity_first": "capacity_first",
    "capacity": "capacity_first",
    "round_robin": "round_robin",
    "rr": "round_robin",
    "random": "random",
    "weighted_k_stripe": "weighted_k_stripe",
    "wks": "weighted_k_stripe",
    "dram_ssd_aware": "dram_ssd_aware",
    "mem_ssd": "dram_ssd_aware",
}


# ===================================================================== common helpers


def fmt_size(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024 or unit == "PiB":
            return "%.1f%s" % (n, unit) if unit != "B" else "%d%s" % (n, unit)
        n /= 1024.0
    return str(n)


def wm_free(total, used, pct):
    """Free bytes under a watermark (same formula as SsdWatermarkFreeBytes / DramWatermarkFreeBytes)."""
    cap = total * pct // 100
    return cap - used if cap > used else 0


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_json_int(v):
    """nlohmann is_number_integer: an integer that is not a bool."""
    return isinstance(v, int) and not isinstance(v, bool)


def failed(msg):
    """ErrorRspData(msg): HTTP 500 once rendered by SetJsonResultResponse."""
    return {"status": "failed", "data": msg}


class TextResponse(object):
    def __init__(self, http_code, text):
        self.http_code = http_code
        self.text = text


# ======================================================== meta: ga_alloc_config
class GaAllocConfigStore(object):
    """Equivalent of GaAllocConfigStore: env baseline -> runtime patch -> published snapshot."""

    def __init__(self, env):
        self.env = env
        self.lock = threading.Lock()
        self.env_baseline = self.load_from_env()
        self.runtime_patch = {}
        self.comment = ""
        self.version = 0
        self.cur = dict(self.env_baseline, version=0, updated_at="", comment="")

    def _multi_active(self):
        v = self.env.get("TAIR_MEMPOOL_META_GA_LOCAL_SERVE", "")
        return not (v == "off" or v == "0")

    def load_from_env(self):
        cfg = {"node_select_policy": "capacity_first", "inflight_adjust": True}
        for key, env_name, lo, hi, default in GA_INT_FIELDS:
            cfg[key] = default
        policy = self.env.get("TAIR_MEMPOOL_GA_NODE_SELECT_POLICY", "")
        if policy:
            if policy in POLICY_ALIASES:
                cfg["node_select_policy"] = POLICY_ALIASES[policy]
            else:
                LOG.warning(
                    "Invalid TAIR_MEMPOOL_GA_NODE_SELECT_POLICY '%s', fallback to %s",
                    policy,
                    cfg["node_select_policy"],
                )
        for key, env_name, lo, hi, default in GA_INT_FIELDS:
            raw = self.env.get(env_name, "")
            if not raw:
                continue
            try:
                parsed = int(raw, 10)
            except ValueError:
                parsed = None
            if parsed is None or not lo <= parsed <= hi:
                LOG.warning(
                    "Invalid %s='%s' (expect integer in [%d, %d]), keep %d",
                    env_name,
                    raw,
                    lo,
                    hi,
                    default,
                )
                continue
            cfg[key] = parsed
        if cfg["tiering_capacity_watermark_pct"] > PROVIDER_DEFAULT_WATERMARK_PCT:
            LOG.warning(
                "TAIR_MEMPOOL_TIERING_CAPACITY_WATERMARK_PCT=%d is above Provider default %d",
                cfg["tiering_capacity_watermark_pct"],
                PROVIDER_DEFAULT_WATERMARK_PCT,
            )
        raw = self.env.get("TAIR_MEMPOOL_GA_INFLIGHT_ADJUST", "")
        if raw:
            if raw in ("0", "1"):
                cfg["inflight_adjust"] = raw == "1"
            else:
                LOG.warning(
                    "Invalid TAIR_MEMPOOL_GA_INFLIGHT_ADJUST '%s', keep %d",
                    raw,
                    cfg["inflight_adjust"],
                )
        if self._multi_active() and not cfg["inflight_adjust"]:
            LOG.warning(
                "multi-active GA local-serve mode requires inflight accounting, forcing inflight_adjust on"
            )
            cfg["inflight_adjust"] = True
        if cfg["node_select_policy"] == "dram_ssd_aware":
            if not cfg["inflight_adjust"]:
                LOG.warning(
                    "dram_ssd_aware requires inflight accounting, forcing inflight_adjust on"
                )
                cfg["inflight_adjust"] = True
            if cfg["dram_gate"] == 0:
                LOG.warning(
                    "dram_ssd_aware requires the DRAM watermark gate, forcing dram_gate on"
                )
                cfg["dram_gate"] = 1
        return cfg

    def _apply_patch(self, base, patch):
        """ApplyGaAllocConfigPatch: returns (cfg, err)."""
        if not isinstance(patch, dict):
            return None, "config patch shall be a JSON object"
        nxt = dict(base)
        for key in patch:
            if (
                key not in ("node_select_policy", "inflight_adjust")
                and key not in GA_INT_SPEC
            ):
                return None, "unknown config key '%s'" % key
        if "node_select_policy" in patch:
            v = patch["node_select_policy"]
            if not isinstance(v, str) or v not in POLICY_ALIASES:
                return None, (
                    "invalid node_select_policy (expect capacity_first / round_robin / random"
                    " / weighted_k_stripe / dram_ssd_aware)"
                )
            nxt["node_select_policy"] = POLICY_ALIASES[v]
        for key, _env, lo, hi, _default in GA_INT_FIELDS:
            if key not in patch:
                continue
            v = patch[key]
            if not is_json_int(v):
                return None, "invalid %s: expect integer" % key
            if v < lo or v > hi:
                return None, "invalid %s: %d out of range [%d, %d]" % (key, v, lo, hi)
            nxt[key] = v
        if (
            "tiering_capacity_watermark_pct" in patch
            and nxt["tiering_capacity_watermark_pct"] > PROVIDER_DEFAULT_WATERMARK_PCT
        ):
            LOG.warning(
                "tiering_capacity_watermark_pct set to %d, above Provider default %d",
                nxt["tiering_capacity_watermark_pct"],
                PROVIDER_DEFAULT_WATERMARK_PCT,
            )
        if "inflight_adjust" in patch:
            if not isinstance(patch["inflight_adjust"], bool):
                return None, "invalid inflight_adjust: expect boolean"
            nxt["inflight_adjust"] = patch["inflight_adjust"]
        if self._multi_active() and not nxt["inflight_adjust"]:
            return None, (
                "inflight_adjust cannot be disabled under multi-active GA local-serve mode "
                "(TAIR_MEMPOOL_META_GA_LOCAL_SERVE != off): multiple instances each account "
                "inflight independently, disabling it causes M-fold over-subscription"
            )
        if nxt["node_select_policy"] == "dram_ssd_aware" and not nxt["inflight_adjust"]:
            return None, (
                "inflight_adjust cannot be disabled under dram_ssd_aware"
                " (its DRAM watermark admission depends on inflight accounting)"
            )
        if nxt["node_select_policy"] == "dram_ssd_aware" and nxt["dram_gate"] == 0:
            return None, (
                "dram_gate cannot be disabled under dram_ssd_aware"
                " (DRAM watermark admission is half of the policy's semantics)"
            )
        return nxt, None

    def _publish(self, cfg):
        self.version += 1
        self.cur = dict(
            cfg, version=self.version, updated_at=now_utc(), comment=self.comment
        )

    def get(self):
        with self.lock:
            return self.cur

    def update(self, patch, comment):
        with self.lock:
            merged = dict(self.runtime_patch)
            merged.update(patch)
            nxt, err = self._apply_patch(self.env_baseline, merged)
            if err:
                return None, err
            self.runtime_patch = merged
            if comment:
                self.comment = comment
            self._publish(nxt)
            return self._view_locked(), None

    def reset(self):
        baseline = self.load_from_env()
        with self.lock:
            self.env_baseline = baseline
            self.runtime_patch = {}
            self.comment = ""
            self._publish(baseline)
            return self._view_locked()

    def view(self):
        with self.lock:
            return self._view_locked()

    def _view_locked(self):
        """BuildGaAllocConfigView."""

        def to_json(cfg):
            out = {
                "node_select_policy": cfg["node_select_policy"],
                "inflight_adjust": cfg["inflight_adjust"],
            }
            for key, *_rest in GA_INT_FIELDS:
                out[key] = cfg[key]
            return out

        sources = {
            "node_select_policy": "runtime"
            if "node_select_policy" in self.runtime_patch
            else "env",
            "inflight_adjust": "runtime"
            if "inflight_adjust" in self.runtime_patch
            else "env",
        }
        limits = {}
        for key, env_name, lo, hi, _default in GA_INT_FIELDS:
            sources[key] = "runtime" if key in self.runtime_patch else "env"
            limits[key] = {"min": lo, "max": hi, "env": env_name}
        return {
            "config": to_json(self.cur),
            "env_baseline": to_json(self.env_baseline),
            "runtime_patch": dict(self.runtime_patch),
            "sources": sources,
            "limits": limits,
            "version": self.cur["version"],
            "updated_at": self.cur["updated_at"],
            "comment": self.cur["comment"],
            "status": "success",
        }


# ============================================================================= meta
def copy_task_http_status(task):
    """CopyTaskHttpStatus."""
    state = task.get("state", "")
    if state == "timed_out_drained":
        return 504
    if state == "cancelled_drained":
        return 409
    if state != "failed":
        return 200
    provider = task.get("provider_http_status", 0)
    if 400 <= provider < 500:
        return provider
    if provider >= 500:
        return 502
    return 500


class Meta(object):
    """tair_mempool_meta_service: node registry, auth, GaAllocConfig, GlobalAddressService and copy.

    Holds no GA ledger: GAs are routed to their Provider by segment_id, listings and frees are
    delegated to Providers, and placement reads each Provider's status() report. Lock order is
    always Meta.lock -> Provider.lock.
    """

    def __init__(self, auth, env):
        self.env = env
        self.lock = threading.RLock()
        self.ga_cfg = GaAllocConfigStore(env)
        self.providers = {}  # pace_node_id -> Provider
        self.consumers = {}  # node UUID -> consumer registration
        self.users = {}
        for item in auth or []:
            user, _, password = item.partition(":")
            self.users[user] = password
        self.rr = 0
        self.copy_seq = 0
        self.shutting_down = False
        self.stats = collections.Counter()
        try:
            self.max_ga_size = int(env.get("TAIR_MEMPOOL_MAX_GA_SIZE", "") or 0)
        except ValueError:
            LOG.warning("ParseSize: invalid TAIR_MEMPOOL_MAX_GA_SIZE, using default 0")
            self.max_ga_size = 0

    # ---------------------------------------------------------------------- auth
    def auth_error(self, header):
        """WithAuth: None when accepted, otherwise the 401 text body."""
        with self.lock:
            users = dict(self.users)
        if not users:
            return None
        if not header:
            return "Missing Authorization header\n"
        if header.startswith("Basic "):
            try:
                user, _, password = base64.b64decode(header[6:]).decode().partition(":")
            except Exception:
                user, password = "", ""
            if user and users.get(user) == password:
                return None
        return "Unauthorized: Invalid admin token\n"

    def add_user(self, user, password):
        with self.lock:
            self.users[user] = password

    def remove_user(self, user):
        with self.lock:
            self.users.pop(user, None)

    # ------------------------------------------------------------------ registry
    def next_pace_id(self):
        with self.lock:
            pid = max(self.providers) + 1 if self.providers else 1
            if pid > MAX_PACE_NODE_ID:
                raise ApiError(400, "pace_node_id exceeds uint16")
            return pid

    def register(self, provider):
        with self.lock:
            pid = provider.pace_id
            if not 1 <= pid <= MAX_PACE_NODE_ID:
                raise ApiError(
                    400, "pace_node_id must be within [1, %d]" % MAX_PACE_NODE_ID
                )
            if pid in self.providers:
                raise ApiError(409, "pace_node_id %d already exists" % pid)
            self.providers[pid] = provider
            return provider

    @staticmethod
    def _send_cluster_info(node, payload):
        url = "http://%s:%d/api/clusterinfo" % (node["ip"], node["port"])
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read() or b"{}")
        if result.get("status") != "success":
            raise ApiError(500, "RegisterNodes failed")
        status_url = "http://%s:%d/api/status" % (node["ip"], node["port"])
        with urllib.request.urlopen(status_url, timeout=5) as response:
            status = json.loads(response.read() or b"{}")
        return status.get("data", {})

    def register_consumers(self, nodes, sender=None):
        """Register native consumers and push their local identity, without virtual Provider RPCs."""
        if not isinstance(nodes, list) or not nodes:
            raise ApiError(400, "Empty node list")
        sender = sender or self._send_cluster_info
        for node in nodes:
            if not isinstance(node, dict):
                raise ApiError(400, "invalid JSON request")
            if node.get("type") != "consumer":
                raise ApiError(400, "mock only supports external consumer registration")
            if (
                not all(
                    isinstance(node.get(key), str) and node[key] for key in ("id", "ip")
                )
                or not is_json_int(node.get("port"))
                or not 0 < node["port"] < 65536
            ):
                raise ApiError(400, "invalid JSON request")
            if node.get("tiering_mode", "off") != "off":
                raise ApiError(
                    400, "consumer tiering_mode must match mock Provider mode off"
                )
            with self.lock:
                existing = self.consumers.get(node["id"])
                if existing is not None:
                    pace_id = existing["pace_node_id"]
                else:
                    used = set(self.providers) | {
                        c["pace_node_id"] for c in self.consumers.values()
                    }
                    pace_id = max(used | {0}) + 1
                    if pace_id > MAX_PACE_NODE_ID:
                        raise ApiError(400, "pace_node_id exceeds uint16")
            payload = {
                "nodes": [node["ip"]],
                "ids": [pace_id],
                "types": ["consumer"],
                "version": 1,
            }
            try:
                reported = sender(node, payload) or {}
            except ApiError:
                raise
            except Exception as exc:
                raise ApiError(500, "RegisterNodes failed: %s" % exc)
            with self.lock:
                self.consumers[node["id"]] = dict(
                    node,
                    pace_node_id=pace_id,
                    healthy=True,
                    status=NODE_STATUS_HEALTHY,
                    total_bytes=reported.get("total_bytes", 0),
                    used_bytes=reported.get("used_bytes", 0),
                    alloced_ga_num=reported.get("alloced_ga_num", 0),
                    ssd_total_bytes=reported.get("ssd_total_bytes", 0),
                    ssd_used_bytes=reported.get("ssd_used_bytes", 0),
                    tiering_enabled=reported.get("tiering_enabled", False),
                    tiering_mode=reported.get("tiering_mode", "off"),
                )

    def unregister(self, pace_id):
        with self.lock:
            provider = self.provider(pace_id)
            del self.providers[provider.pace_id]
            return provider

    def provider(self, pace_id):
        with self.lock:
            try:
                return self.providers[int(pace_id)]
            except (KeyError, ValueError, TypeError):
                raise ApiError(404, "pace_node_id %r not found" % (pace_id,))

    def provider_list(self):
        with self.lock:
            return [p for _, p in sorted(self.providers.items())]

    def nodes_by_params(self, query, prefix=""):
        """ParseNodeinParamsFromCache: id = pace id list / nodeid = uuid list / ip list; None without params."""
        providers = self.provider_list()
        raw = _first(query, prefix + "id")
        if raw is not None:
            ids = set()
            for x in raw.split(","):
                try:
                    ids.add(int(x))
                except ValueError:
                    LOG.warning("str2int: invalid argument '%s'", x)
            return [p for p in providers if p.pace_id in ids]
        raw = _first(query, prefix + "nodeid")
        if raw is not None:
            ids = set(raw.split(","))
            return [p for p in providers if p.uid in ids]
        raw = _first(query, prefix + "ip")
        if raw is not None:
            ips = set(raw.split(","))
            return [p for p in providers if p.ip in ips]
        return None

    def _provider_by_segment(self, segment_id):
        provider = self.providers.get(segment_id >> 4)
        if provider is None or (segment_id & 0xF) not in TIERS:
            return None
        return provider

    # ----------------------------------------------------------------- placement
    @staticmethod
    def _layer_free(st, layer, g):
        """layer_free of GetNodesToAlloc from a Provider status report; None if the layer is not served."""
        dram_total, dram_used = st["total_bytes"], st["used_bytes"]
        ssd_total, ssd_used = st["ssd_total_bytes"], st["ssd_used_bytes"]
        mode, enabled = st["tiering_mode"], st["tiering_enabled"]
        dram_free = max(dram_total - dram_used, 0)
        wm = g["tiering_capacity_watermark_pct"]
        if layer == MEDIA_AUTO:
            if g["dram_gate"]:
                return wm_free(dram_total, dram_used, wm)
            if mode == "explicit":
                return dram_free
            if dram_used > dram_total:
                return 0
            dram_wm_free = wm_free(dram_total, dram_used, wm)
            if not enabled:
                return dram_wm_free
            return max(dram_wm_free, wm_free(ssd_total, ssd_used, wm))
        if mode == "transparent":
            return None
        if mode == "off":
            return dram_free if layer == MEDIA_DRAM else None
        if layer != MEDIA_DRAM and not enabled:
            return None
        return dram_free if layer == MEDIA_DRAM else max(ssd_total - ssd_used, 0)

    @staticmethod
    def _order_free(st, layer_free, g):
        """OrderFreeUnderDramGate."""
        if (
            g["dram_gate"]
            and st["tiering_mode"] == "transparent"
            and st["tiering_enabled"]
        ):
            return max(
                layer_free,
                wm_free(
                    st["ssd_total_bytes"],
                    st["ssd_used_bytes"],
                    g["tiering_capacity_watermark_pct"],
                ),
            )
        return layer_free

    @staticmethod
    def _ssd_blocked(st, layer, g):
        """ShouldBlockSsdAllocation."""
        if (
            layer != MEDIA_AUTO
            or st["tiering_mode"] != "transparent"
            or not st["tiering_enabled"]
        ):
            return False
        pct = g["ssd_alloc_max_ratio_pct"]
        if pct >= 100:
            return False
        total = st["ssd_total_bytes"]
        return total == 0 or wm_free(total, st["ssd_used_bytes"], pct) == 0

    def _candidates(self, size, layer, g):
        """[(provider, layer_free)] eligible for one GA of `size` on `layer`, ordered by policy."""
        eligible = []
        for pid in sorted(self.providers):
            provider = self.providers[pid]
            if not provider.healthy:
                continue
            st = provider.status()
            layer_free = self._layer_free(st, layer, g)
            if layer_free is None or layer_free < size:
                continue
            if self._ssd_blocked(st, layer, g):
                self.stats["ssd_capped_skips"] += 1
                continue
            eligible.append((self._order_free(st, layer_free, g), layer_free, provider))
        if g["candidate_shuffle"]:
            random.shuffle(eligible)
        policy = g["node_select_policy"]
        if policy == "random":
            random.shuffle(eligible)
        elif policy == "round_robin":
            eligible.sort(key=lambda e: e[2].pace_id)
            if eligible:
                k = self.rr % len(eligible)
                eligible = eligible[k:] + eligible[:k]
            self.rr += 1
        else:  # capacity_first (weighted_k_stripe / dram_ssd_aware approximated)
            eligible.sort(key=lambda e: e[0], reverse=True)
        return [(provider, layer_free) for _, layer_free, provider in eligible]

    @staticmethod
    def _ordered(cands, preferred, strict):
        """Affinity: preferred matches first, in hint order; strict keeps only those."""
        if not preferred:
            return cands, False
        pref, seen = [], set()
        for hint in preferred:
            hint = str(hint)
            for provider, lf in cands:
                if provider.pace_id not in seen and hint in (
                    provider.uid,
                    str(provider.pace_id),
                    provider.ip,
                ):
                    pref.append((provider, lf))
                    seen.add(provider.pace_id)
        matched = bool(pref)
        if strict:
            return pref, matched
        rest = [c for c in cands if c[0].pace_id not in seen]
        return pref + rest, matched

    @staticmethod
    def _item(ga, size, provider, shown_media):
        return {
            "offset": ga,
            "size": size,
            "node_id": provider.pace_id,
            "media_type": shown_media,
            "range_id": 0,
        }

    @staticmethod
    def _shown_media(ga, layer, provider):
        """AUTO on a transparent node answers media_type=0, otherwise the GA's segment media."""
        return (
            MEDIA_AUTO
            if (layer == MEDIA_AUTO and provider.tiering_mode == "transparent")
            else ga_media(ga)
        )

    def _check_alloc_args(self, size, preferred, strict):
        if self.max_ga_size > 0 and size > self.max_ga_size:
            return failed(
                "Create GA failed: size %d exceeds TAIR_MEMPOOL_MAX_GA_SIZE limit %d"
                % (size, self.max_ga_size)
            )
        if strict and not preferred:
            return failed("Strict affinity: preferred_nodes shall not be empty")
        return None

    # ------------------------------------------------------ GlobalAddressService
    def alloc_one(self, size, policy, preferred, prefer_policy):
        """GlobalAddressService::AllocGA; the result's status decides the HTTP code."""
        g = self.ga_cfg.get()
        if size <= 0:
            return failed("Create GA failed, GA size shall be positive")
        strict = prefer_policy == "strict"
        err = self._check_alloc_args(size, preferred, strict)
        if err:
            return err
        with self.lock:
            self.stats["alloc_requests"] += 1
            attempted = 0
            for layer in MEDIA_LAYERS[policy]:
                cands, _ = self._ordered(
                    self._candidates(size, layer, g), preferred, strict
                )
                for provider, _lf in cands:
                    attempted += 1
                    ga = provider.alloc(layer, size)
                    if ga is None:
                        continue
                    self.stats["alloc_gas"] += 1
                    self.stats["alloc_bytes"] += size
                    return {
                        "status": "success",
                        "data": self._item(
                            ga, size, provider, self._shown_media(ga, layer, provider)
                        ),
                    }
            self.stats["alloc_failed_requests"] += 1
            if strict and attempted == 0:
                return failed("Strict affinity: no matching node available")
            if attempted == 0:
                if not any(p.healthy for p in self.providers.values()):
                    return failed(
                        "Create GA failed: no healthy provider node available"
                    )
                return failed(
                    "Create GA failed: insufficient memory for request size %d" % size
                )
            return failed(
                "Create GA failed: tried %d provider(s) but all failed, last error: "
                "server_alloc_failed: insufficient memory on this node" % attempted
            )

    def alloc_batch(self, size, count, policy, preferred, prefer_policy):
        """GlobalAddressService::AllocBatchGAs: all or nothing, rolled back on failure."""
        g = self.ga_cfg.get()
        if size <= 0 or count <= 0:
            return failed("Create GA failed, GA size and count shall be positive")
        strict = prefer_policy == "strict"
        err = self._check_alloc_args(size, preferred, strict)
        if err:
            return err
        with self.lock:
            self.stats["alloc_requests"] += 1
            layers = MEDIA_LAYERS[policy]
            observed = [c for layer in layers for c in self._candidates(size, layer, g)]
            if not observed:
                self.stats["alloc_failed_requests"] += 1
                if not self.providers:
                    return failed("Create GA failed: no provider nodes registered")
                if not any(p.healthy for p in self.providers.values()):
                    return failed(
                        "Create GA failed: no healthy provider node available"
                    )
                return failed(
                    "Create GA failed: insufficient capacity for requested media"
                )
            cluster_allocatable = sum(lf for _, lf in observed)
            if count > cluster_allocatable // size and strict:
                self.stats["alloc_failed_requests"] += 1
                return failed(
                    "oversubscribe: requested count*size exceeds cluster allocatable capacity "
                    "(%d bytes within watermark)" % cluster_allocatable
                )
            items, matched_any = [], False
            limit = g["per_node_alloc_limit"]
            for layer in layers:
                remaining = count - len(items)
                if remaining <= 0:
                    break
                progress = True
                # per round, each node takes at most min(per_node_alloc_limit, layer_free / size)
                while remaining > 0 and progress:
                    progress = False
                    cands, matched = self._ordered(
                        self._candidates(size, layer, g), preferred, strict
                    )
                    matched_any = matched_any or matched
                    for provider, layer_free in cands:
                        for _ in range(min(limit, remaining, layer_free // size)):
                            ga = provider.alloc(layer, size)
                            if ga is None:
                                break
                            items.append((ga, provider, layer))
                            remaining -= 1
                            progress = True
                        if remaining == 0:
                            break
            if len(items) < count:
                for ga, provider, _layer in items:
                    provider.free(ga)
                self.stats["alloc_failed_requests"] += 1
                if strict:
                    return failed(
                        "Strict affinity: insufficient capacity on preferred nodes"
                        if matched_any
                        else "Strict affinity: no matching node available"
                    )
                return failed("Create GA failed")
            data = [
                self._item(ga, size, provider, self._shown_media(ga, layer, provider))
                for ga, provider, layer in items
            ]
            self.stats["alloc_gas"] += len(data)
            self.stats["alloc_bytes"] += len(data) * size
            return {"status": "success", "data": data}

    def free_gas(self, gas):
        """FreeGAs: route by segment owner; unknown segments and CONNECT_FAILED nodes are skipped;
        a Provider freeing an unknown GA succeeds (ga_server_operator.cpp Free, not-in-ga_list_)."""
        with self.lock:
            self.stats["free_requests"] += 1
            for ga in gas:
                provider = self._provider_by_segment(ga_segment(ga))
                if provider is None or not provider.healthy:
                    if provider is not None:
                        self.stats["free_skipped_unhealthy"] += 1
                    continue
                size = provider.free(ga)
                if size is None:
                    self.stats["free_not_found"] += 1
                else:
                    self.stats["freed_gas"] += 1
                    self.stats["freed_bytes"] += size

    def free_nodes(self, providers):
        """FreeAllGAsOnNodes (CONNECT_FAILED nodes included)."""
        with self.lock:
            return [
                {"ip": p.ip, "pace_node_id": p.pace_id, "ga_count": p.free_all()}
                for p in providers
            ]

    def list_gas(self):
        """GetAllGAs: snapshots of unhealthy nodes fail and are reported in failed_nodes."""
        gas, failed_nodes = [], []
        for provider in self.provider_list():
            if not provider.healthy:
                failed_nodes.append(provider.pace_id)
                continue
            gas.extend((ga, provider.pace_id) for ga in provider.list_gas())
        return gas, failed_nodes

    def ga_count(self):
        return sum(p.ga_count() for p in self.provider_list())

    def describe_ga(self, ga):
        with self.lock:
            provider = self._provider_by_segment(ga_segment(ga))
        return None if provider is None else provider.describe(ga)

    # ---------------------------------------------------------------------- copy
    def _copy_task(self, req, index, task_id, src, dst):
        """One copy as a CopyTaskToJson-shaped task; the checks run on the source Provider."""
        task = {
            "task_id": task_id,
            "state": "success",
            "src_ga": req["src_ga"],
            "dst_ga": req["dst_ga"],
            "size": req["size"],
            "src_offset": req["src_offset"],
            "dst_offset": req["dst_offset"],
            "src_media_type": req["src_media_type"],
            "dst_media_type": req["dst_media_type"],
            "index": index,
            "provider_http_status": 200,
            "need_link": False,
            "error": "",
            "timeout_ms": req["timeout_ms"],
            "cancel_requested": False,
            "provider_copy_started": True,
            "provider_rpc_returned": True,
            "provider_work_finished": True,
            "provider_request_id": "",
            "provider_incarnation_id": "",
            "drained": True,
            "safe_to_reuse_dst": True,
            "terminal": True,
            "terminal_reason": "",
            "outcome": "success",
        }
        if not src.healthy or not dst.healthy:
            task.update(
                provider_copy_started=False,
                provider_rpc_returned=False,
                provider_work_finished=False,
                state="failed",
                provider_http_status=0,
                outcome="failed",
                error="provider unreachable (mock: node marked unhealthy)",
            )
            return task
        http_status, error = src.copy_to(req, dst)
        if http_status != 200:
            task.update(
                state="failed",
                provider_http_status=http_status,
                error=error,
                outcome="failed",
            )
        return task

    def _next_copy_id(self, src, dst, suffix=""):
        self.copy_seq += 1
        return "copy_%d_%d_%d%s_%d" % (
            int(time.monotonic() * 1000),
            src.pace_id,
            dst.pace_id,
            suffix,
            self.copy_seq,
        )

    def _count_copy(self, tasks):
        self.stats["copy_tasks"] += len(tasks)
        self.stats["copy_failed"] += sum(1 for t in tasks if t["state"] != "success")

    def copy_sync(self, req):
        """SyncCopyGA -> QueryCopyGA: {"status","data":task}; returns (result, http)."""
        with self.lock:
            src = self._provider_by_segment(ga_segment(req["src_ga"]))
            dst = self._provider_by_segment(ga_segment(req["dst_ga"]))
            if src is None or dst is None:
                return failed("src or dst node not found"), 400
            task = self._copy_task(req, 0, self._next_copy_id(src, dst), src, dst)
            self._count_copy([task])
        result = {
            "status": "success" if task["state"] == "success" else "failed",
            "data": task,
        }
        return result, copy_task_http_status(task)

    def copy_batch_sync(self, copies):
        """SyncBatchCopyGA -> QueryBatchCopyGA; returns (result, http)."""
        with self.lock:
            pairs = []
            for c in copies:
                src_seg, dst_seg = ga_segment(c["src_ga"]), ga_segment(c["dst_ga"])
                src, dst = (
                    self._provider_by_segment(src_seg),
                    self._provider_by_segment(dst_seg),
                )
                if src is None or dst is None:
                    return failed(
                        "node not found: src_seg=%d dst_seg=%d" % (src_seg, dst_seg)
                    ), 400
                pairs.append((src, dst))
            self.copy_seq += 1
            batch = self.copy_seq
            tasks = [
                self._copy_task(c, i, self._next_copy_id(s, d, "_b%d" % batch), s, d)
                for i, (c, (s, d)) in enumerate(zip(copies, pairs))
            ]
            self._count_copy(tasks)
        completed = sum(1 for t in tasks if t["state"] == "success")
        failed_n = len(tasks) - completed
        summary = {
            "total": len(tasks),
            "completed": completed,
            "failed": failed_n,
            "pending": 0,
            "running": 0,
            "cancelled": 0,
            "timed_out": 0,
            "unknown": 0,
        }
        result = {
            "status": "success",
            "data": {
                "tasks": tasks,
                "summary": summary,
                "outcome": "failed" if failed_n else "success",
            },
        }
        return result, 200

    # --------------------------------------------------------------------- views
    def memnode_view(self, providers=None):
        """GetNodeMemStatus built from Provider reports and registered consumer capacity."""
        providers = self.provider_list() if providers is None else providers
        memnodes = {}
        total = used = gas = 0
        with self.lock:
            consumers = list(self.consumers.values())
        for consumer in consumers:
            total += consumer["total_bytes"]
            used += consumer["used_bytes"]
            gas += consumer["alloced_ga_num"]
            memnodes[str(consumer["pace_node_id"])] = {
                "node": {
                    "id": consumer["id"],
                    "ip": consumer["ip"],
                    "port": consumer["port"],
                    "type": "consumer",
                },
                "pace_node_id": consumer["pace_node_id"],
                "healthy": consumer["healthy"],
                "status": consumer["status"],
                "peer_registered": True,
                "allocatable": False,
                "total_bytes": consumer["total_bytes"],
                "used_bytes": consumer["used_bytes"],
                "alloced_ga_num": consumer["alloced_ga_num"],
                "ssd_total_bytes": consumer["ssd_total_bytes"],
                "ssd_used_bytes": consumer["ssd_used_bytes"],
                "tiering_enabled": consumer["tiering_enabled"],
                "tiering_mode": consumer["tiering_mode"],
                "tiering_mode_mismatch": False,
            }
        for p in providers:
            st = p.status()
            total += st["total_bytes"]
            used += st["used_bytes"]
            gas += st["alloced_ga_num"]
            # peer_registered: no consumers in the mock, so CheckRegisterStatus is always true
            entry = {
                "node": {"id": p.uid, "ip": p.ip, "port": p.port, "type": "provider"},
                "pace_node_id": p.pace_id,
                "healthy": p.healthy,
                "status": NODE_STATUS_HEALTHY
                if p.healthy
                else NODE_STATUS_CONNECT_FAILED,
                "peer_registered": True,
                "allocatable": p.healthy,
                "total_bytes": st["total_bytes"],
                "used_bytes": st["used_bytes"],
                "alloced_ga_num": st["alloced_ga_num"],
                "ssd_total_bytes": st["ssd_total_bytes"],
                "ssd_used_bytes": st["ssd_used_bytes"],
                "tiering_enabled": st["tiering_enabled"],
                "tiering_mode": st["tiering_mode"],
                "tiering_mode_mismatch": False,
            }
            if st["segment"]:
                entry["segment"] = st["segment"]
            memnodes[str(p.pace_id)] = entry
        result = {
            "mem_usage_stat": {
                "total_bytes": total,
                "used_bytes": used,
                "alloced_ga_num": gas,
            },
            "status": "success",
        }
        if memnodes:
            result["memnodes"] = memnodes
        return result

    def capacity(self):
        """Cluster DRAM / SSD totals over healthy Providers (mock stats only)."""
        out = collections.Counter()
        for p in self.provider_list():
            if not p.healthy:
                continue
            st = p.status()
            out["dram_total_bytes"] += st["total_bytes"]
            out["dram_used_bytes"] += st["used_bytes"]
            out["dram_ga_num"] += st["alloced_ga_num"]
            out["ssd_total_bytes"] += st["ssd_total_bytes"]
            out["ssd_used_bytes"] += st["ssd_used_bytes"]
            out["ssd_ga_num"] += sum(
                s["alloced_ga_num"]
                for s in st["segment"]
                if s["media_type"] == MEDIA_SSD
            )
        return {
            k: out[k]
            for k in (
                "dram_total_bytes",
                "dram_used_bytes",
                "dram_ga_num",
                "ssd_total_bytes",
                "ssd_used_bytes",
                "ssd_ga_num",
            )
        }

    def reset_counters(self):
        with self.lock:
            self.rr = 0
            self.stats.clear()


# ========================================================================== harness
class Faults(object):
    """Mock only: per-endpoint delay and failure injection; the "*" rule provides defaults."""

    MODES = ("status", "http", "drop", "hang")

    def __init__(self):
        self._lock = threading.Lock()
        self._rules = {}

    def set(self, endpoint, rule):
        clean = {}
        if "delay_ms" in rule:
            d = rule["delay_ms"]
            lo, hi = (d, d) if isinstance(d, (int, float)) else (d[0], d[1])
            clean["delay_ms"] = [float(lo), float(hi)]
        if "fail_ratio" in rule:
            ratio = float(rule["fail_ratio"])
            if not 0.0 <= ratio <= 1.0:
                raise ApiError(400, "fail_ratio must be within [0, 1]")
            clean["fail_ratio"] = ratio
        if "fail_mode" in rule:
            if rule["fail_mode"] not in self.MODES:
                raise ApiError(400, "fail_mode must be one of %s" % (self.MODES,))
            clean["fail_mode"] = rule["fail_mode"]
        for key in ("http_code", "hang_s"):
            if key in rule:
                clean[key] = int(rule[key])
        with self._lock:
            self._rules.setdefault(endpoint, {}).update(clean)

    def clear(self, endpoint=None):
        with self._lock:
            if endpoint is None:
                self._rules.clear()
            else:
                self._rules.pop(endpoint, None)

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(self._rules))

    def decide(self, key):
        with self._lock:
            rule = dict(self._rules.get("*", {}))
            rule.update(self._rules.get(key, {}))
        delay = rule.get("delay_ms")
        if delay and (delay[0] > 0 or delay[1] > 0):
            time.sleep(random.uniform(delay[0], delay[1]) / 1000.0)
        ratio = rule.get("fail_ratio", 0.0)
        if ratio > 0 and random.random() < ratio:
            return {
                "mode": rule.get("fail_mode", "status"),
                "http_code": rule.get("http_code", 500),
                "hang_s": rule.get("hang_s", 30),
            }
        return None


class MockPace(object):
    """Harness wiring one Meta and its virtual Providers.

    Plays the roles the real deployment gets from outside the services: starting / resizing /
    killing Providers, their WaterMarkLoop clock, fault injection, request counters and ledger
    snapshots. REST handlers reach the services through `self.meta` only.
    Lock order: MockPace.lock -> Meta.lock -> Provider.lock.
    """

    def __init__(self, cfg, env=None):
        self.cfg = cfg
        self.env = os.environ if env is None else env
        self.lock = threading.RLock()
        self.faults = Faults()
        if cfg.delay_ms:
            self.faults.set("*", {"delay_ms": cfg.delay_ms})
        self.tiering = TieringParams(cfg, self.env)
        self.meta = Meta(cfg.auth, self.env)
        self.started_at = time.time()
        self.stats = collections.Counter()
        self.endpoint_stats = collections.Counter()
        self.unknown_requests = collections.deque(maxlen=100)

    # ---------------------------------------------------------------- providers
    def add_provider(
        self,
        dram,
        ssd,
        ip=None,
        port=None,
        uid=None,
        healthy=True,
        pace_id=None,
        tiering_mode=None,
    ):
        with self.lock:
            pid = int(pace_id) if pace_id is not None else self.meta.next_pace_id()
            if not 1 <= pid <= MAX_PACE_NODE_ID:
                raise ApiError(
                    400, "pace_node_id must be within [1, %d]" % MAX_PACE_NODE_ID
                )
            ip = ip or str(ipaddress.ip_address(self.cfg.node_ip_base) + pid - 1)
            provider = Provider(
                pid,
                parse_size(dram),
                parse_size(ssd),
                ip,
                port or self.cfg.node_port,
                tiering_mode or self.cfg.tiering_mode,
                self.tiering,
                uid=uid,
                healthy=healthy,
            )
            self.meta.register(provider)
        LOG.info(
            "provider added: pace_node_id=%d ip=%s mode=%s dram=%s ssd=%s healthy=%s",
            pid,
            provider.ip,
            provider.tiering_mode,
            fmt_size(provider.tiers[MEDIA_DRAM].total),
            fmt_size(provider.tiers[MEDIA_SSD].total),
            provider.healthy,
        )
        return provider

    def update_provider(
        self, pace_id, healthy=None, dram=None, ssd=None, tiering_mode=None
    ):
        provider = self.meta.provider(pace_id)
        provider.reconfigure(dram=dram, ssd=ssd, tiering_mode=tiering_mode)
        if healthy is not None:
            provider.healthy = bool(healthy)
        LOG.info("provider updated: %s", provider.to_json())
        return provider

    def remove_provider(self, pace_id):
        """Kill a Provider: it leaves the registry and its GAs are gone. Returns the GA count dropped."""
        provider = self.meta.unregister(pace_id)
        dropped = provider.free_all()
        LOG.info(
            "provider removed: pace_node_id=%d dropped_gas=%d",
            provider.pace_id,
            dropped,
        )
        return dropped

    def tiering_tick_all(self):
        """Advance every Provider's WaterMarkLoop by one iteration."""
        for provider in self.meta.provider_list():
            provider.tiering_tick()

    # ------------------------------------------------------------- accounting
    def count_endpoint(self, key):
        with self.lock:
            self.endpoint_stats[key] += 1

    def count_fault(self, key):
        with self.lock:
            self.stats["injected_faults"] += 1
            self.endpoint_stats["fault:" + key] += 1

    def record_unknown(self, entry):
        with self.lock:
            self.unknown_requests.append(entry)
            self.endpoint_stats["unknown"] += 1

    def provider_counters(self):
        total = collections.Counter()
        for provider in self.meta.provider_list():
            with provider.lock:
                total.update(provider.stats)
        return total

    # ------------------------------------------------------------------ views
    def nodes_view(self):
        return [
            dict(p.to_json(), tiering=p.tiering_state(), counters=dict(p.stats))
            for p in self.meta.provider_list()
        ]

    def stats_view(self):
        cap = self.meta.capacity()
        providers = self.meta.provider_list()
        with self.meta.lock:
            meta_counters = dict(self.meta.stats)
        with self.lock:
            harness = {
                "counters": dict(self.stats),
                "endpoints": dict(self.endpoint_stats),
                "unknown_requests": list(self.unknown_requests),
            }
        return {
            "status": "success",
            "version": VERSION,
            "uptime_s": round(time.time() - self.started_at, 1),
            "node_count": len(providers),
            "healthy_node_count": sum(1 for p in providers if p.healthy),
            "ga_count": self.meta.ga_count(),
            "capacity": cap,
            "payload_bytes_moved": 0,
            "capacity_human": {
                k: fmt_size(v) for k, v in cap.items() if k.endswith("_bytes")
            },
            "ga_alloc_config": self.meta.ga_cfg.get(),
            "tiering": self.tiering.to_json(),
            "counters": {
                "meta": meta_counters,
                "provider": dict(self.provider_counters()),
                "harness": harness["counters"],
            },
            "endpoints": harness["endpoints"],
            "faults": self.faults.snapshot(),
            "unknown_requests": harness["unknown_requests"],
            "config": {
                k: v
                for k, v in vars(self.cfg).items()
                if k not in ("auth", "backend_args")
            },
        }

    def summary_line(self):
        cap = self.meta.capacity()
        with self.meta.lock:
            m = collections.Counter(self.meta.stats)
        prov = self.provider_counters()
        return (
            "gas=%d dram=%s/%s ssd=%s/%s alloc_req=%d alloc_fail=%d free_req=%d demoted=%d copy=%d unknown=%d"
            % (
                self.meta.ga_count(),
                fmt_size(cap["dram_used_bytes"]),
                fmt_size(cap["dram_total_bytes"]),
                fmt_size(cap["ssd_used_bytes"]),
                fmt_size(cap["ssd_total_bytes"]),
                m["alloc_requests"],
                m["alloc_failed_requests"],
                m["free_requests"],
                prov["demoted_gas"],
                m["copy_tasks"],
                len(self.unknown_requests),
            )
        )

    # ------------------------------------------------------------ persistence
    def save(self, path=None):
        path = path or self.cfg.state_file
        if not path:
            raise ApiError(400, "state file not configured (--state-file)")
        with self.meta.lock:
            state = {
                "version": VERSION,
                "format": STATE_FORMAT,
                "rr": self.meta.rr,
                "providers": [p.dump() for p in self.meta.provider_list()],
            }
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, separators=(",", ":"))
        os.replace(tmp, path)
        LOG.info(
            "state saved: %s (providers=%d, gas=%d)",
            path,
            len(state["providers"]),
            sum(len(p["gas"]) for p in state["providers"]),
        )
        return path

    def load(self, path):
        """Load a snapshot; format 1 (global "ledger" of [ga, pace_id, media, size]) is still read."""
        with open(path) as fh:
            state = json.load(fh)
        if "providers" in state:
            specs = state["providers"]
            gas_of = {
                s["pace_node_id"]: (s.get("seq", 0), s.get("gas", [])) for s in specs
            }
        else:
            specs, gas_of, seq = state.get("nodes", []), {}, state.get("seq", 0)
            for ga, pid, media, size in state.get("ledger", []):
                gas_of.setdefault(pid, (seq, []))[1].append([ga, media, size])
        with self.lock, self.meta.lock:
            for provider in self.meta.provider_list():
                self.meta.unregister(provider.pace_id)
            for item in specs:
                provider = self.add_provider(
                    item["dram"],
                    item["ssd"],
                    ip=item.get("ip"),
                    port=item.get("port"),
                    uid=item.get("id"),
                    healthy=item.get("healthy", True),
                    pace_id=item["pace_node_id"],
                    tiering_mode=item.get("tiering_mode"),
                )
                seq, gas = gas_of.get(provider.pace_id, (0, []))
                provider.restore(seq, gas, item.get("sim_written", []))
            self.meta.rr = int(state.get("rr", 0))
        LOG.info(
            "state loaded: %s (providers=%d, gas=%d)",
            path,
            len(specs),
            self.meta.ga_count(),
        )

    def reset(self, rebuild_nodes=False):
        with self.lock, self.meta.lock:
            self.meta.reset_counters()
            if rebuild_nodes:
                for provider in self.meta.provider_list():
                    self.meta.unregister(provider.pace_id)
                build_nodes(self, self.cfg)
            else:
                for provider in self.meta.provider_list():
                    provider.clear()
            self.stats.clear()


# ============================================================================= REST
def _first(query, name):
    values = query.get(name)
    if values and values[0] != "":
        return values[0]
    return None


def _parse_json(body):
    """ParseRequestJson / ParseRequestBody: invalid JSON -> 400 "invalid JSON request"."""
    try:
        return json.loads(body or b"")
    except ValueError:
        raise ApiError(400, "invalid JSON request")


def _uint_field(obj, key, default=0):
    """NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT: default when absent, wrong type -> invalid JSON request."""
    if key not in obj:
        return default
    v = obj[key]
    if not is_json_int(v) or v < 0:
        raise ApiError(400, "invalid JSON request")
    return v


def _resolve_media_policy(req):
    """ResolveMediaPolicyFromJson: returns (policy, error)."""
    policy, error = "auto", None
    has_policy, has_media = "media_policy" in req, "media_type" in req
    if has_policy:
        v = req["media_policy"]
        if isinstance(v, str) and v in MEDIA_LAYERS:
            policy = v
        else:
            error = "invalid media_policy"
    if has_media:
        v = req["media_type"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return policy, "media_type must be a number"
        media = int(v) & 0xFF
        if media == MEDIA_DRAM:
            legacy = "only_dram"
        elif media == MEDIA_SSD:
            legacy = "only_ssd"
        else:
            error, legacy = "legacy media_type must be 2 or 5", "auto"
        if error is None and has_policy and policy != legacy:
            error = "media_policy conflicts with media_type"
        elif not has_policy:
            # media_type=2 alone means AUTO (v1 clients serialize DRAM by default); 5 stays strict SSD
            policy = "auto" if media == MEDIA_DRAM else legacy
    return policy, error


def _alloc_request(body, batch):
    req = _parse_json(body)
    if not isinstance(req, dict):
        raise ApiError(400, "invalid JSON request")
    try:
        size = req["size"]
        count = req["count"] if batch else 1
    except KeyError:
        raise ApiError(400, "invalid JSON request")
    if not is_json_int(size) or not is_json_int(count):
        raise ApiError(400, "invalid JSON request")
    policy, error = _resolve_media_policy(req)
    preferred = (
        req.get("preferred_nodes")
        if isinstance(req.get("preferred_nodes"), list)
        else []
    )
    prefer_policy = (
        req.get("prefer_policy") if isinstance(req.get("prefer_policy"), str) else ""
    )
    return size, count, policy, error, [str(x) for x in preferred], prefer_policy


def _parse_ga_list(raw):
    out = []
    for x in raw.split(","):
        try:
            out.append(int(x))
        except ValueError:
            continue
    return out


def _copy_request(obj):
    if not isinstance(obj, dict):
        raise ApiError(400, "invalid JSON request")
    req = {
        k: _uint_field(obj, k)
        for k in (
            "src_ga",
            "dst_ga",
            "size",
            "src_offset",
            "dst_offset",
            "src_ga_size",
            "dst_ga_size",
            "src_media_type",
            "dst_media_type",
            "timeout_ms",
        )
    }
    req["refresh_segment"] = obj.get("refresh_segment", "if_missing")
    return req


def _validate_copy(r):
    """ValidateCopyGARequest."""
    if r["size"] == 0:
        return "size must be > 0"
    if (
        r["src_offset"] > (1 << 64) - 1 - r["size"]
        or r["dst_offset"] > (1 << 64) - 1 - r["size"]
    ):
        return "offset + size overflow"
    if r["src_ga_size"] > 0 and r["src_offset"] + r["size"] > r["src_ga_size"]:
        return "src_offset + size exceeds src_ga_size"
    if r["dst_ga_size"] > 0 and r["dst_offset"] + r["size"] > r["dst_ga_size"]:
        return "dst_offset + size exceeds dst_ga_size"
    return None


def json_result(result, override=None):
    """SetJsonResultResponse."""
    if override is not None:
        return override, result
    return (200 if result.get("status") == "success" else 500), result


# ---- MetaService routes: handlers get the harness and only use app.meta
def r_healthy(app, query, body):
    if app.meta.shutting_down:
        return TextResponse(500, "Server is shutting down")
    return 200, {"status": "success"}


def r_admin_user_add(app, query, body):
    req = _parse_json(body)
    if not isinstance(req, dict):
        raise ApiError(400, "invalid JSON request")
    user, password = str(req.get("username", "")), str(req.get("password", ""))
    if not user:
        raise ApiError(400, "Empty user token")
    if not password:
        raise ApiError(400, "Empty user password")
    app.meta.add_user(user, password)
    return 200, {"status": "success"}


def r_admin_user_del(app, query, body):
    user = _first(query, "username")
    if not user:
        raise ApiError(400, "Empty user token")
    app.meta.remove_user(user)
    return 200, {"status": "success"}


def r_nodes_register(app, query, body):
    app.meta.register_consumers(
        _parse_json(body), sender=getattr(app, "consumer_sender", None)
    )
    return 200, {"status": "success"}


def r_memnode(app, query, body):
    return json_result(app.meta.memnode_view(app.meta.nodes_by_params(query)))


def r_gas_create(app, query, body):
    size, _count, policy, error, preferred, prefer_policy = _alloc_request(
        body, batch=False
    )
    if size <= 0:
        raise ApiError(400, "GA size shall be positive")
    if error:
        raise ApiError(400, error)
    return json_result(app.meta.alloc_one(size, policy, preferred, prefer_policy))


def r_gas_batch(app, query, body):
    size, count, policy, error, preferred, prefer_policy = _alloc_request(
        body, batch=True
    )
    if size <= 0 or count <= 0:
        raise ApiError(400, "GA size and count shall be positive")
    if error:
        raise ApiError(400, error)
    return json_result(
        app.meta.alloc_batch(size, count, policy, preferred, prefer_policy)
    )


def r_gas_get(app, query, body):
    ga_raw = _first(query, "ga")
    wanted = set(_parse_ga_list(ga_raw)) if ga_raw else None
    provider_nodes = app.meta.nodes_by_params(query, "provider_")
    provider_ids = (
        None if provider_nodes is None else {p.pace_id for p in provider_nodes}
    )
    status_raw = _first(query, "status")
    status_set = None
    if status_raw:
        status_set = {
            s
            for s in status_raw.split(",")
            if s in ("NORMAL", "WRITE_NOT_ALLOWED", "INVALID")
        }
    gas, failed_nodes = app.meta.list_gas()
    result = {}
    for ga, pid in gas:
        if provider_ids is not None and pid not in provider_ids:
            continue
        if wanted is not None and ga not in wanted:
            continue
        if status_set is not None and "NORMAL" not in status_set:
            continue
        result[str(ga)] = {"provider_id": pid, "status": "NORMAL", "addr": ga}
    result.update(
        {
            "status": "success",
            "partial": bool(failed_nodes),
            "failed_nodes": failed_nodes,
        }
    )
    return 200, result


def r_gas_delete(app, query, body):
    ga_raw = _first(query, "ga")
    if ga_raw is None:
        providers = app.meta.nodes_by_params(query)
        if providers is None:
            LOG.warning("HandleDeleteGAs input is empty, which means clear all gas")
            providers = app.meta.provider_list()
        elif not providers:
            raise ApiError(400, "No GA found to delete, please check input params")
        nodes_info = app.meta.free_nodes(providers)
        return json_result(
            {
                "status": "success",
                "total": sum(n["ga_count"] for n in nodes_info),
                "nodes": nodes_info,
            }
        )
    gas = _parse_ga_list(ga_raw)
    if not gas:
        raise ApiError(400, "No GA found to delete, please check input params")
    app.meta.free_gas(gas)
    return 200, {"status": "success"}


def r_copy_sync(app, query, body):
    req = _copy_request(_parse_json(body))
    err = _validate_copy(req)
    if err:
        raise ApiError(400, err)
    result, http = app.meta.copy_sync(req)
    return json_result(result, http)


def r_copy_batch_sync(app, query, body):
    raw = _parse_json(body)
    if not isinstance(raw, dict) or not isinstance(raw.get("copies", []), list):
        raise ApiError(400, "invalid JSON request")
    copies = [_copy_request(c) for c in raw.get("copies", [])]
    if not copies:
        raise ApiError(400, "copies must not be empty")
    for i, c in enumerate(copies):
        err = _validate_copy(c)
        if err:
            raise ApiError(400, "copies[%d]: %s" % (i, err))
    result, http = app.meta.copy_batch_sync(copies)
    return json_result(result, http)


def r_ga_cfg_get(app, query, body):
    view = app.meta.ga_cfg.view()
    view["is_leader"] = True
    view["persistence_supported"] = False
    return json_result(view)


def r_ga_cfg_update(app, query, body):
    raw = _parse_json(body)
    if not isinstance(raw, dict):
        raise ApiError(400, "request body shall be a JSON object")
    comment = ""
    if "comment" in raw:
        if not isinstance(raw["comment"], str):
            raise ApiError(400, "comment shall be a string")
        comment = raw["comment"]
    if "patch" in raw:
        patch = raw["patch"]
        if not isinstance(patch, dict):
            raise ApiError(400, "patch shall be a JSON object")
    else:
        patch = {k: v for k, v in raw.items() if k != "comment"}
    if not patch:
        raise ApiError(400, "empty config patch, nothing to update")
    view, err = app.meta.ga_cfg.update(patch, comment)
    if err:
        LOG.warning(
            "HandleUpdateGaAllocConfig rejected: %s, patch: %s", err, json.dumps(patch)
        )
        raise ApiError(400, err)
    LOG.info(
        "GaAllocConfig updated by REST, version: %d, patch: %s, comment: %s",
        view["version"],
        json.dumps(patch),
        comment,
    )
    view.update(
        {
            "applied": patch,
            "persistence_supported": False,
            "persisted": False,
            "warning": "backend has no shared store, change is memory-only and lost on restart",
        }
    )
    return json_result(view)


def r_ga_cfg_reset(app, query, body):
    view = app.meta.ga_cfg.reset()
    LOG.info(
        "GaAllocConfig reset to env baseline, version: %d, persisted: 0",
        view["version"],
    )
    view.update({"persistence_supported": False, "persisted": False})
    return json_result(view)


# ---- mock admin routes (harness)
def _admin_body(body, required=True):
    if not body:
        if required:
            raise ApiError(400, "empty request body")
        return {}
    return _parse_json(body)


def r_admin_stats(app, query, body):
    return 200, app.stats_view()


def r_admin_nodes(app, query, body):
    return 200, {"status": "success", "nodes": app.nodes_view()}


def r_admin_node(app, query, body):
    req = _admin_body(body)
    action = req.get("action", "update")
    if action == "add":
        count = int(req.get("count", 1))
        created = [
            app.add_provider(
                req.get("dram", app.cfg.dram),
                req.get("ssd", app.cfg.ssd),
                ip=req.get("ip") if count == 1 else None,
                port=req.get("port"),
                healthy=req.get("healthy", True),
                pace_id=req.get("pace_node_id") if count == 1 else None,
                tiering_mode=req.get("tiering_mode"),
            ).to_json()
            for _ in range(count)
        ]
        return 200, {"status": "success", "nodes": created}
    if action == "update":
        provider = app.update_provider(
            req.get("pace_node_id"),
            healthy=req.get("healthy"),
            dram=req.get("dram"),
            ssd=req.get("ssd"),
            tiering_mode=req.get("tiering_mode"),
        )
        return 200, {"status": "success", "node": provider.to_json()}
    if action == "remove":
        return 200, {
            "status": "success",
            "dropped_gas": app.remove_provider(req.get("pace_node_id")),
        }
    raise ApiError(400, "action must be add|update|remove")


def r_admin_fault_get(app, query, body):
    return 200, {"status": "success", "faults": app.faults.snapshot()}


def r_admin_fault_set(app, query, body):
    req = _admin_body(body)
    endpoint = req.pop("endpoint", "*")
    if endpoint != "*" and endpoint not in FAULT_KEYS:
        raise ApiError(
            400, "unknown endpoint %r, valid: %s" % (endpoint, sorted(FAULT_KEYS))
        )
    app.faults.set(endpoint, req)
    return 200, {"status": "success", "faults": app.faults.snapshot()}


def r_admin_fault_clear(app, query, body):
    app.faults.clear(_first(query, "endpoint"))
    return 200, {"status": "success", "faults": app.faults.snapshot()}


def r_admin_reset(app, query, body):
    app.reset(
        rebuild_nodes=bool(
            _admin_body(body, required=False).get("rebuild_nodes", False)
        )
    )
    return 200, {"status": "success"}


def r_admin_snapshot(app, query, body):
    return 200, {
        "status": "success",
        "path": app.save(_admin_body(body, required=False).get("path")),
    }


def r_simulated_transfer(app, query, body, write):
    """Mock-only SDK boundary; records readability, not payload contents."""
    request = _admin_body(body)
    items = request.get("items") if isinstance(request, dict) else None
    if not isinstance(items, list) or len(items) > 4096:
        raise ApiError(400, "items must be an array with at most 4096 entries")
    for item in items:
        if (
            not isinstance(item, dict)
            or not is_json_int(item.get("ga"))
            or item["ga"] <= 0
            or not is_json_int(item.get("size"))
            or item["size"] <= 0
        ):
            raise ApiError(400, "each item requires positive integer ga and size")
    results = []
    with app.lock, app.meta.lock:
        for item in items:
            provider = app.meta._provider_by_segment(ga_segment(item["ga"]))
            result = (
                {
                    "ga": item["ga"],
                    "success": False,
                    "reason": "provider not registered",
                }
                if provider is None
                else provider.simulate_io(item["ga"], item["size"], write)
            )
            results.append(result)
    return 200, {"status": "success", "results": results, "payload_bytes_moved": 0}


def r_simulated_save(app, query, body):
    return r_simulated_transfer(app, query, body, True)


def r_simulated_load(app, query, body):
    return r_simulated_transfer(app, query, body, False)


def r_admin_ga(app, query, body):
    raw = _first(query, "ga")
    if not raw:
        raise ApiError(400, "ga is required")
    return 200, {
        "status": "success",
        "gas": {str(ga): app.meta.describe_ga(ga) for ga in _parse_ga_list(raw)},
    }


# (method, path) -> (endpoint key, handler, needs WithAuth)
ROUTES = {
    ("GET", "/v1/api/healthy"): ("healthy", r_healthy, False),
    ("POST", "/v1/api/admin_user"): ("admin_user", r_admin_user_add, True),
    ("DELETE", "/v1/api/admin_user"): ("admin_user", r_admin_user_del, True),
    ("GET", "/v1/api/memnode"): ("memnode", r_memnode, True),
    ("POST", "/v1/api/nodes"): ("nodes_register", r_nodes_register, True),
    ("POST", "/v1/api/gas"): ("gas_create", r_gas_create, True),
    ("POST", "/v1/api/gas/batch"): ("gas_batch", r_gas_batch, True),
    ("GET", "/v1/api/gas"): ("gas_get", r_gas_get, True),
    ("DELETE", "/v1/api/gas"): ("gas_delete", r_gas_delete, True),
    ("POST", "/v1/api/copy_ga/sync"): ("copy_sync", r_copy_sync, True),
    ("POST", "/v1/api/copy_ga/batch_sync"): (
        "copy_batch_sync",
        r_copy_batch_sync,
        True,
    ),
    ("GET", "/v1/api/admin/ga_alloc_config"): ("ga_alloc_config", r_ga_cfg_get, True),
    ("POST", "/v1/api/admin/ga_alloc_config"): (
        "ga_alloc_config",
        r_ga_cfg_update,
        True,
    ),
    ("POST", "/v1/api/admin/ga_alloc_config/reset"): (
        "ga_alloc_config",
        r_ga_cfg_reset,
        True,
    ),
    ("GET", "/admin/stats"): ("admin", r_admin_stats, False),
    ("GET", "/admin/nodes"): ("admin", r_admin_nodes, False),
    ("POST", "/admin/node"): ("admin", r_admin_node, False),
    ("GET", "/admin/fault"): ("admin", r_admin_fault_get, False),
    ("POST", "/admin/fault"): ("admin", r_admin_fault_set, False),
    ("DELETE", "/admin/fault"): ("admin", r_admin_fault_clear, False),
    ("POST", "/admin/reset"): ("admin", r_admin_reset, False),
    ("POST", "/admin/snapshot"): ("admin", r_admin_snapshot, False),
    ("GET", "/admin/ga"): ("admin", r_admin_ga, False),
    ("POST", "/admin/transfer/save"): ("transfer_save", r_simulated_save, False),
    ("POST", "/admin/transfer/load"): ("transfer_load", r_simulated_load, False),
}
FAULT_KEYS = {key for key, _, _ in ROUTES.values() if key != "admin"}


# ============================================================================= HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # KVCM's curl reuses connections
    server_version = "MockPace/3.1"

    def handle_one_request(self):
        # Same as the stdlib, only with a larger request-line limit (DELETE with long GA lists)
        try:
            self.raw_requestline = self.rfile.readline(MAX_REQUEST_LINE + 1)
            if len(self.raw_requestline) > MAX_REQUEST_LINE:
                self.requestline = self.request_version = self.command = ""
                self.send_error(414)
                return
            if not self.raw_requestline:
                self.close_connection = True
                return
            if not self.parse_request():
                return
            method = getattr(self, "do_" + self.command, None)
            if method is None:
                self.send_error(501, "Unsupported method (%r)" % self.command)
                return
            method()
            self.wfile.flush()
        except (socket.timeout, ConnectionError) as exc:
            LOG.debug("connection closed: %s", exc)
            self.close_connection = True

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PUT(self):
        self._dispatch("PUT")

    def log_message(self, fmt, *args):
        LOG.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, code, payload, content_type="application/json"):
        data = (
            payload.encode()
            if isinstance(payload, str)
            else json.dumps(payload, separators=(",", ":")).encode()
        )
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, method):
        app = self.server.app
        started = time.monotonic()
        parts = urlsplit(self.path)
        path = parts.path.rstrip("/") or "/"
        query = parse_qs(parts.query, keep_blank_values=True)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        route = ROUTES.get((method, path))
        if route is None:
            entry = {
                "ts": round(time.time(), 3),
                "method": method,
                "path": self.path[:512],
                "trace_id": self.headers.get("TraceId", ""),
                "body": body[:2048].decode("utf-8", "replace"),
            }
            app.record_unknown(entry)
            LOG.warning(
                "UNKNOWN request %s %s body=%s",
                method,
                self.path[:512],
                entry["body"][:256],
            )
            self._send(
                404,
                {
                    "status": "failed",
                    "data": "not implemented by mock: %s %s" % (method, path),
                },
            )
            return
        key, handler, needs_auth = route
        app.count_endpoint(key)
        if needs_auth:
            err = app.meta.auth_error(self.headers.get("Authorization"))
            if err:
                self._send(401, err, "text/plain")
                return
        if key != "admin":
            action = app.faults.decide(key)
            if action is not None:
                self._apply_fault(app, key, action)
                return
        try:
            out = handler(app, query, body)
        except ApiError as exc:
            out = (exc.http_code, {"status": "failed", "data": exc.message})
        except (
            Exception
        ) as exc:  # last resort: one bad request must not take the server down
            LOG.exception("handler %s crashed", key)
            out = TextResponse(500, "Internal Server Error: %s" % exc)
        if isinstance(out, TextResponse):
            code = out.http_code
            self._send(code, out.text, "text/plain")
        else:
            code, payload = out
            if code != 200 and key in ("gas_batch", "gas_create"):
                LOG.warning("%s failed (%d): %s", key, code, payload.get("data"))
            self._send(code, payload)
        if app.cfg.access_log:
            LOG.info(
                "%s %s -> %d %.1fms trace=%s",
                method,
                self.path[:160],
                code,
                (time.monotonic() - started) * 1000.0,
                self.headers.get("TraceId", ""),
            )

    def _apply_fault(self, app, key, action):
        app.count_fault(key)
        mode = action["mode"]
        if mode == "status":
            self._send(200, {"status": "failed", "data": "injected fault"})
        elif mode == "http":
            self._send(
                action["http_code"], {"status": "failed", "data": "injected fault"}
            )
        elif mode == "hang":
            time.sleep(action["hang_s"])
            self.close_connection = True
        else:  # drop
            self.close_connection = True


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 512

    def __init__(self, addr, app):
        super().__init__(addr, Handler)
        self.app = app

    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), whose reverse lookup can stall for seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def main(argv=None):
    """Kept so `python -m ...mock_meta_service` still works; startup lives in launch_server."""
    from hisim.simulation.hijack.tair_mempool.launch_server import main as launch_main

    return launch_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
