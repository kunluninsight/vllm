"""KVCM preflight checks: zombie instance detection and quota fold sizing.

Run before multi-instance KVCM benchmarks to verify that stale instance_ids
from previous runs are not pinning group quota (GROUP_LRU does not evict
stopped namespaces) and to compute the folded simulation quota.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from urllib.request import urlopen

_METRIC_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?\s+'
    r'(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


def parse_prometheus_metrics(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse Prometheus exposition lines into (name, labels, float_value)."""
    rows: list[tuple[str, dict[str, str], float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        rows.append((m.group("name"), labels, float(m.group("value"))))
    return rows


def check_kvcm_clean(
    metrics_text: str,
    *,
    group_name: str | None = None,
    active_instance_ids: set[str] | None = None,
    max_group_usage_ratio: float = 0.05,
) -> dict:
    """Inspect KVCM Prometheus metrics for zombie namespaces and warm residue."""
    rows = parse_prometheus_metrics(metrics_text)
    active = set(active_instance_ids or ())
    group_usage: dict[str, float] = {}
    zombie_instances: dict[str, float] = {}
    active_bytes: dict[str, float] = {}

    for name, labels, value in rows:
        grp = labels.get("instance_group") or labels.get("group") or ""
        if group_name and grp and grp != group_name:
            continue
        if name == "kvcm_cache_manager_group_usage_ratio":
            group_usage[grp or "<default>"] = value
        elif name == "kvcm_cache_manager_instance_byte_size":
            iid = labels.get("instance_id") or ""
            if not iid:
                continue
            if iid in active:
                active_bytes[iid] = value
            elif value > 0:
                zombie_instances[iid] = value

    issues: list[str] = []
    for grp, ratio in sorted(group_usage.items()):
        if ratio > max_group_usage_ratio:
            issues.append(
                f"group {grp} usage_ratio={ratio:.4f} > {max_group_usage_ratio:.4f}"
            )
    for iid, nbytes in sorted(zombie_instances.items()):
        issues.append(
            f"zombie instance_id={iid} occupies {nbytes / (1 << 30):.2f} GiB"
        )
    return {
        "ok": not issues,
        "group_usage_ratio": group_usage,
        "active_instance_bytes": active_bytes,
        "zombie_instance_bytes": zombie_instances,
        "issues": issues,
    }


def compute_folded_kvcm_quota(
    real_quota_gb: float,
    real_kvcm_writers_per_replica: int,
    sim_kvcm_writers_per_replica: int = 1,
) -> float:
    """Scale shared KVCM group quota when simulation folds multi-node writers."""
    if real_quota_gb <= 0:
        raise ValueError("real_quota_gb must be positive")
    if real_kvcm_writers_per_replica < 1 or sim_kvcm_writers_per_replica < 1:
        raise ValueError("writer counts must be positive integers")
    return real_quota_gb * sim_kvcm_writers_per_replica / real_kvcm_writers_per_replica


def check_predict_cache_hint_env() -> bool:
    return os.getenv("DS_LLM_ENABLE_PREDICT_CACHE_HINT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-url", required=True, help="KVCM /metrics URL")
    parser.add_argument("--group", default=None, help="Target KVCM group name")
    parser.add_argument(
        "--active-instance-id",
        action="append",
        default=[],
        help="Active instance_id allowed to hold data (repeatable)",
    )
    parser.add_argument(
        "--max-usage-ratio",
        type=float,
        default=0.05,
        help="Maximum allowed group_usage_ratio before starting a cold run",
    )
    args = parser.parse_args(argv)
    with urlopen(args.metrics_url, timeout=10.0) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    report = check_kvcm_clean(
        text,
        group_name=args.group,
        active_instance_ids=set(args.active_instance_id),
        max_group_usage_ratio=args.max_usage_ratio,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
