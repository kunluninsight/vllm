"""Account for calibrated host metadata cost at its native scheduler seam."""

import math
import os
import time
from functools import wraps


def install_scheduler_meta_timing(scheduler, config, emit):
    if config is None:
        return
    names = ("base_ms", "cached_request_ms", "query_token_ms")
    if set(config) != set(names):
        raise ValueError(f"scheduler_meta requires exactly {names}")
    base, cached_cost, token_cost = (float(config[k]) for k in names)
    if any(not math.isfinite(x) or x < 0 for x in (base, cached_cost, token_cost)):
        raise ValueError("scheduler_meta coefficients must be finite and nonnegative")
    if float(os.environ.get("HISIM_TPU_PP_PREFILL_HANDOFF_MS", "0")) != 0:
        raise ValueError(
            "scheduler_meta replaces the legacy prefill handoff budget; set it to zero"
        )
    original = scheduler._build_kv_connector_meta

    @wraps(original)
    def build(connector, scheduler_output):
        started = time.monotonic()
        result = original(connector, scheduler_output)
        native_return = time.monotonic()
        tokens = scheduler_output.total_num_scheduled_tokens
        if tokens <= 0:
            return result
        scheduled = scheduler_output.num_scheduled_tokens
        cached = sum(
            scheduled.get(rid, 0) > 0
            for rid in scheduler_output.scheduled_cached_reqs.req_ids
        )
        target_s = (base + cached_cost * cached + token_cost * tokens) / 1000
        # Native metadata still executes; only its unaccounted duration is added.
        remaining = started + target_s - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        ended = time.monotonic()
        emit(
            "scheduler.meta_budget",
            start_mono_ns=int(started * 1e9),
            native_return_mono_ns=int(native_return * 1e9),
            end_mono_ns=int(ended * 1e9),
            target_ms=target_s * 1000,
            added_ms=max(0.0, remaining) * 1000,
            total_num_scheduled_tokens=tokens,
            cached_requests=cached,
            time_kind="calibrated_host_metadata_budget",
        )
        return result

    scheduler._build_kv_connector_meta = build
