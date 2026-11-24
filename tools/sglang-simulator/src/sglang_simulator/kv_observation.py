"""Opt-in host KV observations. Never poll, wait, copy payloads, or alter results."""
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
import inspect
import atexit
import threading
import os
import sys
import time
import uuid

from sglang_simulator.cache_boundaries import (
    _attr, _boolean, _ids, _integer, _observe, _request_hash,
)

_QUERY = ContextVar("hisim_observed_query", default=None)
_LIMIT = 8192
_RECORDER = None
_RECORDER_LOCK = threading.Lock()


def _record(emit, observe, event, **data):
    packet = {"time_kind": "host_observation", **data,
              "device_completion_confirmed": False, "observation_contract": 2}
    _observe(observe, lambda: emit(event, **packet))


def _query_helpers(scheduler):
    """Native class helpers are called directly, so use a context-scoped wrapper."""
    module = sys.modules.get(type(scheduler).__module__)
    manager = getattr(module, "V6dObjectManager", None)
    installed = []
    for name, phase in (("async_lookup_results_many", "lookup"),
                        ("async_fetch_selected_many", "fetch")):
        if manager is None:
            continue
        raw = inspect.getattr_static(manager, name, None)
        if not isinstance(raw, (staticmethod, classmethod)):
            continue
        original = raw.__func__
        if getattr(original, "_hisim_query_phase", False):
            installed.append(name)
            continue
        if not inspect.iscoroutinefunction(original):
            continue

        def wrap(native, label):
            @wraps(native)
            async def call(*args, **kwargs):
                context = _QUERY.get()
                if context is None:
                    return await native(*args, **kwargs)
                context["phase_seq"] += 1
                span = context["query_id"] + ":" + str(context["phase_seq"])
                fields = {k: context[k] for k in (
                    "query_id", "request_id_sha256", "generation", "component_id")}
                fields.update(span_id=span, source=native.__qualname__)
                started = time.monotonic_ns()
                _record(context["emit"], context["observe"],
                        "request.v6d_" + label + "_start",
                        start_mono_ns=started, **fields)
                ok, error_type = False, None
                try:
                    result = await native(*args, **kwargs)
                    ok = True
                    return result
                except BaseException as error:
                    error_type = type(error).__name__
                    raise
                finally:
                    _record(context["emit"], context["observe"],
                            "request.v6d_" + label + "_end",
                            start_mono_ns=started, end_mono_ns=time.monotonic_ns(),
                            native_ok=ok, native_error_type=error_type, **fields)
            call._hisim_query_phase = True
            return call
        setattr(manager, name, type(raw)(wrap(original, phase)))
        installed.append(name)
    return installed


def install_query(scheduler, emit, observe):
    if getattr(scheduler, "_hisim_query_observed", False):
        return
    method = next((n for n in ("_async_match", "_async_match_impl")
                   if callable(getattr(scheduler, n, None))), None)
    consume = getattr(scheduler, "_commit_async_match", None)
    if method is None or not callable(consume):
        _record(emit, observe, "request.v6d_query_capabilities", installed=False,
                reason="query_or_consume_method_missing")
        return
    original = getattr(scheduler, method)
    component = uuid.uuid4().hex
    pending = OrderedDict()
    sequence = 0

    @wraps(original)
    def submitted(*args, **kwargs):
        nonlocal sequence
        created = time.monotonic_ns()
        rid = args[0] if args else kwargs.get("request_id")
        generation = args[1] if len(args) > 1 else kwargs.get("generation")
        native = original(*args, **kwargs)
        sequence += 1
        fields = dict(component_id=component, query_id=component + ":" + str(sequence),
                      request_id_sha256=_request_hash(rid),
                      generation=_integer(generation), created_mono_ns=created,
                      source=type(scheduler).__name__, time_kind="host_native_call")
        key = (fields["request_id_sha256"], fields["generation"])
        # Native generations are integers; unknown identities never share state.
        if None not in key:
            pending[key] = fields
            if len(pending) > _LIMIT:
                _, evicted = pending.popitem(last=False)
                _record(emit, observe, "request.kv_observation_evicted",
                        query_id=evicted["query_id"], state_limit=_LIMIT)
        _record(emit, observe, "request.v6d_query_created", **fields)

        async def execute():
            started = time.monotonic_ns()
            context = dict(fields, emit=emit, observe=observe, phase_seq=0)
            token = _QUERY.set(context)
            _record(emit, observe, "request.v6d_query_start",
                    start_mono_ns=started, **fields)
            result, ok, error_type = None, False, None
            try:
                result = await native
                ok = True
                return result
            except BaseException as error:
                error_type = type(error).__name__
                raise
            finally:
                _QUERY.reset(token)
                ended = time.monotonic_ns()
                fields.update(query_start_mono_ns=started, query_end_mono_ns=ended)
                _record(emit, observe, "request.v6d_query_end",
                        start_mono_ns=started, end_mono_ns=ended,
                        native_ok=ok, native_error_type=error_type,
                        hit_tokens=_integer(_attr(result, "hit_length")),
                        fetch_failed=_boolean(_attr(result, "fetch_failed")), **fields)
                if not ok and pending.get(key) is fields:
                    pending.pop(key, None)
        return execute()

    @wraps(consume)
    def consumed(*args, **kwargs):
        request = args[0] if args else kwargs.get("request")
        result = args[2] if len(args) > 2 else kwargs.get("result")
        rid = _attr(request, "request_id")
        generation = _integer(_attr(result, "generation"))
        key = (_request_hash(rid), generation)
        fields = pending.pop(key, {}).copy()
        fields.update(request_id_sha256=key[0], generation=generation,
                      component_id=component, query_linked=bool(fields))
        started, ok = time.monotonic_ns(), False
        try:
            value = consume(*args, **kwargs)
            ok = True
            return value
        finally:
            _record(emit, observe, "request.v6d_query_consume",
                    start_mono_ns=started, end_mono_ns=time.monotonic_ns(),
                    native_ok=ok, hit_tokens=_integer(_attr(result, "hit_length")),
                    **fields)

    helpers = _query_helpers(scheduler)
    setattr(scheduler, method, submitted)
    scheduler._commit_async_match = consumed
    scheduler._hisim_query_observed = True
    _record(emit, observe, "request.v6d_query_capabilities", installed=True,
            method=method, component_id=component, helpers=helpers,
            correlation_limit=_LIMIT, creation_is_before_runtime_submit=True)


def _wrap_instance(owner, method, before, after, observe):
    native = getattr(owner, method, None)
    if not callable(native) or getattr(native, "_hisim_kv_observed", False):
        return False

    @wraps(native)
    def call(*args, **kwargs):
        started = time.monotonic_ns()
        data = _observe(observe, lambda: before(args, kwargs))
        result, ok = None, False
        try:
            result = native(*args, **kwargs)
            ok = True
            return result
        finally:
            ended = time.monotonic_ns()
            _observe(observe, lambda: after(data, result, started, ended, ok))
    call._hisim_kv_observed = True
    setattr(owner, method, call)
    return True


def _metadata(args, kwargs):
    return args[0] if args else kwargs.get("metadata", kwargs.get("connector_output"))


def install_scheduler_jobs(owner, emit, observe):
    if getattr(owner, "_hisim_job_observed", False):
        return
    owner._hisim_job_observed = True
    raw_emit = emit
    component_source = type(owner).__module__ + "." + type(owner).__qualname__
    def emit(event, **data):
        return raw_emit(event, component_source=component_source, **data)
    ledger = OrderedDict()
    component = uuid.uuid4().hex

    def output(meta, _result, start, end, ok):
        if not ok or meta is None:
            return
        jobs = _attr(meta, "store_job_ids")
        if type(jobs) is dict:
            for rid, job in jobs.items():
                if type(rid) is not str or type(job) is not int:
                    continue
                digest = _request_hash(rid)
                ledger[job] = digest
                if len(ledger) > _LIMIT:
                    ledger.popitem(last=False)
                    _record(emit, observe, "request.kv_observation_evicted",
                            component_id=component, state_limit=_LIMIT)
                _record(emit, observe, "request.kv_store_declared",
                        request_id_sha256=digest, store_job_id=job,
                        component_id=component, start_mono_ns=start, end_mono_ns=end)
        for rid in _ids(_attr(meta, "store_waits")):
            _record(emit, observe, "request.kv_store_wait_declared",
                    request_id_sha256=_request_hash(rid), component_id=component,
                    observed_mono_ns=end)

    def acknowledgements(args, kwargs):
        meta = _metadata(args, kwargs)
        worker = _attr(meta, "kv_connector_worker_meta")
        results = _attr(worker, "store_results")
        if type(results) is not dict:
            return ()
        jobs = _attr(owner, "_store_jobs")
        rows = []
        for job, ranks in results.items():
            if type(job) is not int or type(ranks) is not dict:
                continue
            state = jobs.get(job) if type(jobs) is dict else None
            rid = _attr(state, "req_id")
            rows.append(dict(store_job_id=job,
                             request_id_sha256=_request_hash(rid) or ledger.get(job),
                             ack_rank_count=len(ranks),
                             ack_failed_count=sum(v is not None for v in ranks.values())))
        return rows

    def ack(rows, result, start, end, ok):
        jobs = _attr(owner, "_store_jobs")
        for row in rows or ():
            retired = row["store_job_id"] not in jobs if type(jobs) is dict else None
            _record(emit, observe, "request.kv_job_ack_consumed", **row,
                    component_id=component, ack_observed_mono_ns=start,
                    end_mono_ns=end, native_ok=ok, job_retired=retired)
            if ok and retired:
                ledger.pop(row["store_job_id"], None)

    def decision(data, result, start, end, ok):
        request = data
        wait = result[0] if type(result) is tuple and result and type(result[0]) is bool else None
        _record(emit, observe, "request.kv_release_decision",
                request_id_sha256=_request_hash(_attr(request, "request_id")),
                component_id=component, start_mono_ns=start, end_mono_ns=end,
                native_ok=ok, delay_free=wait)

    def publish(data, result, start, end, ok):
        _record(emit, observe, "request.kv_publication_submit",
                request_id_sha256=_request_hash(_attr(data, "req_id")),
                store_job_id=_integer(_attr(data, "job_id")),
                component_id=component, start_mono_ns=start,
                end_mono_ns=end, native_ok=ok)

    managers = _attr(owner, "managers")
    if type(managers) is dict:
        for group_id, manager in managers.items():
            def summary(args, kwargs):
                hashes = args[0] if args else kwargs.get("block_hashes")
                return dict(request_id_sha256=_request_hash(kwargs.get("request_id")),
                            object_count=len(hashes) if type(hashes) in (tuple, list) else None)
            def published(data, result, start, end, ok, group_id=group_id):
                _record(emit, observe, "request.kv_publication_complete",
                        **(data or {}), group_id=group_id, component_id=component,
                        start_mono_ns=start, end_mono_ns=end, native_ok=ok,
                        seal_result=result if type(result) is bool else None)
            _wrap_instance(manager, "seal", summary, published, observe)

    installed = []
    specs = [
        ("_submit_publication", lambda a, k: a[0] if a else k.get("job"), publish),
        ("build_connector_meta", lambda a, k: None,
         lambda d, r, s, e, ok: output(r, None, s, e, ok)),
        ("update_connector_output", acknowledgements, ack),
        ("request_finished_all_groups", lambda a, k: a[0] if a else k.get("request"), decision),
    ]
    for method, before, after in specs:
        if _wrap_instance(owner, method, before, after, observe):
            installed.append(method)
    _record(emit, observe, "request.kv_job_capabilities", component_id=component,
            installed_methods=installed, state_limit=_LIMIT)


def install_release(scheduler, emit, observe):
    def record(rid, result, start, end, ok):
        _record(emit, observe, "request.kv_blocks_release",
                request_id_sha256=_request_hash(rid), start_mono_ns=start,
                end_mono_ns=end, native_ok=ok)
    _wrap_instance(scheduler, "_free_blocks",
                   lambda a, k: _attr(a[0] if a else k.get("request"), "request_id"),
                   record, observe)



def _close_recorder():
    if _RECORDER is not None:
        try:
            _RECORDER.close()
        except BaseException:
            pass


def _after_fork():
    global _RECORDER, _RECORDER_LOCK
    _RECORDER = None
    _RECORDER_LOCK = threading.Lock()
    _QUERY.set(None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def worker_emitter():
    global _RECORDER
    if os.getenv("SIM_COLLECTOR_ENABLE", "").lower() in ("1", "true", "yes", "on"):
        from sglang_simulator.collector.vllm_hook import trace_v2 as trace
        if trace._RECORDER is None:
            trace.initialize()
        return lambda event, **data: trace.emit(event, trace.ACTIVE.get(), data=data)
    if os.getenv("SGLANG_SIMULATOR_LIFECYCLE_DIR"):
        if _RECORDER is None:
            with _RECORDER_LOCK:
                if _RECORDER is None:
                    from sglang_simulator.lifecycle_recorder import LifecycleRecorder
                    _RECORDER = LifecycleRecorder.from_env("kv_worker")
                    atexit.register(_close_recorder)
        return _RECORDER.emit
    return None


def install_worker(worker, emit, observe):
    """Install after native constructor; handlers are attached after registration."""
    if getattr(worker, "_hisim_transfer_observed", False):
        return
    worker._hisim_transfer_observed = True
    raw_emit = emit
    def emit(event, **data):
        return raw_emit(event, worker_rank=_integer(_attr(worker, "_worker_rank")),
                        node_id=_integer(_attr(worker, "_node_id")), **data)

    def terminal(data, result, start, end, ok):
        if not ok or type(result) is not tuple or len(result) != 2:
            return
        for direction, ids in zip(("send", "recv"), result):
            for rid in _ids(ids):
                _record(emit, observe, "request.kv_worker_terminal",
                        request_id_sha256=_request_hash(rid), direction=direction,
                        start_mono_ns=start, observed_mono_ns=end)

    def exported(data, result, start, end, ok):
        results = _attr(result, "store_results")
        if not ok or type(results) is not dict:
            return
        for job, ranks in results.items():
            if type(job) is int and type(ranks) is dict:
                _record(emit, observe, "request.kv_worker_meta_export",
                        store_job_id=job, ack_rank_count=len(ranks),
                        ack_failed_count=sum(v is not None for v in ranks.values()),
                        start_mono_ns=start, observed_mono_ns=end)

    def plans(args, kwargs):
        meta = _metadata(args, kwargs)
        result = []
        for direction, field in (("load", "reqs_to_load"),
                                 ("store", "reqs_to_store"),
                                 ("handoff", "reqs_to_send")):
            requests = _attr(meta, field)
            if type(requests) is not dict:
                continue
            jobs = _attr(meta, "store_job_ids")
            for rid, item in requests.items():
                result.append(dict(request_id_sha256=_request_hash(rid),
                                   direction=direction,
                                   store_job_id=jobs.get(rid) if type(jobs) is dict else None,
                                   num_tokens=_integer(_attr(item, "num_tokens"))))
        return result

    def submitted_plans(rows, result, start, end, ok):
        for data in rows or ():
            _record(emit, observe, "request.kv_worker_submit", **data,
                    start_mono_ns=start, end_mono_ns=end, native_ok=ok,
                    completion_scope="worker_call_return_not_transfer_completion")

    for method in ("start_load_kv", "start_store_kv", "process_send_load"):
        _wrap_instance(worker, method, plans, submitted_plans, observe)

    _wrap_instance(worker, "get_finished", lambda a, k: None, terminal, observe)
    _wrap_instance(worker, "build_connector_worker_meta", lambda a, k: None, exported, observe)

    def attach_handlers():
        for direction, name in (("load", "_load_handler"), ("store", "_store_handler")):
            handler = _attr(worker, name)
            if handler is None:
                continue

            def submission(args, kwargs):
                job = args[0] if args else kwargs.get("job_id")
                rid = args[3] if len(args) > 3 else kwargs.get("req_id", kwargs.get("request_id"))
                local_map = _attr(worker, "_store_job_ids")
                return dict(native_job_id=_integer(job),
                            store_job_id=local_map.get(job) if type(local_map) is dict else None,
                            request_id_sha256=_request_hash(rid))

            def submitted(data, result, start, end, ok, direction=direction):
                _record(emit, observe, "request.kv_job_submit", **(data or {}),
                        direction=direction, start_mono_ns=start, end_mono_ns=end,
                        native_ok=ok)

            def completed(data, result, start, end, ok, direction=direction):
                if not ok or type(result) not in (tuple, list):
                    return
                pending = _attr(worker, "_pending_jobs")
                stores = _attr(worker, "_store_job_ids")
                for item in result:
                    job = _integer(_attr(item, "job_id"))
                    rid = pending.get(job) if type(pending) is dict else None
                    _record(emit, observe, "request.kv_job_ready_observed",
                            native_job_id=job, store_job_id=stores.get(job) if type(stores) is dict else None,
                            request_id_sha256=_request_hash(rid), direction=direction,
                            poll_start_mono_ns=start, observed_mono_ns=end,
                            failed=_attr(item, "error") is not None,
                            fatal=_boolean(_attr(item, "fatal")),
                            nan_detected=_boolean(_attr(item, "nan_detected")),
                            completion_scope="native_handler_poll_not_device_timestamp")
            _wrap_instance(handler, "swap", submission, submitted, observe)
            _wrap_instance(handler, "get_finished", lambda a, k: None, completed, observe)

    _wrap_instance(worker, "register_kv_caches", lambda a, k: None,
                   lambda d, r, s, e, ok: attach_handlers() if ok else None, observe)
    attach_handlers()
    observed = [name for name in ("register_kv_caches", "get_finished",
                "build_connector_worker_meta", "start_load_kv", "start_store_kv",
                "process_send_load")
                if getattr(getattr(worker, name, None), "_hisim_kv_observed", False)]
    _record(emit, observe, "request.kv_worker_capabilities", installed=bool(observed),
            installed_methods=observed,
            load_handler_present=_attr(worker, "_load_handler") is not None,
            store_handler_present=_attr(worker, "_store_handler") is not None)
