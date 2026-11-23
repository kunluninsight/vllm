"""Unit tests for TPU cache patches and V6D/Raiden/Tair transfer timing."""

from types import SimpleNamespace as NS

import pytest

from sglang_simulator.simulation.vllm import tpu_cache_patches as patches
from sglang_simulator.simulation.vllm.v6d import tair_kvcm
from sglang_simulator.simulation.vllm.v6d import tpu_connector as adapter
from sglang_simulator.simulation.vllm.v6d import v6d_capacity


def test_explicit_cache_checkpoint_overrides_boundary(monkeypatch):
    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")

    class FakeKVCacheManager:
        def __init__(self):
            self.coordinator = NS(scheduler_block_size=3072)

        def prefix_cache_lookup_enabled(self, request):
            return True

        def get_computed_blocks(self, request):
            return ("blocks", 0, 0)

        def get_computed_blocks_for_connector(self, request):
            return ("blocks", 153600, 0, False)

    patches.patch_explicit_cache_checkpoint(FakeKVCacheManager)
    mgr = FakeKVCacheManager()

    req = NS(
        num_prompt_tokens=200000,
        sampling_params=NS(
            extra_args={"cache_control_params": {"token_positions": [3100, 154000, 999999, "bad"]}}
        ),
    )
    assert mgr.get_computed_blocks(req) == ("blocks", 0, 153600)
    assert mgr.get_computed_blocks_for_connector(req) == ("blocks", 153600, 153600, False)

    # Without cache_control_params, original boundary (0) is preserved.
    plain_req = NS(num_prompt_tokens=200000, sampling_params=NS(extra_args=None))
    assert mgr.get_computed_blocks(plain_req) == ("blocks", 0, 0)


def test_same_step_prefix_hits_blocked_until_next_schedule(monkeypatch):
    monkeypatch.setenv("TPU_MTP_PREFILL_ONLY", "1")
    monkeypatch.delenv("HISIM_TPU_DISABLE_SAME_STEP_PREFIX_PATCH", raising=False)

    class FakeBlockPool:
        def __init__(self):
            self.cache = {}

        def _insert_block_hash(self, key, block, num_tokens):
            self.cache[key] = block

        def get_cached_block(self, block_hash, kv_cache_group_ids):
            blocks = [self.cache.get((block_hash, gid)) for gid in kv_cache_group_ids]
            return blocks if all(b is not None for b in blocks) else None

    class FakeScheduler:
        def __init__(self, pool):
            self.kv_cache_manager = NS(block_pool=pool)

        def schedule(self):
            blk = NS(block_id=42)
            self.kv_cache_manager.block_pool._insert_block_hash(("h1", 0), blk, 3072)
            # Same step: must return None (unwritten on TPU).
            return self.kv_cache_manager.block_pool.get_cached_block("h1", [0])

    patches.patch_same_step_prefix_hits(FakeScheduler, FakeBlockPool)
    pool = FakeBlockPool()
    sched = FakeScheduler(pool)

    assert sched.schedule() is None
    # Next schedule clears unwritten set, so previously hashed block hits.
    pool._tpu_unwritten_block_ids.clear()
    hit = pool.get_cached_block("h1", [0])
    assert hit is not None and hit[0].block_id == 42


def test_hybrid_kv_load_failure_recovery_handles_multi_group():
    class FakeScheduler:
        def __init__(self):
            self.kv_cache_config = NS(kv_cache_groups=[NS(), NS(), NS(), NS()])
            self.kv_cache_manager = NS(
                block_pool=NS(null_block=NS(block_id=0)),
                get_block_ids=lambda req_id: [[1, 2, 0], [3, 4], [5], [6]],
            )

        def _update_requests_with_invalid_blocks(
            self, requests, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
        ):
            raise AssertionError("single-group path must not be called")

    patches.patch_hybrid_kv_load_failure_recovery(FakeScheduler)
    sched = FakeScheduler()
    req = NS(request_id="r1", num_computed_tokens=9216)
    affected, tokens, evicted = sched._update_requests_with_invalid_blocks(
        [req], {4}, {"r1": 3072}, evict_blocks=True
    )
    assert affected == {"r1"}
    assert tokens == 6144
    assert req.num_computed_tokens == 0
    assert evicted == {1, 2, 3, 4, 5, 6}


def test_multi_lane_store_avoids_head_of_line_blocking(monkeypatch):
    monkeypatch.setenv("HISIM_TPU_V6D_STORE_LANES", "3")
    q = adapter._TPUStoreQueue()
    g1 = adapter._TPUTransferGate(ready_at=None)
    g2 = adapter._TPUTransferGate(ready_at=10.0)
    q.jobs.append(adapter._TPUStoreJob(1, "r1", g1, 0.5))
    q.jobs.append(adapter._TPUStoreJob(2, "r2", g2, 0.5))
    q.resolve_deadlines()
    assert q.jobs[0].deadline is None
    assert q.jobs[1].deadline == pytest.approx(10.5)


def test_v6d_load_bandwidth_and_raiden_hold_queues(monkeypatch):
    now = NS(value=10.0, sleeps=[])
    monkeypatch.setattr(adapter.time, "monotonic", lambda: now.value)
    monkeypatch.setattr(adapter.time, "sleep", lambda s: now.sleeps.append(s))

    # V6D H2D load queue
    monkeypatch.setenv("HISIM_TPU_V6D_LOAD_BANDWIDTH_GB", "10.0")
    lq = adapter._TimedReqQueue(lanes=1)
    dl = lq.enqueue("req-load", 0.25, now.value)
    assert dl == pytest.approx(10.25)
    assert lq.harvest(10.1) == set()
    adapter.wait_for_tpu_store_progress()
    assert now.sleeps == pytest.approx([0.25])
    assert lq.harvest(10.25) == {"req-load"}

    # Raiden handoff hold
    monkeypatch.setenv("HISIM_TPU_RAIDEN_HANDOFF_BASE_S", "0.5")
    monkeypatch.setenv("HISIM_TPU_RAIDEN_HANDOFF_BANDWIDTH_GB", "10.0")
    monkeypatch.setenv("HISIM_TPU_RAIDEN_BYTES_PER_TOKEN", "1000000")

    class DummyRaidenWorker:
        is_producer = True

        def process_send_load(self, metadata, wait_for_completion=False, report_completion=True):
            pass

        def get_finished(self, *args, **kwargs):
            return set(), set()

        def has_pending_push_work(self):
            return False

    adapter.C_TPURaidenConnectorWorkerHook.hook(DummyRaidenWorker)
    w = DummyRaidenWorker()
    meta = NS(reqs_to_send={"r1": NS(num_tokens=5000)})
    # duration = 0.5 + (5000 * 1e6 / 10e9) = 1.0s
    w.process_send_load(meta)
    assert w.has_pending_push_work() is True
    assert w.get_finished()[0] == set()
    now.value = 11.0
    assert w.get_finished()[0] == {"r1"}
    assert w.has_pending_push_work() is False


def test_tair_and_p2p_optional_bandwidth_sleep(monkeypatch):
    tair_sleeps = []
    monkeypatch.setattr(tair_kvcm.time, "sleep", lambda s: tair_sleeps.append(s))
    monkeypatch.setenv("HISIM_TAIR_KVCM_READ_BANDWIDTH_GB", "2.0")
    tair_kvcm._sleep_for_bytes([1_000_000_000, 1_000_000_000], "HISIM_TAIR_KVCM_READ_BANDWIDTH_GB")
    assert tair_sleeps == pytest.approx([1.0])

    p2p_sleeps = []
    monkeypatch.setattr(v6d_capacity.time, "sleep", lambda s: p2p_sleeps.append(s))
    monkeypatch.setenv("HISIM_V6D_P2P_BANDWIDTH_GB", "4.0")
    v6d_capacity._sim_load_data_noop(sizes=[2_000_000_000])
    assert p2p_sleeps == pytest.approx([0.5])


def test_same_step_guard_allows_direct_pool_lookup_outside_schedule(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "1")
    monkeypatch.delenv("HISIM_TPU_DISABLE_SAME_STEP_PREFIX_PATCH", raising=False)

    class FakeBlockPool:
        def __init__(self):
            self.cache = {}

        def _insert_block_hash(self, key, block, num_tokens):
            self.cache[key] = block

        def get_cached_block(self, block_hash, kv_cache_group_ids):
            blocks = [self.cache.get((block_hash, gid)) for gid in kv_cache_group_ids]
            return blocks if all(b is not None for b in blocks) else None

    class SchedulelessMock:
        pass

    # Must not raise when scheduler class omits schedule().
    patches.patch_same_step_prefix_hits(SchedulelessMock, FakeBlockPool)
    pool = FakeBlockPool()
    pool._insert_block_hash(("h2", 0), NS(block_id=7), 8)
    hit = pool.get_cached_block("h2", [0])
    assert hit is not None and hit[0].block_id == 7


def test_inproc_llm_engine_keeps_stepping_until_kv_cleanup_drains(monkeypatch):
    import inspect
    from collections import deque

    from sglang_simulator.simulation.vllm.engine_core_pipeline import (
        C_VLLMLLMEngineHook,
    )

    monkeypatch.setattr(
        "sglang_simulator.simulation.vllm.engine_core_pipeline._register_tpu_connector_names",
        lambda: None,
    )

    class FakeLLMEngine:
        def __init__(self, core):
            self._output_unfinished = False
            self.engine_core = NS(engine_core=core)

        def has_unfinished_requests(self, local: bool = False) -> bool:
            return self._output_unfinished

    C_VLLMLLMEngineHook.hook(FakeLLMEngine)
    assert "local" in inspect.signature(FakeLLMEngine.has_unfinished_requests).parameters

    # 1. No connector + abort-before-schedule (finished_req_ids non-empty) -> False.
    core_no_conn = NS(
        batch_queue=deque(),
        scheduler=NS(
            connector=None,
            requests={},
            waiting=[],
            skipped_waiting=[],
            running=[],
            finished_req_ids={"abort-1"},
        ),
    )
    eng_no_conn = FakeLLMEngine(core_no_conn)
    assert eng_no_conn.has_unfinished_requests(local=True) is False

    # 2. In-flight empty cleanup step in batch_queue -> True until drained.
    core_no_conn.batch_queue.append("empty-step-E1")
    assert eng_no_conn.has_unfinished_requests() is True
    core_no_conn.batch_queue.popleft()
    assert eng_no_conn.has_unfinished_requests() is False

    # 3. Delayed connector release pinned in scheduler.requests -> True until freed.
    push_pending = [False]
    connector = NS(has_pending_push_work=lambda: push_pending[0])
    core_conn = NS(
        batch_queue=deque(),
        scheduler=NS(
            connector=connector,
            requests={"req-tail": NS()},
            waiting=[],
            skipped_waiting=[],
            running=[],
        ),
    )
    eng_conn = FakeLLMEngine(core_conn)
    assert eng_conn.has_unfinished_requests() is True
    core_conn.scheduler.requests.pop("req-tail")
    assert eng_conn.has_unfinished_requests() is False

    # 4. Connector pending push work (e.g. V6D store / Raiden hold) -> True until done.
    push_pending[0] = True
    assert eng_conn.has_unfinished_requests() is True
    push_pending[0] = False
    assert eng_conn.has_unfinished_requests() is False


def test_v6d_connector_exposes_has_pending_push_work():
    class FakeV6dConnector:
        def __init__(self, sched=None, worker=None):
            self.scheduler = sched
            self.worker = worker

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            return None

    adapter.C_TPUV6dConnectorAdmissionHook.hook(FakeV6dConnector)
    sched = NS(_store_jobs={}, _store_waits=set())
    worker_busy = [False]
    worker = NS(has_pending_push_work=lambda: worker_busy[0])
    conn = FakeV6dConnector(sched=sched, worker=worker)

    assert conn.has_pending_push_work() is False
    sched._store_jobs[1] = "job"
    assert conn.has_pending_push_work() is True
    sched._store_jobs.clear()
    sched._store_waits.add("r1")
    assert conn.has_pending_push_work() is True
    sched._store_waits.clear()
    worker_busy[0] = True
    assert conn.has_pending_push_work() is True
    worker_busy[0] = False
    assert conn.has_pending_push_work() is False


def test_tpu_idle_wait_cap_slices_long_handoff_hold(monkeypatch):
    now = NS(value=100.0)
    sleeps = []
    monkeypatch.setattr(adapter.time, "monotonic", lambda: now.value)
    monkeypatch.setattr(adapter.time, "sleep", lambda s: (sleeps.append(s), setattr(now, "value", now.value + s)))

    tq = adapter._TimedReqQueue(lanes=1)
    tq.enqueue("r-hold", 1.5, now.value)

    with adapter.tpu_idle_wait_cap(0.005):
        adapter.wait_for_tpu_store_progress()
    assert sleeps == pytest.approx([0.005])

    # Uncapped direct call still waits the remaining duration.
    adapter.wait_for_tpu_store_progress()
    assert sleeps == pytest.approx([0.005, 1.495])
