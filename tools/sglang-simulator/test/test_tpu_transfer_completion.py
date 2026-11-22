"""CPU contract tests: no native DMA, RPC, daemon, or wall-clock polling ticks."""

import ast
import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
import weakref
from collections import defaultdict
from types import SimpleNamespace as NS

import pytest

from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.vllm.v6d import tpu_connector as adapter
from test_tpu_local_admission import connector, finished_request, raiden_worker


NODE_BYTES = 37748736
BANDWIDTH_GB = 0.125
DURATION = NODE_BYTES / (BANDWIDTH_GB * 1e9)
NODE_LAYOUT = adapter._tpu_node_object_layout


def native_classes(path, selections, namespace):
    """Execute selected unmodified native protocol bodies, not hardware imports.

    Optional source paths allow offline dependency snapshots without installing
    packages or constructing real workers/controllers. This is not an E2E test.
    """
    tree = ast.parse(Path(path).read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in selections:
            continue
        methods = selections[node.name]
        if methods is not None:
            node.body = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in methods]
            node.bases = [ast.Name(id='NativeBase', ctx=ast.Load())]
        nodes.append(node)
    namespace.update(__name__=__name__, dataclass=dataclass, field=field)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


class WorkerMetadata:
    def __init__(self):
        self.store_results = {}
        self.failed_loads = {}


@pytest.fixture
def clock(monkeypatch):
    now = NS(value=10.0, waits=[])
    monkeypatch.setattr(adapter.time, 'monotonic', lambda: now.value)

    def sleep(delay):
        now.waits.append(delay)
        now.value += delay

    monkeypatch.setattr(adapter.time, 'sleep', sleep)
    return now


@pytest.fixture
def worker_factory(monkeypatch):
    monkeypatch.setattr(ConfigManager, '_get_raw_config', staticmethod(lambda: {
        'platform': {'memory_write_bandwidth_gb': BANDWIDTH_GB},
    }))
    monkeypatch.setattr(adapter, '_tpu_node_object_layout', lambda *_: (
        NODE_BYTES, [8, 1, NODE_BYTES // 8]
    ))

    def make():
        class Native:
            def get_finished(self, finished_req_ids=None):
                # Native contract: cancellation/finished_req_ids is not a store_wait.
                # _saving_req_jobs is the guard for both terminal and cancelled requests.
                assert not self._pending_jobs
                sent = {rid for rid in self._finished_reqs_waiting_for_store
                        if not self._saving_req_jobs.get(rid)}
                self._finished_reqs_waiting_for_store.difference_update(sent)
                return sent, set()

            def build_connector_worker_meta(self):
                result = self._worker_meta
                self._worker_meta = WorkerMetadata()
                return result

            def start_load_kv(self, metadata):
                raise AssertionError('native DMA must not run')

            def start_store_kv(self, metadata, nan_check=None):
                raise AssertionError('native DMA must not run')

        metadata_type = WorkerMetadata
        native_path = os.environ.get('TPU_NATIVE_V6D_SOURCE')
        if native_path:
            ns = native_classes(native_path, {
                'V6dWorkerMetadata': None,
                'V6dObjectConnectorWorker': {'get_finished', 'build_connector_worker_meta'},
            }, {'NativeBase': object, 'KVConnectorWorkerMetadata': object})
            native = ns['V6dObjectConnectorWorker']
            Native.get_finished = native.get_finished
            Native.build_connector_worker_meta = native.build_connector_worker_meta
            metadata_type = ns['V6dWorkerMetadata']
        adapter.C_TPUV6dObjectConnectorWorkerHook.hook(Native)
        worker = Native()
        worker._num_groups = 2
        worker._node_id = 0
        worker._worker_rank = 0
        worker._vllm_config = NS()
        worker._kv_cache_config = NS()
        worker._load_handler = worker._store_handler = None
        worker._pending_jobs = {}
        worker._saving_req_jobs = defaultdict(set)
        worker._loading_req_jobs = {}
        worker._finished_reqs_waiting_for_store = set()
        worker._worker_meta = metadata_type()
        return worker

    return make


def metadata(req='r', job=1, masks=(1,), waits=()):
    groups = {0: ([f'key-{i}' for i in range(len(masks))],
                  list(range(len(masks))), list(masks))}
    return NS(num_groups=2, reqs_to_load={}, reqs_to_store={req: groups},
              store_job_ids={req: job}, store_waits=set(waits))


def submit(worker, meta):
    worker.start_load_kv(meta)
    worker.start_store_kv(meta)


def results(worker):
    meta = worker.build_connector_worker_meta()
    return meta.store_results if meta is not None else {}


def test_gate_stays_bound_after_context_exit_and_ack_needs_deadline(worker_factory, clock):
    worker = worker_factory()
    with adapter.tpu_transfer_step() as gate:
        assert gate.ready_at is None
        submit(worker, metadata(waits={'r'}, masks=(1, 2, 3, 0)))
    clock.value = 50
    assert not worker.get_finished({'r'})[0]
    assert results(worker) == {}
    assert worker._saving_req_jobs == {'r': {1}}
    assert worker.has_pending_push_work()
    gate.ready_at = 100
    clock.value = 100 + 2 * DURATION - 0.00001
    assert not worker.get_finished()[0]
    assert results(worker) == {}
    clock.value = 100 + 2 * DURATION
    assert worker.get_finished()[0] == {'r'}
    assert results(worker) == {1: {0: None}}
    assert not worker._saving_req_jobs and not worker._pending_jobs
    assert not worker.has_pending_push_work()


def test_fifo_and_out_of_order_ready_or_poll_never_advance_lane(worker_factory, clock):
    worker = worker_factory()
    with adapter.tpu_transfer_step() as first:
        submit(worker, metadata('a', 1, waits={'a'}))
    with adapter.tpu_transfer_step() as second:
        submit(worker, metadata('b', 2, waits={'b'}))
    second.ready_at = 5
    clock.value = 100
    assert not worker.get_finished()[0]
    assert results(worker) == {}
    adapter.wait_for_tpu_store_progress()
    assert not clock.waits  # first PP RPC must still progress
    first.ready_at = 100
    clock.value = 100 + DURATION
    assert worker.get_finished()[0] == {'a'}
    assert results(worker) == {1: {0: None}}
    assert not worker.get_finished()[0]
    clock.value += DURATION
    assert worker.get_finished()[0] == {'b'}
    assert results(worker) == {2: {0: None}}
    # Polling late cannot move the completed lane's modeled time to poll time.
    clock.value = 200
    with adapter.tpu_transfer_step() as third:
        submit(worker, metadata('c', 3, waits={'c'}))
    third.ready_at = 100
    assert worker.get_finished()[0] == {'c'}
    assert results(worker) == {3: {0: None}}


@pytest.mark.parametrize('cancelled', [False, True])
def test_same_request_multiple_jobs_and_cancellation_drain(worker_factory, clock, cancelled):
    worker = worker_factory()
    with adapter.tpu_transfer_step() as gate:
        submit(worker, metadata(job=1, waits={'r'}))
        submit(worker, metadata(job=2))
    gate.ready_at = clock.value
    clock.value += DURATION
    assert not worker.get_finished({'r'} if cancelled else set())[0]
    assert results(worker) == {1: {0: None}}
    assert worker._saving_req_jobs == {'r': {2}}
    clock.value += DURATION
    assert worker.get_finished({'r'} if cancelled else set())[0] == {'r'}
    assert results(worker) == {2: {0: None}}
    assert not worker.get_finished({'r'})[0]


def test_job_ack_does_not_invent_request_terminal(worker_factory, clock):
    worker = worker_factory()
    submit(worker, metadata())  # fallback is submit monotonic, no PP gate
    clock.value += DURATION
    assert not worker.get_finished({'r'})[0]
    assert worker.has_pending_push_work()  # unreported worker metadata
    assert results(worker) == {1: {0: None}}
    assert not worker.has_pending_push_work()
    worker.start_load_kv(NS(num_groups=2, reqs_to_load={}, store_waits={'r'}, store_job_ids={}))
    assert worker.has_pending_push_work()
    assert worker.get_finished()[0] == {'r'}


def test_mask_zero_ack_ignores_gate_and_queued_dma(worker_factory, clock):
    worker = worker_factory()
    with adapter.tpu_transfer_step():
        submit(worker, metadata('slow', 1, waits={'slow'}))
        submit(worker, metadata('empty', 2, masks=(0, 2), waits={'empty'}))
    assert worker.get_finished()[0] == {'empty'}
    assert results(worker) == {2: {0: None}}
    assert len(worker._sim_tpu_store_queue.jobs) == 1
    assert worker._sim_tpu_store_queue.lane_ready_at == 0


def test_same_request_zero_mask_cannot_bypass_pending_job(worker_factory, clock):
    worker = worker_factory()
    with adapter.tpu_transfer_step():
        submit(worker, metadata(job=1, waits={'r'}))
        submit(worker, metadata(job=2, masks=()))
    assert results(worker) == {2: {0: None}}
    assert not worker.get_finished()[0]


def test_duplicate_job_does_not_queue_or_ack_twice(worker_factory, clock):
    worker = worker_factory()
    m = metadata(waits={'r'})
    with adapter.tpu_transfer_step() as gate:
        submit(worker, m)
        worker.start_store_kv(m)
    assert len(worker._sim_tpu_store_queue.jobs) == 1
    gate.ready_at = clock.value
    clock.value += DURATION
    assert worker.get_finished()[0] == {'r'}
    snapshot = worker.build_connector_worker_meta()
    worker.start_store_kv(m)
    assert results(worker) == {}
    assert not worker.get_finished()[0]
    assert snapshot.store_results == {1: {0: None}}
    worker.start_store_kv(metadata('other', 1))
    assert not worker._sim_tpu_store_queue.seen


def test_worker_metadata_snapshots_are_not_cleared(worker_factory, clock):
    worker = worker_factory()
    submit(worker, metadata(job=1, masks=(0,)))
    first = worker.build_connector_worker_meta()
    submit(worker, metadata(job=2, masks=(0,)))
    second = worker.build_connector_worker_meta()
    assert first is not second
    assert first.store_results == {1: {0: None}}
    assert second.store_results == {2: {0: None}}
    assert results(worker) == {}


@pytest.mark.parametrize('native', ['_pending_jobs', '_saving_req_jobs', '_loading_req_jobs',
                                  '_load_handler', '_store_handler', '_store_job_ids'])
def test_validation_still_rejects_real_native_jobs(worker_factory, native):
    worker = worker_factory()
    setattr(worker, native, {'alien': {99}})
    with pytest.raises(RuntimeError, match='DMA'):
        worker.start_load_kv(metadata())


@pytest.mark.parametrize('bandwidth', [None, 0, -1, float('nan'), float('inf'), True, 'unknown'])
def test_invalid_bandwidth_cannot_silently_zero_duration(worker_factory, monkeypatch, bandwidth):
    worker = worker_factory()
    monkeypatch.setattr(ConfigManager, '_get_raw_config', staticmethod(lambda: {
        'platform': {'memory_write_bandwidth_gb': bandwidth},
    }))
    with pytest.raises(ValueError, match='positive and finite'):
        submit(worker, metadata())
    assert not worker._saving_req_jobs
    assert not results(worker)


def test_non_tpu_path_does_not_use_node_masks_or_bandwidth(worker_factory, monkeypatch):
    worker = worker_factory()
    del worker._node_id
    monkeypatch.setattr(adapter, '_tpu_node_object_layout', NODE_LAYOUT)
    monkeypatch.setattr(ConfigManager, '_get_raw_config', staticmethod(dict))
    with adapter.tpu_transfer_step():
        submit(worker, metadata(waits={'r'}))
    assert worker.get_finished()[0] == {'r'}
    assert results(worker) == {1: {0: None}}


def test_event_wait_requires_every_gate_and_uses_earliest_deadline(worker_factory, clock):
    a, b = worker_factory(), worker_factory()
    with adapter.tpu_transfer_step() as ga:
        submit(a, metadata())
    with adapter.tpu_transfer_step() as gb:
        submit(b, metadata(masks=(1, 1)))
    ga.ready_at = 10
    adapter.wait_for_tpu_store_progress()
    assert not clock.waits
    gb.ready_at = 10
    adapter.wait_for_tpu_store_progress()
    assert clock.waits == pytest.approx([DURATION])
    adapter.wait_for_tpu_store_progress()  # harvestable head: do not sleep
    assert len(clock.waits) == 1
    a.get_finished()
    adapter.wait_for_tpu_store_progress()
    assert clock.waits == pytest.approx([DURATION, DURATION])
    b.get_finished()
    adapter.wait_for_tpu_store_progress()
    assert len(clock.waits) == 2


def test_weak_registry_does_not_keep_worker_or_block_other_engines(worker_factory, clock):
    worker = worker_factory()
    with adapter.tpu_transfer_step():
        submit(worker, metadata())
    old = weakref.ref(worker._sim_tpu_store_queue)
    del worker
    gc.collect()
    assert old() is None
    live = worker_factory()
    submit(live, metadata())
    adapter.wait_for_tpu_store_progress()
    assert clock.waits == pytest.approx([DURATION])


def test_nested_context_and_exception_restore_gate(worker_factory, clock):
    worker = worker_factory()
    with adapter.tpu_transfer_step() as outer:
        with pytest.raises(RuntimeError):
            with adapter.tpu_transfer_step() as inner:
                submit(worker, metadata('a', 1))
                raise RuntimeError('step failed')
        submit(worker, metadata('b', 2))
    submit(worker, metadata('c', 3))
    a, b, c = worker._sim_tpu_store_queue.jobs
    assert a.gate is inner and b.gate is outer
    assert c.gate.ready_at == clock.value


@pytest.mark.parametrize('raiden_first', [True, False])
def test_native_two_child_async_save_protocol_never_releases_early(
    monkeypatch, worker_factory, clock, raiden_first
):
    path = os.environ.get('TPU_NATIVE_MULTI_SOURCE')
    if not path:
        pytest.skip('set TPU_NATIVE_MULTI_SOURCE to an offline native source snapshot')

    class NativeBase:
        def bind_connector_metadata(self, metadata):
            self.bound = metadata

    ns = native_classes(path, {
        'MultiKVConnectorMetadata': None,
        'MultiKVConnectorWorkerMetadata': None,
        'MultiConnector': {'_aggregate_request_finished', 'build_connector_meta',
                           'bind_connector_metadata', 'get_finished', 'update_connector_output'},
    }, {'NativeBase': NativeBase, 'KVConnectorMetadata': object,
        'KVConnectorWorkerMetadata': object})
    Multi = ns['MultiConnector']
    scheduler = connector(monkeypatch)
    request = finished_request()
    v6d_meta = metadata(waits={'r'})
    received_meta = []
    sibling = NS(
        request_finished_all_groups=lambda *args: (True, None),
        build_connector_meta=lambda *args: v6d_meta,
        update_connector_output=lambda output: received_meta.append(output.kv_connector_worker_meta),
    )
    scheduler_multi = Multi()
    scheduler_multi._connectors = [scheduler, sibling]
    scheduler_multi._extra_async_saves = {}
    scheduler_multi._requests_to_connector = {'r': 0}
    result = scheduler_multi._aggregate_request_finished(
        request, lambda c: c.request_finished_all_groups(request, ([1], [2]))
    )
    assert result == (True, None)
    assert scheduler_multi._extra_async_saves == {'r': 1}
    meta = scheduler_multi.build_connector_meta(None)
    assert meta.extra_async_saves == {'r': 1}
    assert not scheduler_multi._extra_async_saves
    assert scheduler_multi.build_connector_meta(None).extra_async_saves is None

    raiden, v6d = raiden_worker(), worker_factory()
    worker_multi = Multi()
    worker_multi._extra_async_saves = {}
    worker_multi._connectors = [
        NS(bind_connector_metadata=lambda m: None, get_finished=raiden.get_finished),
        NS(bind_connector_metadata=lambda m: None, get_finished=v6d.get_finished),
    ]
    worker_multi.bind_connector_metadata(meta)
    assert worker_multi._extra_async_saves == {'r': 1}
    with adapter.tpu_transfer_step() as gate:
        submit(v6d, meta.metadata[1])
    if raiden_first:
        raiden.process_send_load(meta.metadata[0])
    else:
        gate.ready_at = clock.value
        clock.value += DURATION
    assert not worker_multi.get_finished({'r'})[0]
    assert scheduler.has_pending_push_work()
    first_meta = v6d.build_connector_worker_meta()
    if raiden_first:
        gate.ready_at = clock.value
        clock.value += DURATION
    else:
        raiden.process_send_load(meta.metadata[0])
    assert worker_multi.get_finished({'r'})[0] == {'r'}
    assert scheduler.has_pending_push_work()  # PP output not consumed yet
    last_meta = v6d.build_connector_worker_meta()
    v6d_results = last_meta if raiden_first else first_meta
    assert v6d_results.store_results == {1: {0: None}}
    output = NS(finished_sending={'r'}, kv_connector_worker_meta=
                ns['MultiKVConnectorWorkerMetadata'](metadata=(None, v6d_results)))
    scheduler_multi.update_connector_output(output)
    assert received_meta == [v6d_results]
    assert output.kv_connector_worker_meta.metadata[1] is v6d_results
    assert not scheduler.has_pending_push_work()
    assert not worker_multi.get_finished({'r'})[0]


@pytest.mark.parametrize('mtp', [False, True])
def test_shared_layout_uses_projected_node_page_for_scheduler_and_worker(
    worker_factory, monkeypatch, clock, mtp
):
    # Real simulator partition/projection, lightweight native pool metadata stubs.
    raw = {
        'platform': {'accelerator': {'name': 'tpu7x-32'}, 'num_device_per_node': 8,
                     'memory_write_bandwidth_gb': BANDWIDTH_GB},
        'scheduler': {'pp_size': 4, 'tp_size': 1, 'pcp_size': 8,
                      'pp_layer_partition': [24, 24, 24, 20]},
    }
    monkeypatch.setattr(ConfigManager, '_get_raw_config', staticmethod(lambda: raw))
    monkeypatch.setattr(adapter, '_tpu_node_object_layout', NODE_LAYOUT)
    calls = []

    def project(groups, specs):
        calls.append(len(specs))
        return [NS(layer_names=[n for n in g.layer_names if n in specs],
                   kv_cache_spec=g.kv_cache_spec) for g in groups]

    monkeypatch.setitem(sys.modules, 'vllm.v1.core.kv_cache_utils', NS(
        _project_kv_cache_groups_to_worker=project,
        _pool_bytes_per_block=lambda groups: max(len(g.layer_names) for g in groups) * 786432,
    ))
    groups = [NS(layer_names=[f'model.layers.{i}' for i in range(k, 92, 4)],
                 kv_cache_spec=NS()) for k in range(4)]
    if mtp:
        groups[-1].layer_names.append('mtp.layers.0')
    config = NS(model_config=NS(hf_text_config=NS(num_hidden_layers=92)),
                parallel_config=NS(local_world_size=1))
    cache = NS(kv_cache_groups=groups)

    class NativeScheduler:
        def _compute_group_block_bytes(self, *args):
            return 18087936, [1, 1, 18087936], 'uint8'

    adapter.C_TPUV6dObjectLayoutHook.hook(NativeScheduler)
    scheduler = NativeScheduler()
    scheduler.vllm_config, scheduler.kv_cache_config = config, cache
    assert scheduler._compute_group_block_bytes(None, 0) == (NODE_BYTES, [8, 1, 4718592], 'uint8')
    worker = worker_factory()
    worker._vllm_config, worker._kv_cache_config = config, cache
    submit(worker, metadata())
    assert worker._sim_tpu_store_queue.jobs[0].duration == DURATION
    assert calls == [24, 24, 24, 20 + int(mtp)] * 2
    clock.value += DURATION
    worker.get_finished()
    assert results(worker) == {1: {0: None}}


@pytest.mark.parametrize('raw', [{}, {
    'platform': {'accelerator': {'name': 'tpu7x-32'}}, 'scheduler': {'pp_size': 1},
}])
def test_non_tpu_or_non_pp_does_not_require_geometry(monkeypatch, raw):
    monkeypatch.setattr(ConfigManager, '_get_raw_config', staticmethod(lambda: raw))
    assert NODE_LAYOUT(None, None) is None


def test_native_scheduler_seals_and_frees_only_after_due_job_ack(worker_factory, clock):
    path = os.environ.get('TPU_NATIVE_V6D_SOURCE')
    if not path:
        pytest.skip('set TPU_NATIVE_V6D_SOURCE to an offline native source snapshot')
    ns = native_classes(path, {
        'V6dObjectConnectorScheduler': {'update_connector_output'},
    }, {'NativeBase': object, 'time': adapter.time})
    scheduler = ns['V6dObjectConnectorScheduler']()
    worker = worker_factory()
    ns['V6dWorkerMetadata'] = type(worker._worker_meta)
    calls = []
    scheduler._drain_publications = lambda: None
    scheduler._publication_executor = None
    scheduler._try_drop_state = lambda rid: None
    scheduler._store_ranks = {0}
    scheduler._store_jobs = {1: NS(req_id='r', hashes={0: ['key']},
                                  protected_blocks=[17, 18], results={}, started_at=clock.value)}
    scheduler._request_states = {'r': NS(nan_store_failed=False, pending_store_jobs={1})}

    def seal(hashes, request_id):
        calls.append(('seal', hashes, request_id))
        return True

    scheduler.managers = {0: NS(seal=seal)}
    scheduler._block_pool = NS(free_blocks=lambda blocks: calls.append(('free', blocks)))
    scheduler._stats = NS(finish_job=lambda *args: None, observe=lambda *args: None)

    def consume():
        meta = worker.build_connector_worker_meta()
        scheduler.update_connector_output(NS(kv_connector_worker_meta=meta,
                                             finished_recving=None, finished_sending=None))
        return meta

    with adapter.tpu_transfer_step() as gate:
        submit(worker, metadata(waits={'r'}))
    assert not worker.get_finished()[0]
    consume()
    assert not calls and 1 in scheduler._store_jobs
    gate.ready_at = clock.value
    clock.value += DURATION / 2
    assert not worker.get_finished()[0]
    consume()
    assert not calls and 1 in scheduler._store_jobs
    clock.value += DURATION / 2
    assert worker.get_finished()[0] == {'r'}
    assert not calls  # adapter itself never seals or changes block protection
    snapshot = consume()
    assert calls == [('seal', ['key'], 'r'), ('free', [17, 18])]
    assert not scheduler._store_jobs
    scheduler.update_connector_output(NS(kv_connector_worker_meta=snapshot,
                                         finished_recving=None, finished_sending=None))
    assert len(calls) == 2


@pytest.mark.parametrize('unresolved', [False, True])
def test_preemption_fences_store_without_consuming_ack(worker_factory, clock, unresolved):
    worker = worker_factory()
    with adapter.tpu_transfer_step() as gate:
        submit(worker, metadata('a', 1, waits={'a'}))
    resolved = []
    def resolve():
        resolved.append(True)
        gate.ready_at = clock.value + 0.1
    if unresolved:
        gate.wait_ready = resolve
    else:
        gate.ready_at = clock.value
    worker.handle_preemptions(NS(preempted_req_ids={'other'}))
    assert not resolved and not clock.waits
    worker.handle_preemptions(NS(preempted_req_ids={'a'}))
    assert resolved == ([True] if unresolved else [])
    assert clock.waits == pytest.approx([DURATION + (0.1 if unresolved else 0)])
    assert results(worker) == {}
    assert worker._saving_req_jobs == {'a': {1}}
    assert worker.get_finished()[0] == {'a'}
    assert results(worker) == {1: {0: None}}
    assert not worker._sim_tpu_store_queue.seen


def test_retired_job_history_is_bounded(worker_factory, clock):
    worker = worker_factory()
    for job in range(100):
        submit(worker, metadata(job=job, masks=(0,)))
        results(worker)
    assert worker._sim_tpu_store_queue.highest_job_id == 99
    assert not worker._sim_tpu_store_queue.seen
    submit(worker, metadata(job=100))
    with pytest.raises(ValueError, match='Conflicting'):
        worker.start_store_kv(metadata('other', 100))
    clock.value += DURATION
    worker.get_finished()
    results(worker)
    assert not worker._sim_tpu_store_queue.seen
    worker.start_store_kv(metadata(job=50))
    assert results(worker) == {}


def test_scheduler_finish_guard_does_not_reset_multi_ack_count():
    from sglang_simulator.simulation.vllm.v6d.tpu_connector import install_tpu_v6d_completion_guard
    Child = type('TPUV6dObjectConnector', (), {'__module__': 'kvs_connector.v6d_object_connector'})
    calls = []
    request = NS(request_id='r', is_finished=lambda: True)
    scheduler = NS(requests={'r': request},
                   connector=NS(_connectors=[Child()], update_connector_output=lambda output: None),
                   _connector_finished=lambda req: (calls.append(req) or True, None))
    install_tpu_v6d_completion_guard(scheduler)
    assert scheduler._connector_finished(request) == (True, None)
    assert scheduler._connector_finished(request) == (True, None)
    assert calls == [request]
