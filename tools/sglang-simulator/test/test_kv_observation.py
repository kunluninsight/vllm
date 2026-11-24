import asyncio
import importlib.util
import sys
from types import SimpleNamespace as NS, ModuleType

import pytest

from sglang_simulator import kv_observation as obs
from sglang_simulator.cache_boundaries import install_connector, install_scheduler_transitions
from sglang_simulator.lifecycle_event_kinds import LIFECYCLE_EVENT_KINDS


def recorder():
    rows = []
    return rows, lambda event, **data: rows.append((event, data)), lambda fn: fn()


def query_fixture(monkeypatch, modern=False):
    module = ModuleType('hisim_query_fixture')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    lookup_result, fetched = object(), object()

    class Manager:
        @staticmethod
        async def async_lookup_results_many(*args, **kwargs):
            calls.append(('lookup', args, kwargs))
            await asyncio.sleep(0)
            return lookup_result

        @classmethod
        async def async_fetch_selected_many(cls, *args, **kwargs):
            calls.append(('fetch', args, kwargs))
            return fetched

    class Scheduler:
        async def _async_match_impl(self, request_id, generation):
            assert await Manager.async_lookup_results_many({'group': 1}, request_id=request_id) is lookup_result
            assert await Manager.async_fetch_selected_many(iter([1])) is fetched
            return NS(generation=generation, hit_length=64)

        async def _async_match(self, request_id, generation):
            return await self._async_match_impl(request_id, generation)

        def _commit_async_match(self, request, state, result, **kwargs):
            calls.append(('consume', request, state, result, kwargs))
            return result
    Scheduler.__module__ = module.__name__
    module.V6dObjectManager = Manager
    if modern:
        del Scheduler._async_match
    return Scheduler(), calls, Manager


@pytest.mark.parametrize('modern', [False, True])
def test_query_stages_correlate_without_extra_calls(monkeypatch, modern):
    scheduler, calls, _ = query_fixture(monkeypatch, modern)
    rows, emit, observe = recorder()
    obs.install_query(scheduler, emit, observe)
    obs.install_query(scheduler, emit, observe)
    method = scheduler._async_match_impl if modern else scheduler._async_match

    async def run():
        result = await method('r', 2)
        assert scheduler._commit_async_match(NS(request_id='r'), None, result, flag=True) is result
    asyncio.run(run())
    assert [c[0] for c in calls] == ['lookup', 'fetch', 'consume']
    selected = [(event, d) for event, d in rows if event != 'request.v6d_query_capabilities']
    assert len({d['query_id'] for _, d in selected}) == 1
    consume = dict(selected)['request.v6d_query_consume']
    assert consume['query_linked'] is True
    assert consume['created_mono_ns'] <= consume['query_start_mono_ns'] <= consume['query_end_mono_ns'] <= consume['start_mono_ns']
    assert {e for e, _ in selected} <= LIFECYCLE_EVENT_KINDS


def test_concurrent_queries_have_separate_context(monkeypatch):
    scheduler, calls, Manager = query_fixture(monkeypatch)
    rows, emit, observe = recorder()
    obs.install_query(scheduler, emit, observe)
    async def run():
        await asyncio.gather(scheduler._async_match('a', 1), scheduler._async_match('b', 1))
        # An unrelated native call has no query context and creates no events.
        n = len(rows)
        await Manager.async_lookup_results_many({})
        assert len(rows) == n
    asyncio.run(run())
    by_id = {}
    for e, d in rows:
        if e.endswith('_start') and 'query_id' in d:
            by_id.setdefault(d['query_id'], set()).add(d['request_id_sha256'])
    assert len(by_id) == 2 and all(len(ids) == 1 for ids in by_id.values())


def test_observer_errors_do_not_change_native_query(monkeypatch):
    scheduler, calls, _ = query_fixture(monkeypatch)
    def broken(*args, **kwargs):
        raise RuntimeError('observer')
    obs.install_query(scheduler, broken, lambda fn: fn())
    assert asyncio.run(scheduler._async_match('r', 1)).hit_length == 64
    assert len(calls) == 2


def test_scheduler_job_ack_and_release_keep_result_identity():
    rows, emit, observe = recorder()
    request = NS(request_id='r')
    job = NS(req_id='r')
    metadata = NS(store_job_ids={'r': 7}, store_waits={'r'})
    marker = object()
    class Scheduler:
        def __init__(self):
            self._store_jobs = {7: job}
            self.calls = []
        def build_connector_meta(self, output):
            self.calls.append(('build', output)); return metadata
        def update_connector_output(self, output):
            self.calls.append(('ack', output)); del self._store_jobs[7]; return marker
        def request_finished_all_groups(self, request, blocks):
            self.calls.append(('finish', request, blocks)); return True, marker
    owner = Scheduler()
    obs.install_scheduler_jobs(owner, emit, observe)
    assert owner.build_connector_meta(marker) is metadata
    ack = NS(kv_connector_worker_meta=NS(store_results={7: {0: None, 1: None}}))
    assert owner.update_connector_output(ack) is marker
    assert owner.request_finished_all_groups(request, marker) == (True, marker)
    assert len(owner.calls) == 3
    event = next(d for e, d in rows if e == 'request.kv_job_ack_consumed')
    assert event['request_id_sha256'] == obs._request_hash('r')
    assert event['store_job_id'] == 7 and event['ack_rank_count'] == 2 and event['job_retired'] is True
    assert event['ack_observed_mono_ns'] <= event['end_mono_ns']


def test_worker_observes_existing_poll_and_returns_same_objects():
    rows, emit, observe = recorder()
    done = [NS(job_id=3, error=None)]
    class Handler:
        def __init__(self): self.calls = []
        def swap(self, *args, **kwargs): self.calls.append(('swap', args, kwargs)); return done
        def get_finished(self): self.calls.append(('poll',)); return done
    handler = Handler()
    class Worker:
        _worker_rank = 0
        _node_id = 1
        def __init__(self):
            self._load_handler = None; self._store_handler = None
            self._pending_jobs = {3: 'r'}; self._store_job_ids = {3: 7}
        def register_kv_caches(self, marker): self._store_handler = handler; return marker
        def get_finished(self, ids): return set(), set()
        def build_connector_worker_meta(self): return None
    worker = Worker()
    obs.install_worker(worker, emit, observe)
    marker = object()
    assert worker.register_kv_caches(marker) is marker
    obs.install_worker(worker, emit, observe)
    assert handler.swap(3, [], [], 'r', option=marker) is done
    assert handler.get_finished() is done
    assert [c[0] for c in handler.calls] == ['swap', 'poll']
    event = next(d for e, d in rows if e == 'request.kv_job_ready_observed')
    assert event['store_job_id'] == 7 and event['native_job_id'] == 3
    assert event['worker_rank'] == 0 and event['node_id'] == 1


def test_generator_completion_is_not_consumed_by_observer():
    rows, emit, observe = recorder()
    advanced = []
    def gen():
        advanced.append(True); yield NS(job_id=3)
    iterator = gen()
    class Handler:
        def get_finished(self): return iterator
    worker = NS(_load_handler=Handler(), _store_handler=None)
    obs.install_worker(worker, emit, observe)
    assert worker._load_handler.get_finished() is iterator
    assert advanced == []


def test_native_exception_identity_preserved_for_sync_observer():
    rows, emit, observe = recorder()
    error = ValueError('native')
    class Scheduler:
        def _free_blocks(self, request): raise error
    scheduler = Scheduler()
    obs.install_release(scheduler, emit, observe)
    with pytest.raises(ValueError) as raised:
        scheduler._free_blocks(NS(request_id='r'))
    assert raised.value is error
    assert rows[-1][0] == 'request.kv_blocks_release' and rows[-1][1]['native_ok'] is False


def test_missing_query_contract_is_explicit():
    rows, emit, observe = recorder()
    obs.install_query(NS(), emit, observe)
    assert rows[-1][1]['installed'] is False


def test_fork_reset_drops_inherited_recorder():
    old = obs._RECORDER
    try:
        obs._RECORDER = object()
        obs._after_fork()
        assert obs._RECORDER is None and obs._QUERY.get() is None
    finally:
        obs._RECORDER = old


def test_constructor_hook_runs_native_once(monkeypatch):
    from sglang_simulator import transfer_observation_hook as hooks
    rows, emit, _ = recorder()
    monkeypatch.setattr(hooks, 'worker_emitter', lambda: emit)
    calls = []
    class Worker:
        def __init__(self, value): calls.append(value)
    hooks.C_TransferWorkerObservationHook.hook(Worker)
    hooks.C_TransferWorkerObservationHook.hook(Worker)
    marker = object()
    Worker(marker)
    assert calls == [marker]
    assert any(e == 'request.kv_worker_capabilities' for e, _ in rows)


def test_bounded_query_state_reports_eviction(monkeypatch):
    monkeypatch.setattr(obs, '_LIMIT', 2)
    scheduler, _, _ = query_fixture(monkeypatch)
    rows, emit, observe = recorder()
    obs.install_query(scheduler, emit, observe)
    async def run():
        results = [await scheduler._async_match('r' + str(i), i) for i in range(3)]
        scheduler._commit_async_match(NS(request_id='r0'), None, results[0])
        scheduler._commit_async_match(NS(request_id='r2'), None, results[2])
    asyncio.run(run())
    assert any(e == 'request.kv_observation_evicted' for e, _ in rows)
    consumed = [d for e, d in rows if e == 'request.v6d_query_consume']
    assert [d['query_linked'] for d in consumed] == [False, True]


def test_raiden_handoff_records_native_call_not_completion():
    rows, emit, observe = recorder()
    marker = object()
    calls = []
    class Worker:
        def process_send_load(self, metadata, wait_for_completion=False):
            calls.append((metadata, wait_for_completion)); return marker
    worker = Worker()
    obs.install_worker(worker, emit, observe)
    metadata = NS(reqs_to_send={'r': NS(num_tokens=3072)})
    assert worker.process_send_load(metadata, wait_for_completion=True) is marker
    assert calls == [(metadata, True)]
    data = next(d for e, d in rows if e == 'request.kv_worker_submit')
    assert data['num_tokens'] == 3072 and data['direction'] == 'handoff'
    assert data['completion_scope'] == 'worker_call_return_not_transfer_completion'
