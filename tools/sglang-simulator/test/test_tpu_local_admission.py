import sys
from types import SimpleNamespace as NS
from enum import IntEnum

import pytest

from sglang_simulator.simulation.vllm.v6d.tpu_connector import (
    C_TPURaidenConnectorHook,
    C_TPURaidenConnectorWorkerHook,
)


class RequestStatus(IntEnum):
    FINISHED_LENGTH_CAPPED = 1
    FINISHED_STOPPED = 2
    FINISHED_ABORTED = 3


class TPUConnectorMetadata:
    def __init__(self):
        self.reqs_to_send = {}
        self.reqs_to_load = {}


class MambaSpec:
    pass


def connector(monkeypatch, cache='mamba', producer=True, native_stage3=False):
    # CPU transport does not initialize native Stage-3 geometry or truncate.
    module = NS(MambaSpec=MambaSpec, TPUConnectorMetadata=TPUConnectorMetadata,
                _stage3_group_window_tokens=lambda spec: getattr(spec, 'sliding_window', None))
    monkeypatch.setitem(sys.modules,
                        'vllm_torchtpu.distributed.kv_transfer.tpu_connector', module)

    monkeypatch.setitem(sys.modules, 'vllm.v1.request', NS(RequestStatus=RequestStatus))

    class Scheduler:
        @staticmethod
        def _stage3_transfer_tokens(request, computed_tokens):
            prompt = request.num_prompt_tokens
            truncated = (request.kv_transfer_params or {}).get('_p_side_truncated')
            return min(computed_tokens, prompt - int(not truncated))

        def _maybe_truncate_prompt_for_handoff(self, request):
            params = request.kv_transfer_params or {}
            if request.num_prompt_tokens > 1 and not params.get('_p_side_truncated'):
                token = request.prompt_token_ids.pop()
                request._all_token_ids.pop()
                request.num_prompt_tokens -= 1
                request.kv_transfer_params = dict(params, _p_side_truncated=True,
                                                 _p_side_next_prompt_token_id=token)

    class Native:
        def __init__(self, config, role, groups):
            self.connector_scheduler = Scheduler()
            self.admitted = []

        def on_new_request(self, request):
            self.admitted.append(request)
            if producer and native_stage3 and cache != 'full':
                self.connector_scheduler._maybe_truncate_prompt_for_handoff(request)

        def get_num_new_matched_tokens(self, *args):
            return 0, False

        def update_state_after_alloc(self, *args):
            pass

        def build_connector_meta(self, *args):
            pass

        def update_connector_output(self, output):
            raise AssertionError('local handoff must not enter native Stage-3')

        def request_finished(self, *args):
            return False, None

        def request_finished_all_groups(self, *args):
            return False, None

    C_TPURaidenConnectorHook.hook(Native)
    config = NS(kv_transfer_config=NS(kv_role='kv_producer' if producer else 'kv_consumer'),
                parallel_config=NS())
    spec = MambaSpec() if cache == 'mamba' else NS(sliding_window=4096 if cache == 'window' else None)
    return Native(config, None, NS(kv_cache_groups=[NS(kv_cache_spec=NS()), NS(kv_cache_spec=spec)]))


@pytest.mark.parametrize('cache,length,native_stage3', [
    ('mamba',4,False), ('mamba',1,False), ('full',4,False),
    ('window',4,False), ('mamba',4,True),
])
def test_prefill_extent_without_stage3_transport(monkeypatch, cache, length, native_stage3):
    client_tokens = list(range(length))
    request = NS(prompt_token_ids=client_tokens, _all_token_ids=client_tokens.copy(),
                 num_prompt_tokens=length, kv_transfer_params=None)
    c = connector(monkeypatch, cache, native_stage3=native_stage3)
    c.on_new_request(request)
    expected = length - int(cache != 'full' and length > 1)
    assert request.num_prompt_tokens == expected
    assert request.prompt_token_ids == list(range(expected))
    assert request._all_token_ids == list(range(expected))
    assert client_tokens == list(range(length))
    assert c.admitted == [request]
    c.on_new_request(request)
    assert request.num_prompt_tokens == expected


def test_remote_handoff_is_still_rejected_before_admission(monkeypatch):
    c = connector(monkeypatch)
    request = NS(prompt_token_ids=[1, 2], kv_transfer_params={'remote_host': 'other'})
    with pytest.raises(ValueError, match='remote PD handoff'):
        c.on_new_request(request)
    assert not c.admitted and request.prompt_token_ids == [1, 2]


def test_nonlocal_connector_keeps_native_admission(monkeypatch):
    c = connector(monkeypatch, producer=False)
    request = NS(prompt_token_ids=[1, 2], _all_token_ids=[1, 2],
                 num_prompt_tokens=2, kv_transfer_params=None)
    c.on_new_request(request)
    assert c.admitted == [request] and request.num_prompt_tokens == 2


def raiden_worker(native_pending=True):
    class NativeWorker:
        def get_finished(self, *args):
            raise AssertionError('real transport must not run')

        if native_pending:
            def has_pending_push_work(self):
                return False

        def process_send_load(self, *args):
            raise AssertionError('real transport must not run')

    C_TPURaidenConnectorWorkerHook.hook(NativeWorker)
    worker = NativeWorker()
    worker.is_producer = True
    return worker


def test_raiden_worker_without_native_pending_method(monkeypatch):
    monkeypatch.setenv('HISIM_TPU_RAIDEN_HANDOFF_BASE_S', '0')
    monkeypatch.setenv('HISIM_TPU_RAIDEN_HANDOFF_BANDWIDTH_GB', '0')
    worker = raiden_worker(native_pending=False)
    assert not worker.has_pending_push_work()
    worker.process_send_load(NS(reqs_to_send={'r': NS(num_tokens=1)}))
    assert worker.has_pending_push_work()
    assert worker.get_finished() == ({'r'}, set())
    assert not worker.has_pending_push_work()


def finished_request(status=RequestStatus.FINISHED_LENGTH_CAPPED, tokens=7):
    return NS(request_id='r', status=status, num_computed_tokens=tokens,
              num_prompt_tokens=8, kv_transfer_params=None)


@pytest.mark.parametrize('status', [RequestStatus.FINISHED_LENGTH_CAPPED,
                                    RequestStatus.FINISHED_STOPPED])
def test_local_handoff_pin_metadata_ack_and_scheduler_consumption(monkeypatch, status):
    c = connector(monkeypatch)
    request = finished_request(status)
    blocks = ([4, 5], [9])
    assert c.request_finished_all_groups(request, blocks) == (True, None)
    assert request.kv_transfer_params is None
    assert c.has_pending_push_work()
    assert c.request_finished_all_groups(request, blocks) == (True, None)
    meta = c.build_connector_meta(None)
    assert isinstance(meta, TPUConnectorMetadata)
    assert meta.reqs_to_send['r'].group_block_ids == ((4, 5), (9,))
    assert meta.reqs_to_send['r'].num_tokens == 7
    blocks[0].append(6)
    assert meta.reqs_to_send['r'].group_block_ids == ((4, 5), (9,))
    assert not c.build_connector_meta(None).reqs_to_send
    worker = raiden_worker()
    worker.process_send_load(meta)
    sent, received = worker.get_finished()
    assert sent == {'r'} and not received
    assert not worker.get_finished()[0]
    # Worker ACK alone is not scheduler consumption (nor a manual block free).
    assert c.has_pending_push_work()
    c.update_connector_output(NS(finished_sending=sent))
    assert not c.has_pending_push_work()
    assert c.request_finished_all_groups(request, blocks) == (False, None)
    assert not c.build_connector_meta(None).reqs_to_send


@pytest.mark.parametrize('status,tokens,blocks', [
    (RequestStatus.FINISHED_ABORTED, 7, ([1], [2])),
    (RequestStatus.FINISHED_STOPPED, 0, ([1], [2])),
    (RequestStatus.FINISHED_STOPPED, 7, ([], [])),
])
def test_local_abort_empty_or_no_tokens_never_pins(monkeypatch, status, tokens, blocks):
    c = connector(monkeypatch)
    assert c.request_finished_all_groups(finished_request(status, tokens), blocks) == (False, None)
    assert not c.has_pending_push_work()
    assert not c.build_connector_meta(None).reqs_to_send


def test_local_handoff_validates_all_group_tables(monkeypatch):
    c = connector(monkeypatch)
    with pytest.raises(ValueError, match='group count mismatch'):
        c.request_finished_all_groups(finished_request(), ([1],))
