import sys
from types import SimpleNamespace as NS

import pytest

from sglang_simulator.simulation.vllm.v6d.tpu_connector import C_TPURaidenConnectorHook


class MambaSpec:
    pass


def connector(monkeypatch, cache='mamba', producer=True, native_stage3=False):
    # CPU transport does not initialize native Stage-3 geometry or truncate.
    module = NS(MambaSpec=MambaSpec,
                _stage3_group_window_tokens=lambda spec: getattr(spec, 'sliding_window', None))
    monkeypatch.setitem(sys.modules,
                        'vllm_torchtpu.distributed.kv_transfer.tpu_connector', module)

    class Scheduler:
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
