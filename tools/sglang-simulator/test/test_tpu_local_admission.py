from types import SimpleNamespace as NS

import pytest

from sglang_simulator.simulation.vllm.v6d.tpu_connector import C_TPURaidenConnectorHook


def connector(truncates=True, producer=True):
    class Native:
        def __init__(self, config, role, groups):
            self.connector_scheduler = object()
            self.admitted = []

        def on_new_request(self, request):
            self.admitted.append(request)
            params = request.kv_transfer_params or {}
            if producer and truncates and request.num_prompt_tokens > 1 and not params.get("_p_side_truncated"):
                token = request.prompt_token_ids.pop()
                request._all_token_ids.pop()
                request.num_prompt_tokens -= 1
                request.kv_transfer_params = dict(params, _p_side_truncated=True,
                                                 _p_side_next_prompt_token_id=token)

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
    config = NS(kv_transfer_config=NS(kv_role="kv_producer" if producer else "kv_consumer"),
                parallel_config=NS())
    return Native(config, None, NS(kv_cache_groups=[object(), object()]))


@pytest.mark.parametrize("truncates,length", [(True, 4), (True, 1), (False, 4)])
def test_native_prefill_extent_and_frontend_input_are_preserved(truncates, length):
    client_tokens = list(range(length))
    request = NS(prompt_token_ids=client_tokens, _all_token_ids=client_tokens.copy(),
                 num_prompt_tokens=length, kv_transfer_params=None)
    c = connector(truncates)
    c.on_new_request(request)
    expected = length - int(truncates and length > 1)
    assert request.num_prompt_tokens == expected
    assert request.prompt_token_ids == list(range(expected))
    assert request._all_token_ids == list(range(expected))
    assert client_tokens == list(range(length))
    assert c.admitted == [request]
    c.on_new_request(request)
    assert request.num_prompt_tokens == expected


def test_remote_handoff_is_still_rejected_before_admission():
    c = connector()
    request = NS(prompt_token_ids=[1, 2], kv_transfer_params={"remote_host": "other"})
    with pytest.raises(ValueError, match="remote PD handoff"):
        c.on_new_request(request)
    assert not c.admitted and request.prompt_token_ids == [1, 2]


def test_nonlocal_connector_keeps_native_admission():
    c = connector(producer=False)
    request = NS(prompt_token_ids=[1, 2], _all_token_ids=[1, 2],
                 num_prompt_tokens=2, kv_transfer_params=None)
    c.on_new_request(request)
    assert c.admitted == [request] and request.num_prompt_tokens == 2
