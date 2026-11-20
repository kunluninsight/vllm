from collections import deque
from types import SimpleNamespace as NS

import pytest

from sglang_simulator.simulation.vllm.tpu_pp.stage_runtime import VirtualPPExecutor


@pytest.mark.parametrize('prefill_only', [False, True])
def test_mtp_prefill_draft_read_does_not_drain_the_pipeline(monkeypatch, prefill_only):
    monkeypatch.setenv('TPU_MTP_PREFILL_ONLY', '1' if prefill_only else '0')
    calls = []
    sentinel = object()

    class Reply:
        def _wait_for_response(self):
            calls.append('drain')

    class Inner:
        def collective_rpc(self, method):
            calls.append(method)
            return sentinel

        def take_draft_token_ids(self):
            return self.collective_rpc('take_draft_token_ids')

    executor = VirtualPPExecutor.__new__(VirtualPPExecutor)
    executor.inner = Inner()
    executor.native = NS(STEP_METHODS=frozenset({'execute_model', 'sample_tokens'}))
    executor._tpu_wave_state = NS(settle=lambda owner: calls.append('settle'))
    reply = Reply()
    executor.futures_queue = deque([reply])
    executor._patch_inner_collective_rpc()
    result = executor.take_draft_token_ids()
    if prefill_only:
        assert result is None
        assert calls == []
        assert list(executor.futures_queue) == [reply]
    else:
        assert result is sentinel
        assert calls == ['settle', 'drain', 'take_draft_token_ids']
        assert not executor.futures_queue
