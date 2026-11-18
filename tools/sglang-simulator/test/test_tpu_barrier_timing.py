from unittest.mock import Mock

import pytest

from sglang_simulator.simulation.vllm.tpu_pp import stage_runtime as rt


@pytest.mark.parametrize('overhead', [0.0, 0.00058, 0.0013])
def test_barrier_blocks_host_but_payload_readiness_can_overlap(monkeypatch, overhead):
    clock = [100.0]
    monkeypatch.setattr(rt.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(rt.time, 'sleep', lambda delay: clock.__setitem__(0, clock[0] + delay))
    barrier = Mock()
    barrier.wait.side_effect = lambda **kw: clock.__setitem__(0, clock[0] + 0.002)
    incoming, outgoing, events = Mock(), Mock(), Mock()
    incoming.recv.return_value = ((0, 0), (rt.Descriptor(step_id=7),))
    cfg = rt.WorkerConfig(4, [], 0.020, False, 60, barrier_overhead_s=overhead)
    wave = rt._make_wave_class(object)(0, cfg, barrier, (incoming, outgoing), events)
    wave.device_ready = 100.050
    result = wave._launch_blocking((rt.Descriptor(step_id=3),))
    assert clock[0] == pytest.approx(100.052 + overhead)
    assert result[0].ready_at == pytest.approx(clock[0] + 0.020)
    event = events.put.call_args.args[0]
    assert event['native_return_at'] == pytest.approx(100.052)
    assert event['exit_at'] - event['native_return_at'] == pytest.approx(overhead)
    assert event['outgoing_step_id'] == 3 and event['incoming_step_id'] == 7
    # Host work after the call hides payload latency, but cannot hide barrier overhead.
    rt.Descriptor(ready_at=clock[0] + 0.030).wait()
    before = clock[0]
    result[0].wait()
    assert clock[0] == before


def test_barrier_failure_preserves_abort(monkeypatch):
    barrier, incoming, outgoing, events = Mock(), Mock(), Mock(), Mock()
    barrier.wait.side_effect = RuntimeError('barrier aborted')
    cfg = rt.WorkerConfig(4, [], 0.020, False, 60, barrier_overhead_s=0.0013)
    wave = rt._make_wave_class(object)(0, cfg, barrier, (incoming, outgoing), events)
    with pytest.raises(RuntimeError, match='barrier aborted'):
        wave._launch_blocking(())
    outgoing.send.assert_not_called()
    events.put.assert_not_called()
