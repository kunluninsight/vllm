from unittest.mock import Mock

import pytest

from sglang_simulator.simulation.vllm.tpu_pp import stage_runtime as rt


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("padding", [2048, 4096, 8192])
def test_worker_uses_model_duration_without_padding_correction(monkeypatch, rank, padding):
    prediction = 0.125
    model = Mock()
    model.predict.return_value = prediction
    monkeypatch.setattr(rt.StageModel, "from_file", lambda *args: model)
    monkeypatch.setattr(rt.native_control, "load", lambda *args: Mock(PPWave=object))
    monkeypatch.setattr(rt, "emit_hook_event", Mock())
    monkeypatch.setattr(rt.time, "monotonic", lambda: 100.0)
    wave = Mock(device_ready=101.0)
    wave.start_forward.return_value = {"hidden_states": rt.Descriptor(7, 102.0)}
    monkeypatch.setattr(rt, "_make_wave_class", lambda base: lambda *args: wave)
    features = {"global_padded_tokens": padding}
    payload = rt.StagePayload(7, 1024, features, {"prompt_complete_request_count": 1})
    control, reply, barrier, events = Mock(), Mock(), Mock(), Mock()
    control.recv.side_effect = [(1, "execute_model", payload, True),
                                (2, "shutdown", None, False)]
    cfg = rt.WorkerConfig(4, ["model"] * 4, 0.0008, True, 60)

    rt.worker_main(rank, cfg, control, reply, barrier, None, events)

    expected_features = {**features, **payload.tail} if rank == 3 else features
    model.predict.assert_called_once_with(expected_features)
    start = 101.0 if rank == 0 else 102.0
    wave.event.assert_called_once_with("compute", step_id=7, start_at=start,
                                      end_at=start + prediction, predicted_s=prediction)
    assert wave.device_ready == start + prediction
    reply.send.assert_any_call((1, True, (7, start + prediction)))
    barrier.abort.assert_not_called()


@pytest.mark.parametrize("all_ranks", [False, True])
def test_transfer_gate_uses_reliable_reply_ready_time(all_ranks):
    from types import SimpleNamespace as NS

    executor = object.__new__(rt.VirtualPPExecutor)
    executor._next_call = 0
    executor._next_step = 7
    executor.stages = 4
    executor.output_rank = 3
    executor.emulate_connector_queue = False
    executor._commands = [Mock() for _ in range(4)]
    executor.response_mqs = [
        NS(remote_socket=Mock(recv=Mock(return_value=(1, True, (7, 101.0 + rank)))))
        for rank in range(4)
    ]
    executor.futures_queue = []
    executor.native = NS(FutureWrapper=lambda queue, response: NS(result=response))
    gate = NS(ready_at=None)
    output = object()

    future = executor._stage_rpc("sample_tokens", None, all_ranks, output,
                                 transfer_step=gate)
    assert gate.ready_at is None
    assert future.result() is output
    assert gate.ready_at == 104.0
    for rank, connection in enumerate(executor._commands):
        connection.send.assert_called_once_with((1, "sample_tokens", None,
                                                all_ranks or rank == 3))


def test_failed_transfer_reply_does_not_open_gate():
    from types import SimpleNamespace as NS

    executor = object.__new__(rt.VirtualPPExecutor)
    executor._next_call = 0
    executor._next_step = 7
    executor.stages = 1
    executor.output_rank = 0
    executor.emulate_connector_queue = False
    executor._commands = [Mock()]
    executor.response_mqs = [NS(remote_socket=Mock(recv=Mock(return_value=(1, False, "failed"))))]
    executor.futures_queue = []
    executor.native = NS(FutureWrapper=lambda queue, response: NS(result=response))
    gate = NS(ready_at=None)
    future = executor._stage_rpc("sample_tokens", None, False, None,
                                 transfer_step=gate)
    with pytest.raises(RuntimeError, match="stage worker failed"):
        future.result()
    assert gate.ready_at is None


def test_empty_worker_reply_has_current_ready_time(monkeypatch):
    model = Mock()
    monkeypatch.setattr(rt.StageModel, "from_file", lambda *args: model)
    monkeypatch.setattr(rt.native_control, "load", lambda *args: Mock(PPWave=object))
    monkeypatch.setattr(rt, "emit_hook_event", Mock())
    monkeypatch.setattr(rt.time, "monotonic", lambda: 103.0)
    wave = Mock(device_ready=101.0)
    monkeypatch.setattr(rt, "_make_wave_class", lambda base: lambda *args: wave)
    payload = rt.StagePayload(0, 0)
    control, reply, barrier, events = Mock(), Mock(), Mock(), Mock()
    control.recv.side_effect = [(1, "execute_model", payload, True),
                                (2, "shutdown", None, False)]
    cfg = rt.WorkerConfig(4, ["model"] * 4, 0.0008, True, 60)
    rt.worker_main(3, cfg, control, reply, barrier, None, events)
    reply.send.assert_any_call((1, True, (0, 103.0)))
    model.predict.assert_not_called()
    barrier.abort.assert_not_called()


@pytest.mark.parametrize("tokens", [0, 128])
def test_execute_binds_store_gate_until_sample_reply(monkeypatch, tokens):
    from types import SimpleNamespace as NS
    from sglang_simulator.simulation.vllm.v6d import tpu_connector as connector

    gates = []
    calls = []
    output = object()
    def execute(scheduled, non_block):
        calls.append("execute")
        gates.append(connector._transfer_gate.get())
        return output

    executor = object.__new__(rt.VirtualPPExecutor)
    executor.inner = NS(execute_model=execute, sample_tokens=lambda *a, **kw: output)
    executor.lifecycle = None
    executor._next_step = 0
    executor._empty_host_s = None
    executor._aggregated = lambda: False
    executor._spans = lambda scheduled: []
    executor.predictor = NS(features_for=lambda spans: ({}, {}), host_prep={})
    executor._tpu_wave_state = Mock()
    executor.step_records = {}
    executor._emit_lifecycle = Mock()
    executor._on_step_done = None
    executor._stage_rpc = Mock(side_effect=lambda method, payload, aggregate, value, **kw:
                               NS(result=lambda: value, add_done_callback=Mock()))
    monkeypatch.setattr(rt, "host_prep_s", lambda *args: 0.0)
    monkeypatch.setattr(connector, "wait_for_tpu_store_progress", lambda: calls.append("wait"))
    scheduled = NS(total_num_scheduled_tokens=tokens)

    executor.execute_model(scheduled, non_block=True)

    assert len(gates) == 1 and gates[0] is not None
    assert gates[0].ready_at is None
    assert connector._transfer_gate.get() is None
    if tokens:
        assert calls == ["execute"]
        assert "transfer_step" not in executor._stage_rpc.call_args.kwargs
        assert executor._pending_sample[3] is gates[0]
        executor.sample_tokens(None, non_block=True)
        assert executor._stage_rpc.call_args.args[0] == "sample_tokens"
    else:
        assert calls == ["wait", "execute"]
    assert executor._stage_rpc.call_args.kwargs["transfer_step"] is gates[0]
    assert gates[0].ready_at is None
