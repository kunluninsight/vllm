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
    reply.send.assert_any_call((1, True, 7))
    barrier.abort.assert_not_called()
