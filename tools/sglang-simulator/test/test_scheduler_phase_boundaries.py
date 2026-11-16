from types import SimpleNamespace

import pytest

from sglang_simulator.cache_boundaries import install_connector, install_scheduler_transitions


class Connector:
    def __init__(self):
        self.calls = []
        self.result = object()

    def build_connector_meta(self, scheduler_output):
        self.calls.append(scheduler_output)
        if getattr(scheduler_output, "fail", False):
            raise ValueError("native metadata failure")
        return self.result


class Scheduler:
    def __init__(self):
        self.calls = []

    def _build_kv_connector_meta(self, connector, scheduler_output):
        self.calls.append(("meta", scheduler_output))
        return connector.build_connector_meta(scheduler_output)

    def _make_cached_request_data(self, running, resumed, num_scheduled_tokens, spec, blocks):
        self.calls.append(("cached", num_scheduled_tokens))
        return blocks

    def _update_after_schedule(self, scheduler_output):
        self.calls.append(("update", scheduler_output))
        return scheduler_output


def setup(emit=None):
    rows = []
    emit = emit or (lambda event, **data: rows.append((event, data)))
    scheduler, connector = Scheduler(), Connector()
    for _ in range(2):
        install_scheduler_transitions(scheduler, emit, lambda f: f())
        install_connector(connector, emit, lambda f: f())
    return scheduler, connector, rows


def phases(rows):
    return [d for event, d in rows if event == "scheduler.native_phase"]


@pytest.mark.parametrize("tokens", [0, None])
def test_empty_or_unknown_batches_are_not_observed(tokens):
    scheduler, connector, rows = setup()
    output = SimpleNamespace(total_num_scheduled_tokens=tokens)
    assert scheduler._build_kv_connector_meta(connector, output) is connector.result
    assert scheduler._update_after_schedule(output) is output
    blocks = object()
    assert scheduler._make_cached_request_data([], [], {}, {}, blocks) is blocks
    assert len(scheduler.calls) == 3 and connector.calls == [output]
    assert phases(rows) == []


def test_nonempty_phase_boundaries_preserve_results_and_nesting():
    scheduler, connector, rows = setup()
    output = SimpleNamespace(total_num_scheduled_tokens=512)
    assert scheduler._build_kv_connector_meta(connector, output) is connector.result
    assert scheduler._update_after_schedule(scheduler_output=output) is output
    counts, blocks = {"r": 512}, object()
    assert scheduler._make_cached_request_data([], [], counts, {}, blocks) is blocks
    events = phases(rows)
    assert len(events) == 4  # two metadata levels, cached data, update; no duplicate install
    child, parent = events[:2]
    assert child["phase"] == "build_connector_meta"
    assert parent["phase"] == "_build_kv_connector_meta"
    assert parent["start_mono_ns"] <= child["start_mono_ns"] <= child["end_mono_ns"] <= parent["end_mono_ns"]
    assert all(e["native_ok"] and e["total_num_scheduled_tokens"] == 512 for e in events)
    assert all(e["batch_shared"] and e["host_span_inclusive"] for e in events)
    assert counts == {"r": 512} and len(scheduler.calls) == 3


def test_native_failure_and_observer_failure_are_preserved():
    scheduler, connector, rows = setup()
    output = SimpleNamespace(total_num_scheduled_tokens=512, fail=True)
    with pytest.raises(ValueError, match="native metadata failure"):
        scheduler._build_kv_connector_meta(connector, output)
    assert len(phases(rows)) == 2
    assert all(not e["native_ok"] and e["native_error_type"] == "ValueError" for e in phases(rows))

    def broken(*args, **kwargs):
        raise RuntimeError("observer failure")

    scheduler, connector, _ = setup(broken)
    output.fail = False
    assert scheduler._build_kv_connector_meta(connector, output) is connector.result
    assert connector.calls == [output]


@pytest.mark.parametrize("training,tick", [(False, False), (True, False), (True, True)])
def test_phase_events_survive_real_stage_focus(tmp_path, monkeypatch, training, tick):
    from sglang_simulator.collector.vllm_hook import stage_collection, trace_v2

    monkeypatch.setenv("SIM_TRACE_FOCUS", "pp_stage")
    monkeypatch.setattr(stage_collection, "TRAINING", training)
    monkeypatch.setattr(stage_collection, "TICK", tick)
    recorder = trace_v2.Recorder(tmp_path)
    try:
        scheduler, connector, _ = setup(
            lambda event, **data: recorder.emit(event, data=data))
        before = recorder.accepted
        output = SimpleNamespace(total_num_scheduled_tokens=512)
        assert scheduler._build_kv_connector_meta(connector, output) is connector.result
        # Exercise the real emitter, including its tick-mode retention filter.
        assert recorder.accepted == before + 2
        recorder.identity = (0, 0, 0, 0, 0, 0, 0)
        assert not recorder.emit("scheduler.native_phase", data={})
    finally:
        recorder.close()
    assert recorder.written == before + 2 and recorder.failure is None
