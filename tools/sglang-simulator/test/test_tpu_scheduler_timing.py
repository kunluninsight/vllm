from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sglang_simulator.simulation.vllm.tpu_pp import scheduler_timing as timing


@pytest.mark.parametrize("native_ms,expected_ms", [(3, 12), (20, 20)])
def test_budget_replaces_measured_cost_without_double_count(
    monkeypatch, native_ms, expected_ms
):
    clock = [100.0]
    monkeypatch.setattr(timing.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        timing.time, "sleep", lambda x: clock.__setitem__(0, clock[0] + x)
    )
    monkeypatch.delenv("HISIM_TPU_PP_PREFILL_HANDOFF_MS", raising=False)
    result = object()
    connector_sentinel = object()

    def native(connector, output):
        assert connector is connector_sentinel
        clock[0] += native_ms / 1000
        return result

    scheduler = NS(_build_kv_connector_meta=native)
    emit = Mock()
    timing.install_scheduler_meta_timing(
        scheduler, dict(base_ms=2, cached_request_ms=3, query_token_ms=0.001), emit
    )
    output = NS(
        total_num_scheduled_tokens=4000,
        num_scheduled_tokens={"a": 2000, "b": 2000},
        scheduled_cached_reqs=NS(req_ids=["a", "b", "not_scheduled"]),
    )
    assert scheduler._build_kv_connector_meta(connector_sentinel, output) is result
    assert (clock[0] - 100) * 1000 == pytest.approx(expected_ms)
    assert emit.call_args.kwargs["target_ms"] == pytest.approx(12)
    assert emit.call_args.kwargs["added_ms"] == pytest.approx(max(0, 12 - native_ms))


def test_empty_and_failed_calls_keep_native_behavior(monkeypatch):
    monkeypatch.delenv("HISIM_TPU_PP_PREFILL_HANDOFF_MS", raising=False)
    sleep, emit = Mock(), Mock()
    monkeypatch.setattr(timing.time, "sleep", sleep)
    result = object()
    scheduler = NS(_build_kv_connector_meta=Mock(return_value=result))
    original = scheduler._build_kv_connector_meta
    timing.install_scheduler_meta_timing(
        scheduler, dict(base_ms=2, cached_request_ms=3, query_token_ms=0.001), emit
    )
    assert (
        scheduler._build_kv_connector_meta(None, NS(total_num_scheduled_tokens=0))
        is result
    )
    original.side_effect = RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        scheduler._build_kv_connector_meta(None, NS(total_num_scheduled_tokens=4000))
    sleep.assert_not_called()
    emit.assert_not_called()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_calibration_rejected(monkeypatch, value):
    monkeypatch.delenv("HISIM_TPU_PP_PREFILL_HANDOFF_MS", raising=False)
    with pytest.raises(ValueError):
        timing.install_scheduler_meta_timing(
            NS(), dict(base_ms=value, cached_request_ms=1, query_token_ms=0), Mock()
        )


def test_legacy_budget_must_be_removed(monkeypatch):
    monkeypatch.setenv("HISIM_TPU_PP_PREFILL_HANDOFF_MS", "8.185")
    with pytest.raises(ValueError, match="legacy"):
        timing.install_scheduler_meta_timing(
            NS(), dict(base_ms=1, cached_request_ms=1, query_token_ms=0), Mock()
        )


def test_disabled_preserves_original():
    original = Mock()
    scheduler = NS(_build_kv_connector_meta=original)
    timing.install_scheduler_meta_timing(scheduler, None, Mock())
    assert scheduler._build_kv_connector_meta is original
