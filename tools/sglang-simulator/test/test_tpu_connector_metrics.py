from dataclasses import dataclass
import sys
from types import SimpleNamespace

import pytest

from sglang_simulator.simulation.vllm.v6d.tpu_connector import C_MultiKVConnectorPromMetricsHook


@dataclass
class Stats:
    data: dict


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        SimpleNamespace(KVConnectorStats=Stats),
    )

    class NativeProm:
        def __init__(self):
            self.samples = []

        def observe(self, data, engine_idx=0):
            # r58 MultiKVConnectorPromMetrics consumes serialized child stats.
            for connector_id, stats in data.items():
                assert connector_id in ("V6D", "Nested")
                self.samples.append((connector_id, stats["data"], engine_idx))
            return "observed"

    C_MultiKVConnectorPromMetricsHook.hook(NativeProm)
    return NativeProm()


@pytest.mark.parametrize("in_process", [True, False])
def test_typed_and_ipc_stats_keep_metric_values(metrics, in_process):
    values = {"query_ms": [1.25, 4.5], "hits": 0, "failures": 2}
    child = Stats(values) if in_process else {"data": values}
    data = {"V6D": child}
    assert metrics.observe(data, engine_idx=7) == "observed"
    assert metrics.samples == [("V6D", values, 7)]
    assert data["V6D"] is child
    assert (child.data if in_process else child["data"]) == values


def test_mixed_and_nested_stats_are_not_reduced_or_reset(metrics):
    leaf = Stats({"query_ms": [1.5, 3.5]})
    nested = Stats({"V6D": leaf})
    data = {"Nested": nested, "V6D": {"data": {"hits": 0}}}
    metrics.observe(data)
    assert metrics.samples == [("Nested", {"V6D": leaf}, 0), ("V6D", {"hits": 0}, 0)]
    assert nested.data["V6D"] is leaf and leaf.data["query_ms"] == [1.5, 3.5]


def test_native_validation_is_preserved(metrics):
    with pytest.raises(AssertionError):
        metrics.observe({"unknown": Stats({})})
    with pytest.raises(KeyError):
        metrics.observe({"V6D": {"invalid": {}}})
