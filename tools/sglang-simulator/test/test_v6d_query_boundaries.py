import asyncio
from types import SimpleNamespace

import pytest

from sglang_simulator.cache_boundaries import install_connector


class Scheduler:
    async def _async_match(self, request_id, generation, gate=None):
        if gate is not None:
            await gate.wait()
        if request_id == "error":
            raise ValueError("native")
        return SimpleNamespace(generation=generation, hit_length=0, fetch_failed=False)

    def _commit_async_match(self, request, state, result):
        return result.hit_length, False

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return None, False


def setup(emit=None):
    rows = []
    emit = emit or (lambda event, **data: rows.append((event, data)))
    scheduler = Scheduler()
    child = SimpleNamespace(scheduler=scheduler)
    root = SimpleNamespace(_connectors=[child])
    install_connector(root, emit, lambda fn: fn())
    install_connector(root, emit, lambda fn: fn())
    return scheduler, rows


def test_background_execution_and_consumption_are_separate():
    scheduler, rows = setup()

    async def run():
        coro = scheduler._async_match("r1", 2)
        assert not any(e == "request.v6d_query_start" for e, _ in rows)
        await asyncio.sleep(0.01)
        result = await coro
        await asyncio.sleep(0.01)
        assert scheduler._commit_async_match(SimpleNamespace(request_id="r1"), None, result) == (0, False)

    asyncio.run(run())
    events = {e: d for e, d in rows if e.startswith("request.v6d_query_")}
    end = events["request.v6d_query_end"]
    consume = events["request.v6d_query_consume"]
    assert end["start_mono_ns"] - end["created_mono_ns"] >= 5_000_000
    assert consume["start_mono_ns"] - end["end_mono_ns"] >= 5_000_000
    assert end["hit_tokens"] == 0 and end["native_ok"]
    assert end["request_id_sha256"] == consume["request_id_sha256"]
    assert sum(e == "request.v6d_query_created" for e, _ in rows) == 1


def test_native_failure_and_cancellation_preserved():
    scheduler, rows = setup()

    async def run():
        with pytest.raises(ValueError, match="native"):
            await scheduler._async_match("error", 1)
        task = asyncio.create_task(scheduler._async_match("cancel", 2, asyncio.Event()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    errors = [d["native_error_type"] for e, d in rows if e == "request.v6d_query_end"]
    assert errors == ["ValueError", "CancelledError"]


def test_broken_observer_does_not_change_native_result():
    def broken(*a, **kw):
        raise RuntimeError("observer")
    scheduler, _ = setup(broken)
    assert asyncio.run(scheduler._async_match("r", 1)).hit_length == 0
