"""Rate arrivals must not inherit response latency or silently queue at the client."""
import asyncio
import hashlib
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("pressure", Path(__file__).parents[1] / "tools/prefill_pressure.py")
pressure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pressure)

class Session:
    def __init__(self, **kwargs):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass

class Sender:
    aiohttp = SimpleNamespace(ClientTimeout=lambda **k: None, TCPConnector=lambda **k: None, ClientSession=Session)
    encode = staticmethod(lambda value: json.dumps(value).encode())
    sha = staticmethod(lambda raw: hashlib.sha256(raw).hexdigest())
    def __init__(self, delay, status=200):
        self.delay, self.status, self.starts, self.bodies = delay, status, [], []
    async def send(self, item, session, sem, target, run_id, mono0, wall0, log, records):
        async with sem:
            start = time.monotonic()
            self.starts.append(start-mono0)
            self.bodies.append(json.loads(item["body"]))
            await asyncio.sleep(self.delay)
            rec = dict(id=item["id"], http_status=self.status, outcome="success" if self.status==200 else "error",
                       actual_send_unix_s=wall0+start-mono0, send_delay_s=start-mono0-item["at_s"])
            records[item["id"]] = rec
            log.write(json.dumps(rec)+"\n")

class RateTests(unittest.TestCase):
    def invoke(self, sender, limit=16, rate=20, duration=.25, max_requests=20):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), label="rate-test", duration=duration,
                                   max_requests=max_requests, seed=1, gap=0, timeout=1, max_send_lag=.25)
            return asyncio.run(pressure.case(args, sender, {"model":"fake"}, {"messages":[{}]}, "fixture", limit, 0, rate))
    def test_slow_responses_do_not_change_arrival_schedule(self):
        sender=Sender(.3)
        result=self.invoke(sender)
        self.assertEqual(result["sent"], 5)
        self.assertEqual(result["scheduled_requests"], 5)
        self.assertIsNone(result["stop_reason"])
        self.assertGreaterEqual(result["peak_client_inflight"], 4)
        for i,start in enumerate(sender.starts):
            self.assertLess(abs(start-i/20), .05)
        self.assertEqual(len({x["messages"][0]["content"] for x in sender.bodies}), 5)
    def test_inflight_guard_stops_instead_of_throttling(self):
        result=self.invoke(Sender(.3), limit=2)
        self.assertEqual(result["sent"], 2)
        self.assertEqual(result["stop_reason"], "inflight_limit")
        self.assertEqual(result["outcomes"]["success"], 2)
    def test_auth_failure_stops_new_arrivals_and_drains(self):
        result=self.invoke(Sender(.005,403))
        self.assertEqual(result["sent"], 1)
        self.assertEqual(result["stop_reason"], "auth_or_route")
        self.assertFalse(result["all_success"])
    def test_closed_loop_and_request_cap_remain_available(self):
        result=self.invoke(Sender(.01), limit=2, rate=None, max_requests=5)
        self.assertEqual(result["sent"],5)
        self.assertTrue(result["request_limit_reached"])
        self.assertEqual(result["arrival_mode"],"closed_loop")

if __name__ == "__main__":
    unittest.main()
