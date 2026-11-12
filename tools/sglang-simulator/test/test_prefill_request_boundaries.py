import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src/sglang_simulator/request_boundaries.py"


class PrefillBoundaries(unittest.TestCase):
    def setUp(self):
        hook = ModuleType("sglang_simulator.hook")
        hook.BaseHook = object
        spec = importlib.util.spec_from_file_location("_pd_boundaries_test", SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"sglang_simulator.hook": hook}):
            spec.loader.exec_module(self.module)
        self.events = []
        self.module._enabled = lambda: True
        self.module._emit = lambda name, **d: self.events.append((name, d))

    def test_frames_keep_native_delay_and_do_not_claim_delivery(self):
        class Session:
            request_id = "business"
            request_uuid = "session"
            def prefill_running(self): pass
            def handshake(self, extra): return extra
            def prefill_result(self, outputs, extra, *, first_token_delay_ms): return outputs
            def kv_ready(self, cached): raise LookupError("original")

        self.module.C_PrefillProtocolHook.hook(Session)
        self.module.C_PrefillProtocolHook.hook(Session)
        s, sentinel = Session(), object()
        s.prefill_running()
        self.assertIs(s.handshake(sentinel), sentinel)
        for delay in (367, 0, None):
            self.assertIs(s.prefill_result(sentinel, sentinel, first_token_delay_ms=delay), sentinel)
        with self.assertRaisesRegex(LookupError, "original"): s.kv_ready(0)
        frames = [(e,d) for e,d in self.events if e.startswith("request.pd_")]
        self.assertEqual([e for e,d in frames], ["request.pd_prefill_running",
                         "request.pd_handshake_built", *(["request.pd_first_token_built"]*3),
                         "request.pd_kv_ready_built"])
        self.assertEqual([d["native_prefill_first_token_delay_ms"] for e,d in frames[2:5]],
                         [367, 0, None])
        for _, d in frames:
            self.assertFalse(d["delivery_observed"])
            self.assertEqual(d["business_request_id_sha256"], hashlib.sha256(b"business").hexdigest())
            self.assertNotIn(sentinel, d.values())
        self.assertFalse(frames[-1][1]["native_ok"])

    def test_native_generator_preserves_control_frame_and_token(self):
        class Engine:
            def generate(self, *, request_context):
                yield [control]
                yield [token]
            def _parse_output(self, *, request_id, request_uuid): return [token]
        self.module.C_PrivateAsyncBackendHook.hook(Engine)
        context = SimpleNamespace(request_id="business", request_uuid="service",
                                  disaggregated_request_id=None)
        control = SimpleNamespace(engine_output=SimpleNamespace(request_id="capacity_info_pack"),
                                  token_ids=[0])
        token = SimpleNamespace(token_ids=[1])
        gen = Engine().generate(request_context=context)
        self.assertIs(next(gen)[0], control)
        self.assertIs(next(gen)[0], token)
        with self.assertRaises(StopIteration): next(gen)
        yields = [d for e,d in self.events if e=="request.private_generate_yield"]
        self.assertEqual([d["capacity_info_frame"] for d in yields], [True, False])
        self.assertEqual([d["output_token_counts"] for d in yields], [(1,), (1,)])
