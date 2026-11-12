import importlib.util
import json
import tempfile
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "tools/compare_prefill_ttft.py"
spec = importlib.util.spec_from_file_location("_compare", path)
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)


class PairedWindow(unittest.TestCase):
    def test_two_clocks_chunked_request_and_missing_mirror(self):
        with tempfile.TemporaryDirectory() as temp:
            real, sim = [Path(temp) / name for name in ("real", "sim")]
            for side, offset in ((real, 1000), (sim, 99)):
                side.mkdir()
                is_real = side == real
                rid = "internal-" + side.name
                ext = "external-" + side.name
                ns = lambda t: (offset + t) * 1_000_000
                rows = []
                def emit(e, **d):
                    rows.append({"event":e, **({"data":d} if is_real else {"process_id":"s",**d})})
                emit("request.identity_alias", request_id_sha256=rid, external_request_id_sha256=ext)
                emit("request.business_external_alias", external_request_id_sha256=ext, business_request_id_sha256="common")
                emit("request.add" if is_real else "request.scheduler_add", request_id_sha256=rid, end_mono_ns=ns(0))
                if is_real:
                    for t in (3,5):
                        emit("request.step_membership", members=[{"request_id_sha256":rid}], end_mono_ns=ns(t))
                    emit("request.output_observed", request_id_sha256=rid, new_token_count=0, end_mono_ns=ns(6))
                    emit("request.output_observed", request_id_sha256=rid, new_token_count=1, end_mono_ns=ns(15))
                    emit("request.add", request_id_sha256="not-mirrored", end_mono_ns=ns(1))
                else:
                    emit("scheduler.scheduled", requests=[{"request_id_sha256":rid}], schedule_return_mono_ns=ns(2))
                    emit("scheduler.output_update", step_id=7, end_mono_ns=ns(10))
                    emit("request.output_accepted", request_id_sha256=rid, new_token_count=1, step_id=7)
                file = side / ("events.000000.jsonl" if is_real else "lifecycle-s.jsonl")
                file.write_text("".join(json.dumps(r)+"\n" for r in rows))
            result = cmp.compare(cmp.window(real),cmp.window(sim))
            self.assertEqual(result["admissions"],[2,1])
            self.assertEqual(result["paired_business_ids"],1)
            for phase, diff in (("core_ms",5),("to_first_schedule_ms",1),("schedule_to_token_ms",4)):
                self.assertEqual(result["phases"][phase]["paired_mean_real_minus_sim"],diff)
            self.assertIsNone(result["phases"]["pd_native_ms"]["paired_mean_real_minus_sim"])
