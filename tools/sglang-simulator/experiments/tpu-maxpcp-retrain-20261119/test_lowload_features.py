import json
import tempfile
import unittest
from pathlib import Path

from build_lowload_features import build, reuse_cases


class LowloadFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.windows = [dict(case="c1", concurrency=1, start=100, end=101),
                        dict(case="c4", concurrency=4, start=102, end=103)]
        self.events = []
        for sid, mono, q in [(10, 100_500_000_000, 1300), (20, 102_500_000_000, 3072)]:
            member = dict(request_id_sha256=str(sid), query_length=q, past_kv_length=0,
                          input_length=q, duplicate_id=False, membership_complete=True)
            self.events.append(dict(seq=len(self.events), event="request.step_membership", mono_ns=mono - 1,
                                    data=dict(schedule_id=sid, members=[member], membership_complete=True,
                                              native_schedule_ok=True)))
            self.events.append(dict(seq=len(self.events), event="step.inputs", mono_ns=mono,
                                    context=["run", "core", sid, "real"],
                                    data=dict(schedule_id=sid, members=[member], membership_complete=True,
                                              native_dispatch_ok=True, query_sum_valid=True, past_complete=True,
                                              total_num_scheduled_tokens=q)))

    def write(self):
        (self.root / "windows.json").write_text(json.dumps(self.windows))
        (self.root / "core-metadata.json").write_text(json.dumps(dict(wall_anchor_ns=0, mono_anchor_ns=0)))
        (self.root / "core-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in self.events))

    def test_features_and_context_are_same_scheduled_batch(self):
        self.write()
        result = build(self.root)
        self.assertEqual(result["padding_counts"], {2048: 1, 4096: 1})
        cases = json.loads((self.root / "cases.json").read_text())
        self.assertEqual([c["steps"][0]["context"] for c in cases],
                         [["run", "core", 10, "real"], ["run", "core", 20, "real"]])
        self.assertEqual(result["steps"], 2)

    def test_missing_membership_fails(self):
        del self.events[0]
        self.write()
        with self.assertRaisesRegex(AssertionError, "missing schedule membership"):
            build(self.root)

    def test_mismatched_membership_fails(self):
        self.events[0]["data"]["members"] = []
        self.write()
        with self.assertRaisesRegex(AssertionError, "membership differs"):
            build(self.root)

    def test_overlapping_case_assignment_fails(self):
        self.windows[1]["start"] = 100
        self.write()
        with self.assertRaisesRegex(AssertionError, "overlapping case windows"):
            build(self.root)

    def test_reuse_completed_case_exports_without_duplicate_events(self):
        sources = []
        for w in self.windows:
            trace = self.root / (w["case"] + ".jsonl")
            trace.write_text("".join(json.dumps(e) + "\n" for e in self.events))
            metadata = self.root / (w["case"] + ".metadata.jsons")
            metadata.write_text(json.dumps(dict(pid=1644, process_id="core", written=100,
                                                wall_anchor_ns=0, mono_anchor_ns=0)) + "\n")
            sources.append(dict(w, core_jsonl=str(trace), metadata_jsons=str(metadata)))
        out = self.root / "reused"
        reuse_cases(out, sources)
        self.assertEqual(build(out)["steps"], 2)
        self.assertEqual(len((out / "core-events.jsonl").read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
