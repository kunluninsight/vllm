import asyncio
import importlib.util
from pathlib import Path
import time
import unittest

path = Path(__file__).with_name('replay_test4_phases.py')
if not path.exists():
    path = Path(__file__).parents[1] / 'scripts/replay_test4_phases.py'
spec = importlib.util.spec_from_file_location('phases', path)
phases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phases)


class PhaseBoundaries(unittest.IsolatedAsyncioTestCase):
    async def test_no_sends_in_pause_and_concurrency_is_bounded(self):
        started, ended, active, peak = [], [], {}, {}
        counter = 0
        async def take():
            nonlocal counter
            counter += 1
            return counter
        async def send(row, phase, deadline):
            started.append((row, phase, time.monotonic()))
            active[phase] = active.get(phase, 0) + 1
            peak[phase] = max(peak.get(phase, 0), active[phase])
            await asyncio.sleep(.018)
            active[phase] -= 1
            ended.append(row)
        t0 = time.monotonic()
        await phases.run_windows([('c2', 0, .07, 2), ('c8', .12, .19, 8)], take, send, lambda *a: None)
        self.assertGreater(len(started), 8)
        self.assertEqual(peak, {'c2': 2, 'c8': 8})
        self.assertEqual(set(ended), {x[0] for x in started})
        self.assertTrue(all(t0 <= t < t0 + .09 for _, p, t in started if p == 'c2'))
        self.assertTrue(all(t0 + .12 <= t < t0 + .21 for _, p, t in started if p == 'c8'))
        self.assertEqual([r for r, _, _ in started], list(range(1, len(started) + 1)))

    async def test_row_read_across_deadline_is_kept_for_next_phase(self):
        sent, counter = [], 0
        async def take():
            nonlocal counter
            counter += 1
            await asyncio.sleep(.055)
            return counter
        async def send(row, phase, deadline):
            sent.append((row, phase))
        await phases.run_windows([('first', 0, .025, 2), ('second', .08, .11, 8)], take, send, lambda *a: None)
        self.assertEqual(sent, [(1, 'second')])

    async def test_phase_change_does_not_cancel_inflight_requests(self):
        completed, events = [], []
        async def take():
            return object()
        async def send(row, phase, deadline):
            await asyncio.sleep(.1)
            completed.append(phase)
        await phases.run_windows([('first', 0, .025, 2), ('second', .04, .065, 3)], take, send,
                                 lambda kind, row: events.append((kind, row)))
        self.assertEqual(completed.count('first'), 2)
        self.assertEqual(completed.count('second'), 1)
        self.assertEqual([r['active'] for k, r in events if k == 'start'], [0, 2])

    async def test_expired_gate_never_enters_send_body(self):
        entered = False
        with self.assertRaises(asyncio.CancelledError):
            async with phases.DeadlineGate(time.monotonic() - 1):
                entered = True
        self.assertFalse(entered)


if __name__ == '__main__':
    unittest.main()
