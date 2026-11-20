"""Finite wall-clock phases using an unchanged, independently pinned sender."""
import argparse
import asyncio
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import uuid


class DeadlineGate:
    def __init__(self, deadline):
        self.deadline = deadline

    async def __aenter__(self):
        if time.monotonic() >= self.deadline:
            raise asyncio.CancelledError()
        return self

    async def __aexit__(self, *args):
        return False


async def run_windows(windows, take, send, emit):
    origin, active, pending = time.monotonic(), set(), None
    def reap():
        for task in list(active):
            if task.done():
                active.remove(task)
                if not task.cancelled():
                    task.result()
    try:
        for phase, start, end, limit in windows:
            await asyncio.sleep(max(0, origin + start - time.monotonic()))
            reap()
            emit('start', dict(phase=phase, elapsed_s=time.monotonic() - origin, active=len(active),
                               planned_start_s=start, planned_end_s=end, concurrency=limit))
            deadline = origin + end
            while time.monotonic() < deadline:
                reap()
                if len(active) >= limit:
                    await asyncio.wait(active, timeout=max(0, deadline - time.monotonic()),
                                       return_when=asyncio.FIRST_COMPLETED)
                    continue
                if pending is None:
                    pending = await take()
                if time.monotonic() >= deadline:
                    break
                if pending is None:
                    raise ValueError('Input source is empty')
                item, pending = pending, None
                active.add(asyncio.create_task(send(item, phase, deadline)))
            reap()
            emit('end', dict(phase=phase, elapsed_s=time.monotonic() - origin, active=len(active),
                             planned_start_s=start, planned_end_s=end, concurrency=limit))
        await asyncio.gather(*active)
    finally:
        for task in active:
            if not task.done():
                task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CyclicRows:
    def __init__(self, path, paced, sender, model, expected):
        self.path, self.paced, self.sender, self.model = path, paced, sender, model
        self.expected, self.cycle, self.line, self.context = expected, 0, 0, None

    def close(self, abandon=False):
        if self.context is not None:
            if abandon:
                self.context.__exit__(GeneratorExit, GeneratorExit(), None)
            else:
                self.context.__exit__(None, None, None)
            self.context = None

    def next(self):
        while True:
            if self.context is None:
                self.context = self.paced.lines(self.path)
                self.stream = self.context.__enter__()
                self.line = 0
            raw = next(self.stream, None)
            if raw is None:
                assert self.line == self.expected and self.line > 0
                self.close()
                self.cycle += 1
                continue
            if not raw.strip():
                continue
            self.line += 1
            source = json.loads(raw)
            row = self.sender.load_plan(self.paced.OneLine(raw.encode()), 1, self.model)[0][0]
            source_id = row['id']
            row.update(id=source_id + '-cycle-' + str(self.cycle), at_s=0.0, input_line=self.line)
            row['source'] = dict(source_id=source_id, cycle=self.cycle, source_line=self.line,
                                 original_at_s=source['at_s'], trace=source.get('trace', {}))
            return row


async def replay(args, paced, sender, target, manifest):
    args.output.mkdir(mode=0o700, exist_ok=False)
    windows = [('c2', 0, args.c2_seconds, 2),
               ('c8', args.c2_seconds + args.pause_seconds,
                args.c2_seconds + args.pause_seconds + args.c8_seconds, 8)]
    run_id, mono0, wall0 = str(uuid.uuid4()), time.monotonic(), time.time()
    records, logs, admissions = {}, {}, collections.Counter()
    manifests, dirs = {}, {}
    source = CyclicRows(args.input, paced, sender, target['model'], args.count)
    manifest.update(run_id=run_id, start_unix_s=wall0, windows=windows, source_count=args.count,
                    concurrency_scope='real client outstanding; simulator receives asynchronous mirrors',
                    input_file_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    sender_sha256=hashlib.sha256(args.sender.read_bytes()).hexdigest(),
                    source_policy='continuous original order, cycle only on exhaustion, unchanged wire payload')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output / 'phase-events.jsonl').open('x') as events, (args.output / 'source-ledger.jsonl').open('x') as ledger:
        def emit(kind, row):
            event = dict(kind=kind, unix_s=time.time(), **row)
            events.write(json.dumps(event) + '\n'); events.flush()
            (args.output / 'status.json').write_text(json.dumps(event) + '\n')
            if kind == 'start':
                phase = row['phase']; directory = args.output / phase; directory.mkdir()
                dirs[phase] = directory; records[phase] = {}
                logs[phase] = (directory / 'requests.jsonl').open('x')
                manifests[phase] = dict(manifest, phase=phase, start_unix_s=time.time(),
                                       concurrency=row['concurrency'], count=None)
                (directory / 'manifest.json').write_text(json.dumps(manifests[phase], indent=2) + '\n')
        async def take():
            return await asyncio.to_thread(source.next)
        try:
            async with sender.aiohttp.ClientSession(
                timeout=sender.aiohttp.ClientTimeout(total=args.timeout),
                connector=sender.aiohttp.TCPConnector(limit=8),
                headers={'Authorization': 'Bearer ' + os.environ['DASHSCOPE_API_KEY'],
                         'Content-Type': 'application/json', 'Accept': 'text/event-stream'}, trust_env=False) as session:
                async def send(row, phase, deadline):
                    admissions[phase] += 1
                    ledger.write(json.dumps(dict(id=row['id'], phase=phase, payload_sha256=row['payload_sha256'],
                                                 **row['source'])) + '\n'); ledger.flush()
                    row['at_s'] = time.monotonic() - mono0
                    await sender.send(row, session, DeadlineGate(deadline), target, run_id, mono0, wall0,
                                      logs[phase], records[phase])
                await run_windows(windows, take, send, emit)
        finally:
            source.close(abandon=True)
            results = {}
            for phase in logs:
                logs[phase].close()
                manifests[phase]['count'] = admissions[phase]
                (dirs[phase] / 'manifest.json').write_text(json.dumps(manifests[phase], indent=2) + '\n')
                outcomes = dict(collections.Counter(r['outcome'] for r in records[phase].values()))
                result = dict(planned=admissions[phase], recorded=len(records[phase]), outcomes=outcomes,
                              sent=sum(r['actual_send_unix_s'] is not None for r in records[phase].values()),
                              all_success=outcomes.get('success', 0) == admissions[phase])
                (dirs[phase] / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
                results[phase] = result
            (args.output / 'summary.json').write_text(json.dumps(dict(completed_unix_s=time.time(), phases=results), indent=2) + '\n')
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'sender', 'paced', 'target', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--c2-seconds', type=float, default=900)
    parser.add_argument('--pause-seconds', type=float, default=300)
    parser.add_argument('--c8-seconds', type=float, default=900)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--local-fixture', action='store_true')
    args = parser.parse_args()
    assert args.c2_seconds > 0 and args.pause_seconds >= 0 and args.c8_seconds > 0 and args.timeout > 0
    sender, paced = load_module('pinned_sender', args.sender), load_module('pinned_paced', args.paced)
    target, manifest = sender.load_target(args.target, args.local_fixture)
    count = sum(1 for _ in paced.iter_plan(args.input, sender, target['model']))
    assert count == args.count and count > 0
    if args.validate_only:
        print(json.dumps(dict(validated=count, original_payloads=True, phases=[args.c2_seconds,args.pause_seconds,args.c8_seconds])))
        return
    assert os.environ.get('DASHSCOPE_API_KEY')
    asyncio.run(replay(args, paced, sender, target, manifest))


if __name__ == '__main__':
    main()
