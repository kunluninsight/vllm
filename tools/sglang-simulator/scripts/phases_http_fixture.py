"""Loopback-only integration with the original sender and compressed source."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from aiohttp import web


async def main(root, source_count=400):
    observed = []
    async def handler(request):
        body = await request.read()
        observed.append((time.monotonic(), hashlib.sha256(body).hexdigest()))
        request_index = len(observed)
        await asyncio.sleep(.08)
        event = dict(id='fixture-response', choices=[dict(index=0, delta=dict(content='x'), finish_reason='stop')],
                     usage=dict(prompt_tokens=10, completion_tokens=1, total_tokens=11))
        return web.Response(text='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n',
                            content_type='text/event-stream', headers={'x-request-id': 'fixture-' + str(request_index)})
    app = web.Application(); app.router.add_post('/chat', handler)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        with tempfile.TemporaryDirectory(prefix='test4-phases-fixture-') as name:
            directory = Path(name)
            raw = b''.join((json.dumps(dict(id='row-' + str(i), at_s=i, payload=dict(
                messages=[dict(role='user', content='fixture ' + str(i) + 'x' * 4096)], max_tokens=1))) + '\n').encode() for i in range(source_count))
            source = directory / 'plan.jsonl.zst'
            source.write_bytes(subprocess.check_output(['zstd', '-q', '-c'], input=raw))
            target = directory / 'target.json'; target.write_text(json.dumps(dict(kind='local_fixture', model='fixture', endpoint='http://127.0.0.1:' + str(port) + '/chat')))
            process = await asyncio.create_subprocess_exec(sys.executable, str(root / 'replay_test4_phases.py'),
                '--input', str(source), '--count', str(source_count), '--sender', str(root / 'sender-test4.py'),
                '--paced', str(root / 'replay_plan_paced.py'), '--target', str(target), '--output', str(directory / 'out'),
                '--local-fixture', '--c2-seconds', '.2', '--pause-seconds', '.15', '--c8-seconds', '.2', '--timeout', '2',
                env=dict(os.environ, DASHSCOPE_API_KEY='loopback-fixture-no-credential'),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
            assert process.returncode == 0, stderr.decode()[-2200:]
            out = directory / 'out'
            summary = json.loads((out / 'summary.json').read_text())
            events = [json.loads(x) for x in (out / 'phase-events.jsonl').read_text().splitlines()]
            ledger = [json.loads(x) for x in (out / 'source-ledger.jsonl').read_text().splitlines()]
            receipts = [json.loads(x) for phase in ('c2', 'c8') for x in (out / phase / 'requests.jsonl').read_text().splitlines()]
            assert all(r['outcome'] == 'success' for r in receipts)
            assert len(receipts) == len(observed) == len(ledger)
            assert sorted(r['payload_sha256'] for r in receipts) == sorted(h for _, h in observed)
            assert len({r['id'] for r in receipts}) == len(receipts)
            assert [r['source_line'] for r in ledger] == [i % source_count + 1 for i in range(len(ledger))]
            assert [r['cycle'] for r in ledger] == [i // source_count for i in range(len(ledger))]
            start = next(r['unix_s'] for r in events if r['kind'] == 'start' and r['phase'] == 'c2')
            assert all(0 <= r['actual_send_unix_s'] - start < .23 for r in receipts if r['id'] in {x['id'] for x in ledger if x['phase'] == 'c2'})
            assert all(.34 <= r['actual_send_unix_s'] - start < .58 for r in receipts if r['id'] in {x['id'] for x in ledger if x['phase'] == 'c8'})
            print(json.dumps(dict(loopback_http_verified=True, compressed_prefix_close_verified=True,
                                 wire_hashes_match=True, source_count=source_count, rows=len(receipts), phases=summary['phases'])))
    finally:
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 400))
