import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def exporter():
    path = Path(__file__).resolve().parents[1] / 'scripts/export_lifecycle_window.py'
    spec = importlib.util.spec_from_file_location('lifecycle_export_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_query_and_job_events_survive_real_grep_and_cache(tmp_path, monkeypatch):
    module = exporter()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path / 'events.000001.jsonl'
    names = ['request.v6d_fetch_end', 'request.v6d_query_consume',
             'request.kv_job_ack_consumed', 'request.kv_blocks_release',
             'request.business_external_alias']
    path.write_text(''.join(json.dumps(dict(mono_ns=100_000_000_001,
                    event=e, context=None, seq=i, data={'query_id': 'test'})) + '\n'
                    for i, e in enumerate(names)))
    calls = []
    def call(pod, *args):
        calls.append(args)
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout
    monkeypatch.setattr(module, 'call', call)
    rows = module.grep_records('pod', [str(path)], 100, 101)
    assert [r['event'] for r in rows] == names
    assert all(r['_source_file'] == str(path) for r in rows)
    assert module.grep_records('pod', [str(path)], 100, 101) == rows
    assert len(calls) == 1


def test_transport_error_does_not_bisect_or_claim_empty_success(tmp_path, monkeypatch):
    module = exporter()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    calls = []
    def failed(*args):
        calls.append(args)
        raise RuntimeError('DNS unavailable')
    monkeypatch.setattr(module, 'call', failed)
    with pytest.raises(RuntimeError, match='DNS unavailable'):
        module.grep_records('pod', ['one', 'two'], 100, 500)
    assert len(calls) == 1
    assert not list((tmp_path / 'read_cache').glob('*.jsonl.gz'))


def test_truncated_read_is_not_accepted(tmp_path, monkeypatch):
    module = exporter()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'call', lambda *a: '')
    with pytest.raises(RuntimeError, match='Cannot prove complete'):
        module.grep_records('pod', ['one'], 100, 100)
    assert not list((tmp_path / 'read_cache').glob('*.jsonl.gz'))
