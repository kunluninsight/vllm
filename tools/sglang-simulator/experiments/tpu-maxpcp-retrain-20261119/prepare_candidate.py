#!/usr/bin/env python3
"""Build private Spectrum candidate files; never publishes or starts traffic.

The model tar.gz must contain these four files at its root:
DIAGNOSTIC_ONLY_stage-{0,1,2,3}_native_return.pkl
Optional provenance files are ignored by bootstrap. Existing base artifact stays
unchanged; supply a fresh download URL for it because rollout downloads it again.
"""
import argparse
import copy
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit


MODEL_BOOTSTRAP = '''"$SIM_PYTHON" -S - <<'PYMAXPCP'
import os, pathlib, tarfile, urllib.request
root = pathlib.Path('/home/admin/hisim-pp')
archive = root / 'stage-models-maxpcp.tar.gz'
try:
    with urllib.request.urlopen(os.environ['HISIM_STAGE_MODELS_URL'], timeout=120) as source, archive.open('wb') as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
except Exception as exc:
    raise RuntimeError('all-PCP model bundle download failed: ' + type(exc).__name__) from None
names = [f'DIAGNOSTIC_ONLY_stage-{rank}_native_return.pkl' for rank in range(4)]
with tarfile.open(archive) as bundle:
    members = [bundle.getmember(name) for name in names]
    assert all(member.isfile() for member in members), 'expected four regular stage model files'
    assert all(sum(item.name == member.name for item in bundle.getmembers()) == 1 for member in members), 'duplicate stage model member'
    payloads = [bundle.extractfile(member).read() for member in members]
assert all(payloads), 'empty stage model file'
for name, payload in zip(names, payloads):
    (root / 'models' / name).write_bytes(payload)
print('[hisim-pp] four max-PCP models installed; no legacy model overlay', flush=True)
PYMAXPCP
unset HISIM_STAGE_MODELS_URL
'''


def write_private(path, obj):
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
    path.chmod(0o600)


def make_candidate(dep, runtime, release):
    version = release['runtime_version']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', version):
        raise ValueError('invalid runtime version')
    if version == runtime['metadata']['version']:
        raise ValueError('runtime is immutable: choose a new version')
    for key in ('wheel_url', 'models_url', 'artifact_url'):
        parsed = urlsplit(release[key])
        if parsed.scheme != 'https' or not parsed.netloc:
            raise ValueError(key + ' must be a complete HTTPS download URL')
    new_runtime = copy.deepcopy(runtime)
    new_runtime['kind'] = 'runtime'
    new_runtime['metadata'] = {
        'name': runtime['metadata']['name'],
        'version': version,
        'labels': copy.deepcopy(runtime['metadata'].get('labels', {})),
        'annotations': copy.deepcopy(runtime['metadata'].get('annotations', {})),
    }
    old = runtime['containers'][0]['command']
    begin = old.index('"$SIM_PYTHON" -S - <<\'PYP3\'')
    end_marker = 'unset HISIM_P0_MODEL_URL\n'
    end = old.index(end_marker, begin) + len(end_marker)
    command = old[:begin] + MODEL_BOOTSTRAP + old[end:]
    before = "c['predictor']['barrier_overhead_s'] = 0.0013"
    assert command.count(before) == 1, 'unexpected barrier bootstrap'
    command = command.replace(before, "c['predictor']['barrier_overhead_s'] = 0.00058")
    command = command.replace(
        '[hisim-pp] scheduler meta component budget enabled; barrier/payload unchanged',
        '[hisim-pp] scheduler meta unchanged; barrier 0.58 ms; payload 0.8 ms')
    command = command.replace(
        '[hisim-pp] integrated fee5a79 recorder loaded',
        '[hisim-pp] max-PCP candidate recorder loaded')
    assert 'HISIM_P0_MODEL_URL' not in command and 'HISIM_P3_MODEL_URL' not in command
    assert command.count('HISIM_WHEEL_URL') == old.count('HISIM_WHEEL_URL')
    new_runtime['containers'][0]['command'] = command

    new_dep = copy.deepcopy(dep)
    new_dep['kind'] = 'dep'
    new_dep.pop('status', None)
    new_dep['metadata'] = {
        'name': dep['metadata']['name'],
        'labels': copy.deepcopy(dep['metadata'].get('labels', {})),
    }
    new_dep['runtime']['version'] = version
    env = new_dep['environment_variables']
    env['HISIM_WHEEL_URL'] = release['wheel_url']
    env['HISIM_STAGE_MODELS_URL'] = release['models_url']
    env['HISIM_ARTIFACT_URL'] = release['artifact_url']
    # Only change the digest if the caller explicitly changes the base artifact.
    if 'artifact_sha256' in release:
        if not re.fullmatch('[0-9a-f]{64}', release['artifact_sha256']):
            raise ValueError('invalid base artifact digest')
        env['HISIM_ARTIFACT_SHA256'] = release['artifact_sha256']
    env.pop('HISIM_P0_MODEL_URL', None)
    env.pop('HISIM_P3_MODEL_URL', None)
    assert 'DS_DEPENDENCY_WHEELS' not in env, 'inspect unexpected wheel overlay'
    assert 'package_dependencies' not in new_dep['runtime_config'], 'inspect unexpected dependency overlay'
    assert env['V6D_ENABLE_TRACKER'] == '0'
    assert env['V6D_KVCM_MODE'] == 'disabled'
    assert new_dep['instances'] == dep['instances'] == 1

    audit = {
        'before_revision': dep['metadata']['annotations']['dashscope.ai/revision'],
        'before_runtime': runtime['metadata']['version'],
        'candidate_runtime': version,
        'deployment': dep['metadata']['name'],
        'barrier_overhead_s': {'before': 0.0013, 'after': 0.00058},
        'communication_s': {'before': 0.0008, 'after': 0.0008},
        'env_updated': ['HISIM_ARTIFACT_URL', 'HISIM_WHEEL_URL'],
        'env_added': ['HISIM_STAGE_MODELS_URL'],
        'env_removed': ['HISIM_P0_MODEL_URL', 'HISIM_P3_MODEL_URL'],
        'model_install': 'all four files from one tar.gz after base-artifact extraction',
        'wheel_install': 'download overwrites base artifact wheel before pip --force-reinstall --no-deps',
        'other_package_dependency_overlay_present': False,
        'unchanged': ['model', 'resource', 'instances', 'engine_parameters', 'scheduler_meta',
                      'host_prep', 'KVCM disabled', 'tracker 0', 'local V6D', 'real collector'],
        'requires_renewed_base_artifact_url': True,
        'published': False,
    }
    return new_runtime, new_dep, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--release', type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    dep = json.loads((root / 'sim-dep-before.private.json').read_text())
    runtime = json.loads((root / 'sim-runtime-before.private.json').read_text())
    release = json.loads(args.release.read_text())
    outputs = make_candidate(dep, runtime, release)
    for name, value in zip(('runtime-candidate.private.json', 'deployment-candidate.private.json', 'audit.json'), outputs):
        write_private(root / name, value)
    print(json.dumps(outputs[2], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
