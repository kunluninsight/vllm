#!/usr/bin/env python3
"""Compare old+offset, old bare, and new direct models on identical max-PCP labels.

Example (inside the established Python 3.12 container):
  python3 evaluate_baseline.py --input /path/to/candidate --output /path/to/comparison

Input is labels.csv or its containing aggregation/training output directory.
Candidate bundles default to that directory; override with --candidate-dir.
No fitting, parameter selection or deployment occurs in this script.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from sglang_simulator.simulation.vllm.tpu_pp.stage_model import StageModel

ROOT = Path('/host/qwen38-tpu-pp/work')
DEFAULT_BASELINE = [
    ROOT/'pressure-1116/candidate-p0-native-1117/DIAGNOSTIC_ONLY_stage-0_native_return.pkl',
    ROOT/'cadence-replay-1113/pp4_install_artifact/models/DIAGNOSTIC_ONLY_stage-1_native_return.pkl',
    ROOT/'cadence-replay-1113/pp4_install_artifact/models/DIAGNOSTIC_ONLY_stage-2_native_return.pkl',
    ROOT/'pressure-1116/candidate-p3-native-r2-1117/DIAGNOSTIC_ONLY_stage-3_native_return.pkl',
]
LEGACY_OFFSETS_MS = {2048: (5, 3, 2, -1), 4096: (4, 5, 3, 4), 8192: (1, 3, 6, 0)}
VARIANTS = ('baseline_with_offset_ms', 'baseline_bare_ms', 'candidate_ms')


def predict_batched_ms(model, x):
    """Preserve StageModel.predict: clip each component before summing."""
    return sum(np.maximum(np.asarray(m.predict(x), dtype=float), 0.)
               for m in model.models)*1000


def with_legacy_offsets(bare_ms, paddings, stage):
    offsets = np.array([LEGACY_OFFSETS_MS.get(int(p), (0, 0, 0, 0))[stage]
                        for p in paddings], dtype=float)
    return np.maximum(np.asarray(bare_ms)+offsets, 0.)


def metrics(rows, variant):
    actual = np.array([r['target_ms'] for r in rows])
    predicted = np.array([r[variant] for r in rows])
    error = predicted-actual
    return dict(n=len(rows), real_mean_ms=float(actual.mean()), predicted_mean_ms=float(predicted.mean()),
                bias_ms=float(error.mean()), mae_ms=float(np.abs(error).mean()),
                p95_absolute_error_ms=float(np.quantile(np.abs(error), .95)),
                mape_pct=float(np.mean(np.abs(error)/actual)*100))


def summarize(rows, holdout_only):
    groups = defaultdict(list)
    for row in rows:
        if holdout_only and row['split'] != 'holdout':
            continue
        stage = row['stage']
        case = f'{row["dataset"]}/{row["case"]}'
        padding = str(int(row['global_padded_tokens']))
        for kind, case_key, padding_key in (
            ('stage', 'all', 'all'), ('case', case, 'all'),
            ('padding', 'all', padding), ('case_padding', case, padding)):
            groups[stage, kind, case_key, padding_key].append(row)
    return [dict(stage=stage, group=kind, case=case, padding=padding,
                 variant=variant, **metrics(group, variant))
            for (stage, kind, case, padding), group in sorted(groups.items())
            for variant in VARIANTS]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'no data to write: {path}')
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='labels.csv or containing output directory')
    parser.add_argument('--candidate-dir', type=Path)
    parser.add_argument('--baseline-models', type=Path, nargs=4, default=DEFAULT_BASELINE,
                        metavar=('P0', 'P1', 'P2', 'P3'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    label_path = args.input/'labels.csv' if args.input.is_dir() else args.input
    candidate_dir = args.candidate_dir or label_path.parent
    with label_path.open() as handle:
        sources = list(csv.DictReader(handle))
    if not sources:
        raise ValueError('empty label input')
    rows, seen = [], set()
    for i, source in enumerate(sources):
        stage = int(source['stage'])
        key = (source['dataset'], stage, source.get('context', source['step']), source.get('chunk', '0'))
        if key in seen:
            raise ValueError(f'duplicate label identity at CSV row {i+2}: {key}')
        seen.add(key)
        target = float(source['target_ms'])
        if not math.isfinite(target) or target <= 0:
            raise ValueError(f'invalid target at CSV row {i+2}')
        if source['split'] not in ('train', 'gap', 'holdout'):
            raise ValueError(f'invalid split at CSV row {i+2}')
        rows.append(dict(source_csv_row=i+2, dataset=source['dataset'], case=source['case'],
                         stage=stage, step=int(source['step']), context=source.get('context', ''),
                         split=source['split'], global_padded_tokens=int(float(source['global_padded_tokens'])),
                         batch_size=int(float(source['batch_size'])), target_ms=target,
                         max_pcp=source.get('max_pcp', ''), start_skew_ms=source.get('start_skew_ms', ''),
                         outlier_flags=source.get('outlier_flags', '')))
    if {r['stage'] for r in rows} != set(range(4)):
        raise ValueError('comparison requires all four stages')
    candidate_paths = []
    for stage in range(4):
        old = StageModel.from_file(str(args.baseline_models[stage]), stage, 3)
        path = candidate_dir/f'DIAGNOSTIC_ONLY_stage-{stage}_native_return.pkl'
        bundle = joblib.load(path)
        if bundle.get('label_aggregation') != 'max_pcp_local_duration':
            raise ValueError(f'candidate stage {stage}: expected max_pcp_local_duration label contract')
        new = StageModel(bundle, stage, 3)
        if len(new.models) != 1:
            raise ValueError(f'candidate stage {stage}: expected one direct model')
        candidate_paths.append(str(path))
        indexes = [i for i,r in enumerate(rows) if r['stage'] == stage]
        x = np.array([[float(sources[i][name]) for name in old.names] for i in indexes])
        if not np.isfinite(x).all():
            raise ValueError(f'stage {stage}: non-finite features')
        bare = predict_batched_ms(old, x)
        legacy = with_legacy_offsets(bare, [rows[i]['global_padded_tokens'] for i in indexes], stage)
        candidate = predict_batched_ms(new, x)
        if not all(np.isfinite(v).all() for v in (bare, legacy, candidate)):
            raise ValueError(f'stage {stage}: non-finite predictions')
        # One runtime-equivalence check per stage catches feature/clamp drift.
        first = {name: float(sources[indexes[0]][name]) for name in old.names}
        np.testing.assert_allclose(bare[0], old.predict(first)*1000, rtol=0, atol=1e-9)
        np.testing.assert_allclose(candidate[0], new.predict(first)*1000, rtol=0, atol=1e-9)
        for index, old_with_offset, old_bare, new_pred in zip(indexes, legacy, bare, candidate):
            rows[index].update(baseline_with_offset_ms=float(old_with_offset),
                               baseline_bare_ms=float(old_bare), candidate_ms=float(new_pred))
    holdout = summarize(rows, holdout_only=True)
    if not holdout:
        raise ValueError('no holdout observations; refusing to label training results as main comparison')
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'holdout-summary.csv', holdout)
    write_csv(args.output/'all-summary.csv', summarize(rows, holdout_only=False))
    write_csv(args.output/'paired-predictions.csv', rows)
    report = dict(label_csv=str(label_path), baseline_models=[str(p) for p in args.baseline_models],
                  candidate_models=candidate_paths, legacy_offsets_ms=LEGACY_OFFSETS_MS,
                  label='same max of eight local PCP intervals for all three variants',
                  main_table='holdout-summary.csv: only previously assigned holdout observations',
                  supplementary_table='all-summary.csv: all observations including train and gap; not acceptance evidence',
                  error_sign='prediction minus observed maximum local duration',
                  rows=len(rows), holdout_rows=sum(r['split']=='holdout' for r in rows),
                  no_outliers_filtered=True,
                  main_stage_summary=[r for r in holdout if r['group']=='stage'])
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('rows','holdout_rows','main_stage_summary')}, indent=2))


if __name__ == '__main__':
    main()
