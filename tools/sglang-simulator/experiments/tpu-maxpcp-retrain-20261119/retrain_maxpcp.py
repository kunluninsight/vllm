#!/usr/bin/env python3
"""Aggregate complete PCP observations and fit one direct host-window model/stage.

No TTFT, concurrency, old-model predictions or hand-written millisecond offsets
are used as features/targets. ``--train`` is explicit; default only assembles and
audits the data. See README.md for the manifest and label contract.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from sglang_simulator.simulation.vllm.tpu_pp.stage_model import (
    FEATURE_NAMES, TAIL_FEATURES, MODEL_KIND, StageModel,
)

ALL_FEATURES = FEATURE_NAMES + TAIL_FEATURES
RANKS = set(range(8))


def endpoint(stage):
    return 'native.output_ready_observed' if stage == 3 else 'native.local_drain.ready'


def clock_identity(metadata):
    clock = metadata.get('host_clock', {})
    fields = (metadata.get('clock_domain'), clock.get('source'),
              clock.get('boot_id'), clock.get('time_namespace'))
    return fields if all(fields) else None


def aggregate_stage(features, events_by_pcp, metadata_by_pcp, stage, dataset):
    """Join exact context/chunk across 8 PCPs; retain local durations unchanged."""
    if set(events_by_pcp) != RANKS or set(metadata_by_pcp) != RANKS:
        raise ValueError(f'{dataset} stage {stage}: PCP 0..7 required')
    selected = {int(row['step']) for row in features}
    if len(selected) != len(features):
        raise ValueError(f'{dataset}: duplicate feature step; split distinct sessions into datasets')
    indexes = {}
    for pcp, events in events_by_pcp.items():
        index = {}
        for event in events:
            if event.get('event') != endpoint(stage):
                continue
            context = event.get('context')
            if not isinstance(context, list) or len(context) != 4:
                raise ValueError(f'{dataset} stage {stage} PCP {pcp}: invalid context')
            step = int(context[2])
            if step not in selected:
                continue
            if event.get('rank', {}).get('pp') != stage or event['rank'].get('pcp') != pcp:
                raise ValueError(f'{dataset} step {step}: rank identity mismatch')
            if metadata_by_pcp[pcp].get('collection_id') != context[0]:
                raise ValueError(f'{dataset} step {step}: metadata/context collection mismatch')
            if step in index:
                raise ValueError(f'{dataset} stage {stage} PCP {pcp}: duplicate step/context {step}')
            if event.get('data', {}).get('native_ok') is False:
                raise ValueError(f'{dataset} step {step}: native operation failed')
            index[step] = event
        if set(index) != selected:
            missing = sorted(selected-set(index))
            raise ValueError(f'{dataset} stage {stage} PCP {pcp}: missing {len(missing)} steps {missing[:8]}')
        indexes[pcp] = index
    identities = [clock_identity(metadata_by_pcp[p]) for p in range(8)]
    same_clock = identities[0] is not None and len(set(identities)) == 1
    result = []
    for source in features:
        step = int(source['step'])
        events = [indexes[p][step] for p in range(8)]
        context = events[0]['context']
        chunk = events[0].get('chunk', 0)
        if any(e['context'] != context or e.get('chunk', 0) != chunk for e in events):
            raise ValueError(f'{dataset} stage {stage} step {step}: context/chunk mismatch')
        expected = source.get('_expected_context')
        if expected is not None and expected != context:
            raise ValueError(f'{dataset} step {step}: feature context mismatch')
        starts, ends = [], []
        for pcp, event in enumerate(events):
            data = event['data']
            start, end = data.get('start_mono_ns'), data.get('observed_mono_ns')
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                raise ValueError(f'{dataset} stage {stage} step {step} PCP {pcp}: invalid local interval')
            starts.append(start)
            ends.append(end)
        durations = [(end-start)/1e6 for start, end in zip(starts, ends)]
        row = dict(dataset=dataset, case=str(source.get('case', 'c'+str(source.get('concurrency', 'unknown')))),
                   concurrency=source.get('concurrency', ''), stage=stage, step=step,
                   context=json.dumps(context, separators=(',', ':')), chunk=chunk,
                   reference_start_ns=starts[0], target_ms=max(durations), pcp0_ms=durations[0],
                   max_pcp=int(np.argmax(durations)), max_minus_pcp0_ms=max(durations)-durations[0],
                   shared_monotonic_clock=same_clock,
                   start_skew_ms=(max(starts)-min(starts))/1e6 if same_clock else None,
                   end_skew_ms=(max(ends)-min(ends))/1e6 if same_clock else None,
                   envelope_ms=(max(ends)-min(starts))/1e6 if same_clock else None,
                   from_pcp0_start_to_last_end_ms=(max(ends)-starts[0])/1e6 if same_clock else None)
        for name in ALL_FEATURES:
            value = float(source[name])
            if not math.isfinite(value):
                raise ValueError(f'{dataset} step {step}: non-finite feature {name}')
            row[name] = value
        for pcp in range(8):
            row[f'pcp{pcp}_ms'] = durations[pcp]
            row[f'pcp{pcp}_start_ns'] = starts[pcp]
            row[f'pcp{pcp}_end_ns'] = ends[pcp]
        result.append(row)
    return result


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dataset(spec):
    with Path(spec['features_csv']).open() as handle:
        features = list(csv.DictReader(handle))
    if not features:
        raise ValueError(f'{spec["name"]}: empty feature data')
    if spec.get('contexts_json'):
        contexts = {}
        for case in json.loads(Path(spec['contexts_json']).read_text()):
            for step in case['steps']:
                if int(step['step']) in contexts:
                    raise ValueError('duplicate step in contexts_json')
                contexts[int(step['step'])] = step['context']
        for row in features:
            row['_expected_context'] = contexts[int(row['step'])]
    if len(spec['stage_dirs']) != 4:
        raise ValueError('stage_dirs must list all four PP stage directories')
    rows, metadata_report = [], {}
    stage0_contexts = None
    for stage, directory in enumerate(spec['stage_dirs']):
        directory = Path(directory)
        events = {pcp: read_jsonl(directory/f'pcp{pcp}-events.jsonl') for pcp in range(8)}
        metadata = {pcp: json.loads((directory/f'pcp{pcp}-metadata.json').read_text()) for pcp in range(8)}
        stage_rows = aggregate_stage(features, events, metadata, stage, spec['name'])
        contexts = {r['step']: (r['context'], r['chunk']) for r in stage_rows}
        if stage0_contexts is None:
            stage0_contexts = contexts
        elif contexts != stage0_contexts:
            raise ValueError(f'{spec["name"]}: context mismatch across PP stages')
        rows.extend(stage_rows)
        metadata_report[stage] = {pcp: {k: value.get(k) for k in (
            'collection_id', 'build_id', 'clock_domain', 'host_clock',
            'dropped_full', 'dropped_busy', 'observer_errors', 'failure')}
            for pcp, value in metadata.items()}
    return rows, metadata_report


def assign_splits(rows, holdout_fraction=.2, gap_steps=4):
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['case'], row['stage']].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda r: (r['reference_start_ns'], r['step']))
        boundary = len(ordered)-max(1, math.ceil(len(ordered)*holdout_fraction))
        for index, row in enumerate(ordered):
            row['split'] = ('holdout' if index >= boundary else
                            'gap' if index >= max(0, boundary-gap_steps) else 'train')


def flag_outliers(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['case'], row['stage'], row['global_padded_tokens']].append(row)
    for group in groups.values():
        values = np.array([r['target_ms'] for r in group])
        median = float(np.median(values))
        mad = float(np.median(np.abs(values-median)))
        threshold = max(20., 6*1.4826*mad)
        for row in group:
            flags = []
            if abs(row['target_ms']-median) > threshold:
                flags.append('local_duration_outlier')
            if row['max_minus_pcp0_ms'] > 20:
                flags.append('max_minus_pcp0_over_20ms')
            if row['start_skew_ms'] is not None and row['start_skew_ms'] > 20:
                flags.append('start_skew_over_20ms')
            row['outlier_flags'] = '|'.join(flags)
            row['outlier_group_median_ms'] = median
            row['outlier_threshold_ms'] = threshold


def fit_stage(
    rows,
    stage,
    n_estimators=400,
    alpha=0.95,
    min_samples_leaf=6,
    batch_size_weight_power=1.0,
):
    names = FEATURE_NAMES + (TAIL_FEATURES if stage == 3 else [])
    selected = [r for r in rows if r['stage'] == stage and r['split'] == 'train']
    if not selected:
        raise ValueError(f'stage {stage}: no training rows')
    x = np.array([[r[name] for name in names] for r in selected])
    y = np.array([r['target_ms']/1000 for r in selected])
    model = GradientBoostingRegressor(
        loss='huber', alpha=alpha, n_estimators=n_estimators, learning_rate=.05,
        max_depth=6, min_samples_leaf=min_samples_leaf, subsample=.8, random_state=42)
    sample_weight = None
    if batch_size_weight_power > 0:
        sample_weight = np.array([
            max(1.0, float(r.get('batch_size', 1.0))) ** batch_size_weight_power
            for r in selected
        ])
    model.fit(x, y, sample_weight=sample_weight)
    return dict(models=[model], mode='direct', features=names,
                feature_schema='tpu_stage3_tail_26_v1' if stage == 3 else 'tpu_stage_22_v1',
                feature_dimension=len(names), stage=stage, time_unit='seconds',
                model_kind=MODEL_KIND, production_usable=False, valid_service_label=False,
                label_scope='effective_host_forward_to_native_ready_max_local_duration',
                label_aggregation='max_pcp_local_duration', pcp_size=8,
                endpoint=endpoint(stage), external_offset_ms=0,
                train_rows=len(selected), train_cases=dict(Counter(f'{r["dataset"]}/{r["case"]}' for r in selected)),
                outlier_policy='No latency outlier removed or capped; Huber loss downweights large residuals',
                batch_size_weight_power=batch_size_weight_power,
                numpy_version=np.__version__, sklearn_version=sklearn.__version__,
                training_parameters=model.get_params(), split='last 20% per dataset/case; preceding 4 steps excluded from fit')


def metrics(rows):
    errors = np.array([r['predicted_ms']-r['target_ms'] for r in rows])
    relative = np.abs(errors)/np.array([r['target_ms'] for r in rows])*100
    return dict(n=len(rows), real_mean_ms=float(np.mean([r['target_ms'] for r in rows])),
                predicted_mean_ms=float(np.mean([r['predicted_ms'] for r in rows])),
                bias_ms=float(errors.mean()), mae_ms=float(np.abs(errors).mean()),
                p95_absolute_error_ms=float(np.quantile(np.abs(errors), .95)),
                mape_pct=float(relative.mean()), p95_ape_pct=float(np.quantile(relative, .95)))


def grouped_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        for kind, value in [('all', 'all'), ('case', f'{row["dataset"]}/{row["case"]}'),
                            ('padding', int(row['global_padded_tokens'])),
                            ('batch_size', int(row['batch_size'])),
                            ('outlier', bool(row['outlier_flags']))]:
            groups[row['stage'], row['split'], kind, str(value)].append(row)
    return [dict(stage=stage, split=split, group=kind, value=value, **metrics(group))
            for (stage, split, kind, value), group in sorted(groups.items())]


def write_csv(path, rows, fieldnames=None):
    if not fieldnames:
        fieldnames = list(rows[0]) if rows else ['empty']
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--min-train-per-padding', type=int, default=20,
                        help='Fail fitting if any stage has fewer actual train rows in 2048/4096/8192')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--min-samples-leaf', type=int, default=6)
    parser.add_argument('--batch-size-weight-power', type=float, default=1.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    names = [s['name'] for s in manifest['datasets']]
    if len(set(names)) != len(names):
        raise ValueError('dataset names must be unique')
    all_rows, metadata = [], {}
    for spec in manifest['datasets']:
        rows, audit = load_dataset(spec)
        all_rows.extend(rows)
        metadata[spec['name']] = audit
    assign_splits(all_rows)
    flag_outliers(all_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'labels.csv', all_rows)
    outliers = [r for r in all_rows if r['outlier_flags']]
    write_csv(args.output/'outliers.csv', outliers, list(all_rows[0]))
    coverage = Counter((r['stage'],r['dataset'],r['case'],r['split'],int(r['global_padded_tokens'])) for r in all_rows)
    counts = [dict(stage=k[0],dataset=k[1],case=k[2],split=k[3],padding=k[4],n=n)
              for k,n in sorted(coverage.items())]
    write_csv(args.output/'coverage.csv', counts)
    report = dict(label='max_i(observed_mono_ns_i - start_mono_ns_i); durations computed locally',
                  end_events=[endpoint(s) for s in range(4)], outliers_retained=True,
                  outlier_diagnostic='per dataset/case/stage/padding: abs(target-median)>max(20ms,6*1.4826*MAD); also rank gap/start skew>20ms',
                  rows=len(all_rows), unique_steps=len(all_rows)//4, outliers=len(outliers),
                  split='last 20% chronologically per dataset/case/stage; 4-step gap before holdout',
                  manifest=manifest, clock_and_collector_metadata=metadata, trained=False)
    (args.output/'aggregation.json').write_text(json.dumps(report, indent=2))
    if args.train:
        for stage in range(4):
            for padding in (2048,4096,8192):
                n=sum(r['stage']==stage and r['split']=='train' and r['global_padded_tokens']==padding for r in all_rows)
                if n < args.min_train_per_padding:
                    raise ValueError(f'stage {stage}, padding {padding}: only {n} actual training rows; need {args.min_train_per_padding}; aggregation saved, no training performed')
        for stage in range(4):
            bundle=fit_stage(
                all_rows,
                stage,
                n_estimators=args.n_estimators,
                alpha=args.alpha,
                min_samples_leaf=args.min_samples_leaf,
                batch_size_weight_power=args.batch_size_weight_power,
            )
            bundle['source_manifest']=manifest
            path=args.output/f'DIAGNOSTIC_ONLY_stage-{stage}_native_return.pkl'
            joblib.dump(bundle,path)
            loaded=StageModel.from_file(str(path),stage,3)
            names=bundle['features']
            selected=[r for r in all_rows if r['stage']==stage]
            direct=bundle['models'][0].predict([[r[k] for k in names] for r in selected])
            runtime=np.array([loaded.predict(r) for r in selected])
            np.testing.assert_allclose(runtime,np.maximum(direct,0),rtol=0,atol=1e-12)
            for row,value in zip(selected,runtime):
                row['predicted_ms']=float(value*1000)
        write_csv(args.output/'predictions.csv',all_rows)
        write_csv(args.output/'metrics.csv',grouped_metrics(all_rows))
        report['trained']=True
        report['models']=[f'DIAGNOSTIC_ONLY_stage-{s}_native_return.pkl' for s in range(4)]
        report['all_stage_metrics']=[r for r in grouped_metrics(all_rows) if r['group']=='all']
        (args.output/'training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(rows=len(all_rows),unique_steps=len(all_rows)//4,outliers=len(outliers),
                         trained=args.train,output=str(args.output)),indent=2))


if __name__ == '__main__':
    main()
