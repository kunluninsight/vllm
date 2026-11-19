#!/usr/bin/env python3
"""Flatten existing per-case paired summaries; does not read/query online traces.

Expected layout: ANALYSIS_ROOT/<client case name without mry->/analysis/
paired-summary.json. Values retain the analyzer's successful paired-request scope.
This script does not turn paired token counters into throughput measurements.
"""
import argparse
import csv
import json
import math
from pathlib import Path

PARTS = ('admit_to_dispatch', 'between_dispatch', 'last_dispatch_to_ready', 'ready_to_consume')


def summarize(case, paired, success_samples=None, paired_samples=None):
    if paired['case'] != case['case'].removeprefix('mry-'):
        raise ValueError(f'{case["case"]}: paired summary belongs to another case')
    coverage = paired['coverage']
    count = paired['pairs']
    if count != coverage['pairs']:
        raise ValueError(f'{case["case"]}: inconsistent pair counts')
    missing = {side: len(coverage['missing'][side]) for side in ('real', 'sim')}
    success = case['outcomes']['success']
    complete = count == success == coverage['receipts'] and not any(missing.values())
    paired_success = paired_failures = None
    missing_success = None
    if success_samples is not None and paired_samples is not None:
        if len(success_samples) != success or len(paired_samples) != count:
            raise ValueError(f'{case["case"]}: receipt/pair identity counts disagree')
        paired_success = len(success_samples & paired_samples)
        paired_failures = len(paired_samples - success_samples)
        missing_success = {side: len(success_samples & set(coverage['missing'][side])) for side in ('real', 'sim')}
        complete = success_samples <= paired_samples and not any(missing_success.values())
    row = {
        'case': case['case'], 'concurrency': case['concurrency'],
        'start_unix_s': case['start_unix_s'],
        'admission_stop_unix_s': case['admission_stop_unix_s'],
        'completed_unix_s': case['completed_unix_s'],
        'sent': case['sent'], 'real_success': success,
        'real_errors': case['sent'] - success,
        'receipt_scope_count': coverage['receipts'], 'pairs': count,
        'missing_real': missing['real'], 'missing_sim': missing['sim'],
        'complete_successful_request_coverage': complete,
        'clean_comparison_prerequisites': complete and case['sent'] == success,
        'paired_successful_requests': paired_success,
        'paired_client_failed_requests': paired_failures,
        'missing_success_real': missing_success['real'] if missing_success is not None else None,
        'missing_success_sim': missing_success['sim'] if missing_success is not None else None,
        'equal_input_tokens': paired['equal_input_tokens'],
        'equal_scheduled_query_tokens': paired['equal_scheduled_query_tokens'],
        'same_batch_members': paired['same_batch_members'],
        'same_batch_shapes': paired['same_batch_shapes'],
    }
    for side in ('real', 'sim'):
        values = paired[side]
        for field in ('core', 'core_p95_ms', *PARTS, 'trace_input_tokens', 'trace_hit_tokens', 'trace_hit_fraction'):
            value = values[field]
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{case["case"]}: invalid {side}.{field}')
            row[side + '_' + field] = value
        if not math.isclose(sum(values[k] for k in PARTS), values['core'], abs_tol=.001, rel_tol=0):
            raise ValueError(f'{case["case"]}: {side} TTFT decomposition does not add up')
    row['engine_ttft_error_pct'] = (paired['sim']['core'] / paired['real']['core'] - 1) * 100 if paired['real']['core'] else None
    for field in PARTS:
        row[field + '_delta_ms'] = paired['sim'][field] - paired['real'][field]
    row['paired_prefix_difference_pp'] = (paired['sim']['trace_hit_fraction'] - paired['real']['trace_hit_fraction']) * 100
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a nonempty client summary list')
    rows = []
    for case in cases:
        name = case['case'].removeprefix('mry-')
        if Path(name).name != name:
            raise ValueError('case must be a single directory name')
        path = args.analysis_root / name / 'analysis' / 'paired-summary.json'
        receipts = args.analysis_root / name / ('receipts-' + name + '.csv')
        pairs = path.parent / ('engine-pairs-' + name + '-paired.csv')
        success_samples = paired_samples = None
        if receipts.exists() and pairs.exists():
            with receipts.open(newline='') as handle:
                success_samples = {r['sample'] for r in csv.DictReader(handle) if r['outcome'] == 'success'}
            with pairs.open(newline='') as handle:
                paired_samples = {r['sample'] for r in csv.DictReader(handle)}
        row = summarize(case, json.loads(path.read_text()), success_samples, paired_samples)
        row['source_summary'] = str(path)
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'engine-summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    with (args.output / 'engine-summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'cases': len(rows), 'pairs': sum(row['pairs'] for row in rows),
                      'complete_successful_request_coverage': all(row['complete_successful_request_coverage'] for row in rows),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
