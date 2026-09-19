"""Offline, transport-neutral local-AI fitness evaluator. Never calls a model or CUDA.

A result file is a JSONL ledger of independently captured trials. Simulated
records are for harness tests only and are never ranked beside measurements.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


class InvalidLedger(ValueError):
    pass


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def require(ok: bool, why: str) -> None:
    if not ok:
        raise InvalidLedger(why)


def read_suite(path: Path) -> dict:
    suite = json.loads(path.read_text(encoding='utf-8'))
    require(suite.get('schema_version') == 1 and isinstance(suite.get('cases'), list),
            'Unsupported or malformed suite')
    ids = set()
    for case in suite['cases']:
        ident = case.get('id')
        require(isinstance(ident, str) and ident and ident not in ids,
                'Missing or repeated case ID')
        ids.add(ident)
        require(case.get('kind') in ('exact', 'json_contract', 'tool_call', 'python_syntax', 'repo_task'),
                f'Unsupported case kind: {ident}')
        require(isinstance(case.get('prompt'), str) and case['prompt'],
                f'Missing test prompt: {ident}')
        require(isinstance(case.get('critical'), bool), f'Critical flag missing: {ident}')
    require(bool(ids), 'Suite has no cases')
    return suite


def matches(value: object, expected: object) -> bool:
    """Subset contract, with exact equality at scalar leaves."""
    if isinstance(expected, dict):
        return isinstance(value, dict) and all(k in value and matches(value[k], v)
                                                 for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) == len(expected) and all(
            matches(a, b) for a, b in zip(value, expected))
    return type(value) is type(expected) and value == expected


def judge(case: dict, response: str, code_receipt: dict | None = None,
          repo_receipt: dict | None = None) -> tuple[str, str]:
    try:
        if case['kind'] == 'repo_task':
            from bench.repo_case import judge_repo
            return judge_repo(case, repo_receipt)
        if case['kind'] == 'exact':
            return ('pass', 'exact match') if response.strip() == case['expected'] else ('fail', 'answer mismatch')
        if case['kind'] in ('json_contract', 'tool_call'):
            parsed = json.loads(response)
            if case['kind'] == 'tool_call':
                require(isinstance(parsed, dict) and set(parsed) == {'name', 'arguments'},
                        'Tool call requires only name and arguments')
                require(parsed['name'] == case['tool']['name'], 'Incorrect tool selected')
                require(isinstance(parsed['arguments'], dict), 'Tool arguments must be an object')
                require(set(parsed['arguments']) == set(case['tool']['arguments']),
                        'Missing or extra tool argument')
                expected = case['tool']['arguments']
                return ('pass', 'tool name and arguments match') if matches(parsed['arguments'], expected) else ('fail', 'tool argument mismatch')
            return ('pass', 'JSON contract satisfied') if matches(parsed, case['expected']) else ('fail', 'JSON contract mismatch')
        if case['kind'] == 'python_syntax':
            program = ast.parse(response)
            functions = [n for n in program.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            names = {n.name for n in functions}
            require(case['function'] in names, 'Required function definition absent')
            if code_receipt is not None:
                require(isinstance(code_receipt, dict) and isinstance(case.get('test_source'), str),
                        'Code receipt requires benchmark-owned behavioral tests')
                require(code_receipt.get('solution_sha256') == hashlib.sha256(response.encode('utf-8')).hexdigest() and
                        code_receipt.get('test_sha256') == hashlib.sha256(case['test_source'].encode('utf-8')).hexdigest(),
                        'Code receipt does not match the tested source/fixture')
                require(type(code_receipt.get('tests_ran')) is int and code_receipt['tests_ran'] > 0 and
                        isinstance(code_receipt.get('container_image'), str) and '@sha256:' in code_receipt['container_image'],
                        'Missing pinned container or behavioral test execution')
                return ('pass', 'Pinned-container behavioral tests passed') if code_receipt.get('passed') is True else ('fail', 'Behavioral tests failed')
            return 'unverified', 'Syntax only; isolated behavioral tests are required'
    except (ValueError, SyntaxError, TypeError, KeyError, InvalidLedger) as exc:
        return 'fail', str(exc)
    raise InvalidLedger('Unknown case kind')


REQUIRED_CONTROL = ('model_revision', 'engine_revision', 'quant', 'cache',
                    'mtp_depth', 'context_tokens', 'sampler', 'power_mode',
                    'template_sha256', 'output_limit')
REQUIRED_METRIC = ('ttft_ms', 'e2e_ms', 'decode_ms', 'output_tokens')


def validate_trial(trial: dict, cases: dict, suite_hash: str) -> tuple[str, int]:
    require(trial.get('schema_version') == 1, 'Unknown trial schema')
    require(trial.get('suite_sha256') == suite_hash, 'Suite digest mismatch')
    require(trial.get('case_id') in cases, 'Unknown case')
    require(trial.get('provenance') in ('simulated', 'measured'), 'Unknown provenance')
    require(isinstance(trial.get('arm'), str) and trial['arm'], 'Missing run arm')
    require(type(trial.get('repeat')) is int and 0 <= trial['repeat'] <= 10000,
            'Invalid trial repeat')
    require(isinstance(trial.get('response'), str), 'Response must be text')
    controls = trial.get('controls')
    require(isinstance(controls, dict) and all(key in controls for key in REQUIRED_CONTROL),
            'Missing controlled configuration identity')
    metrics = trial.get('metrics')
    require(isinstance(metrics, dict) and all(key in metrics for key in REQUIRED_METRIC),
            'Missing required timing/token telemetry')
    for key in REQUIRED_METRIC:
        val = metrics[key]
        if cases[trial['case_id']]['kind'] == 'repo_task' and key in ('ttft_ms', 'decode_ms') and val is None:
            continue  # non-streaming multi-turn agent has no observed first-token/decode interval
        require(type(val) in (int, float) and math.isfinite(val) and val >= 0,
                'Invalid metric ' + key)
    require(all(metrics['e2e_ms'] >= metrics[key] for key in ('ttft_ms', 'decode_ms')
                if metrics[key] is not None), 'Inconsistent time spans')
    require(type(metrics['output_tokens']) is int, 'Token count must be integer')
    if trial['provenance'] == 'measured':
        require(isinstance(trial.get('receipt'), dict) and
                all(trial['receipt'].get(k) for k in
                    ('engine_commit', 'checkpoint_revision', 'hardware_sku', 'run_id')),
                'Measured trial requires hardware and revision receipt')
    return trial['case_id'], trial['repeat']


def evaluate(suite: dict, trials: list[dict]) -> dict:
    cases = {case['id']: case for case in suite['cases']}
    digest = canonical_digest(suite)
    observed = set()
    output = []
    for trial in trials:
        key = validate_trial(trial, cases, digest)
        identity = (trial['arm'], *key)
        require(identity not in observed, 'Repeated (arm, case, repeat) record')
        observed.add(identity)
        status, why = judge(cases[trial['case_id']], trial['response'], trial.get('code_receipt'),
                            trial.get('repo_receipt'))
        if cases[trial['case_id']]['kind'] == 'repo_task' and trial.get('agent_failed') is True:
            status, why = 'fail', 'Agent failed to complete the bounded tool session'
        output.append({'arm': trial['arm'], 'case_id': key[0], 'repeat': key[1],
                       'critical': cases[key[0]]['critical'], 'status': status,
                       'reason': why, 'provenance': trial['provenance'],
                       'controls': trial['controls'], 'metrics': trial['metrics'],
                       'gpu_telemetry': trial.get('gpu_telemetry', {'collected': False}),
                       'response_sha256': hashlib.sha256(trial['response'].encode('utf-8')).hexdigest()})
    return {'suite_sha256': digest, 'records': output, 'static_analysis_only': True,
            'model_or_gpu_executed_by_tool': False}


def compare(report: dict, suite: dict, arm_a: str, arm_b: str,
            changed_controls: tuple[str, ...] = ()) -> dict:
    records = report['records']
    arms = {arm: {(r['case_id'], r['repeat']): r for r in records if r['arm'] == arm}
            for arm in (arm_a, arm_b)}
    keys = set(arms[arm_a])
    require(keys and keys == set(arms[arm_b]), 'A/B must contain identical case and repeat keys')
    require({key[0] for key in keys} == {case['id'] for case in suite['cases']},
            'A/B omitted suite cases')
    pair_results = []
    controls_a = {canonical_digest(r['controls']) for r in arms[arm_a].values()}
    controls_b = {canonical_digest(r['controls']) for r in arms[arm_b].values()}
    require(len(controls_a) == len(controls_b) == 1, 'Controls changed within one arm')
    a0, b0 = next(iter(arms[arm_a].values())), next(iter(arms[arm_b].values()))
    differences = {key for key in REQUIRED_CONTROL
                   if a0['controls'][key] != b0['controls'][key]}
    require(differences == set(changed_controls), 'Uncontrolled A/B configuration difference')
    origins = {r['provenance'] for arm in arms.values() for r in arm.values()}
    require(len(origins) == 1, 'Cannot compare simulated and measured results')
    telemetry_modes = {r['gpu_telemetry'].get('collected', False) for arm in arms.values() for r in arm.values()}
    require(len(telemetry_modes) == 1, 'Instrumentation differs: GPU telemetry must match on A and B')
    for key in sorted(keys):
        a, b = arms[arm_a][key], arms[arm_b][key]
        pair_results.append({'case_id': key[0], 'repeat': key[1],
                             'a_quality': a['status'], 'b_quality': b['status'],
                             'a_e2e_ms': a['metrics']['e2e_ms'],
                             'b_e2e_ms': b['metrics']['e2e_ms'],
                             'delta_e2e_ms': b['metrics']['e2e_ms'] - a['metrics']['e2e_ms']})
    quality_complete = all(p['a_quality'] != 'unverified' and p['b_quality'] != 'unverified'
                           for p in pair_results)
    critical_ok = all(not arms[arm_b][key]['critical'] or
                      arms[arm_b][key]['status'] == 'pass' for key in keys)
    summaries = {}
    for arm in (arm_a, arm_b):
        rows = list(arms[arm].values())
        decode_known = all(r['metrics']['decode_ms'] is not None for r in rows)
        total_decode = sum(r['metrics']['decode_ms'] for r in rows) if decode_known else 0
        ttfts = [r['metrics']['ttft_ms'] for r in rows if r['metrics']['ttft_ms'] is not None]
        gpu_peaks = [r['gpu_telemetry'].get('peak_used_mib') for r in rows
                     if r['gpu_telemetry'].get('collected') and r['gpu_telemetry'].get('peak_used_mib') is not None]
        temp_peaks = [r['gpu_telemetry'].get('peak_temperature_c') for r in rows
                      if r['gpu_telemetry'].get('collected') and r['gpu_telemetry'].get('peak_temperature_c') is not None]
        summaries[arm] = {
            'sampled_whole_gpu_peak_used_mib': max(gpu_peaks) if gpu_peaks else None,
            'sampled_gpu_peak_temp_c': max(temp_peaks) if temp_peaks else None,
            'trials': len(rows), 'quality_pass': sum(r['status']=='pass' for r in rows),
            'quality_fail': sum(r['status']=='fail' for r in rows),
            'quality_unverified': sum(r['status']=='unverified' for r in rows),
            'median_e2e_ms': statistics.median(r['metrics']['e2e_ms'] for r in rows),
            'median_ttft_ms': statistics.median(ttfts) if len(ttfts) == len(rows) else None,
            'client_observed_decode_tokens_per_second': (
                round(1000*sum(r['metrics']['output_tokens'] for r in rows)/total_decode, 3)
                if total_decode > 0 else None),
            'token_rate_warning': ('Client-observed streaming decode time; not GPU compute throughput'
                                   if decode_known else 'Multi-turn agent has no observed intertoken decode interval'),
        }
    return {'arms': [arm_a, arm_b], 'changed_controls': sorted(differences),
            'arm_summaries': summaries,
            'provenance': next(iter(origins)), 'quality_fully_verified': quality_complete,
            'candidate_critical_cases_pass': critical_ok,
            'eligible_for_production_decision': False,
            'requires_final_hardware_and_human_review': True,
            'pairs': pair_results, 'note': 'Time deltas are descriptive, never a quality/speed winner.'}


def read_trials(path: Path) -> list[dict]:
    records = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise InvalidLedger(f'Invalid JSONL at line {i}: {exc}') from exc
    require(bool(records), 'No trial records')
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True,
                        help='Read recorded trials; never contacts an inference endpoint')
    parser.add_argument('--arm-a', help='Optional paired comparison baseline ID')
    parser.add_argument('--arm-b', help='Optional paired comparison candidate ID')
    parser.add_argument('--changed-control', action='append', default=[],
                        choices=REQUIRED_CONTROL)
    parser.add_argument('--output', type=Path, help='Write JSON report (private by default)')
    args = parser.parse_args()
    try:
        suite = read_suite(args.suite)
        report = evaluate(suite, read_trials(args.records))
        if args.arm_a or args.arm_b:
            require(bool(args.arm_a and args.arm_b and args.arm_a != args.arm_b),
                    'Specify two distinct arms')
            report['comparison'] = compare(report, suite, args.arm_a, args.arm_b,
                                           tuple(args.changed_control))
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + '\n', encoding='utf-8')
        else:
            print(rendered)
    except (InvalidLedger, KeyError, OSError, TypeError) as exc:
        parser.exit(2, 'FITNESS EVALUATION REJECTED: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
