"""Generate a balanced A/B/B/A campaign schedule. PLAN ONLY; never launch models.

Each arm receives identical suite cases at matching repeat identifiers. The
actual run requires a separately authorized final-stage collector invocation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from bench.fitness import REQUIRED_CONTROL, canonical_digest, read_suite


def build_campaign(suite: dict, a: dict, b: dict, *, repeats: int = 5,
                   changed_control: tuple[str, ...] = ()) -> dict:
    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError('A/B/B/A block count must be between 1 and 100')
    for label, controls in (('A', a), ('B', b)):
        if not isinstance(controls, dict) or not all(k in controls for k in REQUIRED_CONTROL):
            raise ValueError(f'Missing explicit {label} controls')
    differences = {k for k in REQUIRED_CONTROL if a[k] != b[k]}
    if differences != set(changed_control):
        raise ValueError('Uncontrolled profile difference: ' + ', '.join(sorted(differences)))
    if any(k not in REQUIRED_CONTROL for k in changed_control):
        raise ValueError('Unknown changed control')
    ordered = []
    for block in range(repeats):
        for arm, rep in (('A', 2*block), ('B', 2*block),
                         ('B', 2*block+1), ('A', 2*block+1)):
            ordered.append({'ordinal': len(ordered)+1, 'arm': arm, 'repeat': rep,
                            'suite_sha256': canonical_digest(suite),
                            'expected_cases': [case['id'] for case in suite['cases']],
                            'action': 'PLAN ONLY - no GPU or endpoint contacted'})
    return {'schema_version': 1, 'suite_sha256': canonical_digest(suite),
            'blocks': repeats, 'paired_repeats_per_arm': repeats*2,
            'changed_control': sorted(differences), 'controls': {'A': a, 'B': b},
            'warmup': 'Warm each arm equally; exclude warmup from scored samples',
            'thermal_rule': 'Match AC/fan/power mode; record temperature and clocks in final stage',
            'runs': ordered, 'execution_authorized': False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', required=True, type=Path)
    ap.add_argument('--controls-a', required=True, type=Path)
    ap.add_argument('--controls-b', required=True, type=Path)
    ap.add_argument('--changed-control', action='append', default=[], choices=REQUIRED_CONTROL)
    ap.add_argument('--blocks', type=int, default=5)
    ap.add_argument('--output', type=Path, default=Path('.local/benchmark/campaign.json'))
    args = ap.parse_args()
    try:
        suite = read_suite(args.suite)
        a = json.loads(args.controls_a.read_text(encoding='utf-8'))
        b = json.loads(args.controls_b.read_text(encoding='utf-8'))
        result = build_campaign(suite, a, b, repeats=args.blocks,
                                changed_control=tuple(args.changed_control))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n',
                               encoding='utf-8')
        print(json.dumps({'status': 'PLAN_ONLY', 'model_or_gpu_contacted': False,
                          'scheduled_arms': len(result['runs']),
                          'paired_repeats_per_arm': result['paired_repeats_per_arm'],
                          'output': str(args.output)}))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        ap.exit(2, 'CAMPAIGN PLAN REJECTED: '+str(exc)+'\n')


if __name__ == '__main__':
    main()
