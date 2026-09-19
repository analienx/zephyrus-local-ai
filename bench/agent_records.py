"""Convert a trusted multi-turn session to the existing paired fitness ledger.

No model or generated-source execution. Per-token TTFT/decode remain explicitly
unknown for a non-streaming session; only total task wall time and usage exist.
"""
from __future__ import annotations
from bench.fitness import REQUIRED_CONTROL, canonical_digest


class AgentRecordError(ValueError):
    pass


def make_trial(suite: dict, result: dict, controls: dict, hardware: dict, *,
               arm: str, repeat: int, provenance: str = 'measured') -> dict:
    cases = [c for c in suite['cases'] if c['kind'] == 'repo_task' and
             c['id'] == result.get('task_id')]
    if len(cases) != 1 or type(repeat) is not int or repeat < 0 or not arm:
        raise AgentRecordError('Unknown repository task, arm or repeat')
    if provenance not in ('measured', 'simulated'):
        raise AgentRecordError('Unknown trial provenance')
    if not isinstance(controls, dict) or any(k not in controls for k in REQUIRED_CONTROL):
        raise AgentRecordError('Missing immutable experiment controls')
    if not isinstance(hardware, dict) or any(not hardware.get(k) for k in
                     ('engine_commit', 'checkpoint_revision', 'hardware_sku', 'run_id')):
        raise AgentRecordError('Incomplete revision/hardware receipt')
    if hardware['engine_commit'] != controls['engine_revision'] or \
       hardware['checkpoint_revision'] != controls['model_revision']:
        raise AgentRecordError('Runtime/checkpoint receipts differ from pinned controls')
    if type(result.get('model_completion_tokens')) is not int or \
       type(result.get('elapsed_ms')) not in (float, int) or result['elapsed_ms'] < 0:
        raise AgentRecordError('Missing measured task time or reported output tokens')
    return {'schema_version': 1, 'suite_sha256': canonical_digest(suite),
            'case_id': cases[0]['id'], 'arm': arm, 'repeat': repeat,
            'provenance': provenance, 'response': '', 'controls': controls,
            'receipt': hardware, 'agent_failed': result.get('agent_failed') is True,
            'repo_receipt': result.get('repo_receipt'),
            'metrics': {'ttft_ms': None, 'decode_ms': None,
                        'e2e_ms': result['elapsed_ms'],
                        'output_tokens': result['model_completion_tokens'],
                        'input_tokens': result['model_prompt_tokens'],
                        'model_turns': result['model_turns']},
            'measurement_scope': 'Whole multi-turn tool session; no observed intertoken rate',
            'gpu_telemetry': {'collected': False}}
