"""Future-stage OpenAI-compatible local endpoint collector; DRY RUN BY DEFAULT.

Importing this module neither contacts a model nor initializes CUDA. Live mode
requires an explicit flag AND separate environment authorization. The endpoint
is constrained to IPv4 loopback, with no redirects or arbitrary tool execution.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from bench.fitness import REQUIRED_CONTROL, canonical_digest, read_suite
from bench.telemetry import GpuSampler

LIVE_GATE = 'LOCAL_AI_FINAL_BENCHMARKS_AUTHORIZED'


class CollectorError(ValueError):
    pass


def assert_local_endpoint(url: str) -> None:
    u = urllib.parse.urlsplit(url)
    if u.scheme != 'http' or u.hostname != '127.0.0.1' or u.username or u.password or u.fragment:
        raise CollectorError('Endpoint must be explicit http://127.0.0.1:<port>/v1/chat/completions')
    if not (1 <= (u.port or 0) <= 65535) or u.path != '/v1/chat/completions' or u.query:
        raise CollectorError('Invalid local API route, query or port')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CollectorError('No redirects are permitted for local benchmark requests')


def build_request(case: dict, model: str, max_tokens: int) -> dict:
    body = {'model': model, 'messages': [{'role': 'user', 'content': case['prompt']}],
            'temperature': 0, 'max_tokens': max_tokens, 'stream': True,
            'stream_options': {'include_usage': True}}
    if case['kind'] == 'tool_call':
        tool = case['tool']
        props = {key: {'type': 'string'} for key in tool['arguments']}
        body['tools'] = [{'type': 'function', 'function': {
            'name': tool['name'], 'description': 'Select the requested local test tool.',
            'parameters': {'type': 'object', 'properties': props,
                           'required': list(props), 'additionalProperties': False}}}]
        body['tool_choice'] = 'required'
    return body


def collect_stream(stream) -> dict:
    """Consumes SSE bytes in a testable way; does not invent missing token counts."""
    began = time.monotonic()
    first = None
    content = []
    tool_calls = {}
    usage = None
    finish = None
    done = False
    for raw in stream:
        line = raw.decode('utf-8').strip()
        if not line.startswith('data:'):
            continue
        value = line[5:].strip()
        if value == '[DONE]':
            done = True
            break
        if not value:
            continue
        chunk = json.loads(value)
        if chunk.get('error'):
            raise CollectorError('Endpoint returned an error chunk')
        if chunk.get('usage') is not None:
            usage = chunk['usage']
        for choice in chunk.get('choices', []):
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
            delta = choice.get('delta') or {}
            text = delta.get('content') or ''
            if text:
                first = time.monotonic() if first is None else first
                content.append(text)
            for tc in delta.get('tool_calls') or []:
                first = time.monotonic() if first is None else first
                index = tc['index']
                entry = tool_calls.setdefault(index, {'name': '', 'arguments': ''})
                function = tc.get('function') or {}
                entry['name'] += function.get('name') or ''
                entry['arguments'] += function.get('arguments') or ''
    finished = time.monotonic()
    if not done or first is None or finish is None or not isinstance(usage, dict):
        raise CollectorError('Incomplete SSE, missing first output, finish or token usage')
    out_tokens = usage.get('completion_tokens')
    in_tokens = usage.get('prompt_tokens')
    if type(out_tokens) is not int or out_tokens < 0 or type(in_tokens) is not int or in_tokens < 0:
        raise CollectorError('Server must return actual prompt and completion token counts')
    if tool_calls:
        if len(tool_calls) != 1:
            raise CollectorError('This one-tool fixture requires exactly one tool call')
        tc = next(iter(tool_calls.values()))
        try:
            response = json.dumps({'name': tc['name'], 'arguments': json.loads(tc['arguments'])},
                                  separators=(',', ':'), sort_keys=True)
        except json.JSONDecodeError as exc:
            raise CollectorError('Server returned invalid tool arguments') from exc
    else:
        response = ''.join(content)
    return {'response': response, 'finish_reason': finish,
            'metrics': {'ttft_ms': round((first - began) * 1000, 3),
                        'e2e_ms': round((finished - began) * 1000, 3),
                        'decode_ms': round((finished - first) * 1000, 3),
                        'output_tokens': out_tokens, 'input_tokens': in_tokens},
            'measurement_scope': 'client-observed streaming wall clock; not GPU kernel time'}


def request_case(url: str, case: dict, model: str, max_tokens: int,
                 timeout: int = 300) -> dict:
    assert_local_endpoint(url)
    body = json.dumps(build_request(case, model, max_tokens)).encode('utf-8')
    req = urllib.request.Request(url, data=body, method='POST',
                                 headers={'Content-Type': 'application/json',
                                          'Accept': 'text/event-stream'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=timeout) as response:
            if response.status != 200:
                raise CollectorError('Local API returned non-200 HTTP status')
            return collect_stream(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CollectorError('Local API unavailable or timed out') from exc


def enforce_final_gate(*, execute: bool, authorization: str | None,
                       armed: dict | None) -> None:
    if not execute:
        return
    if authorization != 'I_AUTHORIZE_FINAL_GPU_BENCHMARKS':
        raise CollectorError('Live collection requires separate final-stage authorization')
    if not isinstance(armed, dict) or armed.get('status') != 'static-contract-armed':
        raise CollectorError('Live collection requires an armed profile preflight receipt')
    if armed.get('inference_or_gpu_testing') != 'NOT PERFORMED':
        raise CollectorError('Armed preflight receipt must be from static validation')


def capture(suite: dict, url: str, model: str, arm: str, repeat: int,
            controls: dict, receipt: dict, output: Path,
            docker_image: str | None = None, gpu_telemetry: bool = False) -> list[dict]:
    """Called exclusively after final-stage gate; writes private JSONL receipts."""
    if not all(key in controls for key in REQUIRED_CONTROL):
        raise CollectorError('All configuration controls must be explicitly pinned')
    if not isinstance(receipt, dict) or not all(receipt.get(key) for key in
                ('engine_commit', 'checkpoint_revision', 'hardware_sku', 'run_id')):
        raise CollectorError('Incomplete immutable run receipt')
    if receipt['engine_commit'] != controls['engine_revision'] or receipt['checkpoint_revision'] != controls['model_revision']:
        raise CollectorError('Controls and source/model receipt disagree')
    assert_local_endpoint(url)
    records = []
    for case in suite['cases']:
        with GpuSampler(enabled=gpu_telemetry) as sampler:
            observed = request_case(url, case, model, controls['output_limit'])
        telemetry = sampler.summary()
        code_receipt = None
        if case['kind'] == 'python_syntax' and docker_image:
            from bench.code_judge import run_in_container
            code_receipt = run_in_container(observed['response'], case['test_source'],
                                            image=docker_image, private_root=Path('.local/benchmark/code-sandbox'))
        records.append({'schema_version': 1, 'suite_sha256': canonical_digest(suite),
                        'case_id': case['id'], 'arm': arm, 'repeat': repeat,
                        'provenance': 'measured', 'response': observed['response'],
                        'controls': controls, 'receipt': receipt, 'code_receipt': code_receipt,
                        'metrics': observed['metrics'], 'finish_reason': observed['finish_reason'],
                        'measurement_scope': observed['measurement_scope'],
                        'gpu_telemetry': telemetry})
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('a', encoding='utf-8') as f:
            f.write(json.dumps(records[-1], sort_keys=True) + '\n')
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', type=Path, required=True)
    ap.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    ap.add_argument('--model', default='pinned-local-model')
    ap.add_argument('--arm', default='baseline')
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--controls', type=Path, help='Immutable run settings JSON for live mode')
    ap.add_argument('--receipt', type=Path, help='Local GPU/source receipt JSON for live mode')
    ap.add_argument('--armed-report', type=Path, help='Latest armed preflight JSON')
    ap.add_argument('--output', type=Path, default=Path('.local/benchmark/live-trials.jsonl'))
    ap.add_argument('--gpu-telemetry', action='store_true', help='Final-stage optional 1s whole-card GPU sampling')
    ap.add_argument('--docker-image', help='Optional locally cached image pinned by @sha256 for isolated code behavior checks')
    ap.add_argument('--execute', action='store_true', help='Final-stage only: actually contact local model')
    args = ap.parse_args()
    try:
        assert_local_endpoint(args.url)
        suite = read_suite(args.suite)
        report = json.loads(args.armed_report.read_text(encoding='utf-8')) if args.armed_report else None
        enforce_final_gate(execute=args.execute, authorization=os.environ.get(LIVE_GATE), armed=report)
        if not args.execute:
            print(json.dumps({'mode': 'DRY_RUN', 'model_or_gpu_contacted': False,
                              'suite_sha256': canonical_digest(suite), 'case_ids': [c['id'] for c in suite['cases']],
                              'live_collection': 'BLOCKED until separately authorized final-stage validation'},
                             indent=2))
            return
        if not args.controls or not args.receipt:
            raise CollectorError('Live mode requires explicit controls and receipt files')
        if args.output.resolve().is_relative_to(Path.cwd().resolve()) and not args.output.resolve().is_relative_to((Path.cwd()/'.local').resolve()):
            raise CollectorError('Raw benchmark output inside the repo must be under ignored .local/')
        settings = json.loads(args.controls.read_text(encoding='utf-8'))
        receipt = json.loads(args.receipt.read_text(encoding='utf-8'))
        count = len(capture(suite, args.url, args.model, args.arm, args.repeat,
                            settings, receipt, args.output, args.docker_image, args.gpu_telemetry))
        print(json.dumps({'completed_cases': count, 'private_record_path': str(args.output),
                          'note': 'Server usage and client timings are not GPU kernel metrics'}))
    except (CollectorError, OSError, ValueError, KeyError, TypeError) as exc:
        ap.exit(2, 'BENCHMARK COLLECTOR REJECTED: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
