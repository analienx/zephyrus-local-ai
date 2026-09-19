"""Final-stage-only OpenAI tool-call adapter; importing/dry-run never contacts a model.

The model sees bounded workspace tool results, never evaluator-owned grading receipts.
Use with a disposable, unprivileged machine; Docker is not a hardened sandbox.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from bench.agent_loop import TOOLS, run_session
from bench.agent_replay import AgentTaskError, load_task
from bench.collector import (LIVE_GATE, CollectorError, NoRedirect,
                             assert_local_endpoint, enforce_final_gate)

MAX_RESPONSE = 1 << 20


def tool_schema(name: str) -> dict:
    fields = {'read_file': {'path': {'type': 'string'}},
              'write_file': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
              'finish': {'summary': {'type': 'string'}}}.get(name, {})
    return {'type': 'function', 'function': {'name': name,
            'description': 'Bounded disposable-workspace operation; never access grader files.',
            'parameters': {'type': 'object', 'properties': fields,
                           'required': list(fields), 'additionalProperties': False}}}


def encode_messages(messages: list[dict]) -> list[dict]:
    """Translate internal tool transcript into valid OpenAI tool_call_id exchanges."""
    encoded = []
    for message in messages:
        role = message['role']
        if role in ('system', 'user'):
            encoded.append({'role': role, 'content': message['content']})
        elif role == 'assistant':
            call = message['tool_calls'][0]
            encoded.append({'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': f'call_{len(encoded):04d}', 'type': 'function', 'function': {
                 'name': call['tool'], 'arguments': json.dumps(call['arguments'])}}]})
        elif role == 'tool':
            if not encoded or encoded[-1]['role'] != 'assistant':
                raise AgentTaskError('Orphan tool response')
            ident = encoded[-1]['tool_calls'][0]['id']
            encoded.append({'role': 'tool', 'tool_call_id': ident, 'content': message['content']})
        else:
            raise AgentTaskError('Unsupported message role')
    return encoded

class OpenAITransport:
    """Non-streaming multi-turn transport; instrument task wall-clock, not GPU time."""
    def __init__(self, url: str, model: str, max_tokens: int = 1024, *, sender=None):
        assert_local_endpoint(url)
        if not isinstance(model, str) or not model or not 1 <= max_tokens <= 16384:
            raise AgentTaskError('Invalid model name or response-token limit')
        self.url, self.model, self.max_tokens = url, model, max_tokens
        self.sender = sender or self._send
        self.usage = []

    def _send(self, payload: dict) -> dict:
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(self.url, body, method='POST', headers={
            'Content-Type': 'application/json', 'Accept': 'application/json'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=180) as response:
                if response.status != 200:
                    raise AgentTaskError('Unexpected model HTTP status')
                raw = response.read(MAX_RESPONSE + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise AgentTaskError('Loopback inference endpoint failed') from exc
        if len(raw) > MAX_RESPONSE:
            raise AgentTaskError('Model response exceeds size limit')
        return json.loads(raw)

    def complete(self, messages: list[dict]) -> dict:
        body = {'model': self.model, 'messages': encode_messages(messages),
                'tools': [tool_schema(t) for t in TOOLS], 'tool_choice': 'required',
                'parallel_tool_calls': False, 'temperature': 0,
                'max_tokens': self.max_tokens, 'stream': False}
        began = time.monotonic()
        answer = self.sender(body)
        elapsed_ms = round((time.monotonic() - began) * 1000, 3)
        if not isinstance(answer, dict) or len(answer.get('choices', [])) != 1:
            raise AgentTaskError('Expected exactly one assistant completion')
        choice = answer['choices'][0]
        message = choice.get('message') or {}
        calls = message.get('tool_calls')
        if choice.get('finish_reason') != 'tool_calls' or not isinstance(calls, list) or len(calls) != 1:
            raise AgentTaskError('Expected exactly one structured tool call, not prose')
        item = calls[0]
        function = item.get('function') or {}
        name = function.get('name')
        if item.get('type') != 'function' or name not in TOOLS or not isinstance(item.get('id'), str):
            raise AgentTaskError('Unknown or malformed model tool call')
        arguments = json.loads(function.get('arguments', ''))
        if not isinstance(arguments, dict):
            raise AgentTaskError('Tool arguments must be an object')
        usage = answer.get('usage') or {}
        prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if type(prompt) is not int or type(completion) is not int or prompt < 0 or completion < 0:
            raise AgentTaskError('Missing actual per-turn token usage')
        self.usage.append({'elapsed_ms': elapsed_ms, 'prompt_tokens': prompt,
                           'completion_tokens': completion, 'tool': name})
        return {'tool_calls': [{'tool': name, 'arguments': arguments}], 'content': ''}

def run_live(task_root: Path, transport: OpenAITransport, scratch: Path, *,
             image: str, allow_execution: bool) -> dict:
    """Only the trusted collector supplies grader callbacks; tool args contain no receipts."""
    if not allow_execution:
        raise AgentTaskError('Live agent execution not authorized')
    if not image:
        raise AgentTaskError('Pinned, locally cached isolated-grader image is required')
    from bench.repo_judge import grade, grade_visible, image_is_pinned
    from bench.fitness import read_suite
    from bench.repo_case import judge_repo
    if not image_is_pinned(image):
        raise AgentTaskError('Grader image must be pinned by digest')
    suite = read_suite(Path(__file__).resolve().parent / 'fixtures/agent-suite.json')
    task, _ = load_task(task_root.resolve())
    cases = [c for c in suite['cases'] if c['id'] == task['id']]
    if len(cases) != 1:
        raise AgentTaskError('Task is not registered in evaluator-owned suite')
    def visible(root, working):
        return grade_visible(root, working, image=image, private_root=scratch,
                             authorized=True)
    def final(root, working):
        return grade(root, working, image=image, private_root=scratch,
                     authorized=True)
    started = time.monotonic()
    try:
        result = run_session(task_root, transport, scratch, visible_runner=visible,
                             final_grader=final)
        status, reason = judge_repo(cases[0], result['repo_receipt'])
    except (AgentTaskError, CollectorError, json.JSONDecodeError) as exc:
        result = {'task_id': task['id'], 'repo_receipt': None,
                  'elapsed_ms': round((time.monotonic()-started)*1000, 3),
                  'agent_failed': True, 'failure_class': type(exc).__name__}
        status, reason = 'fail', 'Agent failed a bounded tool or response contract'
    except (OSError, RuntimeError) as exc:
        result = {'task_id': task['id'], 'repo_receipt': None,
                  'elapsed_ms': round((time.monotonic()-started)*1000, 3),
                  'agent_failed': False, 'failure_class': type(exc).__name__}
        status, reason = 'unverified', 'Grader or infrastructure failed; not a model quality result'
    result.update(status=status, reason=reason,
                  model_turns=len(transport.usage),
                  model_prompt_tokens=sum(x['prompt_tokens'] for x in transport.usage),
                  model_completion_tokens=sum(x['completion_tokens'] for x in transport.usage),
                  model_turn_timings=transport.usage,
                  quality_source=('trusted isolated repository grader' if result.get('repo_receipt')
                                  else 'no completed grader receipt'))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', required=True, type=Path)
    ap.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions')
    ap.add_argument('--model', default='pinned-local-model')
    ap.add_argument('--image', help='Pinned locally cached grader container digest')
    ap.add_argument('--armed-report', type=Path)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('.local/benchmark/agent-live.json'))
    ap.add_argument('--controls', type=Path, help='Explicit model/engine/quant/sampler controls')
    ap.add_argument('--hardware-receipt', type=Path, help='Local hardware and revisions receipt')
    ap.add_argument('--arm', default='baseline')
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--record-output', type=Path, default=Path('.local/benchmark/agent-trials.jsonl'))
    args = ap.parse_args()
    try:
        task, _ = load_task(args.task.resolve())
        assert_local_endpoint(args.url)
        armed = json.loads(args.armed_report.read_text(encoding='utf-8')) if args.armed_report else None
        enforce_final_gate(execute=args.execute, authorization=os.environ.get(LIVE_GATE), armed=armed)
        if not args.execute:
            print(json.dumps({'status': 'DRY_RUN', 'task_id': task['id'],
                              'model_or_gpu_contacted': False, 'generated_code_executed': False}))
            return
        output = args.output.resolve()
        scratch = (Path.cwd() / '.local/benchmark/agent').resolve()
        if not output.is_relative_to((Path.cwd()/'.local').resolve()):
            raise AgentTaskError('Raw agent results must be written under ignored .local/')
        records = args.record_output.resolve()
        if not records.is_relative_to((Path.cwd()/'.local').resolve()):
            raise AgentTaskError('Raw trial ledger must remain under ignored .local/')
        if not args.controls or not args.hardware_receipt:
            raise AgentTaskError('Live trials need explicit controls and hardware/revision receipt')
        controls = json.loads(args.controls.read_text(encoding='utf-8'))
        hardware = json.loads(args.hardware_receipt.read_text(encoding='utf-8'))
        result = run_live(args.task, OpenAITransport(args.url, args.model, args.max_tokens),
                          scratch, image=args.image, allow_execution=True)
        from bench.fitness import read_suite
        from bench.agent_records import make_trial
        suite = read_suite(Path(__file__).resolve().parent / 'fixtures/agent-suite.json')
        trial = make_trial(suite, result, controls, hardware, arm=args.arm, repeat=args.repeat)
        records.parent.mkdir(parents=True, exist_ok=True)
        with records.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(trial, sort_keys=True)+'\n')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        print(json.dumps({'task_id': result['task_id'], 'status': result['status'],
                          'elapsed_ms': result['elapsed_ms'], 'receipt': str(output)}))
    except (AgentTaskError, CollectorError, ValueError, KeyError, TypeError, OSError) as exc:
        ap.exit(2, 'AGENT ADAPTER REJECTED: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
