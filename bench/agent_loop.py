"""Transport-injected multi-turn agent runtime; offline scripted mode ONLY for now.

The model receives only workspace files and bounded tool replies. Private tests
remain outside the workspace and are never passed into the conversation.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from bench.agent_replay import AgentTaskError, digest, inspect_workspace, load_task, require, tool_step

TOOLS = ('list_files', 'read_file', 'write_file', 'run_tests', 'finish')


class ScriptedTransport:
    """CPU-only protocol rehearsal. These tool calls are test data, NOT inference."""
    def __init__(self, calls: list[dict]):
        self.calls = calls
        self.turn = 0
        self.seen_messages = []

    def complete(self, messages: list[dict]) -> dict:
        self.seen_messages.append(json.loads(json.dumps(messages)))
        if self.turn >= len(self.calls):
            raise AgentTaskError('Scripted model ran out of calls without finishing')
        call = self.calls[self.turn]
        self.turn += 1
        return {'tool_calls': [call], 'content': ''}


def run_session(task_root: Path, transport, private_root: Path, *,
                visible_runner=None, final_grader=None) -> dict:
    """No network, model import, subprocess or execution of generated Python."""
    started = time.monotonic()
    task, template = load_task(task_root.resolve())
    inspect_workspace(template)
    private_root = private_root.resolve()
    require('.local' in private_root.parts, 'Agent workspace must be in ignored .local/')
    private_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root, prefix='agent-session-') as temporary:
        working = Path(temporary) / 'workspace'
        shutil.copytree(template, working)
        instructions = (working / task['prompt_file']).read_text(encoding='utf-8')
        messages = [{'role':'system', 'content':'Use only the five bounded local tools. '
                     'A replay test response is not proof of correctness. Never access private grading files.'},
                    {'role':'user', 'content':instructions}]
        inspected, writes, tested, tool_names, result_hashes = set(), set(), False, [], []
        ended = False
        for turn in range(task['max_tool_calls']):
            reply = transport.complete(messages)
            require(isinstance(reply, dict) and isinstance(reply.get('tool_calls'), list) and
                    len(reply['tool_calls']) == 1, 'One unambiguous tool call required per turn')
            call = reply['tool_calls'][0]
            require(isinstance(call, dict) and set(call) == {'tool','arguments'},
                    'Unexpected model tool-call shape')
            require(call['tool'] in TOOLS, 'Unsupported agent tool')
            result = tool_step(task, working, inspected, call)
            if call['tool'] == 'run_tests' and visible_runner is not None:
                result = visible_runner(task_root, working)
                require(isinstance(result, dict) and 'private' not in str(result).lower(),
                        'Visible-test tool must not reveal private grading content')
            tool_names.append(call['tool'])
            if call['tool'] == 'write_file':
                writes.add(call['arguments']['path'])
            if call['tool'] == 'run_tests':
                tested = True
            result_hashes.append(hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest())
            messages.append({'role':'assistant', 'content':'', 'tool_calls':[call]})
            messages.append({'role':'tool', 'name':call['tool'], 'content':json.dumps(result, sort_keys=True)})
            if call['tool'] == 'finish':
                ended = True
                break
        require(ended and writes and (tested or not task['required_test_call']),
                'Task was not completed inside the tool-call budget')
        changed = [name for name in task['writable_files']
                   if digest(working / name) != digest(template / name)]
        require(bool(changed), 'Session did not change allowlisted source')
        final_receipt = final_grader(task_root, working) if final_grader is not None else None
        return {'task_id':task['id'], 'status':'unverified', 'turns':len(tool_names),
                'elapsed_ms':round((time.monotonic()-started)*1000,3),
                'provenance':'simulated' if isinstance(transport, ScriptedTransport) else 'unverified',
                'tool_names':tool_names, 'source_hashes':{n:digest(working/n) for n in changed},
                'private_tests_exposed':False,
                'generated_code_executed':False if visible_runner is None and final_grader is None else None,
                'model_or_gpu_contacted':False if isinstance(transport, ScriptedTransport) else None,
                'repo_receipt':final_receipt,
                'reason':'Scripted transport validates end-to-end multi-turn tool I/O only.'}
