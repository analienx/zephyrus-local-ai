"""Offline deterministic coding-agent tool-session rehearsal; NEVER runs generated code.

Live inference/agent execution is deliberately absent. The task workspace is
copied into ignored .local; grader files are never put in the agent filesystem.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath


class AgentTaskError(ValueError):
    pass


def require(ok: bool, message: str):
    if not ok:
        raise AgentTaskError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_name(name: str) -> str:
    require(isinstance(name, str) and len(name) <= 160 and '\\' not in name and '\x00' not in name,
            'Invalid relative POSIX path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.parts and all(x not in ('', '.', '..') for x in p.parts)
            and ':' not in name and name == str(p), 'Unsafe or noncanonical task path')
    return name


def load_task(root: Path) -> tuple[dict, Path]:
    task = json.loads((root / 'task.json').read_text(encoding='utf-8'))
    require(task.get('schema_version') == 1 and task.get('root') == 'workspace',
            'Unsupported task or workspace layout')
    workspace = root / 'workspace'
    require(workspace.is_dir() and not workspace.is_symlink(), 'Missing safe task workspace')
    require(task.get('max_tool_calls', 0) in range(1, 101), 'Tool budget not bounded')
    require(task.get('max_source_bytes', 0) in range(1, 262145), 'Source limit not bounded')
    require(isinstance(task.get('writable_files'), list) and task['writable_files'],
            'Missing allowlisted writable files')
    for name in task['writable_files'] + task['required_inspections'] + [task['prompt_file'], task['visible_test']]:
        safe_name(name)
        require((workspace / name).is_file() and not (workspace / name).is_symlink(),
                'Missing, linked, or unavailable workspace file')
    require((root / task['private_test']).is_file() and
            not (root / task['private_test']).is_symlink(), 'Missing private verifier')
    return task, workspace


def inspect_workspace(workspace: Path):
    for item in workspace.rglob('*'):
        require(not item.is_symlink(), 'Symlink in disposable agent fixture')
        require(item.is_dir() or item.is_file(), 'Unexpected workspace object')


def tool_step(task: dict, working: Path, seen: set, step: dict) -> dict:
    require(isinstance(step, dict) and isinstance(step.get('tool'), str), 'Invalid tool step')
    tool = step['tool']
    args = step.get('arguments', {})
    require(isinstance(args, dict), 'Tool arguments must be an object')
    if tool == 'list_files':
        require(not args, 'list_files takes no arguments')
        return {'files': sorted(p.relative_to(working).as_posix() for p in working.rglob('*') if p.is_file())}
    if tool == 'read_file':
        require(set(args) == {'path'}, 'read_file requires path only')
        name = safe_name(args['path'])
        p = working / name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(working.resolve()),
                'Read outside exposed task workspace is forbidden')
        seen.add(name)
        text = p.read_text(encoding='utf-8')
        require(len(text.encode('utf-8')) <= task['max_source_bytes'], 'Read exceeds task bound')
        return {'path': name, 'content': text}
    if tool == 'write_file':
        require(set(args) == {'path', 'content'}, 'write_file requires path and content only')
        name = safe_name(args['path'])
        require(name in task['writable_files'], 'Agent attempted to alter protected or unapproved file')
        require(all(p in seen for p in task['required_inspections']),
                'Agent must inspect task instructions and source before editing')
        source = args['content']
        require(isinstance(source, str) and 0 < len(source.encode('utf-8')) <= task['max_source_bytes'],
                'Invalid replacement source size')
        p = working / name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(working.resolve()),
                'Write outside exposed task workspace is forbidden')
        p.write_text(source, encoding='utf-8')  # never execute this untrusted source on host
        return {'path': name, 'sha256': digest(p), 'bytes': len(source.encode('utf-8'))}
    if tool == 'run_tests':
        require(not args, 'run_tests takes no arguments')
        return {'status': 'unverified', 'reason': 'Replay is CPU-only file-operation rehearsal; '
                'generated code requires separately authorized isolated verifier'}
    if tool == 'finish':
        require(set(args) == {'summary'} and isinstance(args['summary'], str),
                'finish requires a summary string')
        return {'status': 'finished', 'summary_sha256': hashlib.sha256(args['summary'].encode()).hexdigest()}
    raise AgentTaskError('Unknown agent tool: ' + tool)


def replay(task_root: Path, transcript: list[dict], private_root: Path) -> dict:
    """Only file I/O and a synthetic test-tool response; no subprocess/model/network."""
    task, template = load_task(task_root.resolve())
    require(isinstance(transcript, list) and 1 <= len(transcript) <= task['max_tool_calls'],
            'Tool-call budget exceeded or transcript missing')
    private_root = private_root.resolve()
    require('.local' in private_root.parts, 'Disposable workspace must be under ignored .local/')
    inspect_workspace(template)
    private_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root, prefix='agent-') as temporary:
        working = Path(temporary) / 'workspace'
        shutil.copytree(template, working, symlinks=False)
        seen, trace, writes, tested, ended = set(), [], set(), False, False
        for index, step in enumerate(transcript):
            require(not ended, 'Tool call after finish')
            result = tool_step(task, working, seen, step)
            if step['tool'] == 'write_file':
                writes.add(step['arguments']['path'])
            if step['tool'] == 'run_tests':
                tested = True
            if step['tool'] == 'finish':
                ended = True
                require(index == len(transcript) - 1, 'finish must be the last action')
            # Only a hash, not model-visible content, is emitted in receipts.
            trace.append({'index': index, 'tool': step['tool'],
                          'result_sha256': hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()})
        require(ended and writes and (tested or not task['required_test_call']),
                'Agent did not edit, invoke tests or finish task')
        changed = sorted(name for name in task['writable_files']
                         if digest(working / name) != digest(template / name))
        require(changed, 'Task produced no source change')
        return {'task_id': task['id'], 'status': 'unverified',
                'source_file_hashes': {name: digest(working / name) for name in changed},
                'tool_calls': len(trace), 'tool_trace': trace,
                'private_tests_exposed': False, 'generated_code_executed': False,
                'model_or_gpu_contacted': False,
                'reason': 'Offline replay checks workspace isolation and tool flow; '
                          'behavioral correctness needs an isolated live-stage grader.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', type=Path, required=True)
    ap.add_argument('--transcript', type=Path, required=True)
    ap.add_argument('--private-root', type=Path, default=Path('.local/benchmark/agent'))
    args = ap.parse_args()
    try:
        steps = json.loads(args.transcript.read_text(encoding='utf-8'))
        result = replay(args.task, steps, args.private_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as error:
        ap.exit(2, 'AGENT REPLAY REJECTED: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
