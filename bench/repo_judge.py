"""Final-stage-only hermetic repository task grader. Never executes generated code on host.

Docker with resource limits is defense in depth, NOT a secure sandbox for hostile
code. Run only on a disposable unprivileged machine after final-stage approval.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from bench.agent_replay import AgentTaskError, digest, inspect_workspace, load_task, require


class RepoJudgeError(ValueError):
    pass


def image_is_pinned(image: str) -> bool:
    return isinstance(image, str) and re.fullmatch(
        r'[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}', image) is not None


def stage_candidate(task_root: Path, candidate: Path, scratch: Path) -> dict:
    task, template = load_task(task_root)
    candidate = candidate.resolve()
    require(candidate.is_dir() and candidate != template.resolve(), 'Candidate must be a separate workspace')
    inspect_workspace(candidate)
    initial = {p.relative_to(template).as_posix(): digest(p) for p in template.rglob('*') if p.is_file()}
    submitted = {p.relative_to(candidate).as_posix(): digest(p) for p in candidate.rglob('*') if p.is_file()}
    require(initial.keys() == submitted.keys(), 'Candidate added or removed an unapproved file')
    changed = {name for name in initial if submitted[name] != initial[name]}
    require(bool(changed) and changed <= set(task['writable_files']),
            'Candidate touched protected tests, task instructions or unapproved files')
    for name in changed:
        require((candidate / name).stat().st_size <= task['max_source_bytes'],
                'Changed source exceeds bound')
    workspace = scratch / 'workspace'
    shutil.copytree(candidate, workspace)
    grader = scratch / 'grader'
    grader.mkdir()
    protected = task_root / task['private_test']
    shutil.copy2(protected, grader / 'test_private.py')
    return {'task_id': task['id'], 'source_hashes': {n: submitted[n] for n in sorted(changed)},
            'protected_tests_sha256': digest(protected),
            'public_tests_sha256': digest(template / task['visible_test']),
            'changed_files': sorted(changed)}


def build_command(scratch: Path, image: str) -> list[str]:
    if not image_is_pinned(image) or not (scratch/'workspace').is_dir() or not (scratch/'grader/test_private.py').is_file():
        raise RepoJudgeError('Pinned image and staged source/private tests required')
    root = scratch.resolve()
    # Public and private tests are evaluated in the same fresh container; no network.
    checks = ('python -B -m unittest discover -s tests -p test_public.py -q && '
              'python -B -m unittest discover -s ../grader -p test_private.py -q')
    return ['docker', 'run', '--rm', '--pull=never', '--network=none', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=64',
            '--memory=512m', '--cpus=2', '--user=65534:65534',
            '--tmpfs=/tmp:rw,nosuid,noexec,size=32m',
            '--mount', f'type=bind,source={root},destination=/bench,readonly',
            '--workdir=/bench/workspace', '--env=PYTHONDONTWRITEBYTECODE=1',
            image, 'sh', '-c', checks]


def grade(task_root: Path, candidate: Path, *, image: str,
          private_root: Path, authorized: bool = False, timeout_seconds: int = 45) -> dict:
    """Will not start a container or run candidate source without explicit authorization."""
    if not authorized:
        raise RepoJudgeError('Final-stage isolated execution is not authorized')
    if not image_is_pinned(image) or not 1 <= timeout_seconds <= 300:
        raise RepoJudgeError('Pinned locally cached container and bounded timeout required')
    private_root = private_root.resolve()
    if '.local' not in private_root.parts:
        raise RepoJudgeError('Scratch must be inside ignored .local')
    private_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root, prefix='graded-repo-') as temporary:
        receipt = stage_candidate(task_root, candidate, Path(temporary))
        command = build_command(Path(temporary), image)
        try:
            process = subprocess.run(command, capture_output=True, text=True,
                                     timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            return dict(receipt, passed=False, public_tests_ran=0, private_tests_ran=0,
                        reason='Isolated repository grader timed out', container_image=image,
                        verification_provenance='isolated-container')
        report = (process.stdout + '\n' + process.stderr)[-8000:]
        counts = [int(n) for n in re.findall(r'Ran (\d+) tests?', report)]
        public_count, private_count = (counts + [0, 0])[:2]
        return dict(receipt, passed=(process.returncode == 0 and public_count >= 2 and
                                     private_count >= 5 and len(counts) == 2),
                    public_tests_ran=public_count, private_tests_ran=private_count,
                    container_image=image, exit_code=process.returncode,
                    verification_provenance='isolated-container',
                    reason='Isolated public and evaluator-owned tests; no host code execution')
