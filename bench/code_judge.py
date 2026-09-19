"""OPTIONAL FINAL-STAGE code grader: never execute generated code on the host.

Docker provides resource and filesystem restrictions, not a hardened security
boundary for hostile code. Run only in a disposable, non-privileged machine.
No container is started unless the caller explicitly invokes run_in_container.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path


class SandboxError(ValueError):
    pass


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_docker_command(workspace: Path, image: str) -> list[str]:
    if not re.fullmatch(r'[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}', image):
        raise SandboxError('Require an explicitly pinned local container image digest')
    root = workspace.resolve()
    if not root.is_dir() or not (root / 'solution.py').is_file() or not (root / 'tests/test_solution.py').is_file():
        raise SandboxError('Sandbox workspace or fixtures missing')
    return ['docker', 'run', '--rm', '--pull=never', '--network=none',
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--pids-limit=64', '--memory=512m', '--cpus=2', '--user=65534:65534',
            '--tmpfs=/tmp:rw,nosuid,noexec,size=32m',
            '--mount', f'type=bind,source={root},destination=/app,readonly',
            '--workdir=/app', '--env=PYTHONDONTWRITEBYTECODE=1',
            image, 'python', '-B', '-m', 'unittest', 'discover',
            '-s', 'tests', '-p', 'test_solution.py', '-v']


def run_in_container(source: str, test_source: str, *, image: str,
                     private_root: Path, timeout_seconds: int = 30) -> dict:
    """Runs only inside an explicit pinned-image Docker container. NEVER host exec."""
    if not source or not test_source or timeout_seconds < 1 or timeout_seconds > 300:
        raise SandboxError('Invalid source, tests or test timeout')
    private_root = private_root.resolve()
    if '.local' not in private_root.parts:
        raise SandboxError('Sandbox workspaces must reside under the ignored .local directory')
    private_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private_root, prefix='graded-') as temp:
        workspace = Path(temp)
        (workspace / 'tests').mkdir()
        (workspace / 'solution.py').write_text(source, encoding='utf-8')
        (workspace / 'tests/test_solution.py').write_text(test_source, encoding='utf-8')
        cmd = build_docker_command(workspace, image)
        try:
            process = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            return {'passed': False, 'tests_ran': 0, 'reason': 'Container execution timed out',
                    'solution_sha256': digest(source), 'test_sha256': digest(test_source),
                    'container_image': image}
        text = (process.stdout + '\n' + process.stderr)[-4000:]
        m = re.search(r'Ran (\d+) tests?', text)
        ran = int(m.group(1)) if m else 0
        return {'passed': process.returncode == 0 and ran > 0,
                'tests_ran': ran, 'reason': 'Isolated unittest result',
                'solution_sha256': digest(source), 'test_sha256': digest(test_source),
                'container_image': image, 'exit_code': process.returncode}
