"""Verify sandbox construction only. NEVER launch Docker in these tests."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.code_judge import SandboxError, build_docker_command, digest, run_in_container

IMAGE = 'python:3.12-slim@sha256:' + '1' * 64
TESTS = 'import unittest\nfrom solution import add\nclass T(unittest.TestCase):\n def test_one(self): self.assertEqual(add(2,3),5)\n'


class SandboxConstructionTests(unittest.TestCase):
    def test_command_enforces_nonnetwork_nonprivileged_pinned_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'solution.py').write_text('def add(a,b): return a+b')
            (root/'tests').mkdir()
            (root/'tests/test_solution.py').write_text(TESTS)
            cmd = build_docker_command(root, IMAGE)
            for flag in ('--network=none', '--read-only', '--cap-drop=ALL',
                         '--pull=never', '--security-opt=no-new-privileges',
                         '--memory=512m', '--pids-limit=64'):
                self.assertIn(flag, cmd)
            self.assertEqual(cmd[cmd.index('--workdir=/app')+1], '--env=PYTHONDONTWRITEBYTECODE=1')
            with self.assertRaises(SandboxError):
                build_docker_command(root, 'python:latest')

    def test_simulated_docker_result_binds_test_and_solution_hash(self):
        source = 'def add(a,b): return a+b\n'
        fake_result = subprocess.CompletedProcess(args=['docker'], returncode=0,
                                                   stdout='', stderr='Ran 1 test in 0.001s\n\nOK')
        with tempfile.TemporaryDirectory() as parent:
            private = Path(parent)/'.local'/'bench'
            with patch('bench.code_judge.subprocess.run', return_value=fake_result) as runner:
                receipt = run_in_container(source, TESTS, image=IMAGE, private_root=private)
            self.assertTrue(receipt['passed'])
            self.assertEqual(receipt['tests_ran'], 1)
            self.assertEqual(receipt['solution_sha256'], digest(source))
            self.assertEqual(receipt['test_sha256'], digest(TESTS))
            self.assertEqual(runner.call_count, 1)

    def test_timeout_and_outside_private_tree(self):
        with tempfile.TemporaryDirectory() as parent:
            with self.assertRaisesRegex(SandboxError, 'ignored'):
                run_in_container('def add(a,b): return a+b', TESTS,
                                 image=IMAGE, private_root=Path(parent))
            with patch('bench.code_judge.subprocess.run', side_effect=subprocess.TimeoutExpired(['docker'],2)):
                receipt = run_in_container('def add(a,b): return a+b', TESTS,
                                           image=IMAGE, private_root=Path(parent)/'.local', timeout_seconds=2)
            self.assertFalse(receipt['passed'])
