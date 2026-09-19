"""Tests simulate Docker output; no real Docker or generated-source execution."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from bench.repo_judge import RepoJudgeError, build_command, grade, stage_candidate
from bench.agent_replay import AgentTaskError

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'bench/agent_tasks/energy_rollup_v1'
PRIVATE = ROOT / '.local/benchmark/repo-judge-unit'
IMAGE = 'python:3.12-slim@sha256:' + 'a' * 64


class RepoJudgeTests(unittest.TestCase):
    def candidate(self, base):
        dest = base / 'candidate'
        shutil.copytree(TASK / 'workspace', dest)
        source = dest / 'src/usage.py'
        source.write_text(source.read_text(encoding='utf-8') + '\n# controlled edit\n', encoding='utf-8')
        return dest

    def test_staging_separates_hidden_tests_and_restricts_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self.candidate(root)
            receipt = stage_candidate(TASK, candidate, root / 'staged')
            self.assertEqual(receipt['changed_files'], ['src/usage.py'])
            self.assertTrue((root / 'staged/grader/test_private.py').is_file())
            self.assertFalse((candidate / 'grader').exists())
            command = build_command(root / 'staged', IMAGE)
            self.assertIn('--network=none', command)
            self.assertIn('--read-only', command)
            self.assertIn('--pull=never', command)

    def test_rejects_protected_modification_and_extra_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self.candidate(root)
            target = candidate / 'tests/test_public.py'
            target.write_text(target.read_text(encoding='utf-8') + '\n# tampered\n', encoding='utf-8')
            with self.assertRaises(AgentTaskError):
                stage_candidate(TASK, candidate, root / 'staged')
            target.write_bytes((TASK / 'workspace/tests/test_public.py').read_bytes())
            (candidate / 'grader').mkdir()
            (candidate / 'grader/test_private.py').write_text('pass', encoding='utf-8')
            with self.assertRaises(AgentTaskError):
                stage_candidate(TASK, candidate, root / 'staged')

    def test_container_is_explicitly_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self.candidate(Path(tmp))
            with self.assertRaises(RepoJudgeError):
                grade(TASK, candidate, image=IMAGE, private_root=PRIVATE)

    def test_mocked_pass_requires_both_test_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self.candidate(Path(tmp))
            simulated = subprocess.CompletedProcess(args=['docker'], returncode=0,
                stdout='', stderr='Ran 2 tests in 0.01s\nOK\nRan 5 tests in 0.01s\nOK\n')
            with patch('bench.repo_judge.subprocess.run', return_value=simulated) as called:
                receipt = grade(TASK, candidate, image=IMAGE, private_root=PRIVATE, authorized=True)
            self.assertTrue(receipt['passed'])
            self.assertEqual(receipt['private_tests_ran'], 5)
            self.assertEqual(called.call_count, 1)
            self.assertEqual(called.call_args.args[0][0], 'docker')
