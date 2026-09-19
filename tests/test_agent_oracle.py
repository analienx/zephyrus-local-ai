"""Fixture calibration: only repository-authored trusted reference runs on host."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from tests.test_agent_replay import TASK


class FixtureCalibrationTests(unittest.TestCase):
    def test_reference_passes_visible_and_evaluator_owned_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            shutil.copytree(TASK / 'workspace', sandbox / 'workspace')
            shutil.copytree(TASK / 'grader', sandbox / 'grader')
            trusted = (TASK / 'grader/reference_usage.py').read_text(encoding='utf-8')
            (sandbox / 'workspace/src/usage.py').write_text(trusted, encoding='utf-8')
            for suite, expected_count in [('tests', 2), ('../grader', 5)]:
                process = subprocess.run(
                    [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', suite, '-q'],
                    cwd=sandbox / 'workspace', capture_output=True, text=True, timeout=10)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn(f'Ran {expected_count} tests', process.stderr)


if __name__ == '__main__':
    unittest.main()
