"""No model, GPU, or execution of agent-supplied source in these tests."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from bench.agent_replay import AgentTaskError, load_task, replay, tool_step

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'bench/agent_tasks/energy_rollup_v1'
PRIVATE = ROOT / '.local/benchmark/agent-unit-tests'


def transcript():
    text = (TASK / 'workspace/src/usage.py').read_text(encoding='utf-8')
    changed = text.replace('if ident in seen:', 'if (event["device"], ident) in seen:')
    return [{'tool':'list_files', 'arguments':{}},
            {'tool':'read_file', 'arguments':{'path':'README.md'}},
            {'tool':'read_file', 'arguments':{'path':'src/usage.py'}},
            {'tool':'write_file', 'arguments':{'path':'src/usage.py', 'content':changed}},
            {'tool':'run_tests', 'arguments':{}},
            {'tool':'finish', 'arguments':{'summary':'Edited source and requested tests'}}]


class AgentReplayTests(unittest.TestCase):
    def test_fixture_baseline_passes_visible_but_fails_protected(self):
        root = TASK / 'workspace'
        public = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'],
                                cwd=root, capture_output=True, text=True, timeout=10)
        private = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', '../grader', '-q'],
                                 cwd=root, capture_output=True, text=True, timeout=10)
        self.assertEqual(public.returncode, 0)
        self.assertNotEqual(private.returncode, 0)
        self.assertIn('FAILED', private.stderr)

    def test_replay_is_unverified_and_never_exposes_grader(self):
        result = replay(TASK, transcript(), PRIVATE)
        self.assertEqual(result['status'], 'unverified')
        self.assertFalse(result['private_tests_exposed'])
        self.assertFalse(result['generated_code_executed'])
        self.assertFalse(result['model_or_gpu_contacted'])
        self.assertEqual(result['tool_calls'], 6)

    def test_path_escape_and_protected_write_rejected(self):
        for path in ('../grader/test_private.py', '/tmp/a', 'src\\usage.py',
                     'tests/test_public.py', 'README.md'):
            with self.subTest(path=path):
                steps = transcript()
                steps[3]['arguments']['path'] = path
                with self.assertRaises(AgentTaskError):
                    replay(TASK, steps, PRIVATE)

    def test_cannot_read_private_grader(self):
        steps = transcript()
        steps[1]['arguments']['path'] = '../grader/test_private.py'
        with self.assertRaises(AgentTaskError):
            replay(TASK, steps, PRIVATE)

    def test_no_fake_tests_or_extra_steps(self):
        steps = transcript()
        steps[4]['arguments'] = {'success': True}
        with self.assertRaises(AgentTaskError):
            replay(TASK, steps, PRIVATE)
        steps = transcript() + [{'tool':'read_file', 'arguments':{'path':'README.md'}}]
        with self.assertRaises(AgentTaskError):
            replay(TASK, steps, PRIVATE)

    def test_inspection_and_real_edit_required(self):
        for steps in ([s for s in transcript() if s['tool'] != 'read_file'],
                      [s for s in transcript() if s['tool'] != 'write_file']):
            with self.assertRaises(AgentTaskError):
                replay(TASK, steps, PRIVATE)


if __name__ == '__main__':
    unittest.main()
