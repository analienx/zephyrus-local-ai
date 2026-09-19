"""End-to-end scripted multi-turn tool loop; no model/API/code execution."""
import unittest
from pathlib import Path
from bench.agent_loop import ScriptedTransport, run_session
from bench.agent_replay import AgentTaskError
from tests.test_agent_replay import TASK, PRIVATE, transcript


class ScriptedAgentSessionTests(unittest.TestCase):
    def test_tools_really_exchange_outputs_without_exposing_grader(self):
        scripted = ScriptedTransport(transcript())
        result = run_session(TASK, scripted, PRIVATE)
        self.assertEqual(result['status'], 'unverified')
        self.assertEqual(result['turns'], 6)
        self.assertEqual(len(scripted.seen_messages), 6)
        self.assertIn('device-energy rollup', scripted.seen_messages[0][1]['content'])
        for turn in scripted.seen_messages:
            self.assertNotIn('test_private', str(turn))
        self.assertFalse(result['generated_code_executed'])

    def test_model_must_not_claim_fake_test_feedback_is_success(self):
        scripted = ScriptedTransport(transcript())
        run_session(TASK, scripted, PRIVATE)
        last = scripted.seen_messages[-1]
        test_results = [m for m in last if m.get('role') == 'tool' and m.get('name') == 'run_tests']
        self.assertEqual(len(test_results), 1)
        self.assertIn('unverified', test_results[0]['content'])

    def test_unfinished_or_unauthorized_agent_is_rejected(self):
        with self.assertRaises(AgentTaskError):
            run_session(TASK, ScriptedTransport(transcript()[:-1]), PRIVATE)
        tamper = transcript()
        tamper[3]['arguments']['path'] = 'tests/test_public.py'
        with self.assertRaises(AgentTaskError):
            run_session(TASK, ScriptedTransport(tamper), PRIVATE)


if __name__ == '__main__':
    unittest.main()
