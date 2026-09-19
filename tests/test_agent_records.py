"""Offline conversion tests; fake receipts never represent model or GPU measurements."""
import copy
import unittest
from pathlib import Path
from bench.agent_records import AgentRecordError, make_trial
from bench.fitness import InvalidLedger, compare, evaluate, read_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = read_suite(ROOT / 'bench/fixtures/agent-suite.json')
CONTROLS = {'model_revision':'model', 'engine_revision':'engine', 'quant':'exl3',
 'cache':'6/5', 'mtp_depth':4, 'context_tokens':32768, 'sampler':'greedy',
 'power_mode':'ac', 'template_sha256':'a'*64, 'output_limit':1024}
HW = {'engine_commit':'engine','checkpoint_revision':'model',
      'hardware_sku':'SYNTHETIC-DO-NOT-REPORT','run_id':'simulated-run'}
RESULT = {'task_id':'energy-rollup-v1', 'elapsed_ms':140.0,
          'model_completion_tokens':20, 'model_prompt_tokens':100,
          'model_turns':6, 'repo_receipt':None}


class AgentRecordTests(unittest.TestCase):
    def trial(self, result=None, arm='A'):
        return make_trial(SUITE, result or RESULT, CONTROLS, HW,
                          arm=arm, repeat=0, provenance='simulated')

    def test_unknown_ttft_is_not_fabricated(self):
        trial = self.trial()
        self.assertIsNone(trial['metrics']['ttft_ms'])
        self.assertIsNone(trial['metrics']['decode_ms'])
        self.assertEqual(trial['metrics']['e2e_ms'], 140)
        output = evaluate(SUITE, [trial])
        self.assertEqual(output['records'][0]['status'], 'unverified')

    def test_agent_failure_is_scored_even_when_fast(self):
        first = self.trial()
        second = self.trial(dict(RESULT, elapsed_ms=50,
                                 agent_failed=True), arm='B')
        # Use the one-case subset to satisfy identical suite-case matching.
        subset = dict(SUITE, cases=[SUITE['cases'][0]])
        from bench.fitness import canonical_digest
        for record in (first, second):
            record['suite_sha256'] = canonical_digest(subset)
        comparison = compare(evaluate(subset, [first, second]), subset, 'A', 'B')
        self.assertEqual(comparison['arm_summaries']['B']['quality_fail'], 1)
        self.assertIsNone(comparison['arm_summaries']['B']['median_ttft_ms'])
        self.assertIsNone(comparison['arm_summaries']['B']['client_observed_decode_tokens_per_second'])
        self.assertFalse(comparison['quality_fully_verified'])
