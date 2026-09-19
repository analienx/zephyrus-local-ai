"""No model imports, network requests, generated-code execution or GPU work."""
import unittest

from bench.fitness import (InvalidLedger, canonical_digest, compare, evaluate,
                           judge, matches, read_suite)

SUITE = {'schema_version': 1, 'cases': [
    {'id': 'exact', 'kind': 'exact', 'critical': True, 'prompt': 'Answer 42.', 'expected': '42'},
    {'id': 'tool', 'kind': 'tool_call', 'critical': True,
     'prompt': 'Invoke a tool with a JSON object.',
     'tool': {'name': 'file_search', 'arguments': {'filename': 'main.py'}}},
    {'id': 'code', 'kind': 'python_syntax', 'critical': False,
     'prompt': 'Write a pure function def add(a,b): ...', 'function': 'add'},
]}
CONTROL = dict(model_revision='rev', engine_revision='eng', quant='exl3',
               cache='6/5', mtp_depth=4, context_tokens=32768, sampler='greedy',
               power_mode='plugged', template_sha256='0'*64, output_limit=128)


def record(case, response, arm='A', repeat=0, provenance='simulated', controls=None):
    return {'schema_version': 1, 'suite_sha256': canonical_digest(SUITE),
            'case_id': case, 'arm': arm, 'repeat': repeat, 'provenance': provenance,
            'response': response, 'controls': controls or CONTROL,
            'metrics': {'ttft_ms': 2, 'decode_ms': 8, 'e2e_ms': 12,
                        'output_tokens': 2}}


class FitnessTests(unittest.TestCase):
    def test_subset_schema_has_exact_scalar_types(self):
        self.assertTrue(matches({'a': {'b': 1, 'c': 0}}, {'a': {'b': 1}}))
        self.assertFalse(matches({'a': {'b': True}}, {'a': {'b': 1}}))

    def test_exact_and_tool_judges(self):
        self.assertEqual(judge(SUITE['cases'][0], ' 42 ')[0], 'pass')
        self.assertEqual(judge(SUITE['cases'][1], '{"name":"file_search","arguments":{"filename":"main.py"}}')[0], 'pass')
        self.assertEqual(judge(SUITE['cases'][1], '{"name":"file_search","arguments":{"filename":"main.py","danger":true}}')[0], 'fail')

    def test_code_syntax_is_not_false_quality_pass(self):
        self.assertEqual(judge(SUITE['cases'][2], 'def add(a,b): return a-b')[0], 'unverified')
        self.assertEqual(judge(SUITE['cases'][2], 'def plus(a,b): return a+b')[0], 'fail')

    def test_simulated_speedup_cannot_promote_bad_quality(self):
        a = [record('exact', '42'), record('tool', '{"name":"file_search","arguments":{"filename":"main.py"}}'),
             record('code', 'def add(a,b): return a+b')]
        b = [record('exact', '41', 'B'), record('tool', '{"name":"file_search","arguments":{"filename":"main.py"}}','B'),
             record('code', 'def add(a,b): return a-b', 'B')]
        for x in b: x['metrics']['e2e_ms'] = 9
        result = compare(evaluate(SUITE, a+b), SUITE, 'A', 'B')
        self.assertFalse(result['candidate_critical_cases_pass'])
        self.assertFalse(result['eligible_for_production_decision'])
        self.assertFalse(result['quality_fully_verified'])
        self.assertTrue(all(pair['delta_e2e_ms'] < 0 for pair in result['pairs']))

    def test_uncontrolled_config_change_is_rejected(self):
        a, b = record('exact', '42'), record('exact', '42', 'B')
        b['controls'] = dict(CONTROL, context_tokens=65536)
        report = evaluate(SUITE, [a,b])
        with self.assertRaisesRegex(InvalidLedger, 'omitted suite cases'):
            compare(report, SUITE, 'A','B')

    def test_uncontrolled_difference_is_rejected_for_complete_pairs(self):
        a = [record('exact','42'), record('tool','{}'),record('code','def add(a,b): pass')]
        b = [record(t['case_id'],t['response'],'B',controls=dict(CONTROL, context_tokens=65536))
             for t in a]
        report = evaluate(SUITE, a+b)
        with self.assertRaisesRegex(InvalidLedger, 'Uncontrolled'):
            compare(report, SUITE, 'A','B')
        result = compare(report, SUITE, 'A','B', ('context_tokens',))
        self.assertEqual(result['changed_controls'], ['context_tokens'])

    def test_duplicate_record_mismatched_suite_and_fake_measurement_rejected(self):
        a = record('exact','42')
        with self.assertRaisesRegex(InvalidLedger, 'Repeated'):
            evaluate(SUITE, [a,a])
        wrong = dict(a, suite_sha256='bad')
        with self.assertRaisesRegex(InvalidLedger, 'digest'):
            evaluate(SUITE, [wrong])
        fake = dict(a, provenance='measured')
        with self.assertRaisesRegex(InvalidLedger, 'receipt'):
            evaluate(SUITE, [fake])

    def test_invalid_timing_and_unmatched_provenance_rejected(self):
        a, b = record('exact','42'), record('exact','42','B')
        bad = dict(a, metrics=dict(a['metrics'], e2e_ms=1))
        with self.assertRaisesRegex(InvalidLedger, 'Inconsistent'):
            evaluate(SUITE, [bad])
        b['provenance'] = 'measured'
        b['receipt'] = dict(engine_commit='r', checkpoint_revision='m',
                            hardware_sku='fake', run_id='fake')
        with self.assertRaisesRegex(InvalidLedger, 'omitted suite'):
            compare(evaluate(SUITE,[a,b]),SUITE,'A','B')

    def test_behavioral_receipt_is_bound_to_exact_code_and_tests(self):
        from bench.code_judge import digest
        source = 'def add(a,b): return a+b\n'
        testcase = dict(SUITE['cases'][2], test_source='import unittest\n')
        receipt = dict(passed=True, tests_ran=3, solution_sha256=digest(source),
                       test_sha256=digest(testcase['test_source']),
                       container_image='python:3.12-slim@sha256:'+'1'*64)
        self.assertEqual(judge(testcase, source, receipt)[0], 'pass')
        self.assertEqual(judge(testcase, 'def add(a,b): return a-b\n', receipt)[0], 'fail')
        self.assertEqual(judge(testcase, source, dict(receipt, tests_ran=0))[0], 'fail')
        self.assertEqual(judge(testcase, source, dict(receipt, passed=False))[0], 'fail')
