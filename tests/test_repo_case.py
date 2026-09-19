"""Repository-task decisions must use evaluator receipts, not model statements."""
import json
import unittest
from pathlib import Path
from bench.fitness import canonical_digest, evaluate, read_suite
from bench.repo_case import judge_repo

ROOT = Path(__file__).resolve().parents[1]
SUITE = read_suite(ROOT / 'bench/fixtures/agent-suite.json')


def receipt(case):
    return {'task_id':case['task_id'],
            'protected_tests_sha256':case['expected_private_sha256'],
            'public_tests_sha256':case['expected_public_sha256'],
            'changed_files':case['writable_files'],
            'source_hashes':{name:'f'*64 for name in case['writable_files']},
            'public_tests_ran':2,'private_tests_ran':5,'passed':True,
            'verification_provenance':'isolated-container',
            'container_image':'python:3.12-slim@sha256:'+'a'*64}


class RepoCaseTests(unittest.TestCase):
    def test_two_tasks_only_pass_with_complete_independent_receipts(self):
        self.assertEqual(len(SUITE['cases']), 2)
        for case in SUITE['cases']:
            with self.subTest(task=case['id']):
                self.assertEqual(judge_repo(case, None)[0], 'unverified')
                self.assertEqual(judge_repo(case, receipt(case))[0], 'pass')
                bad = receipt(case)
                bad['protected_tests_sha256'] = '0'*64
                self.assertEqual(judge_repo(case, bad)[0], 'fail')
                bad = receipt(case)
                bad['private_tests_ran'] = 0
                self.assertEqual(judge_repo(case, bad)[0], 'fail')

    def test_simulated_or_forged_speed_cannot_promote(self):
        simulated=[]
        for case in SUITE['cases']:
            for arm in ('A','B'):
                simulated.append({'schema_version':1,'suite_sha256':canonical_digest(SUITE),
                    'case_id':case['id'],'arm':arm,'repeat':0,'provenance':'simulated',
                    'response':'I fixed everything, tests passed',
                    'repo_receipt':receipt(case) if arm=='A' else None,
                    'controls':{'model_revision':'r','engine_revision':'e','quant':'q',
                        'cache':'6/5','mtp_depth':4,'context_tokens':32768,
                        'sampler':'greedy','power_mode':'ac','template_sha256':'0'*64,
                        'output_limit':100},
                    'metrics':{'ttft_ms':1,'decode_ms':2,'e2e_ms':3,'output_tokens':2}})
        evaluation=evaluate(SUITE, simulated)
        from bench.fitness import compare
        result=compare(evaluation,SUITE,'A','B')
        self.assertFalse(result['eligible_for_production_decision'])
        self.assertFalse(result['quality_fully_verified'])
        self.assertEqual(sum(row['status']=='unverified' for row in evaluation['records']),2)


if __name__ == '__main__':
    unittest.main()
