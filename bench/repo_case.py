"""Grade an agent's repository task from evaluator-owned receipts, not its prose.

Receipts MUST be attached by a trusted local collector, never accepted as
model-supplied tool arguments. A passing response alone proves nothing.
"""
from __future__ import annotations
import re


def judge_repo(case: dict, receipt: dict | None) -> tuple[str, str]:
    if receipt is None:
        return 'unverified', 'Missing isolated repository test receipt'
    if not isinstance(receipt, dict):
        return 'fail', 'Malformed repository receipt'
    if receipt.get('task_id') != case['task_id']:
        return 'fail', 'Task identity mismatch'
    if receipt.get('protected_tests_sha256') != case['expected_private_sha256']:
        return 'fail', 'Evaluator-owned tests were changed or substituted'
    if receipt.get('public_tests_sha256') != case['expected_public_sha256']:
        return 'fail', 'Public tests were changed or substituted'
    if not isinstance(receipt.get('changed_files'), list) or not receipt['changed_files']:
        return 'fail', 'No source change in isolated workspace'
    if not set(receipt['changed_files']) <= set(case['writable_files']):
        return 'fail', 'Unauthorized source or tests changed'
    hashes = receipt.get('source_hashes')
    if not isinstance(hashes, dict) or set(hashes) != set(receipt['changed_files']):
        return 'fail', 'Changed file digests missing'
    if not all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in hashes.values()):
        return 'fail', 'Source hash format invalid'
    if receipt.get('verification_provenance') != 'isolated-container':
        return 'unverified', 'Only trusted isolated execution can verify generated code'
    if not isinstance(receipt.get('container_image'), str) or not re.fullmatch(
            r'[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}', receipt['container_image']):
        return 'fail', 'Container image must be pinned by digest'
    if any(type(receipt.get(name)) is not int or receipt[name] < bound
           for name, bound in (('public_tests_ran', case['min_public_tests']),
                               ('private_tests_ran', case['min_private_tests']))):
        return 'fail', 'Public or private test execution missing'
    if receipt.get('passed') is not True:
        return 'fail', 'Repository acceptance tests failed'
    return 'pass', 'Both visible and evaluator-owned tests passed inside pinned container'
