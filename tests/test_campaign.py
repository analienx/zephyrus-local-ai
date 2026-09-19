"""Pure CPU scheduling tests; never launch a serving process."""
import unittest
from bench.campaign import build_campaign
from bench.fitness import InvalidLedger, canonical_digest
from tests.test_fitness import CONTROL, SUITE


class CampaignTests(unittest.TestCase):
    def test_balanced_abba_and_matching_repeats(self):
        plan = build_campaign(SUITE, CONTROL, CONTROL, repeats=5)
        self.assertEqual(len(plan['runs']), 20)
        self.assertEqual(plan['paired_repeats_per_arm'], 10)
        self.assertEqual([x['arm'] for x in plan['runs'][:4]], ['A','B','B','A'])
        for repeat in range(10):
            self.assertEqual([x['arm'] for x in plan['runs'] if x['repeat']==repeat].count('A'),1)
            self.assertEqual([x['arm'] for x in plan['runs'] if x['repeat']==repeat].count('B'),1)
        self.assertFalse(plan['execution_authorized'])
        self.assertTrue(all(x['suite_sha256']==canonical_digest(SUITE) for x in plan['runs']))

    def test_only_declared_variable_can_change(self):
        candidate = dict(CONTROL, quant='exl3-3.5bpw')
        with self.assertRaisesRegex(ValueError, 'Uncontrolled'):
            build_campaign(SUITE, CONTROL, candidate)
        planned = build_campaign(SUITE, CONTROL, candidate, changed_control=('quant',))
        self.assertEqual(planned['changed_control'],['quant'])

    def test_invalid_repeat_count_rejected(self):
        for blocks in (0, 101, True):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                build_campaign(SUITE, CONTROL, CONTROL, repeats=blocks)
