"""Evaluator-owned tests. Do not expose through agent file-read or tool results."""
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workspace' / 'src'))
from usage import rollup


def event(device, ident, when, kwh):
    return {'device': device, 'event_id': ident, 'when': when, 'kwh': kwh}


class PrivateTests(unittest.TestCase):
    def test_cross_device_duplicate_id(self):
        rows = rollup([event('A','x','2026-01-01T12:01:00Z',1),
                       event('B','x','2026-01-01T12:05:00Z',2)])
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['device'] for r in rows], ['A','B'])

    def test_midnight_utc_rollover(self):
        rows = rollup([event('p','m','2026-01-01T00:30:00+02:00',0.25),
                       event('p','n','2025-12-31T22:55:00Z',0.5)])
        self.assertEqual(rows, [{'device':'p','hour_utc':'2025-12-31T22:00:00Z','kwh':0.75}])

    def test_reject_invalid_values(self):
        for value in (-1, float('nan'), float('inf'), True, '5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rollup([event('p','n','2026-01-01T00:00:00Z', value)])

    def test_bad_identity_and_timezone(self):
        for e in [event('', 'a', '2026-01-01T01:00:00Z', 1),
                  event('p', '', '2026-01-01T01:00:00Z', 1),
                  event('p', 'a', '2026-01-01T01:00:00', 1),
                  event('p', 'a', 'nonsense', 1),
                  {'device':'p', 'event_id':'a', 'when':'2026-01-01T01:00:00Z'}]:
            with self.subTest(event=e), self.assertRaises(ValueError):
                rollup([e])

    def test_invalid_duplicate_does_not_hide_error(self):
        with self.assertRaises(ValueError):
            rollup([event('p','a','2026-01-01T00:00:00Z',1),
                    event('p','a','2026-01-01T00:00:00Z',-1)])


if __name__ == '__main__':
    unittest.main()
