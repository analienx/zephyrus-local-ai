"""Visible tests only; the agent cannot view the private grader cases."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usage import rollup


class PublicTests(unittest.TestCase):
    def test_basic_and_duplicate(self):
        data = [
            {"device": "panel", "event_id": "1", "when": "2026-01-01T10:12:00+00:00", "kwh": 1},
            {"device": "panel", "event_id": "1", "when": "2026-01-01T10:12:00+00:00", "kwh": 99},
            {"device": "panel", "event_id": "2", "when": "2026-01-01T10:40:00+00:00", "kwh": 2},
        ]
        self.assertEqual(rollup(data), [{"device": "panel", "hour_utc": "2026-01-01T10:00:00Z", "kwh": 3.0}])

    def test_empty(self):
        self.assertEqual(rollup([]), [])


if __name__ == "__main__":
    unittest.main()
