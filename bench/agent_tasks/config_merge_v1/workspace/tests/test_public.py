"""Visible minimal sanity tests."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import merge_layers


class PublicTests(unittest.TestCase):
    def test_basic_precedence(self):
        self.assertEqual(merge_layers({'timeout':30, 'trace':True},
                                      {'timeout':10}, {'timeout':5}),
                         {'timeout':5, 'trace':True})

    def test_no_overrides(self):
        self.assertEqual(merge_layers({'a':1}, None, {}), {'a':1})


if __name__ == '__main__':
    unittest.main()
