"""Evaluator tests: inaccessible to the agent's file tools."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workspace' / 'src'))
from config import merge_layers


class PrivateTests(unittest.TestCase):
    def test_explicit_falsy_values_win(self):
        self.assertEqual(merge_layers({'n':10,'flag':True,'name':'default'},
                                      {}, {'n':0,'flag':False,'name':''}),
                         {'n':0,'flag':False,'name':''})

    def test_recursive_merge_at_each_layer(self):
        defaults = {'limits':{'cpu':4,'ram':8},'mode':'normal'}
        self.assertEqual(merge_layers(defaults, {'limits':{'cpu':2}},
                                      {'limits':{'ram':16}}),
                         {'limits':{'cpu':2,'ram':16},'mode':'normal'})

    def test_unknown_top_and_nested_keys_rejected(self):
        for override in ({'surprise':1}, {'limits':{'gpu':2}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                merge_layers({'limits':{'cpu':4}}, override, {})

    def test_input_immutability_deep_copy(self):
        defaults = {'limits':{'cpu':4},'tags':['a']}
        files = {'limits':{'cpu':2},'tags':['b']}
        merged = merge_layers(defaults, files, {})
        merged['limits']['cpu'] = 99
        merged['tags'].append('x')
        self.assertEqual(defaults, {'limits':{'cpu':4},'tags':['a']})
        self.assertEqual(files, {'limits':{'cpu':2},'tags':['b']})

    def test_none_skips_nested_override(self):
        self.assertEqual(merge_layers({'limits':{'cpu':4, 'ram':8},'enabled':True},
                                      {'limits':{'cpu':None,'ram':2}},
                                      {'limits':{'ram':None},'enabled':None}),
                         {'limits':{'cpu':4,'ram':2},'enabled':True})


if __name__ == '__main__':
    unittest.main()
