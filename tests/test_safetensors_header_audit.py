"""CPU-only safetensors-header and quantization-metadata tests. No model weights."""
import copy
import unittest
from unittest.mock import patch
from tools.safetensors_header_audit import audit, inventory, ranged


def fixture():
    h = {'a.safetensors': {
        'x.trellis': {'dtype': 'I16', 'shape': [8, 16, 48], 'data_offsets': [0, 12288]},
        'x.suh': {'dtype': 'F16', 'shape': [128], 'data_offsets': [12288, 12544]},
        'x.svh': {'dtype': 'F16', 'shape': [256], 'data_offsets': [12544, 13056]},
        'x.mul1': {'dtype': 'I32', 'shape': [], 'data_offsets': [13056, 13060]},
        'mtp.norm.weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [13060, 13064]},
    }}
    idx = {'weight_map': {k: 'a.safetensors' for k in h['a.safetensors']},
           'metadata': {'total_size': 13064}}
    logical = {k: {'shape': v['shape'], 'n_bytes': v['data_offsets'][1]-v['data_offsets'][0],
                   'dtype': {'I16': 'torch.int16', 'I32': 'torch.int32',
                             'F16': 'torch.float16'}[v['dtype']]}
               for k, v in h['a.safetensors'].items() if k.startswith('x.')}
    q = {'tensor_storage': {'x': {'quant_format': 'exl3', 'bits_per_weight': 3,
                                  'stored_tensors': logical}}}
    return idx, q, h


class HeaderAuditTests(unittest.TestCase):
    def test_reconcile_all_bytes_and_mark_unlisted(self):
        idx, q, h = fixture()
        tensors, described = audit(idx, q, h, {'a.safetensors': 14000})
        self.assertEqual(len(tensors), 5)
        self.assertEqual(len(described), 4)
        rows = inventory(tensors, described)
        self.assertEqual(next(r for r in rows if r['tensor'] == 'x.trellis')['bits_per_weight'], 3)
        self.assertEqual(next(r for r in rows if r['tensor'] == 'x.trellis')['in_features'], 128)
        self.assertEqual(next(r for r in rows if r['tensor'] == 'mtp.norm.weight')['bits_per_weight'], '')

    def test_missing_shard_entry_is_rejected(self):
        idx, q, h = fixture()
        del h['a.safetensors']['mtp.norm.weight']
        with self.assertRaisesRegex(ValueError, 'Index/header tensor mismatch'):
            audit(idx, q, h, {'a.safetensors': 14000})

    def test_wrong_quantization_bytes_fail_closed(self):
        idx, q, h = fixture()
        q['tensor_storage']['x']['stored_tensors']['x.svh']['n_bytes'] -= 2
        with self.assertRaisesRegex(ValueError, 'Quantization metadata differs'):
            audit(idx, q, h, {'a.safetensors': 14000})

    def test_shard_or_shape_mismatch_fail_closed(self):
        idx, q, h = fixture()
        h['a.safetensors']['x.trellis']['shape'] = [8, 8, 47]
        with self.assertRaisesRegex(ValueError, 'Invalid tensor byte size'):
            audit(idx, q, h, {'a.safetensors': 14000})

    def test_inferred_trellis_is_explicitly_labeled(self):
        idx, q, h = fixture()
        q['tensor_storage'] = {}
        tensors, described = audit(idx, q, h, {'a.safetensors': 14000})
        row = next(r for r in inventory(tensors, described) if r['tensor'] == 'x.trellis')
        self.assertEqual(row['bits_per_weight'], 3)
        self.assertEqual(row['quant_format'], 'exl3_header_inferred')

    def test_unlisted_trellis_without_companion_tensors_is_rejected(self):
        tensors = {'x.trellis': ('a.safetensors',
                    {'dtype': 'I16', 'shape': [8, 16, 48], 'data_offsets': [0, 12288]}, 12288)}
        with self.assertRaisesRegex(ValueError, 'lacks'):
            inventory(tensors, {})

    def test_range_server_must_honor_partial_content(self):
        class Dummy:
            status = 200
            headers = {'Content-Length': '12000000000'}
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, *_): raise AssertionError('Must never read a full weight file')
        with patch('urllib.request.urlopen', return_value=Dummy()):
            with self.assertRaisesRegex(ValueError, 'abort.*weights'):
                ranged('https://example.invalid/model.safetensors', 0, 7)


if __name__ == '__main__':
    unittest.main()
