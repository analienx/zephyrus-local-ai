"""Static model-residency planning only. Never initializes torch or CUDA."""
import csv
import tempfile
import unittest
from pathlib import Path

from tools.model_residency_plan import plan, read_manifest
from tests.test_static_memory_budget import CONFIG


COMPONENT_BYTES = {'text': 1000, 'output_head': 200, 'mtp': 150, 'vision': 300}


class ModelPlanTests(unittest.TestCase):
    def test_component_switches_are_exact(self):
        base = plan(CONFIG, COMPONENT_BYTES, context=32768)
        vision = plan(CONFIG, COMPONENT_BYTES, context=32768, vision=True)
        no_draft = plan(CONFIG, COMPONENT_BYTES, context=32768, with_mtp=False)
        self.assertEqual(base['serialized_weight_bytes'], 1350)
        self.assertEqual(vision['serialized_weight_bytes']-base['serialized_weight_bytes'], 300)
        self.assertEqual(base['serialized_weight_bytes']-no_draft['serialized_weight_bytes'], 150)
        self.assertEqual(base['known_cache_and_recurrent_bytes'] -
                         no_draft['known_cache_and_recurrent_bytes'], 32768 * 1536)

    def test_manifest_rejects_duplicate_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tensors.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['tensor', 'component', 'bytes'])
                writer.writeheader()
                writer.writerow({'tensor': 'x', 'component': 'text', 'bytes': 12})
                writer.writerow({'tensor': 'x', 'component': 'mtp', 'bytes': 14})
            with self.assertRaisesRegex(ValueError, 'duplicated'):
                read_manifest(path)
