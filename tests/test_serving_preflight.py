"""CPU-only profile, fail-closed metadata and patch tests. No torch import."""
import copy
import json
import unittest
from pathlib import Path

from tests.test_static_memory_budget import CONFIG
from tools.build_quant_cache_guard_patch import (make_patch, IMPORT, ANCHOR_FLAG,
                                                 OLD_BRANCH, NEW_BRANCH)
from tools.serving_preflight import PreflightError, validate_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'profiles/qwen38-27b-exl3-3bpw-text-mtp.json').read_text())
FAKE_INVENTORY = {
    'serialized': {'text': 1000, 'output_head': 200, 'mtp': 150, 'vision': 300},
    'cpu_embedding': 100,
    'max_reconstructed_matrix': {'text': 1024, 'output_head': 2048, 'mtp': 512, 'vision': 512},
    'tensor_count': 3080,
}


class ProfilePreflightTests(unittest.TestCase):
    def validate(self, **modifications):
        p = copy.deepcopy(PROFILE)
        for dotted, value in modifications.items():
            cur = p
            keys = dotted.split('__')
            for key in keys[:-1]:
                cur = cur[key]
            cur[keys[-1]] = value
        return validate_profile(p, CONFIG, FAKE_INVENTORY)

    def test_static_candidate_reports_unproven_memory(self):
        row = self.validate()
        self.assertTrue(row['static_only'])
        self.assertEqual(row['reserved_tokens'], 32768)
        self.assertGreater(row['unbudgeted_mib'], 0)

    def test_fails_closed_on_dtype_and_varlen(self):
        for key, value in (('attention_contract__query_dtype', 'torch.float32'),
                           ('attention_contract__varlen', True),
                           ('attention_contract__head_dim', 512)):
            with self.subTest(key=key), self.assertRaises(PreflightError):
                self.validate(**{key: value})

    def test_fails_closed_on_expansion_modes(self):
        for key, value in (('runtime__qc_staging', 1),
                           ('runtime__qc_staging', 2),
                           ('cache__compand_a', 0.1),
                           ('runtime__strict_quant_cache', False),
                           ('cache__layer_type', 'CacheLayer_fp16')):
            with self.subTest(key=key), self.assertRaises(PreflightError):
                self.validate(**{key: value})

    def test_rejects_unsupported_geometry_and_components(self):
        for key, value in (('components__vision', True),
                           ('cache__slots', 2), ('cache__history', 3),
                           ('cache__k_bits', 8)):
            with self.subTest(key=key), self.assertRaises(PreflightError):
                self.validate(**{key: value})
        bad = copy.deepcopy(CONFIG)
        bad['text_config']['head_dim'] = 128
        with self.assertRaises(PreflightError):
            validate_profile(PROFILE, bad, FAKE_INVENTORY)

    def test_rejects_static_shortfall(self):
        with self.assertRaisesRegex(PreflightError, 'residual'):
            self.validate(**{'memory__static_unbudgeted_floor_mib': 16000})

    def test_patch_generation_is_exact_and_opt_in(self):
        original = IMPORT + ANCHOR_FLAG + OLD_BRANCH
        patch = make_patch(original)
        self.assertIn('+_qc_strict = os.environ.get(', patch)
        self.assertIn('+        if _qc_strict and isinstance(layer, CacheLayer_quant)', patch)
        self.assertIn('+            raise RuntimeError(', patch)
        self.assertIn('+        if quant_direct_eligible:', patch)
        self.assertIn('-        if (', patch)
        self.assertIn('+++ b/exllamav3/modules/attention_fn/dispatch.py', patch)

    def test_patch_generator_rejects_unsupported_source(self):
        for source in ('', IMPORT + ANCHOR_FLAG,
                       IMPORT + ANCHOR_FLAG + OLD_BRANCH + OLD_BRANCH):
            with self.subTest(source=source[:20]), self.assertRaises(ValueError):
                make_patch(source)
