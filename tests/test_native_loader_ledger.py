"""CPU-only loader budget checks; no model, torch, or GPU imports."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from tools.native_loader_ledger import inventory, scenario
from tests.test_static_memory_budget import CONFIG
from tools.static_memory_budget import MIB


class NativeLoaderLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest = Path(self.temp.name) / "synthetic.csv"
        self.config = copy.deepcopy(CONFIG)
        self.config["text_config"].update(vocab_size=4, hidden_size=128)
        self.rows = [
            {"tensor": "model.language_model.embed_tokens.weight", "component": "text", "bytes": 1024,
             "shape": "4x128", "dtype": "BF16", "quant_format": "", "in_features": "", "out_features": ""},
            {"tensor": "lm_head.trellis", "component": "output_head", "bytes": 4096,
             "shape": "", "dtype": "I16", "quant_format": "exl3", "in_features": 5120, "out_features": 248320},
            {"tensor": "mtp.fc.trellis", "component": "mtp", "bytes": 100,
             "shape": "", "dtype": "I16", "quant_format": "exl3_header_inferred", "in_features": 5120, "out_features": 5120},
        ]

    def write(self):
        with self.manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            w.writeheader()
            w.writerows(self.rows)
        return inventory(self.manifest, self.config)

    def test_gpu_input_excludes_cpu_embedding_once(self):
        inv = self.write()
        row = scenario(self.config, inv, context=65536)
        self.assertEqual(row["disk_payload_bytes"], 1024 + 4096 + 100)
        self.assertEqual(row["gpu_serialized_weight_input_bytes"], 4096 + 100)
        self.assertEqual(row["cpu_embedding_bytes"], 1024)
        self.assertEqual(row["fp16_reconstruct_single_matrix_bytes"], 320 * MIB)
        self.assertEqual(row["fp16_fallback_single_layer_pool_bytes"], 256 * MIB)

    def test_mtp_and_vision_options_do_not_replicate_embedding(self):
        inv = self.write()
        with_mtp = scenario(self.config, inv, context=16384)
        no_mtp = scenario(self.config, inv, context=16384, mtp=False, history=0)
        self.assertEqual(with_mtp["disk_payload_bytes"] - no_mtp["disk_payload_bytes"], 100)
        self.assertEqual(with_mtp["cpu_embedding_bytes"], no_mtp["cpu_embedding_bytes"])

    def test_staging_scales_with_reserved_pages(self):
        inv = self.write()
        online = scenario(self.config, inv, context=32769, staging=0)
        staged = scenario(self.config, inv, context=32769, staging=1)
        self.assertEqual(staged["context_reserved"], 33024)
        self.assertEqual(online["fp16_staged_single_attention_window_bytes"], 0)
        self.assertEqual(staged["fp16_staged_single_attention_window_bytes"], 256 * MIB)
        self.assertEqual(staged["fp16_fallback_single_layer_pool_bytes"], 129 * MIB)
        with self.assertRaisesRegex(ValueError, "Only native quant-direct"):
            scenario(self.config, inv, context=32768, staging=2)

    def test_pinned_checkpoint_reconciles_without_model_load(self):
        config = copy.deepcopy(CONFIG)
        config["text_config"].update(vocab_size=248320, hidden_size=5120)
        manifest = Path(__file__).resolve().parents[1] / "data" / "qwen38-27b-exl3-3bpw-tensor-inventory.csv"
        inv = inventory(manifest, config)
        row = scenario(config, inv, context=65536, staging=1)
        self.assertEqual(inv["tensor_count"], 3080)
        self.assertEqual(inv["cpu_embedding"], 2542796800)
        self.assertEqual(row["disk_payload_bytes"], 12526588516)
        self.assertEqual(row["fp16_reconstruct_single_matrix_bytes"], 320 * MIB)
        self.assertEqual(row["fp16_staged_single_attention_window_bytes"], 256 * MIB)

    def test_embedding_shape_mismatch_fails_closed(self):
        self.rows[0]["shape"] = "4x129"
        with self.assertRaisesRegex(ValueError, "Embedding source"):
            self.write()

    def test_missing_embedding_fails_closed(self):
        self.rows.pop(0)
        with self.assertRaisesRegex(ValueError, "Required text embedding"):
            self.write()
