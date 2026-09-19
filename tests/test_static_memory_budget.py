"""CPU-only arithmetic tests: do not load a model, torch, CUDA or weights."""
import unittest

from tools.static_memory_budget import MIB, estimate, geometry


CONFIG = {"text_config": {
    "num_hidden_layers": 64,
    "layer_types": ["full_attention" if (i + 1) % 4 == 0
                    else "linear_attention" for i in range(64)],
    "num_key_value_heads": 4, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "mtp_num_hidden_layers": 1,
}}


class TestStaticModel(unittest.TestCase):
    def test_official_geometry(self):
        g = geometry(CONFIG)
        self.assertEqual((g["gdn"], g["full"], g["gv"]), (48, 16, 48))

    def test_mtp4_allocation_floor(self):
        b = estimate(CONFIG, context=32768, k_bits=6, v_bits=5)
        self.assertEqual(b["kv_bytes_per_token_per_attention_layer"], 1536)
        self.assertEqual(b["recurrent_matrix_bytes"], 720 * MIB)
        self.assertEqual(b["recurrent_conv_bytes"], 7.5 * MIB)
        self.assertEqual(b["kv_target_bytes"], 32768 * 1536 * 16)
        self.assertEqual(b["kv_draft_bytes"], 32768 * 1536)

    def test_short_history_and_page_rounding(self):
        b = estimate(CONFIG, context=32769, history=3, slots=1)
        self.assertEqual(b["context_reserved"], 33024)
        self.assertEqual(b["recurrent_matrix_bytes"], 576 * MIB)
        self.assertEqual(b["recurrent_checkpoint_payload_bytes"], 147.75 * MIB)

    def test_no_drafter_and_multiple_slots(self):
        one = estimate(CONFIG, mtp=False)
        two = estimate(CONFIG, mtp=False, slots=2)
        self.assertEqual(one["kv_draft_bytes"], 0)
        self.assertEqual(two["recurrent_matrix_bytes"], 2 * one["recurrent_matrix_bytes"])
        self.assertEqual(two["kv_target_bytes"], one["kv_target_bytes"])

    def test_bad_shapes_fail_closed(self):
        with self.assertRaises(ValueError):
            estimate(CONFIG, k_bits=1)
        bad = {"text_config": {**CONFIG["text_config"], "layer_types": ["full_attention"]}}
        with self.assertRaises(ValueError):
            geometry(bad)
        bad_values = {"text_config": {**CONFIG["text_config"], "head_dim": 30}}
        with self.assertRaises(ValueError):
            estimate(bad_values)


if __name__ == "__main__":
    unittest.main()
