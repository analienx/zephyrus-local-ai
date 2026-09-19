"""GPU telemetry parser tests only; nvidia-smi is NEVER invoked."""
import unittest
from unittest.mock import patch
from bench.telemetry import GpuSampler, parse_sample


class TelemetryTests(unittest.TestCase):
    def test_parser_is_exact_and_rejects_unexpected_data(self):
        row = parse_sample('2000, 14000, 68, 84.3, 2100, 91')
        self.assertEqual(row['memory.used'], 2000)
        self.assertEqual(row['temperature.gpu'], 68)
        self.assertEqual(parse_sample('2000, 14000, N/A, N/A, 2100, 80')['power.draw'], None)
        with self.assertRaises(ValueError):
            parse_sample('2000, 14000, 68')

    def test_disabled_sampler_does_nothing(self):
        with patch('bench.telemetry.sample_once', side_effect=AssertionError('Should not sample')):
            with GpuSampler(enabled=False) as sampler:
                self.assertFalse(sampler.summary()['collected'])

    def test_summary_marks_whole_card_scope(self):
        sampler = GpuSampler(enabled=True)
        sampler.samples = [parse_sample('3000,13000,62,71,1000,55'),
                           parse_sample('4200,11800,73,95,1500,90')]
        summary = sampler.summary()
        self.assertEqual(summary['peak_used_mib'], 4200)
        self.assertEqual(summary['min_free_mib'], 11800)
        self.assertEqual(summary['peak_temperature_c'], 73)
        self.assertIn('whole-card', summary['scope'])
