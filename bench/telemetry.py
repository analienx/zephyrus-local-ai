"""Optional FINAL-STAGE whole-card GPU telemetry; nothing starts on import.

A 1s nvidia-smi sampler may itself perturb latency measurements. Record its
presence, report whole-card metrics, and run identical telemetry on A and B.
"""
from __future__ import annotations
import statistics
import subprocess
import threading
from dataclasses import dataclass, field

FIELDS = ('memory.used', 'memory.free', 'temperature.gpu', 'power.draw',
          'clocks.sm', 'utilization.gpu')


def parse_sample(line: str) -> dict:
    values = [x.strip() for x in line.split(',')]
    if len(values) != len(FIELDS):
        raise ValueError('Unexpected nvidia-smi telemetry field count')
    out = {}
    for field, value in zip(FIELDS, values):
        try:
            out[field] = float(value) if value not in ('N/A', '[Not Supported]', '') else None
        except ValueError as exc:
            raise ValueError('Unexpected GPU telemetry value for ' + field) from exc
    return out


def sample_once() -> dict:
    proc = subprocess.run(['nvidia-smi', '--query-gpu='+','.join(FIELDS),
                           '--format=csv,noheader,nounits', '--id=0'],
                          capture_output=True, text=True, timeout=4, check=False)
    if proc.returncode != 0 or len(proc.stdout.splitlines()) != 1:
        raise ValueError('Cannot sample a unique local GPU')
    return parse_sample(proc.stdout.strip())


@dataclass
class GpuSampler:
    enabled: bool = False
    interval_seconds: float = 1.0
    samples: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self):
        if self.enabled:
            # Fail before any model request if requested telemetry is unavailable.
            self.samples.append(sample_once())
            self._stop.clear()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def _poll(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self.samples.append(sample_once())
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                self.errors.append(type(exc).__name__)
                return

    def __exit__(self, _type, _value, _traceback):
        if self.enabled:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=6)
            if not self.errors:
                self.samples.append(sample_once())

    def summary(self) -> dict:
        if not self.enabled:
            return {'collected': False, 'scope': 'whole GPU; not process-specific'}
        if self.errors or not self.samples:
            raise ValueError('Requested GPU telemetry was incomplete')
        def numbers(key):
            return [row[key] for row in self.samples if row[key] is not None]
        used, free, temp, power, clock = (numbers(x) for x in
            ('memory.used','memory.free','temperature.gpu','power.draw','clocks.sm'))
        return {'collected': True, 'samples': len(self.samples),
                'peak_used_mib': max(used) if used else None,
                'min_free_mib': min(free) if free else None,
                'peak_temperature_c': max(temp) if temp else None,
                'median_power_w': statistics.median(power) if power else None,
                'median_sm_clock_mhz': statistics.median(clock) if clock else None,
                'scope': 'whole-card nvidia-smi sampling; may perturb timings; not process VRAM'}
