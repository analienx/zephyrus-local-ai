"""Combine a complete safetensors header inventory with source-derived cache shapes.

Output is a *paper budget*, not a VRAM allocation, throughput, or fit test.
"""
import argparse
import csv
import json
from pathlib import Path

from tools.static_memory_budget import MIB, estimate

COMPONENTS = {'text', 'output_head', 'mtp', 'vision'}


def read_manifest(path):
    totals = {c: 0 for c in COMPONENTS}
    names = set()
    with Path(path).open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name, group, size = row['tensor'], row['component'], int(row['bytes'])
            if name in names or group not in COMPONENTS or size < 0:
                raise ValueError(f'Invalid, duplicated or unknown tensor: {name}')
            names.add(name)
            totals[group] += size
    if not names:
        raise ValueError('Tensor manifest is empty')
    return totals, len(names)


def plan(config, component_bytes, *, context, with_mtp=True, vision=False,
         kv_k=6, kv_v=5, history=4, slots=1):
    active = ['text', 'output_head']
    if with_mtp:
        active.append('mtp')
    if vision:
        active.append('vision')
    weights = sum(component_bytes[key] for key in active)
    cache = estimate(config, context=context, k_bits=kv_k, v_bits=kv_v,
                     history=history, slots=slots, mtp=with_mtp)
    return {'components': active, 'serialized_weight_bytes': weights,
            'known_cache_and_recurrent_bytes': cache['known_resident_bytes'],
            'optimistic_accounted_bytes': weights + cache['known_resident_bytes'],
            'context_reserved': cache['context_reserved'],
            'excluded': ['graph allocations', 'reconstruction/prefill temporary allocations',
                         'driver/display reservations', 'allocator overhead and fragmentation',
                         'checkpoint stashes', 'weight decoding and runtime tensor copies']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--nominal-vram-mib', type=int, required=True,
                    help='Physical total (NOT available/free VRAM); no fit claim is produced')
    ap.add_argument('--contexts', nargs='+', type=int, default=[16384, 32768, 65536, 131072])
    ap.add_argument('--include-vision', action='store_true')
    ap.add_argument('--without-mtp', action='store_true')
    ap.add_argument('--history', type=int, default=4)
    ap.add_argument('--slots', type=int, default=1)
    ap.add_argument('--kv-k', type=int, default=6)
    ap.add_argument('--kv-v', type=int, default=5)
    args = ap.parse_args()
    if args.nominal_vram_mib < 1:
        ap.error('Nominal VRAM must be positive')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    components, count = read_manifest(args.manifest)
    print('INVENTORY', count, 'tensor payloads; physical VRAM is NOT a residency certificate')
    print('COMPONENTS', ' '.join(f'{k}={components[k]/MIB:.2f}MiB' for k in sorted(components)))
    for context in args.contexts:
        row = plan(config, components, context=context, with_mtp=not args.without_mtp,
                   vision=args.include_vision, history=args.history, slots=args.slots,
                   kv_k=args.kv_k, kv_v=args.kv_v)
        unbudgeted = args.nominal_vram_mib * MIB - row['optimistic_accounted_bytes']
        print(f"context={row['context_reserved']} serialized_weights={row['serialized_weight_bytes']/MIB:.2f}MiB "
              f"cache+recurrent={row['known_cache_and_recurrent_bytes']/MIB:.2f}MiB "
              f"optimistic_unbudgeted={unbudgeted/MIB:.2f}MiB")
    print('UNBUDGETED: ' + '; '.join(row['excluded']))
    print('WARNING: weights-on-disk are a modeling input, not observed GPU resident bytes; '
          'negative headroom rejects the modeled plan but positive headroom never proves fit.')


if __name__ == '__main__':
    main()
