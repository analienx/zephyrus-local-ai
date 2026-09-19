"""Read public safetensors HEADERS only and reconcile quantization metadata. No weights/inference."""
import argparse
import collections
import csv
import json
import math
import pathlib
import re
import urllib.request

DTYPE_BYTES = {'F64': 8, 'I64': 8, 'U64': 8, 'F32': 4, 'I32': 4, 'U32': 4,
               'F16': 2, 'BF16': 2, 'I16': 2, 'U16': 2, 'I8': 1, 'U8': 1,
               'BOOL': 1, 'F8_E4M3': 1, 'F8_E5M2': 1}
MAX_HEADER = 2 * 1024 * 1024
REPO = 'turboderp/Qwen3.8-27B-exl3'
REV = '004a887127d8304ca2d5475d3a3c41f1761fdd27'


def ranged(url, start, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}',
                                               'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(req, timeout=30) as response:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if response.status != 206 or match is None or (int(match[1]), int(match[2])) != (start, end):
            raise ValueError('Server did not honor the exact byte range: aborting without downloading weights')
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise ValueError('Incomplete or excessive range response')
        return data, int(match[3])


def read_headers(directory, shards, fetch=False):
    result, sizes = {}, {}
    local = directory / 'headers'
    if fetch:
        local.mkdir(parents=True, exist_ok=True)
    for shard in sorted(shards):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors', shard):
            raise ValueError(f'Unsafe shard name: {shard}')
        path = local / (shard + '.header.json')
        if fetch:
            url = f'https://huggingface.co/{REPO}/resolve/{REV}/{shard}'
            prefix, length = ranged(url, 0, 7)
            n = int.from_bytes(prefix, 'little')
            if n < 2 or n > MAX_HEADER or n + 8 >= length:
                raise ValueError(f'Header out of bounds: {shard} ({n} bytes)')
            raw, second_length = ranged(url, 8, 7 + n)
            if second_length != length:
                raise ValueError('Remote shard changed between requests')
            path.write_bytes(raw)
            (local / (shard + '.size.txt')).write_text(str(length), encoding='ascii')
        if not path.is_file():
            raise FileNotFoundError(f'Missing metadata-only header: {path}; use --fetch-headers')
        result[shard] = json.loads(path.read_text(encoding='utf-8'))
        size_path = local / (shard + '.size.txt')
        sizes[shard] = int(size_path.read_text()) if size_path.exists() else None
    return result, sizes


def component(name):
    if name.startswith('model.language_model.'):
        return 'text'
    if name.startswith('model.visual.'):
        return 'vision'
    if name.startswith('mtp.'):
        return 'mtp'
    if name.startswith('lm_head.'):
        return 'output_head'
    return 'other'


def audit(index, quant, headers, shard_sizes):
    index_names = set(index['weight_map'])
    by_name = {}
    for shard, entries in headers.items():
        max_end = 0
        for name, meta in entries.items():
            if name == '__metadata__':
                continue
            if name in by_name:
                raise ValueError(f'Duplicate tensor: {name}')
            start, end = meta['data_offsets']
            expected = math.prod(meta['shape']) * DTYPE_BYTES[meta['dtype']]
            if start < 0 or end < start or end - start != expected:
                raise ValueError(f'Invalid tensor byte size/offset: {name}')
            if index['weight_map'].get(name) != shard:
                raise ValueError(f'Shard/index mismatch: {name}')
            by_name[name] = (shard, meta, expected)
            max_end = max(max_end, end)
        if shard_sizes[shard] is not None:
            if max_end > shard_sizes[shard]:
                raise ValueError(f'Tensor extends beyond shard: {shard}')
    if set(by_name) != index_names:
        raise ValueError(f'Index/header tensor mismatch: {len(index_names - set(by_name))} missing')
    if sum(v[2] for v in by_name.values()) != index['metadata']['total_size']:
        raise ValueError('Index total_size does not equal summed tensor payload')
    from_quant = {}
    for logical_name, entry in quant['tensor_storage'].items():
        for name, meta in entry['stored_tensors'].items():
            if name in from_quant:
                raise ValueError(f'Quantization metadata duplicates tensor: {name}')
            from_quant[name] = (logical_name, entry, meta)
            if name not in by_name:
                raise ValueError(f'Quantization metadata names missing tensor: {name}')
            shard, actual, n = by_name[name]
            dtype = {'torch.float16': 'F16', 'torch.bfloat16': 'BF16',
                     'torch.int16': 'I16', 'torch.int32': 'I32'}.get(meta['dtype'])
            if dtype != actual['dtype'] or meta['shape'] != actual['shape'] or meta['n_bytes'] != n:
                raise ValueError(f'Quantization metadata differs from header: {name}')
    return by_name, from_quant


def inventory(by_name, from_quant):
    rows = []
    for name, (shard, meta, n) in sorted(by_name.items()):
        logical, entry, _ = from_quant.get(name, ('', {}, {}))
        bpw = entry.get('bits_per_weight', '')
        shape = meta['shape']
        if not logical and name.endswith('.trellis'):
            logical = name[:-len('.trellis')]
            if len(shape) != 3 or shape[2] % 16:
                raise ValueError(f'Unlisted EXL3 trellis has invalid shape: {name}')
            bpw = shape[2] // 16
            for suffix in ('.mul1', '.suh', '.svh'):
                if logical + suffix not in by_name:
                    raise ValueError(f'Unlisted EXL3 trellis lacks {suffix}: {name}')
            entry = {'quant_format': 'exl3_header_inferred', 'stored_tensors': {logical + '.mul1': {}}}
        k, out, gemv = '', '', ''
        if name.endswith('.trellis') and entry.get('quant_format') in ('exl3', 'exl3_header_inferred'):
            if len(shape) != 3 or shape[2] != 16 * bpw:
                raise ValueError(f'Trellis shape disagrees with bitrate: {name}')
            k, out = shape[0] * 16, shape[1] * 16
            if k % 128 or out % 128:
                raise ValueError(f'EXL3 matrix dimensions violate Hadamard tile: {name}')
            has_cb = logical + '.mul1' in entry['stored_tensors']
            gemv = 'hard-eligible; heuristic/SM-dependent' if bpw in (2, 3, 4) and has_cb else 'general GEMM'
        rows.append(dict(tensor=name, component=component(name), shard=shard,
                         dtype=meta['dtype'], shape='x'.join(map(str, shape)), bytes=n,
                         logical_group=logical, quant_format=entry.get('quant_format', ''),
                         bits_per_weight=bpw, in_features=k, out_features=out,
                         small_m_dispatch=gemv))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', required=True, type=pathlib.Path)
    parser.add_argument('--fetch-headers', action='store_true', help='Fetch ONLY byte-range safetensors headers')
    parser.add_argument('--output-csv', type=pathlib.Path)
    args = parser.parse_args()
    p = args.metadata_dir
    idx = json.loads((p / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    quant = json.loads((p / 'quantization_config.json').read_text(encoding='utf-8'))
    headers, sizes = read_headers(p, set(idx['weight_map'].values()), fetch=args.fetch_headers)
    found, described = audit(idx, quant, headers, sizes)
    for shard, entries in headers.items():
        if sizes[shard] is not None:
            header_len = (p / 'headers' / (shard + '.header.json')).stat().st_size
            data_len = sizes[shard] - 8 - header_len
            last_end = max(v['data_offsets'][1] for k, v in entries.items() if k != '__metadata__')
            if last_end != data_len:
                raise ValueError(f'Header/file byte accounting mismatch: {shard}')
    rows = inventory(found, described)
    print(f'checkpoint={REPO}@{REV} stored={len(found)} quant_described={len(described)}')
    print(f'payload_bytes={sum(r["bytes"] for r in rows)} missing_quant_descriptions={len(found)-len(described)}')
    for group in sorted({r['component'] for r in rows}):
        sub = [r for r in rows if r['component'] == group]
        print(f'{group}: tensor_count={len(sub)} bytes={sum(r["bytes"] for r in sub)}')
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'inventory={args.output_csv}')


if __name__ == '__main__':
    main()
