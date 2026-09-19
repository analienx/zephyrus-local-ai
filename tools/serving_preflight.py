"""Read-only, CPU-only profile preflight. NEVER imports a model or launches inference.

'prepared' verifies an UNPATCHED pinned checkout and an applicable guard patch.
'armed' verifies a PATCHED pinned checkout; neither mode certifies VRAM fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tools.native_loader_ledger import inventory, scenario
from tools.static_memory_budget import MIB, geometry


class PreflightError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                          text=True, check=False)


def validate_profile(p: dict, config: dict, inv: dict) -> dict:
    require(p.get('schema_version') == 1, 'Unknown profile schema version')
    c, rt, a = p['cache'], p['runtime'], p['attention_contract']
    components = p['components']
    require(components == {'text': True, 'mtp': True, 'vision': False},
            'This profile models text + MTP only (no vision)')
    require(c['layer_type'] == 'CacheLayer_quant' and c['k_bits'] == 6 and c['v_bits'] == 5,
            'Expected native quantized 6/5-bit target and draft cache')
    require(c['compand_a'] == 0.0 and rt['qc_staging'] == 0,
            'Companding or FP16 staging violates strict online-cache profile')
    require(rt['strict_quant_cache'] is True and rt['bc_attention'] is True,
            'Explicit strict cache and BC attention required')
    g = geometry(config)
    require((g['full'], g['gdn'], g['mtp_layers'], g['heads'], g['dim']) ==
            (16, 48, 1, 4, 256), 'Unexpected Qwen attention/GDN/MTP geometry')
    require(a == {'query_dtype': 'torch.float16', 'head_dim': g['dim'],
                  'varlen': False, 'requires_quant_direct': True},
            'Static attention contract differs from quant-direct prerequisites')
    require(c['context_tokens'] > 0 and c['history'] == 4 and c['slots'] == 1,
            'Invalid cache capacity, recurrent history or batch slots')
    row = scenario(config, inv, context=c['context_tokens'], mtp=True, vision=False,
                   staging=rt['qc_staging'], history=c['history'], slots=c['slots'],
                   k_bits=c['k_bits'], v_bits=c['v_bits'])
    floor = p['memory']['static_unbudgeted_floor_mib']
    total = p['memory']['nominal_gpu_mib']
    require(isinstance(floor, int) and floor >= 0 and isinstance(total, int) and total > 0,
            'Invalid nominal-GPU or modeled-residual floor')
    residual = total * MIB - row['optimistic_known_gpu_inputs']
    require(residual >= floor * MIB, 'Optimistic static GPU residual below configured floor')
    return {'static_only': True, 'reserved_tokens': row['context_reserved'],
            'candidate_gpu_weight_mib': round(row['gpu_serialized_weight_input_bytes'] / MIB, 2),
            'cache_recurrent_mib': round(row['kv_and_recurrent_bytes'] / MIB, 2),
            'unbudgeted_mib': round(residual / MIB, 2),
            'fp16_fallback_single_layer_mib': round(row['fp16_fallback_single_layer_pool_bytes'] / MIB, 2)}


def verify_source(root: Path, patch: Path, revision: str, phase: str) -> None:
    head = git(root, 'rev-parse', 'HEAD')
    require(head.returncode == 0 and head.stdout.strip() == revision,
            'Upstream source HEAD is not the pinned revision')
    dispatch_path = root / 'exllamav3/modules/attention_fn/dispatch.py'
    text = dispatch_path.read_text(encoding='utf-8')
    require('layer.get_qkv()' in text and 'layer.get_kv(' in text and
            'cu_seqlens is None' in text and 'q.dtype == torch.float16' in text,
            'Upstream compressed-KV dispatcher unexpectedly changed')
    require('_qc_staging = int(os.environ.get("EXL3_QC_STAGING", "1"))' in
            (root / 'exllamav3/modules/attention_fn/triton_paged.py').read_text(encoding='utf-8'),
            'Upstream compressed-attention staging behavior unexpectedly changed')
    require('layer.compand_a == 0.0' in
            (root / 'exllamav3/modules/attention_fn/bc_attn.py').read_text(encoding='utf-8'),
            'Upstream BC attention quantized eligibility unexpectedly changed')
    if phase == 'prepared':
        require('EXL3_QC_STRICT' not in text, 'Source is already patched: use phase armed')
        require(git(root, 'status', '--porcelain').stdout.strip() == '',
                'Pinned reference checkout must remain clean in prepared phase')
        result = git(root, 'apply', '--check', str(patch.resolve()))
        require(result.returncode == 0, 'Guard patch does not apply cleanly: ' + result.stderr)
    elif phase == 'armed':
        require('EXL3_QC_STRICT' in text and 'not quant_direct_eligible' in text,
                'Strict guard absent: do not attempt to arm an unpatched engine')
        result = git(root, 'apply', '--reverse', '--check', str(patch.resolve()))
        require(result.returncode == 0, 'Source does not match the audited patch: ' + result.stderr)
    else:
        raise PreflightError('Unsupported preflight phase')


def preflight(profile_path: Path, metadata_dir: Path, manifest_path: Path,
              source_root: Path, patch_path: Path, phase: str = 'prepared') -> dict:
    p = json.loads(profile_path.read_text(encoding='utf-8'))
    cp = p['checkpoint']
    cfg_path = metadata_dir / 'config.json'
    require(sha256(cfg_path) == cp['config_sha256'], 'Checkpoint config SHA-256 mismatch')
    require(sha256(manifest_path) == cp['manifest_sha256'], 'Tensor manifest SHA-256 mismatch')
    require(sha256(patch_path) == p['runtime']['patch_sha256'], 'Guard patch SHA-256 mismatch')
    config = json.loads(cfg_path.read_text(encoding='utf-8'))
    inv = inventory(manifest_path, config)
    require(inv['tensor_count'] == 3080, 'Expected 3,080 tensors from pinned checkpoint')
    result = validate_profile(p, config, inv)
    verify_source(source_root, patch_path, p['runtime']['revision'], phase)
    result.update({'status': 'static-contract-' + phase,
                   'runtime_validation': 'NOT PERFORMED',
                   'inference_or_gpu_testing': 'NOT PERFORMED',
                   'required_environment': {'EXL3_QC_STRICT': '1',
                                            'EXL3_QC_STAGING': '0', 'EXL3_BC_ATTN': '1'},
                   'required_runtime_assertions': ['query dtype FP16', 'actual packed-cache dispatch',
                                                   'no full-cache FP16 fallback', 'no unsupported BC path'],
                   'warning': 'Not a VRAM-fit, numerical-correctness or performance certificate.'})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--metadata-dir', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--upstream-root', type=Path, required=True)
    ap.add_argument('--patch', type=Path, required=True)
    ap.add_argument('--phase', choices=('prepared', 'armed'), default='prepared')
    args = ap.parse_args()
    try:
        result = preflight(args.profile, args.metadata_dir, args.manifest,
                           args.upstream_root, args.patch, args.phase)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.exit(2, 'STATIC PREFLIGHT REJECTED: ' + str(exc) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
