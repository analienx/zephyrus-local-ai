"""Generate a narrow opt-in EXL3 packed-KV strictness patch. Never import torch.

The pinned upstream checkout is READ ONLY: this script writes only a project patch.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path

SOURCE = Path('exllamav3/modules/attention_fn/dispatch.py')
IMPORT = 'import torch\n'
INSERT_IMPORT = 'import os\nimport torch\n'
ANCHOR_FLAG = '_qc_attn = _qc_staging < 2\n'
GUARD_FLAG = ('_qc_attn = _qc_staging < 2\n'
              '_qc_strict = os.environ.get("EXL3_QC_STRICT", "0") == "1"\n')
OLD_BRANCH = '''        if (
            _qc_attn and
            isinstance(layer, CacheLayer_quant) and
            layer.compand_a == 0.0 and
            q.dtype == torch.float16 and
            dim <= 512 and dim % 32 == 0 and   # packed groups of 32; non-pow2 dims run zero-padded
            cu_seqlens is None
        ):
'''
NEW_BRANCH = '''        quant_direct_eligible = (
            _qc_attn and
            isinstance(layer, CacheLayer_quant) and
            layer.compand_a == 0.0 and
            q.dtype == torch.float16 and
            dim <= 512 and dim % 32 == 0 and   # packed groups of 32; non-pow2 dims run zero-padded
            cu_seqlens is None
        )
        if _qc_strict and isinstance(layer, CacheLayer_quant) and not quant_direct_eligible:
            raise RuntimeError(
                "EXL3_QC_STRICT: quantized attention would allocate full-cache FP16 K/V; "
                "check staging, compand_a, query dtype/head dimension and varlen arguments"
            )
        if quant_direct_eligible:
'''


def replace_exact(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f'Pinned source changed: expected one occurrence, found {text.count(old)}')
    return text.replace(old, new, 1)


def make_patch(original: str) -> str:
    modified = replace_exact(original, IMPORT, INSERT_IMPORT)
    modified = replace_exact(modified, ANCHOR_FLAG, GUARD_FLAG)
    modified = replace_exact(modified, OLD_BRANCH, NEW_BRANCH)
    return ''.join(difflib.unified_diff(
        original.splitlines(keepends=True), modified.splitlines(keepends=True),
        fromfile='a/' + SOURCE.as_posix(), tofile='b/' + SOURCE.as_posix(),
    ))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    source = (args.upstream_root / SOURCE).read_text(encoding='utf-8')
    patch = make_patch(source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('diff --git a/' + SOURCE.as_posix() + ' b/' + SOURCE.as_posix() + '\n' + patch, encoding='utf-8', newline='\n')
    print(f'Wrote opt-in strictness patch: {args.out}')


if __name__ == '__main__':
    main()
