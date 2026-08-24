"""Triton backend exports and its dependency boundary."""

import importlib.util

from .cp_layout import build_attention_indices, compress_compressor_input

compact_compressor_input = compress_compressor_input

__all__ = [
    "build_attention_indices",
    "compact_compressor_input",
]

# Keep the CP-layout reference importable on CPU-only installations.  The
# unified registry imports this backend for fused operations only after this
# dependency probe succeeds.
if importlib.util.find_spec("triton") is not None:
    from .kernels import (
        build_flat_topk_idxs,
        dsa_sparse_attn_sbhd,
        fused_indexer_sparse_attn,
        indexer_topk,
    )

    indexer_sparse_attn = dsa_sparse_attn_sbhd
    __all__ += [
        "build_flat_topk_idxs",
        "fused_indexer_sparse_attn",
        "indexer_sparse_attn",
        "indexer_topk",
    ]
