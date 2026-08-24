"""CUDA aggregate backend for CuTe, cuDNN DSA, and FlashMLA."""

from .. import cute
from ..cute import build_attention_indices, compact_compressor_input

from .kernels import (
    build_flat_topk_idxs,
    dsa_sparse_attn,
    fused_indexer_sparse_attn,
    indexer_topk,
)

indexer_sparse_attn = dsa_sparse_attn


def supports(operation: str) -> bool:
    """Report optional sub-backend capability without importing from Core."""
    if operation in {"build_attention_indices", "compact_compressor_input"}:
        return cute._CUTE_AVAILABLE
    return True

__all__ = [
    "build_attention_indices",
    "build_flat_topk_idxs",
    "compact_compressor_input",
    "fused_indexer_sparse_attn",
    "indexer_sparse_attn",
    "indexer_topk",
    "supports",
]
