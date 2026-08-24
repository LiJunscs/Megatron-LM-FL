"""CUDA backend for non-CP cuDNN DSA and FlashMLA operations."""

from .fused_ops import (
    build_flat_topk_idxs,
    dsa_sparse_attn,
    fused_indexer_sparse_attn,
    indexer_topk,
)

indexer_sparse_attn = dsa_sparse_attn


__all__ = [
    "build_flat_topk_idxs",
    "fused_indexer_sparse_attn",
    "indexer_sparse_attn",
    "indexer_topk",
]
