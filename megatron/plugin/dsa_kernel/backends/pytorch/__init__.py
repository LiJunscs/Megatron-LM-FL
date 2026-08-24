"""Portable PyTorch reference backend for non-CP DSA operations."""

from .reference_ops import (
    build_flat_topk_idxs,
    indexer_sparse_attn,
    indexer_topk,
    unfused_sparse_attn,
)

__all__ = [
    "build_flat_topk_idxs",
    "indexer_sparse_attn",
    "indexer_topk",
    "unfused_sparse_attn",
]
