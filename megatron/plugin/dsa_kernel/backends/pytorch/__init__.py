"""Portable PyTorch reference backend for non-CP DSA operations."""

from .reference_ops import (
    build_flat_topk_idxs,
    indexer_sparse_attn,
    indexer_topk,
    unfused_sparse_attn,
)

def supports(operation: str, **_capabilities) -> bool:
    """Return non-CP operations implemented by the portable reference backend."""
    return operation in {
        "build_flat_topk_idxs",
        "indexer_sparse_attn",
        "indexer_topk",
    }

__all__ = [
    "build_flat_topk_idxs",
    "indexer_sparse_attn",
    "indexer_topk",
    "supports",
    "unfused_sparse_attn",
]
