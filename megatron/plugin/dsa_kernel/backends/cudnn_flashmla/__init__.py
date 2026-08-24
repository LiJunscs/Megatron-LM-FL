"""CUDA backend for non-CP cuDNN DSA and FlashMLA operations."""

from .fused_ops import (
    build_flat_topk_idxs,
    dsa_sparse_attn,
    fused_indexer_sparse_attn,
    indexer_topk,
)

indexer_sparse_attn = dsa_sparse_attn


def supports(
    operation: str, *, device=None, dtype=None, layout=None, features=None
) -> bool:
    """Report non-CP cuDNN/FlashMLA capabilities."""
    if device is not None and not str(device).startswith("cuda"):
        return False
    if layout is not None and str(layout).lower() != "sbhd":
        return False
    if dtype is not None and operation != "build_flat_topk_idxs":
        if str(dtype).lower() not in {"bf16", "bfloat16", "torch.bfloat16"}:
            return False
    return operation in {
        "build_flat_topk_idxs",
        "fused_indexer_sparse_attn",
        "indexer_sparse_attn",
        "indexer_topk",
    }

__all__ = [
    "build_flat_topk_idxs",
    "fused_indexer_sparse_attn",
    "indexer_sparse_attn",
    "indexer_topk",
]
