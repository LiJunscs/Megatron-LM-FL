"""Triton backend for non-CP fused DSA operations."""

import importlib.util

__all__ = []

# Keep the package importable on CPU-only installations. The unified registry
# imports fused operations only after this dependency probe succeeds.
if importlib.util.find_spec("triton") is not None:
    from .fused_ops import (
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


def supports(
    operation: str, *, device=None, dtype=None, layout=None, features=None
) -> bool:
    """Report exported Triton operations and their current layout boundary."""
    if device is not None and not str(device).startswith("cuda"):
        return False
    if layout is not None and str(layout).lower() != "sbhd":
        return False
    if dtype is not None and operation != "build_flat_topk_idxs":
        if str(dtype).lower() not in {"bf16", "bfloat16", "torch.bfloat16"}:
            return False
    return operation in __all__


__all__.append("supports")
