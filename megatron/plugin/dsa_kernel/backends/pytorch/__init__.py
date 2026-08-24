"""Portable PyTorch backend with no optional kernel dependency."""

from .kernels import (
    build_attention_indices,
    build_flat_topk_idxs,
    compress_compressor_input,
    indexer_sparse_attn,
    indexer_topk,
    unfused_sparse_attn,
)

compact_compressor_input = compress_compressor_input

__all__ = [
    "build_attention_indices",
    "build_flat_topk_idxs",
    "compact_compressor_input",
    "indexer_sparse_attn",
    "indexer_topk",
    "unfused_sparse_attn",
]

