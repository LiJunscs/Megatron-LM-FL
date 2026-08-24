"""Triton implementation of the DSv4 CP contract."""

from .cp_kernel import build_attention_indices, compress_compressor_input

compact_compressor_input = compress_compressor_input

__all__ = [
    "build_attention_indices",
    "compact_compressor_input",
]
