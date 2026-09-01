"""Triton implementation of the DSv4 CP contract."""

from .cp_kernel import _TRITON_AVAILABLE, build_attention_indices, compress_compressor_input

compact_compressor_input = compress_compressor_input


def supports(
    operation: str, *, device=None, dtype=None, layout=None, features=None
) -> bool:
    """Report the complete public CP contract owned by this provider."""
    if not _TRITON_AVAILABLE:
        return False
    if device is not None and not str(device).startswith("cuda"):
        return False
    if layout is not None and str(layout).lower() not in {"sbhd", "thd"}:
        return False
    return operation in {"build_attention_indices", "compact_compressor_input"}

__all__ = [
    "build_attention_indices",
    "compact_compressor_input",
    "supports",
]
