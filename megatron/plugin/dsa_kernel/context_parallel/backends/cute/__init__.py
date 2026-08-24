"""NVIDIA CuTe DSL implementation of the DSv4 CP contract."""

from .cp_kernel import CompressorInputCompact, _CUTE_AVAILABLE, build_attention_indices


def compact_compressor_input(
    hidden_local,
    boundary_hidden,
    cu_seqlens,
    global_start,
    ratio,
    d_comp,
    c_cap,
):
    return CompressorInputCompact.apply(
        hidden_local,
        boundary_hidden,
        cu_seqlens,
        global_start,
        ratio,
        d_comp,
        c_cap,
    )


def supports(operation: str) -> bool:
    """Return whether the optional CuTe DSL runtime is available."""
    return (
        operation in {"build_attention_indices", "compact_compressor_input"}
        and _CUTE_AVAILABLE
    )


__all__ = [
    "build_attention_indices",
    "compact_compressor_input",
    "supports",
]
