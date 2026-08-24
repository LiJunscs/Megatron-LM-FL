"""NVIDIA CuTe DSL kernels, imported only by the CUDA aggregate backend."""

from .cp_layout import CompressorInputCompact, _CUTE_AVAILABLE, build_attention_indices


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


__all__ = [
    "build_attention_indices",
    "compact_compressor_input",
]
