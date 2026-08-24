# Copyright (c) 2026 FlagOS / Megatron-LM-FL. All rights reserved.

"""Portable PyTorch reference backend for DSv4 sparse attention.

This module exposes the backend-callable surface while delegating shared
flat-index and contiguous-CP layout semantics to the small core reference
in ``csa_utils.utils``. Optional Triton and legacy CUDA implementations
are validated against these operators.
"""

from typing import Optional, Tuple

import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import utils as dsa_utils

# ---------------------------------------------------------------------------
# Index helpers (pure PyTorch, shared with every backend)
# ---------------------------------------------------------------------------


def build_flat_topk_idxs(
    *idx_groups: torch.Tensor,
    batch_size: int,
    compact: bool = False,
    seqlen_kv: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Combine local index groups and globalize (torch reference).

    ``seqlen_kv`` is accepted for signature parity with the legacy/FlashMLA
    ``dsa_kernels.build_flat_topk_idxs`` (used only for shape assertions
    there) and ignored by the reference.
    """
    return dsa_utils.build_flat_topk_idxs(
        *idx_groups,
        batch_size=batch_size,
        compact=compact,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )


# ---------------------------------------------------------------------------
# CP layout (legacy: CuTeDSL ``cp_layout_kernels``)
# ---------------------------------------------------------------------------


def compress_compressor_input(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress local+boundary hidden rows into fixed-capacity input (torch)."""
    return dsa_utils.compressor_input_compact(
        hidden_local, boundary_hidden, cu_seqlens, global_start, ratio, d_comp, c_cap
    )


def build_attention_indices(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    compressed_topk: Optional[torch.Tensor] = None,
    cu_seqlens_compressed: Optional[torch.Tensor] = None,
    seq_to_rank_row: Optional[torch.Tensor] = None,
    for_indexer_loss: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Lower logical CP indices into physical attention indices (torch)."""
    return dsa_utils.build_attention_indices(
        cu_seqlens,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk,
        cu_seqlens_compressed,
        seq_to_rank_row,
        for_indexer_loss,
    )


# ---------------------------------------------------------------------------
# Fused sparse attention (legacy: FlashMLA + cuDNN DSA ``dsa_kernels``)
# ---------------------------------------------------------------------------


def unfused_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable sparse attention (pure-torch reference).

    Lazy import avoids a circular dependency with ``csa.py``.
    """
    from megatron.core.transformer.experimental_attention_variant.csa import (
        unfused_compressed_sparse_attn,
    )

    return unfused_compressed_sparse_attn(
        query, kv_full, attn_sink, topk_indices, softmax_scale
    )


def indexer_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    topk_length: Optional[torch.Tensor] = None,
    indexer_topk: int = 0,
) -> torch.Tensor:
    """Sparse attention drop-in for the legacy ``dsa_sparse_attn`` contract.

    Accepts the fused SBHD-flat layout used by ``dsa_kernels.dsa_sparse_attn``
    (flat-global top-k indices of shape ``(sq*b, topk)`` with flat row order
    ``s*B + b``) and delegates to the pure-torch reference.

    ``topk_length`` (used for the fused compact fast-path) is ignored: the
    reference masks via ``-1``.
    """
    sq, b, np_ = query.shape[0], query.shape[1], query.shape[2]
    if topk_idxs.ndim == 2 and topk_idxs.shape[0] == sq * b:
        global_idx = topk_idxs.long()
        valid = global_idx >= 0
        batch_ids = torch.arange(sq * b, device=query.device) % b
        local_idx = (global_idx // b).clamp_min(0)
        local_idx = torch.where(valid, local_idx, torch.full_like(local_idx, -1))
        local_idx = local_idx.view(sq, b, -1).permute(1, 0, 2).contiguous()
    else:
        local_idx = topk_idxs
    return unfused_sparse_attn(
        query, kv_full, attn_sink, local_idx, softmax_scale
    )


def indexer_topk(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights_indexer: torch.Tensor,
    index_topk: int,
    *,
    indexer_softmax_scale: float,
    varlen_starts: Optional[torch.Tensor] = None,
    varlen_ends: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """PyTorch indexer scoring + top-k selection reference.

    Mirrors the SBHD fused ``indexer_topk`` API; returns
    ``(scores, topk_indices)`` (scores may be ``None`` for packed paths).
    """
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        fused_qk_topk_naive,
    )

    scores, topk = fused_qk_topk_naive(
        q_indexer,
        k_indexer,
        weights_indexer,
        index_topk,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        use_relu=True,
    )
    return scores, topk


__all__ = [
    "build_attention_indices",
    "build_flat_topk_idxs",
    "compress_compressor_input",
    "indexer_sparse_attn",
    "indexer_topk",
    "unfused_sparse_attn",
]


