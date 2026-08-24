# Copyright (c) 2026 FlagOS / Megatron-LM-FL. All rights reserved.

"""Triton CP-layout kernels for the DSv4 THD context-parallel path (Stage 4 / M3).

These are the operators that upstream implements with CuTeDSL
(``csa_utils`` and the CP kernel contract): fixed-capacity compressor-input
compaction (forward + backward scatter) and final attention-index lowering.

Design (plan §2.1)
------------------
The *semantic* logical-to-physical mappings --- which compact row copies which
hidden row, which sequence a query row belongs to, window/compressed-id
lowering rules --- are defined in the PyTorch/metadata layer (the portable
reference in ``csa_utils/cp_layout.py``) and are therefore host-side and CPU
testable.  Triton executes the frozen local compute only: the masked
column-wise data movement for compaction and the affine per-column index
lowering for attention indices.  This is how the plan keeps "Triton as a
per-op accelerator, never a correctness dependency".

Backward of the compaction is a scatter over a *provably one-to-one* source
mapping (each hidden row is referenced by at most one compact row inside a
rank), so Triton uses plain masked stores --- the atomic-free fast path the plan
allows.  When a duplicate source row is ever observed (defensive guard), the
backward transparently falls back to the accumulating PyTorch
``index_add`` reference so gradient accumulation stays correct.

Triton is optional and lazily imported: without ``triton`` (or on a non-GPU
box) every entry point falls back directly to the core PyTorch reference in
``csa_utils.cp_layout``. Public import of this module never requires Triton.
"""

import math
from typing import Optional, Tuple

import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_layout as dsa_layout,
)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environ varies by platform
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

def _next_pow2(x: int) -> int:
    """Smallest power of two >= ``x`` (``x`` must be >= 1)."""
    return 1 << (max(int(x), 1) - 1).bit_length()


# =============================================================================
# Triton kernels (defined only when Triton is importable)
# =============================================================================


if _TRITON_AVAILABLE:

    @triton.jit
    def _compaction_fwd_kernel(
        src_cat_ptr,  # (d_window + l_local + 1, W)
        src_flat_ptr,  # (compact_len,) int64 physical source row into src_cat
        valid_ptr,  # (compact_len,) bool
        out_ptr,  # (compact_len, W)
        compact_len,
        W,
        W_BLOCK: tl.constexpr,
    ):
        """Copy each compact row's ratio-token source (or zero it) on GPU."""
        row = tl.program_id(0)
        if row < compact_len:
            src_flat = tl.load(src_flat_ptr + row)
            valid = tl.load(valid_ptr + row).to(tl.int1)
            cols = tl.arange(0, W_BLOCK)
            cmask = cols < W
            value = tl.load(src_cat_ptr + src_flat * W + cols, mask=cmask, other=0.0)
            value = tl.where(valid, value, 0.0)
            tl.store(out_ptr + row * W + cols, value, mask=cmask)

    @triton.jit
    def _compaction_bwd_kernel(
        grad_compact_ptr,  # (compact_len, W)
        back_idx_ptr,  # (l_local + d_window,) int32 compact row or -1
        grad_out_ptr,  # (l_local + d_window, W)
        total_rows,
        W,
        W_BLOCK: tl.constexpr,
    ):
        """Scatter each compact-row gradient to its unique source row."""
        row = tl.program_id(0)
        if row < total_rows:
            cidx = tl.load(back_idx_ptr + row)
            ok = cidx >= 0
            cols = tl.arange(0, W_BLOCK)
            cmask = cols < W
            value = tl.load(grad_compact_ptr + cidx * W + cols, mask=ok & cmask, other=0.0)
            tl.store(grad_out_ptr + row * W + cols, value, mask=cmask)

    @triton.jit
    def _build_attention_indices_kernel(
        seq_start_ptr,  # (l_local,) int32, -1 for rows outside every sequence
        seq_comp_start_ptr,  # (l_local,) int32
        seq_comp_len_ptr,  # (l_local,) int32
        window_start_ptr,  # (l_local,) int32
        window_count_ptr,  # (l_local,) int32
        seq_to_rank_row_ptr,  # (seq_major_rows,) int32
        compressed_topk_ptr,  # (l_local, compressed_width) int32 (modes 0/2)
        topk_idxs_ptr,  # (l_local, total_width) int32
        topk_length_ptr,  # (l_local,) int32 (not written in mode 2)
        indexer_rank_major_ptr,  # (l_local, compressed_width) int32 (mode 2)
        global_start,
        d_window,
        compressed_base,
        seq_major_rows,
        l_local,
        total_width,
        ratio,
        compressed_width,
        SW_BLOCK: tl.constexpr,
        WIN_UNITS: tl.constexpr,
        C_BLOCK: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """Lower window + compressed logical ids to physical attention rows.

        Modes match the CuTe kernel's ``index_mode`` (0 selected top-k, 1 all
        visible compressed rows, 2 indexer-loss).  ``topk_idxs`` is pre-filled
        with -1; kernels only write the valid entries.
        """
        row = tl.program_id(0)
        if row < l_local:
            q = global_start + row
            seq_start = tl.load(seq_start_ptr + row)
            in_seq = seq_start >= 0
            window_start = tl.load(window_start_ptr + row)
            window_count = tl.load(window_count_ptr + row)

            if MODE == 2:
                # Compressed ids first, then window positions (contiguous).
                if compressed_width > 0:
                    cids = tl.arange(0, C_BLOCK)
                    cmask = cids < compressed_width
                    cid = tl.load(
                        compressed_topk_ptr + row * compressed_width + cids,
                        mask=cmask,
                        other=-1,
                    )
                    seq_comp_start = tl.load(seq_comp_start_ptr + row)
                    seq_comp_len = tl.load(seq_comp_len_ptr + row)
                    comp_valid = in_seq & (cid >= 0) & (cid < seq_comp_len)
                    seq_major = seq_comp_start + cid
                    safe_major = tl.where(seq_major < 0, 0, seq_major)
                    safe_major = tl.where(
                        safe_major < seq_major_rows, safe_major, seq_major_rows - 1
                    )
                    rank_row = tl.load(
                        seq_to_rank_row_ptr + safe_major, mask=cmask, other=-1
                    )
                    rank_valid = (
                        comp_valid
                        & (seq_major < seq_major_rows)
                        & (rank_row >= 0)
                    )
                    comp_phys = tl.where(rank_valid, compressed_base + rank_row, -1)
                    tl.store(
                        topk_idxs_ptr + row * total_width + cids,
                        comp_phys,
                        mask=cmask,
                    )
                    tl.store(
                        indexer_rank_major_ptr + row * compressed_width + cids,
                        tl.where(rank_valid, rank_row, -1),
                        mask=cmask,
                    )
                # Window part at columns [compressed_width, +window_count).
                for unit in tl.static_range(0, WIN_UNITS):
                    war = unit * SW_BLOCK + tl.arange(0, SW_BLOCK)
                    abs_col = compressed_width + war
                    ok = (war < window_count) & in_seq & (abs_col < total_width)
                    pos = window_start + war
                    phys_win_2 = tl.where(
                        pos < global_start,
                        pos - (global_start - d_window),
                        d_window + pos - global_start,
                    )
                    tl.store(
                        topk_idxs_ptr + row * total_width + abs_col,
                        tl.where(ok, phys_win_2, -1),
                        mask=abs_col < total_width,
                    )
            else:
                # Window part first (modes 0/1), contiguous in [0, window_count).
                for unit in tl.static_range(0, WIN_UNITS):
                    war = unit * SW_BLOCK + tl.arange(0, SW_BLOCK)
                    ok = (war < window_count) & in_seq & (war < total_width)
                    pos = window_start + war
                    phys_win_01 = tl.where(
                        pos < global_start,
                        pos - (global_start - d_window),
                        d_window + pos - global_start,
                    )
                    tl.store(
                        topk_idxs_ptr + row * total_width + war,
                        tl.where(ok, phys_win_01, -1),
                        mask=war < total_width,
                    )

                if MODE == 0:
                    # Selected top-k: compact only the valid compressed entries
                    # right after the window (column of entry j = window_count +
                    # #valid before j), matching the CuTe write-col loop.
                    cids_0 = tl.arange(0, C_BLOCK)
                    cmask_0 = cids_0 < compressed_width
                    cid_0 = tl.load(
                        compressed_topk_ptr + row * compressed_width + cids_0,
                        mask=cmask_0,
                        other=-1,
                    )
                    seq_comp_start_0 = tl.load(seq_comp_start_ptr + row)
                    seq_comp_len_0 = tl.load(seq_comp_len_ptr + row)
                    comp_valid_0 = in_seq & (cid_0 >= 0) & (cid_0 < seq_comp_len_0)
                    seq_major_0 = seq_comp_start_0 + cid_0
                    safe_major_0 = tl.where(seq_major_0 < 0, 0, seq_major_0)
                    safe_major_0 = tl.where(
                        safe_major_0 < seq_major_rows,
                        safe_major_0,
                        seq_major_rows - 1,
                    )
                    rank_row_0 = tl.load(
                        seq_to_rank_row_ptr + safe_major_0,
                        mask=cmask_0 & (safe_major_0 >= 0),
                        other=-1,
                    )
                    rank_valid_0 = (
                        comp_valid_0
                        & (seq_major_0 < seq_major_rows)
                        & (rank_row_0 >= 0)
                    )
                    prefix_0 = tl.cumsum(rank_valid_0.to(tl.int32), axis=0)
                    target_0 = window_count + prefix_0 - 1
                    phys_comp_0 = compressed_base + rank_row_0
                    tl.store(
                        topk_idxs_ptr + row * total_width + target_0,
                        tl.where(rank_valid_0 & cmask_0, phys_comp_0, -1),
                        mask=rank_valid_0 & cmask_0 & (target_0 < total_width),
                    )
                    comp_len_0 = tl.sum(rank_valid_0.to(tl.int32), axis=0)
                    length_0 = window_count + comp_len_0
                    fallback_length_0 = tl.where(total_width > 0, 1, 0)
                    tl.store(
                        topk_length_ptr + row,
                        tl.where(in_seq, length_0, fallback_length_0),
                    )
                else:
                    # Mode 1: every causal-visible compressed row, in a fixed
                    # block starting at window_count.
                    seq_comp_start_1 = tl.load(seq_comp_start_ptr + row)
                    seq_comp_len_1 = tl.load(seq_comp_len_ptr + row)
                    causal_count_1 = (q - seq_start + 1) // ratio
                    comp_count_1 = tl.minimum(
                        tl.minimum(causal_count_1, compressed_width), seq_comp_len_1
                    )
                    comp_count_1 = tl.where(in_seq & (ratio > 1), comp_count_1, 0)
                    cids_1 = tl.arange(0, C_BLOCK)
                    cwidth_mask_1 = cids_1 < compressed_width
                    causal_mask_1 = (cids_1 < comp_count_1) & in_seq
                    seq_major_1 = seq_comp_start_1 + cids_1
                    safe_major_1 = tl.where(
                        seq_major_1 < seq_major_rows,
                        seq_major_1,
                        seq_major_rows - 1,
                    )
                    rank_row_1 = tl.load(
                        seq_to_rank_row_ptr + safe_major_1,
                        mask=causal_mask_1 & (seq_major_1 < seq_major_rows),
                        other=-1,
                    )
                    rank_ok_1 = (
                        causal_mask_1
                        & (seq_major_1 < seq_major_rows)
                        & (rank_row_1 >= 0)
                    )
                    phys_comp_1 = compressed_base + rank_row_1
                    target_1 = window_count + cids_1
                    tl.store(
                        topk_idxs_ptr + row * total_width + target_1,
                        tl.where(rank_ok_1, phys_comp_1, -1),
                        mask=cwidth_mask_1 & (target_1 < total_width),
                    )
                    length_1 = window_count + comp_count_1
                    fallback_length_1 = tl.where(total_width > 0, 1, 0)
                    tl.store(
                        topk_length_ptr + row,
                        tl.where(in_seq, length_1, fallback_length_1),
                    )

                # Rows outside every sequence (CP capacity padding / tail)
                # use index 0 when the output has at least one column.
                tl.store(
                    topk_idxs_ptr + row * total_width,
                    0,
                    mask=(~in_seq) & (total_width > 0),
                )


# =============================================================================
# Host-side wrappers
# =============================================================================


def _launch_compaction_fwd(
    src_cat: torch.Tensor,
    src_flat: torch.Tensor,
    valid: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Launch the Triton compaction forward (grid = one program / compact row)."""
    compact_len, W = out.shape
    src_cat = src_cat.contiguous()
    src_flat = src_flat.contiguous()
    valid = valid.contiguous()
    _compaction_fwd_kernel[(compact_len,)](
        src_cat,
        src_flat,
        valid,
        out,
        compact_len,
        W,
        W_BLOCK=_next_pow2(max(1, W)),
        num_warps=4,
    )


def _launch_compaction_bwd(
    grad_compact: torch.Tensor,
    back_idx: torch.Tensor,
    grad_out: torch.Tensor,
) -> None:
    """Launch the Triton compaction backward (one program / output row)."""
    total_rows, W = grad_out.shape
    grad_compact = grad_compact.contiguous()
    back_idx = back_idx.contiguous()
    _compaction_bwd_kernel[(total_rows,)](
        grad_compact,
        back_idx,
        grad_out,
        total_rows,
        W,
        W_BLOCK=_next_pow2(max(1, W)),
        num_warps=4,
    )


class _TritonCompressorInputCompact(torch.autograd.Function):
    """Autograd compaction backed by the Triton forward/backward kernels.

    Tensor contract matches ``cp_kernel.CompressorInputCompact`` and the
    pure-PyTorch ``CompressorInputCompact`` reference: forward returns
    ``(hidden_compact, comp_ids)`` and backward scatters compact gradients back
    to local and boundary hidden rows (accumulating when the source mapping is
    not one-to-one, via the PyTorch reference).
    """

    @staticmethod
    def forward(
        ctx,
        hidden_local: torch.Tensor,
        boundary_hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        global_start: int,
        ratio: int,
        d_comp: int,
        c_cap: int,
    ):
        l_local = hidden_local.shape[0]
        d_window = boundary_hidden.shape[0]
        W = math.prod(hidden_local.shape[1:])
        ctx.hidden_shape = tuple(hidden_local.shape)
        ctx.boundary_shape = tuple(boundary_hidden.shape)
        ctx.compact_args = (
            int(global_start),
            int(l_local),
            int(ratio),
            int(d_comp),
            int(d_window),
        )
        ctx.save_for_backward(cu_seqlens)

        src_global, comp_ids, valid = dsa_layout._compact_row_to_source(
            cu_seqlens, global_start, l_local, ratio, d_comp, c_cap
        )
        flat = dsa_layout._flat_source_index(src_global, valid, global_start, d_window)

        hidden_flat = hidden_local.reshape(l_local, W)
        boundary_flat = boundary_hidden.reshape(d_window, W)
        zero_row = torch.zeros((1, W), dtype=hidden_local.dtype, device=hidden_local.device)
        src_cat = torch.cat((boundary_flat, hidden_flat, zero_row), dim=0)

        compact_len = int(c_cap) * int(ratio)
        hidden_compact = torch.empty(
            (compact_len, W), dtype=hidden_local.dtype, device=hidden_local.device
        )
        _launch_compaction_fwd(src_cat, flat, valid, hidden_compact)
        hidden_compact = hidden_compact.reshape((compact_len,) + tuple(hidden_local.shape[1:]))
        return hidden_compact, comp_ids

    @staticmethod
    def backward(ctx, grad_hidden_compact: torch.Tensor, _grad_comp_ids: torch.Tensor):
        (cu_seqlens,) = ctx.saved_tensors
        global_start, l_local, ratio, d_comp, d_window = ctx.compact_args
        compact_len = grad_hidden_compact.shape[0]
        W = math.prod(ctx.hidden_shape[1:])
        grad_compact = grad_hidden_compact.reshape(compact_len, W)
        total_rows = l_local + d_window
        range_start = int(global_start)

        src_global, _, valid = dsa_layout._compact_row_to_source(
            cu_seqlens, global_start, l_local, ratio, d_comp, compact_len // ratio
        )
        compact_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1).to(dtype=torch.int64)
        if compact_idx.numel() == 0:
            grad_hidden = grad_compact.new_zeros(ctx.hidden_shape)
            grad_boundary = grad_compact.new_zeros(ctx.boundary_shape)
            return grad_hidden, grad_boundary, None, None, None, None, None

        srcs = src_global[valid]
        out_row = torch.where(
            srcs < range_start,
            srcs - (range_start - d_window),
            d_window + (srcs - range_start),
        ).to(dtype=torch.int64)

        back_idx = torch.full((total_rows,), -1, dtype=torch.int32, device=grad_compact.device)
        grad_out = torch.empty(
            (total_rows, W), dtype=grad_compact.dtype, device=grad_compact.device
        )

        unique_count = torch.unique(out_row).numel()
        if unique_count == out_row.numel():
            # One-to-one mapping: Triton scatter (atomic-free fast path).
            back_idx[out_row] = compact_idx.to(dtype=torch.int32)
            _launch_compaction_bwd(grad_compact, back_idx, grad_out)
        else:
            # Repeated source row (defensive): accumulating PyTorch reference.
            # Contributions must be split by ownership first: local rows index
            # ``grad_hidden`` and boundary rows index ``grad_boundary``; using
            # the combined row map on either output would go out of bounds.
            contrib = grad_compact.index_select(0, compact_idx)
            is_boundary = srcs < range_start
            local_row = (srcs[~is_boundary] - range_start).to(dtype=torch.int64)
            boundary_row = (srcs[is_boundary] - (range_start - d_window)).to(
                dtype=torch.int64
            )
            grad_hidden = torch.zeros(
                (l_local, W), dtype=grad_compact.dtype, device=grad_compact.device
            )
            grad_boundary = torch.zeros(
                (d_window, W), dtype=grad_compact.dtype, device=grad_compact.device
            )
            if local_row.numel():
                grad_hidden = grad_hidden.index_add(0, local_row, contrib[~is_boundary])
            if boundary_row.numel():
                grad_boundary = grad_boundary.index_add(
                    0, boundary_row, contrib[is_boundary]
                )
            grad_out = torch.cat((grad_boundary, grad_hidden), dim=0)

        grad_hidden = grad_out[d_window:].reshape(ctx.hidden_shape)
        grad_boundary = grad_out[:d_window].reshape(ctx.boundary_shape)
        return grad_hidden, grad_boundary, None, None, None, None, None


def compress_compressor_input(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress local+boundary hidden rows into fixed-capacity compressor input.

    Uses the Triton compaction kernels when available; otherwise falls back to
    the core PyTorch reference (``csa_utils.cp_layout``), which is the portable
    correctness truth.
    """
    if not _TRITON_AVAILABLE:
        return dsa_layout.compact_compressor_input(
            hidden_local,
            boundary_hidden,
            cu_seqlens,
            global_start,
            ratio,
            d_comp,
            c_cap,
        )
    return _TritonCompressorInputCompact.apply(
        hidden_local,
        boundary_hidden,
        cu_seqlens,
        global_start,
        ratio,
        d_comp,
        c_cap,
    )


def _row_metadata_torch(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row sequence metadata consumed by the attention-indices kernel.

    Returns ``(seq_start, seq_comp_start, seq_comp_len, window_start,
    window_count)`` --- each ``(l_local,)`` int32.  ``seq_start == -1`` flags a
    row outside every sequence (CP capacity padding / truncated tail).  This is
    the portable semantic layer (plan §2.1): the kernel only lowers indices.
    """
    global_start = int(global_start)
    l_local = int(l_local)
    n_seq = int(cu_seqlens.shape[0]) - 1
    device = cu_seqlens.device
    global_q = torch.arange(global_start, global_start + l_local, device=device, dtype=torch.int64)
    seq_ids = torch.bucketize(global_q, cu_seqlens[1:], right=True).clamp_max(max(n_seq - 1, 0))
    seq_start = cu_seqlens[seq_ids].to(dtype=torch.int32)
    seq_end = cu_seqlens[seq_ids + 1].to(dtype=torch.int32)
    in_seq = (global_q >= seq_start.to(dtype=torch.int64)) & (
        global_q < seq_end.to(dtype=torch.int64)
    )

    seq_comp_start = cu_seqlens_compressed[seq_ids].to(dtype=torch.int32)
    seq_comp_len = (
        cu_seqlens_compressed[seq_ids + 1] - cu_seqlens_compressed[seq_ids]
    ).to(dtype=torch.int32)

    window_start = (global_q - int(window_size) + 1).clamp(min=seq_start.to(dtype=torch.int64))
    window_start = torch.where(in_seq, window_start, seq_start.to(dtype=torch.int64))
    window_count = (global_q - window_start + 1).clamp_min(0).to(dtype=torch.int32)

    seq_start_out = torch.where(in_seq, seq_start, torch.full_like(seq_start, -1))
    return (
        seq_start_out,
        seq_comp_start,
        seq_comp_len,
        window_start.to(dtype=torch.int32),
        window_count,
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
    """Lower logical CP indices into physical attention indices (Triton).

    Same contract as ``csa_utils.cp_layout.build_attention_indices`` and the
    CuTeDSL kernel: three modes (0 selected top-k, 1 all visible compressed rows, 2 indexer loss),
    physical space ``cat((boundary, local, compressed_rank_major))``.  Falls
    back to the PyTorch reference when Triton is unavailable.
    """
    if not _TRITON_AVAILABLE:
        return dsa_layout.build_attention_indices(
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

    global_start = int(global_start)
    l_local = int(l_local)
    d_window = int(d_window)
    window_size = int(window_size)
    ratio = int(ratio)
    compressed_width = int(compressed_width)
    device = cu_seqlens.device

    # Preserve the caller-visible mode before replacing optional inputs with
    # valid dummy device pointers for the Triton launch.  Computing this after
    # the replacement would make ``compressed_topk is None`` permanently
    # false and incorrectly dispatch mode 1 (all visible compressed rows) as
    # mode 0 (caller-selected top-k).
    mode = 2 if for_indexer_loss else int(compressed_topk is None)

    if cu_seqlens_compressed is None:
        cu_seqlens_compressed = cu_seqlens
    if seq_to_rank_row is None:
        seq_to_rank_row = torch.full((1,), -1, dtype=torch.int32, device=device)
    if compressed_topk is None:
        compressed_topk = torch.empty((1, 1), dtype=torch.int32, device=device)

    total_width = window_size + compressed_width

    seq_start, seq_comp_start, seq_comp_len, window_start, window_count = _row_metadata_torch(
        cu_seqlens, cu_seqlens_compressed, global_start, l_local, window_size
    )

    topk_idxs = torch.full((l_local, total_width), -1, dtype=torch.int32, device=device)
    if mode == 2:
        topk_length = torch.empty((0,), dtype=torch.int32, device=device)
        indexer_rank_major = torch.full(
            (l_local, compressed_width), -1, dtype=torch.int32, device=device
        )
    else:
        topk_length = torch.empty((l_local,), dtype=torch.int32, device=device)
        indexer_rank_major = torch.empty((1,), dtype=torch.int32, device=device)

    seq_major_rows = seq_to_rank_row.shape[0]
    compressed_base = d_window + l_local
    SW_BLOCK = _next_pow2(max(1, min(window_size, 256)))
    WIN_UNITS = max(1, (total_width + SW_BLOCK - 1) // SW_BLOCK)
    C_BLOCK = _next_pow2(max(1, compressed_width))

    _build_attention_indices_kernel[(l_local,)](
        seq_start,
        seq_comp_start,
        seq_comp_len,
        window_start,
        window_count,
        seq_to_rank_row,
        compressed_topk,
        topk_idxs,
        topk_length,
        indexer_rank_major,
        global_start,
        d_window,
        compressed_base,
        seq_major_rows,
        l_local,
        total_width,
        ratio,
        compressed_width,
        SW_BLOCK=SW_BLOCK,
        WIN_UNITS=WIN_UNITS,
        C_BLOCK=C_BLOCK,
        MODE=mode,
        num_warps=4,
    )

    if mode == 2:
        return topk_idxs, None, indexer_rank_major
    return topk_idxs, topk_length, None


__all__ = ["build_attention_indices", "compress_compressor_input"]
