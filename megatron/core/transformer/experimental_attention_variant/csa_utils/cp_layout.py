# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Portable PyTorch index and contiguous-CP layout helpers for DSv4.

Only helpers used by the current DSv4 model path and its selectable kernel
backends live here: flat sparse-attention index conversion, compressor-input
compaction, and final attention-index lowering.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "build_attention_indices",
    "build_flat_topk_idxs",
    "compact_compressor_input",
    "compressor_input_compact",
]

# =============================================================================
# from LM: csa_utils/fused_sparse_attention.py 鈥?flat index helpers
# =============================================================================
def batch_of_row(cu_seqlens_q: Tensor, total_q: Optional[int] = None) -> Tensor:
    """For a THD-packed query of length ``total_q``, return a ``(total_q,)``
    int64 tensor where entry ``i`` is the index of the segment that owns
    query row ``i`` (i.e. the unique ``b`` with
    ``cu_seqlens_q[b] <= i < cu_seqlens_q[b+1]``).

    When ``total_q`` exceeds ``cu_seqlens_q[-1]`` (e.g. after padding token
    tensors to a static CUDA-graph capacity), orphan rows are clamped to the
    last segment so the returned indices are always in ``[0, B-1]``.

    Args:
        cu_seqlens_q: ``(B+1,)`` int 鈥?cumulative Q lengths.
        total_q: optional row count override; defaults to
            ``int(cu_seqlens_q[-1].item())`` (forces a GPU鈫扖PU sync).

    Returns:
        ``(total_q,)`` int64.
    """
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    num_sequences = cu_seqlens_q.shape[0] - 1
    row_idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens_q[1:], right=True).clamp(
        max=max(num_sequences - 1, 0)
    )


def local_to_global_flat(
    local_idxs: Tensor,
    batch_size: int,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
) -> Tensor:
    """Convert local per-sequence indices to global flat indices.

    Follows the convention used by FlashMLA / SparseAttentionBackward:
    two layouts are supported:

    * **SBHD-flat** (default, ``cu_seqlens_*=None``): flat row order is
      ``row[s * B + b]``; global index is ``local * B + b`` for valid entries
      and ``-1`` otherwise. Inputs are ``(b, sq, topk)``; outputs are
      ``(sq*b, topk)``.
    * **THD packed** (``cu_seqlens_*`` supplied): both must be 1-D int32
      tensors of length ``B+1``. Flat row order is the natural ``(total_q,)``
      order; global index is ``cu_seqlens_kv[batch_of_q] + local`` for valid
      entries. Inputs are ``(total_q, topk)``; outputs are ``(total_q, topk)``.

    Args:
        local_idxs: SBHD ``(b, sq, topk)`` or THD ``(total_q, topk)`` int.
        batch_size: ``B`` (only consulted in the SBHD branch).
        cu_seqlens_q: optional 1-D ``(B+1,)`` int32 鈥?THD branch selector.
        cu_seqlens_kv: optional 1-D ``(B+1,)`` int32 鈥?same.
    """
    if (cu_seqlens_q is None) != (cu_seqlens_kv is None):
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_kv must both be provided for THD, or "
            "both None for SBHD."
        )

    if cu_seqlens_q is None:
        b, sq, topk = local_idxs.shape

        idxs_sb = local_idxs.permute(1, 0, 2).reshape(sq * b, topk)
        valid = idxs_sb >= 0
        batch_ids = torch.arange(sq * b, device=local_idxs.device) % b
        batch_ids_exp = batch_ids.unsqueeze(1).expand_as(idxs_sb)
        idxs_sb = torch.where(valid, idxs_sb * b + batch_ids_exp, idxs_sb)
        return idxs_sb.int()

    if local_idxs.ndim != 2:
        raise ValueError(f"THD local_idxs must be 2-D (total_q, topk), got {local_idxs.shape}")
    total_q, topk = local_idxs.shape
    if cu_seqlens_q.ndim != 1 or cu_seqlens_kv.ndim != 1:
        raise ValueError("cu_seqlens_q/kv must be 1-D")
    if cu_seqlens_q.shape != cu_seqlens_kv.shape:
        raise ValueError(
            f"cu_seqlens_q.shape={tuple(cu_seqlens_q.shape)} must equal "
            f"cu_seqlens_kv.shape={tuple(cu_seqlens_kv.shape)}"
        )

    row_batch_ids = batch_of_row(cu_seqlens_q, total_q=total_q)
    kv_offset = cu_seqlens_kv[row_batch_ids].unsqueeze(1)  # (total_q, 1)
    valid = local_idxs >= 0
    global_idxs = torch.where(valid, local_idxs + kv_offset, local_idxs)
    return global_idxs.int()


def _compact_flat_topk_idxs(global_idxs: Tensor) -> Tuple[Tensor, Tensor]:
    """Pack valid global indices into a per-row prefix (PyTorch reference).

    The returned ``topk_length`` selects that prefix in fused forward /
    backward paths. Invalid suffix entries remain ``-1``.
    """
    if global_idxs.ndim != 2:
        raise ValueError(f"global_idxs must be 2-D (rows, topk), got {tuple(global_idxs.shape)}")

    valid_mask = global_idxs >= 0
    sorted_indices = valid_mask.int().argsort(dim=-1, descending=True, stable=True)
    compact_idxs = global_idxs.gather(-1, sorted_indices)
    topk_length = valid_mask.sum(dim=-1).int()
    return compact_idxs.int().contiguous(), topk_length.int().contiguous()


def build_flat_topk_idxs(
    *idx_groups: Tensor,
    batch_size: int,
    compact: bool = False,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Combine local per-sequence index groups and convert to flat global form.

    Each *idx_group* contains local per-sequence KV indices (already in
    ``kv_full`` index space, i.e. with any compressed-position offset
    applied). ``-1`` marks invalid positions. The shape of each group
    differs by layout:

    * **SBHD-flat** (``cu_seqlens_*=None``, default): each group is
      ``(b, sq, topk_i)``; outputs are ``(sq*b, total_topk)``.
    * **THD packed** (``cu_seqlens_*`` supplied): each group is
      ``(total_q, topk_i)``; outputs are ``(total_q, total_topk)``.

    Args:
        *idx_groups: one or more index tensors, all of the same layout.
        batch_size: ``B`` (only consulted in SBHD).
        compact: if True, pack valid entries to the front of each row and
            additionally return ``topk_length``; if False, leave as-is and
            return ``None``.
        cu_seqlens_q: optional 1-D ``(B+1,)`` int32 鈥?selects THD branch.
        cu_seqlens_kv: optional 1-D ``(B+1,)`` int32 鈥?selects THD branch.

    Returns:
        ``(topk_idxs, topk_length)`` where the first axis of ``topk_idxs``
        is ``sq*b`` (SBHD) or ``total_q`` (THD), and ``topk_length`` is
        ``(rows,)`` int32 when ``compact``, else ``None``.
    """
    combined = torch.cat(idx_groups, dim=-1)
    global_idxs = local_to_global_flat(
        combined, batch_size, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv
    )

    topk_length_flat = None
    if compact:
        global_idxs, topk_length_flat = _compact_flat_topk_idxs(global_idxs)
    return global_idxs, topk_length_flat

# =============================================================================
# From LM: the CuTeDSL ``cp_kernel.py`` contract -> pure-PyTorch CP layout.
# port. Keeps the tensor contracts used by the optional plugin kernels.
# =============================================================================



def _visible_group_offsets(
    cu_seqlens: torch.Tensor, global_start: int, l_local: int, ratio: int, d_comp: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized per-sequence visible compressed-group ranges.

    Mirrors the serial loop inside ``_compressor_input_compact_fwd_kernel``.
    Returns ``(first_visible_group, visible_group_count,
    total_visible_tokens)``.
    """
    seq_starts = cu_seqlens[:-1].to(dtype=torch.int64)
    seq_ends = cu_seqlens[1:].to(dtype=torch.int64)
    range_start = int(global_start)
    range_end = range_start + int(l_local)
    first_range_group_start = range_start - int(d_comp)

    local_seq_ends = torch.minimum(seq_ends, torch.full_like(seq_ends, range_end))
    intersects = (seq_starts < local_seq_ends) & (
        torch.as_tensor(range_start, device=cu_seqlens.device, dtype=torch.int64)
        < local_seq_ends
    )

    first_visible_numer = (first_range_group_start - seq_starts).clamp_min(0)
    first_visible_group = torch.div(
        first_visible_numer + (ratio - 1), ratio, rounding_mode="floor"
    )
    stop_visible_group = torch.div(local_seq_ends - seq_starts, ratio, rounding_mode="floor")
    visible_group_count = (stop_visible_group - first_visible_group).clamp_min(0)
    visible_group_count = torch.where(
        intersects, visible_group_count, torch.zeros_like(visible_group_count)
    )

    visible_token_count = visible_group_count * ratio
    total_visible_tokens = torch.sum(visible_token_count)
    return first_visible_group, visible_group_count, total_visible_tokens


def _compact_row_to_source(
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(src_global, comp_ids, valid)`` for every compact row.

    ``src_global`` is the global-hidden row each compact row copies from, or
    ``-1`` when the row is beyond the visible token budget (fixed-capacity
    padding). ``comp_ids`` holds each ratio-th row's compressed-group id
    (``-1`` for padding). ``valid`` flags compact rows that carry real data.

    Vectorized version of the ``tidx == 0`` thread-serial loop in the CuTe fwd
    kernel: each compact row is located inside its sequence's visible-token
    interval by binary-searching the running-token offsets.
    """
    ratio = int(ratio)
    compact_len = int(c_cap) * ratio
    device = cu_seqlens.device

    first_vis_group, visible_group_count, total_visible = _visible_group_offsets(
        cu_seqlens, global_start, l_local, ratio, d_comp
    )
    n_seq = first_vis_group.shape[0]

    rows = torch.arange(compact_len, device=device, dtype=torch.int64)
    valid = rows < total_visible
    # Running *ends* of each sequence's visible tokens; bucketize with
    # right=True counts boundaries <= row, which is exactly the sequence id.
    running_ends = torch.cumsum(visible_group_count * ratio, dim=0)
    seq_ids = torch.bucketize(rows, running_ends, right=True).clamp_max(max(n_seq - 1, 0))

    local = rows - (running_ends[seq_ids] - visible_group_count[seq_ids] * ratio)
    local = torch.where(valid, local, torch.zeros_like(local))
    comp_id = first_vis_group[seq_ids] + torch.div(local, ratio, rounding_mode="floor")
    token_in_group = local - torch.div(local, ratio, rounding_mode="floor") * ratio
    valid_in_seq = local < visible_group_count[seq_ids] * ratio
    effective_valid = valid & valid_in_seq

    src_global = cu_seqlens[seq_ids].to(dtype=torch.int64) + comp_id * ratio + token_in_group
    src_global = torch.where(effective_valid, src_global, torch.full_like(src_global, -1))

    comp_ids = torch.full((int(c_cap),), -1, dtype=torch.int32, device=device)
    group_ids = torch.div(rows, ratio, rounding_mode="floor")
    leading = effective_valid & (rows % ratio == 0)
    comp_ids[group_ids[leading].clamp_max(int(c_cap) - 1)] = comp_id[leading].to(
        dtype=torch.int32
    )
    return src_global, comp_ids, effective_valid


def _flat_source_index(
    src_global: torch.Tensor,
    valid: torch.Tensor,
    global_start: int,
    d_window: int,
) -> torch.Tensor:
    """Map global hidden-row ids onto ``cat((boundary, local, zero))`` indices.

    Rows before ``global_start`` index the boundary block; rows at/after index
    the local block. Invalid rows map to 0 and are masked by the caller.
    """
    range_start = int(global_start)
    is_boundary = src_global < range_start
    flat = torch.where(
        is_boundary,
        src_global - (range_start - int(d_window)),
        int(d_window) + (src_global - range_start),
    )
    flat = torch.where(valid, flat, torch.zeros_like(flat)).clamp_min(0)
    return flat.to(dtype=torch.int64)


class CompressorInputCompact(torch.autograd.Function):
    """Compress a CP rank's local+boundary hidden rows into fixed-capacity
    compressor input (pure-PyTorch reference).

    Drop-in tensor-contract replacement for
    ``cp_kernel.CompressorInputCompact``: forward returns
    ``(hidden_compact, comp_ids)`` with ``hidden_compact`` of shape
    ``(c_cap * ratio, *trailing)``; backward scatters compact gradients back
    to the local and boundary hidden rows with accumulation.
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
        ctx.hidden_shape = tuple(hidden_local.shape)
        ctx.boundary_shape = tuple(boundary_hidden.shape)
        ctx.compact_args = (int(global_start), int(l_local), int(ratio), int(d_comp), int(d_window))
        ctx.save_for_backward(cu_seqlens)

        src_global, comp_ids, valid = _compact_row_to_source(
            cu_seqlens, global_start, l_local, ratio, d_comp, c_cap
        )

        zero_row = torch.zeros(
            (1,) + tuple(hidden_local.shape[1:]), dtype=hidden_local.dtype, device=hidden_local.device
        )
        src_cat = torch.cat((boundary_hidden, hidden_local, zero_row), dim=0)
        flat = _flat_source_index(src_global, valid, global_start, d_window)
        gathered = torch.index_select(src_cat, 0, flat)
        valid_shape = (valid.shape[0],) + (1,) * (gathered.ndim - 1)
        hidden_compact = torch.where(valid.view(valid_shape), gathered, zero_row)
        return hidden_compact, comp_ids

    @staticmethod
    def backward(ctx, grad_hidden_compact, _grad_comp_ids):
        (cu_seqlens,) = ctx.saved_tensors
        global_start, l_local, ratio, d_comp, d_window = ctx.compact_args
        grad_hidden = grad_hidden_compact.new_zeros(ctx.hidden_shape)
        grad_boundary = grad_hidden_compact.new_zeros(ctx.boundary_shape)

        src_global, _, valid = _compact_row_to_source(
            cu_seqlens,
            global_start,
            l_local,
            ratio,
            d_comp,
            grad_hidden_compact.shape[0] // ratio,
        )
        range_start = int(global_start)
        compact_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1).to(dtype=torch.int64)
        if compact_idx.numel() == 0:
            return (
                grad_hidden,
                grad_boundary,
                None,
                None,
                None,
                None,
                None,
            )
        contrib = grad_hidden_compact.index_select(0, compact_idx)
        src_global = src_global[valid]

        # Split contributions by ownership: boundary rows scatter into
        # ``grad_boundary`` and local rows into ``grad_hidden``.  Indexing both
        # output tensors with the *combined* row maps is invalid (a local row's
        # ``boundary_row`` exceeds the boundary extent), so each map is applied
        # only to its own rows. Repeated source rows still accumulate.
        is_boundary = src_global < range_start
        local_row = (src_global[~is_boundary] - range_start).to(dtype=torch.int64)
        grad_hidden = grad_hidden.index_add(0, local_row, contrib[~is_boundary])
        boundary_row = (src_global[is_boundary] - (range_start - d_window)).to(
            dtype=torch.int64
        )
        grad_boundary = grad_boundary.index_add(0, boundary_row, contrib[is_boundary])

        return (
            grad_hidden,
            grad_boundary,
            None,
            None,
            None,
            None,
            None,
        )


def compressor_input_compact(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    ratio: int,
    d_comp: int,
    c_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public entry point for compressor-input compaction (PyTorch reference)."""
    return CompressorInputCompact.apply(
        hidden_local, boundary_hidden, cu_seqlens, global_start, ratio, d_comp, c_cap
    )


# Backend-neutral dispatcher contract. Keep the original name above for
# compatibility with the core unfused path.
compact_compressor_input = compressor_input_compact


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
    """Build final sparse-attention / indexer-loss indices (PyTorch reference).

    Vectorized tensor-contract drop-in for
    ``cp_kernel.build_attention_indices``.

    Three modes (mirroring the CuTe kernel's ``index_mode``):

    * ``mode 0`` (selected top-k): ``compressed_topk`` is ``(l_local,
      compressed_width)`` of per-sequence compressed ids; window ids are
      written first, then the selected compressed ids.
    * ``mode 1`` (all visible compressed rows): ``compressed_topk is None``;
      window ids first, then every visible compressed id in sequence order.
    * ``mode 2`` (indexer loss): ``for_indexer_loss=True``; compressed ids
      first (and their rank-major rows returned in ``indexer_rank_major``),
      then window ids.

    Physical index space is ``kv_full_thd = cat((boundary, local,
    compressed_rank_major))``: boundary rows are ``[0, d_window)``, local rows
    ``[d_window, d_window + l_local)``, and compressed row ``r`` in the
    rank-major buffer at ``d_window + l_local + r``.
    """
    global_start = int(global_start)
    l_local = int(l_local)
    d_window = int(d_window)
    window_size = int(window_size)
    ratio = int(ratio)
    compressed_width = int(compressed_width)
    device = cu_seqlens.device
    n_seq = int(cu_seqlens.shape[0]) - 1

    if cu_seqlens_compressed is None:
        cu_seqlens_compressed = cu_seqlens
    if seq_to_rank_row is None:
        seq_to_rank_row = torch.full((1,), -1, dtype=torch.int32, device=device)
    if compressed_topk is None:
        compressed_topk = torch.empty((0, 0), dtype=torch.int32, device=device)

    total_width = window_size + compressed_width
    topk_idxs = torch.full((l_local, total_width), -1, dtype=torch.int32, device=device)
    topk_length = torch.full((l_local,), 0, dtype=torch.int32, device=device)
    indexer_rank_major = torch.full(
        (l_local, compressed_width) if compressed_width > 0 else (l_local, 0),
        -1,
        dtype=torch.int32,
        device=device,
    )

    # Per-query sequence id and starts.
    global_q = torch.arange(
        global_start, global_start + l_local, device=device, dtype=torch.int64
    )
    seq_ids = torch.bucketize(global_q, cu_seqlens[1:], right=True).clamp_max(n_seq - 1)
    seq_starts = cu_seqlens[seq_ids].to(dtype=torch.int64)
    seq_ends = cu_seqlens[seq_ids + 1].to(dtype=torch.int64)
    in_seq = (global_q >= seq_starts) & (global_q < seq_ends)
    seq_comp_starts = cu_seqlens_compressed[seq_ids].to(dtype=torch.int64)
    seq_comp_len = (
        cu_seqlens_compressed[seq_ids + 1] - cu_seqlens_compressed[seq_ids]
    ).to(dtype=torch.int64)

    # Window positions and physical rows.
    window_start = (global_q - window_size + 1).clamp(min=seq_starts)
    window_start = torch.where(in_seq, window_start, seq_starts)
    window_positions = window_start.unsqueeze(1) + torch.arange(
        window_size, device=device, dtype=torch.int64
    ).unsqueeze(0)
    window_valid = (
        (window_positions >= window_start.unsqueeze(1))
        & (window_positions <= global_q.unsqueeze(1))
        & in_seq.unsqueeze(1)
    )
    window_phys = torch.where(
        window_valid,
        _flatten_window_positions(window_positions, global_start, d_window),
        torch.full_like(window_positions, -1),
    )
    window_count = (global_q - window_start + 1).clamp_min(0).to(dtype=torch.int64)

    # If a query row is not in any sequence (CP capacity padding / truncated
    # tail), the CuTe kernel emits ``topk_length = 1`` with index 0.
    fallback_rows = ~in_seq

    # Per-query causal compressed bound (mode 1) or caller-selected ids (mode 0).
    if compressed_width > 0:
        if compressed_topk.shape[0] == l_local and compressed_topk.shape[1] == compressed_width:
            comp_ids = compressed_topk.to(dtype=torch.int64)  # (l_local, w)
            seq_major_ids = seq_comp_starts.unsqueeze(1) + comp_ids
            comp_valid = (comp_ids >= 0) & (comp_ids < seq_comp_len.unsqueeze(1))
        else:
            width_idx = torch.arange(compressed_width, device=device, dtype=torch.int64).unsqueeze(0)
            comp_count = torch.minimum(
                torch.clamp(
                    torch.div(global_q - seq_starts + 1, ratio, rounding_mode="floor"),
                    max=compressed_width,
                ),
                seq_comp_len,
            )
            comp_count = torch.where(in_seq, comp_count, torch.zeros_like(comp_count))
            comp_ids = width_idx.expand(l_local, -1)
            seq_major_ids = seq_comp_starts.unsqueeze(1) + width_idx
            comp_valid = width_idx < comp_count.unsqueeze(1)
        safe_major = seq_major_ids.clamp_min(0).clamp_max(int(seq_to_rank_row.shape[0]) - 1)
        rank_row = seq_to_rank_row[safe_major].to(dtype=torch.int64)
        in_rank_rows = seq_major_ids < int(seq_to_rank_row.shape[0])
        valid_rank = comp_valid & in_rank_rows & (rank_row >= 0) & in_seq.unsqueeze(1)
    else:
        comp_ids = torch.empty(0, device=device, dtype=torch.int64)
        seq_major_ids = torch.empty(0, device=device, dtype=torch.int64)
        rank_row = torch.empty(0, device=device, dtype=torch.int64)
        valid_rank = torch.empty(0, device=device, dtype=torch.bool)

    if for_indexer_loss:
        # Mode 2: compressed ids first, then window; rank-major rows returned.
        if compressed_width > 0:
            base = d_window + l_local
            comp_phys = torch.where(
                valid_rank,
                (base + rank_row).to(dtype=torch.int32),
                torch.full_like(topk_idxs[:, :compressed_width], -1),
            )
            topk_idxs[:, :compressed_width] = comp_phys
            indexer_rank_major = torch.where(
                valid_rank,
                rank_row.to(dtype=torch.int32),
                torch.full_like(indexer_rank_major, -1),
            )
        topk_idxs[:, compressed_width:] = window_phys.to(dtype=torch.int32)
        topk_idxs = torch.where(fallback_rows.unsqueeze(1), -1, topk_idxs)
        indexer_rank_major = torch.where(
            fallback_rows.unsqueeze(1), -1, indexer_rank_major
        )
        return topk_idxs, None, indexer_rank_major

    # Normal attention path (mode 0 / mode 1): window entries first (compact),
    # then compressed entries. Column layout matches the CuTe kernel.
    base = d_window + l_local
    if compressed_width > 0:
        comp_phys = torch.where(
            valid_rank,
            (base + rank_row).to(dtype=torch.int32),
            torch.full_like(topk_idxs[:, :compressed_width], -1),
        )
        selected_mode = (
            compressed_topk.shape[0] == l_local
            and compressed_topk.shape[1] == compressed_width
        )
        comp_len = (
            valid_rank.sum(dim=-1).to(dtype=torch.int64) if selected_mode else comp_count
        )
    else:
        comp_phys = torch.empty((l_local, 0), dtype=torch.int32, device=device)
        comp_len = torch.zeros(l_local, dtype=torch.int64, device=device)

    topk_length = (window_count + comp_len).clamp_max(total_width).to(dtype=torch.int32)

    # Window entries occupy columns [0, window_count); window_phys is -1 for
    # the trailing columns so the initial -1 fill stays intact there.
    window_w = torch.arange(window_size, device=device, dtype=torch.int64).unsqueeze(0)
    topk_idxs = topk_idxs.scatter(
        1, window_w.expand(l_local, -1), window_phys.to(dtype=torch.int32)
    )

    if compressed_width > 0:
        if selected_mode:
            # Mode 0 (selected top-k): compact only the valid compressed
            # entries right after the window (matching the CuTe write_col
            # loop). Column of valid entry j = window_count + (#valid before j).
            # Invalid entries must not occupy a column; route them to a padding
            # column (total_width) sliced off after the scatter, so a leading
            # -1 entry cannot clobber an already-written window entry.
            valid_prefix = torch.cumsum(valid_rank.to(dtype=torch.int64), dim=-1)
            compact_cols = window_count.unsqueeze(1) + valid_prefix - 1
            compact_cols = torch.where(valid_rank, compact_cols, total_width)
            topk_idxs = torch.cat(
                (topk_idxs, torch.full((l_local, 1), -1, dtype=torch.int32, device=device)),
                dim=1,
            )
            topk_idxs = topk_idxs.scatter(1, compact_cols, comp_phys)
            topk_idxs = topk_idxs[:, :total_width]
            valid_mask = valid_rank
        else:
            # Mode 1 (all visible): fixed positions [window_count,
            # window_count + compressed_width) in sequence order.
            compact_cols = window_count.unsqueeze(1) + torch.arange(
                compressed_width, device=device, dtype=torch.int64
            ).unsqueeze(0)
            valid_mask = comp_valid
            topk_idxs = topk_idxs.scatter(1, compact_cols, comp_phys)

    # Rows outside every sequence mirror the CuTe kernel's ``elif``:
    # topk_length = 1 and index 0 (when total_width > 0).
    if total_width > 0:
        fallback_indices = torch.full_like(topk_idxs, -1)
        fallback_indices[:, 0] = 0
        topk_idxs = torch.where(fallback_rows.unsqueeze(1), fallback_indices, topk_idxs)
    topk_length = torch.where(
        fallback_rows,
        torch.full_like(topk_length, 1 if total_width > 0 else 0),
        topk_length,
    )
    return topk_idxs, topk_length, None


def _flatten_window_positions(
    window_positions: torch.Tensor, global_start: int, d_window: int
) -> torch.Tensor:
    """Convert absolute window positions into physical rows of ``kv_full_thd``.

    Positions before ``global_start`` address the boundary block
    ``[0, d_window)``; positions at/after address local rows
    ``[d_window, d_window + l_local)``.
    """
    range_start = int(global_start)
    return torch.where(
        window_positions < range_start,
        window_positions - (range_start - int(d_window)),
        int(d_window) + (window_positions - range_start),
    )
