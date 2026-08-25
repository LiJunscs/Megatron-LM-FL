# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""
Triton-based DSA kernel wrappers 鈥?drop-in replacement for ``dsa_kernels.py``.

Provides the same high-level API as ``dsa_kernels.py`` but uses Triton kernels
instead of cuDNN DSA namespace and FlashMLA. No external CUDA kernel
dependencies required.

Public API:

* ``build_flat_topk_idxs`` / ``local_to_global_flat`` 鈥?index helpers.
* ``dsa_sparse_attn`` 鈥?differentiable sparse attention, flat layout (Path A / Path C step 2).
* ``dsa_sparse_attn_sbhd`` 鈥?sparse attention with SBHD interface (used by csa.py).
* ``indexer_topk`` 鈥?indexer scoring + top-K selection (Path C inference).
* ``fused_indexer_sparse_attn`` 鈥?fused indexer loss + sparse attention (Path B training).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.plugin.dsa_kernel.triton_indexer_kernels import (
    compute_sparse_indexer_predict_state,
    compute_sparse_local_target_head_sum,
    dense_attn_score_recompute,
    dense_indexer_score_recompute,
    fused_dense_indexer_loss_and_backward,
    fused_sparse_indexer_loss_and_backward,
    indexer_topk_selection,
    sparse_indexer_kl_and_backward,
    sparse_indexer_score_recompute,
)
from megatron.plugin.dsa_kernel.triton_sparse_attn import (
    triton_sparse_attn_backward,
    triton_sparse_attn_forward,
)
from megatron.plugin.dsa_kernel.triton_sparse_attn_bwd import fused_dkv, fused_dq, sorted_scatter_add


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profiling utilities (enabled via DSA_PROFILE=1 env var)
# ---------------------------------------------------------------------------

_DSA_PROFILE = os.environ.get("DSA_PROFILE", "0") == "1"

##### FlagScale Add #####
# TP overlap for sparse indexer loss: async all-reduce + predict overlap.
# Default off; enable after distributed correctness is validated.
_DSA_TP_OVERLAP = os.environ.get("MEGATRON_DSA_TP_OVERLAP", "0") == "1"

##### FlagScale End #####

class _CudaProfiler:
    """Lightweight CUDA event profiler for forward pass breakdown."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._events = []  # list of (name, start_event, end_event)
        self._current_start = None
        self._current_name = None

    def start(self, name: str):
        if not self.enabled:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._current_start = start
        self._current_name = name

    def stop(self):
        if not self.enabled or self._current_start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events.append((self._current_name, self._current_start, end))
        self._current_start = None
        self._current_name = None

    def report(self, prefix: str = ""):
        if not self.enabled or not self._events:
            return
        torch.cuda.synchronize()
        total = 0.0
        parts = []
        for name, start, end in self._events:
            elapsed = start.elapsed_time(end)
            total += elapsed
            parts.append(f"    {name}: {elapsed:.3f} ms")
        print(f"{prefix}FusedIndexerSparseAttn forward breakdown (total={total:.3f} ms):")
        for p in parts:
            print(p)
        self._events.clear()


# ---------------------------------------------------------------------------
# Index helpers (pure PyTorch, no kernel dependency)
# ---------------------------------------------------------------------------


def _batch_of_row(cu_seqlens: Tensor, total_rows: int) -> Tensor:
    """Map each packed row to its sequence without synchronizing with the host."""
    rows = torch.arange(total_rows, device=cu_seqlens.device, dtype=torch.int64)
    return torch.bucketize(rows, cu_seqlens[1:], right=True).clamp(
        max=max(cu_seqlens.shape[0] - 2, 0)
    )


def local_to_global_flat(
    local_idxs: Tensor,
    batch_size: int,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
) -> Tensor:
    """Convert per-sequence indices to the flat KV address space.

    SBHD uses sequence-major flattening (``local * B + batch``). THD rows are
    already packed, so each valid local index is shifted by its sequence's
    ``cu_seqlens_kv`` offset. The Triton attention kernel sees the same flat
    ``(rows, topk)`` result in either case.

    Args:
        local_idxs: ``(b, sq, topk)`` int, values in ``[0, seqlen_kv)`` or -1.
        batch_size: ``B``.
        cu_seqlens_q: THD cumulative query lengths, or ``None`` for SBHD.
        cu_seqlens_kv: THD cumulative KV lengths, or ``None`` for SBHD.

    Returns:
        ``(sq*b, topk)`` int32.
    """
    if (cu_seqlens_q is None) != (cu_seqlens_kv is None):
        raise ValueError("cu_seqlens_q and cu_seqlens_kv must be provided together")

    if cu_seqlens_q is not None:
        if local_idxs.ndim != 2:
            raise ValueError(f"THD local indices must be [T, K], got {tuple(local_idxs.shape)}")
        if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape != cu_seqlens_kv.shape:
            raise ValueError("THD cu_seqlens_q/kv must be matching one-dimensional tensors")
        row_batch = _batch_of_row(cu_seqlens_q, local_idxs.shape[0])
        offsets = cu_seqlens_kv[row_batch].unsqueeze(1)
        return torch.where(local_idxs >= 0, local_idxs + offsets, local_idxs).int()

    b, sq, topk = local_idxs.shape
    assert b == batch_size

    # Permute to SB order: (b, sq, topk) -> (sq, b, topk) -> (sq*b, topk)
    idxs_sb = local_idxs.permute(1, 0, 2).reshape(sq * b, topk)
    valid = idxs_sb >= 0
    batch_ids = torch.arange(sq * b, device=local_idxs.device) % b
    batch_ids_exp = batch_ids.unsqueeze(1).expand_as(idxs_sb)
    idxs_sb = torch.where(valid, idxs_sb * b + batch_ids_exp, idxs_sb)
    return idxs_sb.int()


def build_flat_topk_idxs(
    *idx_groups: Tensor,
    batch_size: int,
    compact: bool = False,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Combine local per-batch index groups and convert to flat global form.

    Drop-in replacement for ``dsa_kernels.build_flat_topk_idxs`` that uses
    PyTorch argsort for compact instead of cuDNN's ``compactify_wrapper``.

    Each *idx_group* is ``(b, sq, topk_i)`` with local per-batch KV indices.
    ``-1`` marks invalid positions.

    Args:
        *idx_groups: one or more ``(b, sq, topk_i)`` int tensors.
        batch_size: ``B``.
        compact: if True, pack valid entries to the front of each row and
            additionally return ``topk_length``; if False, leave as-is.
        cu_seqlens_q: THD cumulative query lengths, or ``None`` for SBHD.
        cu_seqlens_kv: THD cumulative KV lengths, or ``None`` for SBHD.

    Returns:
        ``(topk_idxs, topk_length)`` where
        ``topk_idxs`` is ``(sq*b, total_topk)`` int32 (flat global) and
        ``topk_length`` is ``(sq*b,)`` int32 when ``compact``, else ``None``.
    """
    combined = torch.cat(idx_groups, dim=-1)
    global_idxs = local_to_global_flat(
        combined,
        batch_size,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )

    topk_length_flat = None
    if compact:
        valid_mask = global_idxs >= 0
        sorted_indices = valid_mask.int().argsort(dim=-1, descending=True, stable=True)
        global_idxs = global_idxs.gather(-1, sorted_indices)
        topk_length_flat = valid_mask.sum(dim=-1).int()

    return global_idxs, topk_length_flat


# ---------------------------------------------------------------------------
# Helper: SBHD <-> BSHD conversions (matches dsa_kernels.py layout conventions)
# ---------------------------------------------------------------------------


def _sbhd_to_bshd_indexer_inputs(
    q_indexer: Tensor,  # (sq, b, idx_nh, idx_hd)
    k_indexer: Tensor,  # (sk, b, idx_hd)
    weights: Tensor,    # (sq, b, idx_nh)
    indexer_softmax_scale: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Transpose indexer inputs from SBHD to BSHD layout.

    Note: .contiguous() is omitted 鈥?downstream consumers (einsum, topk,
    elementwise ops) all handle strided tensors correctly.
    """
    q_bshd = q_indexer.permute(1, 0, 2, 3)   # (b, sq, nh, hd)
    k_bsd = k_indexer.permute(1, 0, 2)       # (b, sk, hd)
    w_bsh = weights.permute(1, 0, 2)         # (b, sq, nh)
    # Scale weights
    w_bsh_scaled = w_bsh * indexer_softmax_scale
    return q_bshd, k_bsd, w_bsh, w_bsh_scaled


def _packed_to_padded_sb(x: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
    """Scatter a THD tensor into sequence-major ``[S, B, ...]`` storage.

    This is layout glue only. It keeps the existing BSHD Triton indexer kernels
    unchanged while allowing variable-length packed batches to share their code.
    The scatter is differentiable, so gradients are gathered back to THD by
    autograd without a layout-specific kernel backward.
    """
    total_rows = x.shape[0]
    batch = cu_seqlens.shape[0] - 1
    row_batch = _batch_of_row(cu_seqlens, total_rows)
    row_pos = torch.arange(total_rows, device=x.device) - cu_seqlens[row_batch]
    padded = x.new_zeros((max_seqlen, batch, *x.shape[1:]))
    return padded.index_put((row_pos.long(), row_batch.long()), x)


def _padded_sb_to_packed(x: Tensor, cu_seqlens: Tensor, total_rows: int) -> Tensor:
    """Gather ``[S, B, ...]`` storage back into natural THD row order."""
    row_batch = _batch_of_row(cu_seqlens, total_rows)
    row_pos = torch.arange(total_rows, device=x.device) - cu_seqlens[row_batch]
    return x[row_pos.long(), row_batch.long()]


def _packed_full_kv_to_padded(
    kv_full: Tensor,
    cu_seqlens_kv: Tensor,
    cu_seqlens_kv_full: Tensor,
    max_seqlen_kv: int,
    max_seqlen_compressed: int,
) -> Tensor:
    """Repack per-sequence ``[original, compressed]`` THD KV into SBD.

    SBHD kernels require one constant compressed-region offset. Original KV is
    placed in ``[0, max_seqlen_kv)`` and compressed KV in the following region;
    holes are never addressed because local indices are length-masked.
    """
    total_rows = kv_full.shape[0]
    batch = cu_seqlens_kv_full.shape[0] - 1
    row_batch = _batch_of_row(cu_seqlens_kv_full, total_rows)
    local_pos = torch.arange(total_rows, device=kv_full.device) - cu_seqlens_kv_full[row_batch]
    original_lens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    row_original_len = original_lens[row_batch]
    padded_pos = torch.where(
        local_pos < row_original_len,
        local_pos,
        max_seqlen_kv + local_pos - row_original_len,
    )
    padded = kv_full.new_zeros(
        (max_seqlen_kv + max_seqlen_compressed, batch, *kv_full.shape[1:])
    )
    return padded.index_put((padded_pos.long(), row_batch.long()), kv_full)


def _masked_topk_from_scores(
    scores: Tensor,
    topk: int,
    valid_k_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Select a fixed-width top-K after applying per-sequence key lengths."""
    if valid_k_mask is not None:
        if valid_k_mask.ndim == 2:
            valid_k_mask = valid_k_mask[:, None, :]
        scores = scores.masked_fill(~valid_k_mask, float("-inf"))
    selected = min(topk, scores.shape[-1])
    if selected:
        values, indices = torch.topk(scores, selected, dim=-1, sorted=False)
        indices = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1)).int()
    else:
        indices = torch.empty(*scores.shape[:-1], 0, dtype=torch.int32, device=scores.device)
    if selected < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - selected), value=-1)
    return indices, (indices >= 0).sum(dim=-1).int()


def _indexer_topk_bshd(
    q_bshd: Tensor,    # (B, S_q, H_q, D)
    k_bsd: Tensor,     # (B, S_k, D)
    w_bsh: Tensor,     # (B, S_q, H_q) 鈥?already scaled
    topk: int,
    ratio: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute indexer scores and top-K selection.

    Returns:
        topk_indices: (B, S_q, topk) int32
        topk_length: (B, S_q) int32
        full_scores: (B, S_q, S_k) fp32
    """
    scores = indexer_topk_selection(q_bshd, k_bsd, w_bsh, topk, ratio)
    topk_indices = scores["topk_indices"]
    topk_length = scores["topk_length"]
    full_scores = scores["full_scores"]
    return topk_indices, topk_length, full_scores


# ---------------------------------------------------------------------------
# Path A / Path C: dsa_sparse_attn
# ---------------------------------------------------------------------------


class _DSASparseAttnFunc(torch.autograd.Function):
    """Differentiable sparse attention using pure Triton kernels.

    Forward dispatches to HP WGMMA or 2D-tiled Triton kernel (no PyTorch BMM).

    The backward adapts to match the forward path's dot-product method:
    - When forward used the HP WGMMA kernel (tl.dot f16脳f16鈫抐32), backward uses
      cuBLAS BMM with f16 inputs for numerically consistent score recomputation.
    - When forward used the 2D Triton kernel (tl.sum(q*k)), backward uses the
      Triton per-position kernel which shares the same accumulation order.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,     # (total_Sq, H, D)
        kv: Tensor,        # (total_Skv, D_kv) where D_kv >= D
        topk_idxs: Tensor, # (total_Sq, H, TopK) or (total_Sq, 1, TopK)
        softmax_scale: float,
        d_v: int,
        attn_sink: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        out, lse, _ = triton_sparse_attn_forward(
            query, kv, topk_idxs, softmax_scale, d_v, attn_sink
        )
        ctx.has_attn_sink = attn_sink is not None

        # Determine if the HP WGMMA kernel was used (shared + aligned dims).
        # Backward needs this to select numerically consistent score recomputation.
        D = query.shape[-1]
        H = topk_idxs.shape[1]
        shared = (topk_idxs.stride(1) == 0)

        ctx.used_hp_fwd = (
            shared and H >= 16 and (H % 16 == 0)
            and (D % 16 == 0) and (d_v % 16 == 0)
        )

        logger.debug(
            "_DSASparseAttnFunc.forward: total_Sq=%d, H=%d, TopK=%d, shared=%s, "
            "fwd_path=%s",
            topk_idxs.shape[0], H, topk_idxs.shape[-1], shared,
            "hp_wgmma" if ctx.used_hp_fwd else "triton_2d",
        )

        # Downstream fused inverse-RoPE mutates the returned output in-place.
        # Backward needs the original attention result for Di = sum(dO * O),
        # so retain an independent copy inside the DSA kernel boundary.
        out_for_backward = out.clone()
        if attn_sink is not None:
            ctx.save_for_backward(query, kv, topk_idxs, out_for_backward, lse, attn_sink)
        else:
            ctx.save_for_backward(query, kv, topk_idxs, out_for_backward, lse)
        ctx.softmax_scale = softmax_scale
        ctx.d_v = d_v
        return out, lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        if ctx.has_attn_sink:
            query, kv, topk_idxs, out, lse, attn_sink = ctx.saved_tensors
        else:
            query, kv, topk_idxs, out, lse = ctx.saved_tensors
            attn_sink = None

        logger.debug(
            "_DSASparseAttnFunc.backward: used_hp_fwd=%s, bwd_path=%s",
            ctx.used_hp_fwd,
            "bmm_f16" if ctx.used_hp_fwd else "triton",
        )

        if ctx.used_hp_fwd:
            # --- BMM backward with f16 score recomputation ---
            # HP forward used tl.dot(Q_f16, K_f16^T) 鈫?f32 accumulator.
            # cuBLAS BMM with f16 inputs also does f16脳f16鈫抐32 accumulation,
            # so exp(scores_bwd - lse_fwd) is numerically consistent.
            dq, dkv, d_sink = _DSASparseAttnFunc._hp_bmm_backward(
                grad_out, query, kv, topk_idxs, out, lse, attn_sink,
                ctx.softmax_scale, ctx.d_v,
            )
        else:
            # --- Triton backward: forward used 2D Triton kernel ---
            # Both fwd and bwd use tl.sum(q * k) 鈫?same dot product, consistent.
            bwd_result = triton_sparse_attn_backward(
                grad_out, query, kv, out, lse, topk_idxs,
                ctx.softmax_scale, ctx.d_v, attn_sink
            )
            dq, dkv, d_sink = bwd_result["dq"], bwd_result["dkv"], bwd_result["d_sink"]

        return dq, dkv, None, None, None, d_sink

    @staticmethod
    def _hp_bmm_backward(
        grad_out: Tensor,   # (total_Sq, H, d_v) bf16
        query: Tensor,      # (total_Sq, H, D) bf16
        kv: Tensor,         # (total_Skv, D_full) bf16
        topk_idxs: Tensor,  # (total_Sq, H, TopK) int32, shared (stride(1)==0)
        out: Tensor,        # (total_Sq, H, d_v) bf16
        lse: Tensor,        # (total_Sq, H) f32
        attn_sink: Optional[Tensor],  # (H,) f32 or None
        softmax_scale: float,
        d_v: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Fused backward for HP WGMMA forward path.

        Uses cuBLAS BMM with f16 inputs for score recomputation (numerical
        consistency with forward tl.dot), then fused Triton kernels for dQ
        and dKV that keep P, dov, dS in registers 鈥?eliminating ~768MB of
        intermediate GMEM allocations.

        For D==DV (shared latent, always true in training):
          - fused_dq: scores + lse + Di + dO + K 鈫?dQ (no P/dov/dS materialized)
          - fused_dkv: scores + lse + Di + dO + Q + K 鈫?dKV (no P/dov/dS materialized)
          - sorted_scatter_add: local reduction before atomic (less contention)
        """
        total_Sq, H, D = query.shape
        total_Skv = kv.shape[0]
        D_full = kv.shape[-1] if kv.dim() > 1 else D
        TopK = topk_idxs.shape[-1]

        # Shared indices: (total_Sq, H, TopK) with stride(1)==0 鈫?use head 0
        idxs_shared = topk_idxs[:, 0, :]  # (total_Sq, TopK)
        valid_shared = idxs_shared >= 0    # (total_Sq, TopK)
        safe_shared = idxs_shared.clamp(min=0).long()

        # Gather KV once in bf16
        flat_idxs = safe_shared.reshape(-1)  # (total_Sq * TopK)
        kv_gathered = kv[flat_idxs].reshape(total_Sq, TopK, D_full)  # bf16

        # Di = sum(dO * O) per (query, head)
        Di = (grad_out.float() * out.float()).sum(dim=-1)  # (total_Sq, H)

        # Recompute scores via cuBLAS BMM in bf16 鈥?matches HP WGMMA forward
        # (tl.dot(Q_bf16, K_bf16^T) 鈫?f32 accumulator, training dtype is bf16)
        scores = torch.bmm(
            query, kv_gathered[:, :, :D].transpose(1, 2)
        ).float() * softmax_scale  # (S, H, TopK) f32

        # --- Fused dQ: eliminates P, dov, dS materialization ---
        # dQ = sum_k( exp(scores-lse)*valid * (dO@K^T - Di) * scale ) @ K
        dq = fused_dq(
            scores, lse, Di, grad_out, kv_gathered[:, :, :D], valid_shared, softmax_scale
        )

        # --- Fused dKV: eliminates P, dov, dS materialization ---
        # dKV[q,k,:] = sum_h( dS[q,h,k]*Q[q,h,:] + P[q,h,k]*dO[q,h,:] )
        dkv_gathered = fused_dkv(
            scores, lse, Di, grad_out, query, kv_gathered[:, :, :D],
            valid_shared, softmax_scale
        )
        del scores

        # --- Scatter with sorted local reduction ---
        valid_flat = valid_shared.reshape(-1)
        dkv = torch.zeros(total_Skv, D_full, dtype=torch.float32, device=query.device)
        sorted_scatter_add(dkv_gathered, flat_idxs, valid_flat, dkv)

        dq_out = dq.to(query.dtype)
        dkv_out = dkv.to(kv.dtype)

        # d_sink: gradient of the bias-only attention sink
        d_sink = None
        if attn_sink is not None:
            p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, H)
            ds_sink = -p_sink * Di  # (total_Sq, H)
            d_sink = ds_sink.sum(0)  # (H,)

        return dq_out, dkv_out, d_sink


def _dsa_sparse_attn_flat(
    query: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Sparse attention forward (differentiable).

    Drop-in replacement for the FlashMLA-based ``dsa_sparse_attn`` in
    ``dsa_kernels.py``.

    Args:
        query: ``(total_S_q, H, D)`` bf16 鈥?flat queries.
        kv: ``(total_S_kv, D_kv)`` bf16 鈥?flat KV (single-head).
        topk_idxs: ``(total_S_q, H_kv, TopK)`` int32 鈥?global KV indices.
            H_kv is typically 1 for MQA.
        softmax_scale: scaling applied to Q @ K^T.
        d_v: value dimension (typically 512 for DSA).
        attn_sink: ``(H,)`` f32 鈥?per-head sink bias (optional).
        topk_length: ``(total_S_q, H_kv)`` int32 鈥?valid count per query (optional,
            for compact mode). If None, -1 entries in topk_idxs are used as mask.
        indexer_topk: if > 0, compute separate LSE for first ``indexer_topk``
            positions (used by fused indexer path).

    Returns:
        ``(out, lse, lse_indexer)``
        - out: ``(total_S_q, H, d_v)`` bf16.
        - lse: ``(total_S_q, H)`` fp32.
        - lse_indexer: ``(total_S_q, H)`` fp32 or None.
    """
    # Expand topk_idxs to per-head if needed (H_kv=1 -> broadcast)
    total_Sq, H, D = query.shape
    if topk_idxs.shape[1] == 1 and H > 1:
        topk_idxs = topk_idxs.expand(-1, H, -1)

    if indexer_topk > 0:
        # Split computation: full attention + indexer-only LSE
        out, lse = _DSASparseAttnFunc.apply(
            query, kv, topk_idxs, softmax_scale, d_v, attn_sink
        )
        # Compute LSE for first indexer_topk positions
        TopK = topk_idxs.shape[-1]
        if indexer_topk >= TopK:
            lse_indexer = lse.clone()
        else:
            idx_subset = topk_idxs[:, :, :indexer_topk].contiguous()
            _, lse_indexer, _ = triton_sparse_attn_forward(
                query, kv, idx_subset, softmax_scale, d_v, attn_sink
            )
        return out, lse, lse_indexer
    else:
        out, lse = _DSASparseAttnFunc.apply(
            query, kv, topk_idxs, softmax_scale, d_v, attn_sink
        )
        return out, lse, None


def dsa_sparse_attn(
    query: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
    is_thd: bool = False,
) -> Tensor:
    """Sparse attention with a shared SBHD/THD interface.

    Triton always consumes flat Q/KV rows. SBHD is reshaped into that contract;
    THD already satisfies it and therefore bypasses all layout conversion.

    Args:
        query: SBHD ``(sq, b, np, d)`` or THD ``(total_q, np, d)``.
        kv: SBHD ``(skv, b, d)`` or THD ``(total_kv, d)``.
        attn_sink: ``(np,)`` f32.
        topk_idxs: ``(sq*b, topk)`` int32 鈥?flat global indices.
        softmax_scale: scalar float.
        topk_length: ``(sq*b,)`` int32 鈥?optional compact fast-path.
        indexer_topk: int; 0 for Paths A/C, positive for Path B.

    Returns:
        SBHD ``(sq, b, np * d_v)`` or THD ``(total_q, np * d_v)``.
    """
    if is_thd:
        if query.ndim != 3 or kv.ndim != 2:
            raise ValueError(
                "THD dsa_sparse_attn expects query [T,H,D] and kv [Tkv,D], got "
                f"{tuple(query.shape)} and {tuple(kv.shape)}"
            )
        q_flat, kv_flat = query, kv
        np_, d = query.shape[1:]
    else:
        if query.ndim != 4 or kv.ndim != 3:
            raise ValueError(
                "SBHD dsa_sparse_attn expects query [S,B,H,D] and kv [S,B,D], got "
                f"{tuple(query.shape)} and {tuple(kv.shape)}"
            )
        sq, b, np_, d = query.shape
        skv = kv.shape[0]
        q_flat = query.reshape(sq * b, np_, d)
        kv_flat = kv.reshape(skv * b, d)
    # dsa_sparse_attn expects (total_Sq, H_kv, TopK); core produces (total_Sq, TopK)
    idxs = topk_idxs.unsqueeze(1) if topk_idxs.dim() == 2 else topk_idxs
    tlen = topk_length
    if tlen is not None and tlen.dim() == 1:
        tlen = tlen.unsqueeze(1)
    out_flat, _lse, _ = _dsa_sparse_attn_flat(
        q_flat, kv_flat, idxs, softmax_scale, d, attn_sink, tlen, indexer_topk
    )
    d_v = out_flat.shape[-1]
    if is_thd:
        return out_flat.reshape(query.shape[0], np_ * d_v)
    return out_flat.reshape(sq, b, np_ * d_v)


# Compatibility name used by the original FlagScale dispatcher.
dsa_sparse_attn_sbhd = dsa_sparse_attn


# ---------------------------------------------------------------------------
# Path C inference: indexer_topk
# ---------------------------------------------------------------------------


def indexer_topk(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk: int,
    ratio: int,
    indexer_softmax_scale: float = 1.0,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    q_causal_offsets: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Indexer scoring + top-K selection for SBHD or packed THD.

    Drop-in replacement for the cuDNN/TRT-LLM-based ``indexer_topk``.

    Args:
        q_indexer: SBHD ``(sq,b,h,d)`` or THD ``(total_q,h,d)``.
        k_indexer: SBHD ``(sk,b,d)`` or THD ``(total_k,d)``.
        weights: SBHD ``(sq,b,h)`` or THD ``(total_q,h)``.
        topk: number of top-K indices to select.
        ratio: compression ratio for the causal mask.
        indexer_softmax_scale: scale applied to indexer scores.
        q_causal_offsets: optional THD per-segment position of the first local
            query. CP supplies this for segments that start after sequence row 0.

    Returns:
        topk_indices: ``(b, sq, topk)`` int32.
        topk_length:  ``(b, sq)`` int32.
    """
    is_thd = cu_seqlens_q is not None
    if not is_thd:
        if any(x is not None for x in (cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)):
            raise ValueError("THD indexer metadata must be provided together")
        if q_causal_offsets is not None:
            raise ValueError("q_causal_offsets is only valid for THD")
        q_bshd, k_bsd, _w_bsh_raw, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
            q_indexer, k_indexer, weights, indexer_softmax_scale
        )
        topk_indices, topk_length, _ = _indexer_topk_bshd(
            q_bshd, k_bsd, w_bsh_scaled, topk, ratio
        )
        return topk_indices, topk_length

    if cu_seqlens_kv is None or max_seqlen_q is None or max_seqlen_kv is None:
        raise ValueError(
            "THD indexer_topk requires cu_seqlens_kv, max_seqlen_q, and max_seqlen_kv"
        )
    if q_indexer.ndim != 3 or k_indexer.ndim != 2 or weights.ndim != 2:
        raise ValueError("THD indexer inputs must be [T,H,D], [Tk,D], and [T,H]")

    q_sbhd = _packed_to_padded_sb(q_indexer, cu_seqlens_q, int(max_seqlen_q))
    k_sbd = _packed_to_padded_sb(k_indexer, cu_seqlens_kv, int(max_seqlen_kv))
    w_sbh = _packed_to_padded_sb(weights, cu_seqlens_q, int(max_seqlen_q))
    q_bshd, k_bsd, _, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
        q_sbhd, k_sbd, w_sbh, indexer_softmax_scale
    )
    _, _, scores = _indexer_topk_bshd(
        q_bshd, k_bsd, w_bsh_scaled, min(topk, int(max_seqlen_kv)), ratio
    )
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    kv_lens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    q_positions = torch.arange(int(max_seqlen_q), device=q_indexer.device)[None, :]
    if q_causal_offsets is not None:
        q_positions = q_positions + q_causal_offsets[:, None]
    visible_k = torch.div(q_positions + 1, ratio, rounding_mode="floor")
    k_positions = torch.arange(int(max_seqlen_kv), device=k_indexer.device)[None, None, :]
    valid_k = (
        (k_positions < visible_k[:, :, None])
        & (k_positions < kv_lens[:, None, None])
        & (torch.arange(int(max_seqlen_q), device=q_indexer.device)[None, :, None]
           < q_lens[:, None, None])
    )
    topk_padded, length_padded = _masked_topk_from_scores(scores, topk, valid_k)
    total_q = q_indexer.shape[0]
    topk_thd = _padded_sb_to_packed(topk_padded.permute(1, 0, 2), cu_seqlens_q, total_q)
    length_thd = _padded_sb_to_packed(length_padded.permute(1, 0), cu_seqlens_q, total_q)
    return topk_thd, length_thd


# ---------------------------------------------------------------------------
# Path B training: fused_indexer_sparse_attn
# ---------------------------------------------------------------------------


_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny
_SPARSE_KL_EPS = 1e-10


def _kl_loss_from_target_predict(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL(target || predict) reduced and scaled by loss_coeff."""
    # Keep inference/no-grad loss reporting identical to compute_dsa_indexer_loss.
    kl_per_row = (
        target
        * (
            torch.log(target + _SPARSE_KL_EPS)
            - torch.log(predict + _SPARSE_KL_EPS)
        )
    ).sum(dim=-1)

    row_valid = (topk_indices >= 0).any(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return loss_coeff * loss


def _kl_loss_from_dense_scores(
    attn_score: Tensor,
    attn_l1norm: Tensor,
    index_score: Tensor,
    index_lse: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL loss from dense scores (full-KV path)."""
    eps = _CLIP_PROB_MIN
    B, S_q, S_k = attn_score.shape

    row_valid = (topk_indices >= 0).any(dim=-1)  # (B, S_q)
    safe_l1 = attn_l1norm.clamp(min=eps)
    safe_lse = index_lse.clone()
    safe_lse[~row_valid] = 0.0

    target = attn_score / safe_l1.unsqueeze(-1)
    target_clamped = target.clamp(min=eps)
    position_valid = torch.isfinite(index_score)
    safe_index_score = torch.where(position_valid, index_score, torch.zeros_like(index_score))
    log_predict = safe_index_score - safe_lse.unsqueeze(-1)

    kl_terms = target_clamped * (torch.log(target_clamped) - log_predict)
    kl_terms = torch.where(position_valid, kl_terms, torch.zeros_like(kl_terms))
    kl_per_row = kl_terms.sum(dim=-1)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return loss_coeff * loss


class FusedIndexerSparseAttnFunc(torch.autograd.Function):
    """Path B: fused indexer (+KL loss) + sparse attention.

    Differentiable w.r.t. ``query``, ``kv_full``, ``attn_sink``,
    ``q_indexer``, ``k_indexer``, ``weights``.

    Two indexer-loss variants selected by ``sparse_loss``:
    - Sparse: KL over top-K positions only.
    - Dense: KL over all causally valid KV positions.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,       # (sq, b, np, d)
        kv_full: Tensor,     # (skv, b, d)
        attn_sink: Tensor,   # (np,) f32
        window_idxs: Tensor, # (b, sq, win_topk) int32
        q_indexer: Tensor,   # (sq, b, idx_nh, idx_hd)
        k_indexer: Tensor,   # (n_comp, b, idx_hd)
        weights: Tensor,     # (sq, b, idx_nh) 鈥?raw
        indexer_topk: int,
        ratio: int,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        sparse_loss: bool,
        kv_offset: int,
        calculate_per_token_loss: bool,
        tp_group=None,  ##### FlagScale Add #####
        valid_comp_mask: Optional[Tensor] = None,
        loss_q_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        sq, b, np_, d = query.shape
        skv = kv_full.shape[0]
        n_comp = k_indexer.shape[0]
        idx_nh, idx_hd = q_indexer.shape[2], q_indexer.shape[3]

        effective_topk = min(indexer_topk, n_comp)

        logger.debug(
            "FusedIndexerSparseAttnFunc.forward: sq=%d, b=%d, np=%d, d=%d, "
            "skv=%d, n_comp=%d, effective_topk=%d, sparse_loss=%s, "
            "loss_coeff=%.4g",
            sq, b, np_, d, skv, n_comp, effective_topk, sparse_loss, loss_coeff,
        )

        prof = _CudaProfiler(enabled=_DSA_PROFILE)

        # 1. Transpose indexer inputs SBHD -> BSHD
        prof.start("step1_sbhd_to_bshd")
        q_idx_bshd, k_idx_bsd, w_bsh, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
            q_indexer, k_indexer, weights, indexer_softmax_scale
        )
        prof.stop()

        # 2. Indexer scoring + top-K
        prof.start("step2_indexer_topk")
        topk_indices_cmp, _, indexer_scores = _indexer_topk_bshd(
            q_idx_bshd, k_idx_bsd, w_bsh_scaled, effective_topk, ratio
        )
        if valid_comp_mask is not None:
            # THD is padded only at this wrapper boundary. Mask per-sequence K
            # holes and reselect from the Triton-produced score matrix.
            topk_indices_cmp, _ = _masked_topk_from_scores(
                indexer_scores, effective_topk, valid_comp_mask
            )
        prof.stop()

        # 3. Combine indices (compressed + window)
        prof.start("step3_4_combine_flatten")
        # Add kv_offset to compressed indices
        topk_indices_global = topk_indices_cmp.clone()
        valid_cmp = topk_indices_global >= 0
        topk_indices_global[valid_cmp] += kv_offset

        # Combine: compressed first, then window
        combined_idxs = torch.cat([topk_indices_global, window_idxs], dim=-1)  # (b, sq, total_topk)
        total_topk = combined_idxs.shape[-1]

        # 4. Flatten for sparse attention
        # Use SB (seq-major) flat layout: flat[s * b + batch_idx] = orig[s, batch_idx]
        # query: (sq, b, np, d) -> (sq*b, np, d)  鈥?already SB order via reshape
        # kv_full: (skv, b, d) -> (skv*b, d)      鈥?already SB order via reshape
        q_flat = query.reshape(sq * b, np_, d)
        kv_flat = kv_full.reshape(skv * b, -1)

        # Convert local per-batch indices to global flat indices (SB layout).
        # For SB flat KV: global_idx = local_kv_idx * b + batch_idx
        # combined_idxs: (b, sq, total_topk) with local values in [0, skv)
        batch_ids = torch.arange(b, device=query.device, dtype=combined_idxs.dtype)
        global_idxs = combined_idxs.clone()
        valid_mask = global_idxs >= 0
        global_idxs = torch.where(
            valid_mask,
            global_idxs * b + batch_ids.view(b, 1, 1),
            global_idxs,
        )  # (b, sq, total_topk)
        # Permute to SB order then flatten: (b, sq, topk) -> (sq, b, topk) -> (sq*b, topk)
        global_idxs = global_idxs.permute(1, 0, 2).reshape(sq * b, total_topk)
        # MLA: all heads share KV indices. Keep as (sq*b, 1, TopK) to avoid
        # redundant np_ copies in save_for_backward and backward gather.
        global_idxs = global_idxs.unsqueeze(1)  # (sq*b, 1, total_topk)
        # Expand for forward (uses stride trick, no memory allocation)
        global_idxs_expanded = global_idxs.expand(-1, np_, -1)  # (sq*b, np, total_topk)
        prof.stop()

        # 5. Sparse attention forward
        prof.start("step5_sparse_attn_fwd")
        # When sparse_loss is enabled, compute partial LSE for the first
        # effective_topk positions (compressed indices) in a single pass,
        # avoiding a redundant second forward call.
        _indexer_topk_for_lse = effective_topk if sparse_loss else 0
        out_flat, lse, lse_indexer_raw = triton_sparse_attn_forward(
            q_flat, kv_flat, global_idxs_expanded, softmax_scale, d, attn_sink,
            indexer_topk=_indexer_topk_for_lse,
        )
        prof.stop()

        # Artificial SBD padding rows, plus optional CUDA-graph padding rows,
        # participate in sparse attention for static-shape safety but must not
        # contribute to the indexer loss or its eager backward.
        if loss_q_mask is not None:
            topk_indices_cmp = topk_indices_cmp.masked_fill(
                ~loss_q_mask[:, :, None], -1
            )
        # 6. Compute indexer loss
        # P3 optimization: skip step 6+7 entirely when loss_coeff == 0
        if loss_coeff == 0:
            indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)
            precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
            precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
            precomputed_grad_weights = torch.zeros_like(weights)

            # Save for backward
            ctx.save_for_backward(
                q_flat, kv_flat, attn_sink, global_idxs, out_flat.clone(), lse,
                precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights,
            )
            ctx.softmax_scale = softmax_scale
            ctx.sq = sq
            ctx.b = b
            ctx.np_ = np_
            ctx.d = d
            ctx.skv = skv

            d_v = out_flat.shape[-1]
            output = out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)
            prof.report(f"  [sq={sq}, b={b}, np={np_}, topk={total_topk}, loss_coeff=0] ")
            return output, indexer_loss

        # 6+7. Fused: compute indexer loss AND pre-compute indexer backward in one pass.
        # This replaces the separate step 6 (score_recompute + KL) and step 7
        # (indexer backward), eliminating redundant gather/einsum operations.
        prof.start("step6_7_indexer_loss_bwd")
        needs_grad = any(
            t.requires_grad for t in (query, kv_full, attn_sink, q_indexer, k_indexer, weights)
        )

        # Prepare attention tensors in BSHD layout (shared by both paths).
        # .contiguous() omitted: downstream einsum/indexing handle strided tensors.
        q_attn_bshd = query.permute(1, 0, 2, 3)  # (b, sq, np, d)
        lse_indexer_bsh = lse_indexer_raw.reshape(sq, b, np_).permute(1, 0, 2) if sparse_loss else None

        if sparse_loss:
            k_attn_bsd = kv_full[:, :, :d].permute(1, 0, 2)  # (b, skv, d)

            ##### FlagScale Add #####
            # Determine TP size for target reduction
            _tp_size = tp_group.size() if tp_group is not None and hasattr(tp_group, 'size') else 1
            _need_tp_reduce = _tp_size > 1

            ##### FlagScale End #####
            if needs_grad:
                ##### FlagScale Add #####
                if not _need_tp_reduce:
                    # Preserve the original fused TP=1 path. Besides being faster,
                    # its indexer backward avoids the decomposed scatter_add path.
                    (
                        indexer_loss,
                        precomputed_grad_q_indexer,
                        precomputed_grad_k_indexer,
                        precomputed_grad_weights,
                    ) = fused_sparse_indexer_loss_and_backward(
                        q_idx_bshd,
                        k_idx_bsd,
                        w_bsh_scaled,
                ##### FlagScale End #####
                        topk_indices_cmp,
                        ##### FlagScale Add #####
                        q_attn_bshd,
                        k_attn_bsd,
                        lse_indexer_bsh,
                        ##### FlagScale End #####
                        indexer_softmax_scale=indexer_softmax_scale,
                        softmax_scale=softmax_scale,
                        loss_coeff=loss_coeff,
                        calculate_per_token_loss=calculate_per_token_loss,
                        idx_nh=idx_nh,
                        kv_offset=kv_offset,
                    )
                ##### FlagScale Add #####
                else:
                    # Decompose only when TP needs a global target reduction.
                    local_head_sum = compute_sparse_local_target_head_sum(
                        q_attn_bshd,
                        k_attn_bsd,
                        lse_indexer_bsh,
                        topk_indices_cmp,
                        softmax_scale=softmax_scale,
                        kv_offset=kv_offset,
                    ).contiguous()

                    tp_work = None
                    if _DSA_TP_OVERLAP:
                        tp_work = torch.distributed.all_reduce(
                            local_head_sum,
                            op=torch.distributed.ReduceOp.SUM,
                            group=tp_group,
                            async_op=True,
                        )
                    else:
                        torch.distributed.all_reduce(
                            local_head_sum,
                            op=torch.distributed.ReduceOp.SUM,
                            group=tp_group,
                        )

                    predict_state = compute_sparse_indexer_predict_state(
                        q_idx_bshd, k_idx_bsd, w_bsh_scaled, topk_indices_cmp,
                    )
                    if tp_work is not None:
                        tp_work.wait()

                    (
                        indexer_loss,
                        precomputed_grad_q_indexer,
                        precomputed_grad_k_indexer,
                        precomputed_grad_weights,
                    ) = sparse_indexer_kl_and_backward(
                        local_head_sum,
                        predict_state,
                        q_idx_bshd,
                        k_idx_bsd,
                        w_bsh_scaled,
                        loss_coeff=loss_coeff,
                        calculate_per_token_loss=calculate_per_token_loss,
                    )
                ##### FlagScale End #####
                # BSHD -> SBHD (match input layout)
                precomputed_grad_q_indexer = precomputed_grad_q_indexer.permute(1, 0, 2, 3).contiguous()
                precomputed_grad_k_indexer = precomputed_grad_k_indexer.permute(1, 0, 2).contiguous()
                precomputed_grad_weights = precomputed_grad_weights.permute(1, 0, 2).contiguous()
                # Chain rule: w_scaled = w_raw * indexer_softmax_scale (done in
                # _sbhd_to_bshd_indexer_inputs). The fused function returns
                # 鈭侺/鈭倃_scaled; convert to 鈭侺/鈭倃_raw for the autograd Function.
                precomputed_grad_weights = precomputed_grad_weights * indexer_softmax_scale
            else:
                # Inference: only compute loss, no backward
                predict_result = sparse_indexer_score_recompute(
                    q_idx_bshd, k_idx_bsd, w_bsh_scaled, topk_indices_cmp,
                    qhead_per_kv_head=idx_nh,
                )
                ##### FlagScale Add #####
                local_head_sum = compute_sparse_local_target_head_sum(
                    q_attn_bshd,
                    k_attn_bsd,
                    lse_indexer_bsh,
                    topk_indices_cmp,
                    softmax_scale=softmax_scale,
                    kv_offset=kv_offset,
                ).contiguous()
                if _need_tp_reduce:
                    torch.distributed.all_reduce(
                        local_head_sum,
                        op=torch.distributed.ReduceOp.SUM,
                        group=tp_group,
                    )
                target = local_head_sum / local_head_sum.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1e-12)
                ##### FlagScale End #####
                indexer_loss = _kl_loss_from_target_predict(
                    ##### FlagScale Add #####
                    target,
                    predict_result["predict"],
                    ##### FlagScale End #####
                    topk_indices_cmp, loss_coeff, calculate_per_token_loss
                )
                precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
                precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
                precomputed_grad_weights = torch.zeros_like(weights)
        else:
            # Dense path
            k_attn_bsd = kv_full[kv_offset:kv_offset + n_comp, :, :d].permute(1, 0, 2)
            lse_bsh = lse.reshape(sq, b, np_).permute(1, 0, 2)

            if needs_grad:
                # Fused: loss + backward in one pass
                indexer_loss, precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights = (
                    fused_dense_indexer_loss_and_backward(
                        q_idx_bshd, k_idx_bsd, w_bsh_scaled,
                        topk_indices_cmp,
                        q_attn_bshd, k_attn_bsd, lse_bsh,
                        indexer_softmax_scale=indexer_softmax_scale,
                        softmax_scale=softmax_scale,
                        loss_coeff=loss_coeff,
                        ratio=ratio,
                        calculate_per_token_loss=calculate_per_token_loss,
                        idx_nh=idx_nh,
                        tp_group=tp_group,  ##### FlagScale Add #####
                    )
                )
                # BSHD -> SBHD (match input layout)
                precomputed_grad_q_indexer = precomputed_grad_q_indexer.permute(1, 0, 2, 3).contiguous()
                precomputed_grad_k_indexer = precomputed_grad_k_indexer.permute(1, 0, 2).contiguous()
                precomputed_grad_weights = precomputed_grad_weights.permute(1, 0, 2).contiguous()
                # Chain rule: same as sparse path
                precomputed_grad_weights = precomputed_grad_weights * indexer_softmax_scale
            else:
                # Inference: only compute loss
                dense_idx_result = dense_indexer_score_recompute(
                    q_idx_bshd, k_idx_bsd, w_bsh_scaled,
                    qhead_per_kv_head=idx_nh, sm_scale=1.0, ratio=ratio,
                )
                # Pass lse=None so dense_attn_score_recompute uses self-contained
                # softmax over compressed keys only (matching unfused reference).
                # Using the full LSE (which includes window tokens in the
                # denominator) would make compressed-token probabilities too small.
                dense_attn_result = dense_attn_score_recompute(
                    q_attn_bshd, k_attn_bsd, None,
                    qhead_per_kv_head=np_, softmax_scale=softmax_scale, ratio=ratio,
                )
                indexer_loss = _kl_loss_from_dense_scores(
                    dense_attn_result["out"], dense_attn_result["denom"],
                    dense_idx_result["out"], dense_idx_result["denom"],
                    topk_indices_cmp, loss_coeff, calculate_per_token_loss,
                )
                precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
                precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
                precomputed_grad_weights = torch.zeros_like(weights)

        # Save for backward
        prof.stop()
        ctx.save_for_backward(
            q_flat, kv_flat, attn_sink, global_idxs, out_flat.clone(), lse,
            precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights,
        )
        ctx.softmax_scale = softmax_scale
        ctx.sq = sq
        ctx.b = b
        ctx.np_ = np_
        ctx.d = d
        ctx.skv = skv

        # Return
        d_v = out_flat.shape[-1]
        output = out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)
        prof.report(f"  [sq={sq}, b={b}, np={np_}, topk={total_topk}] ")
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss):
        (
            q_flat, kv_flat, attn_sink, global_idxs, out_flat, lse,
            precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights,
        ) = ctx.saved_tensors

        sq, b, np_, d = ctx.sq, ctx.b, ctx.np_, ctx.d
        skv = ctx.skv

        d_v = out_flat.shape[-1]
        total_Sq = sq * b
        TopK = global_idxs.shape[-1]
        d_kv = kv_flat.shape[-1]

        prof = _CudaProfiler(enabled=_DSA_PROFILE)

        logger.debug(
            "FusedIndexerSparseAttnFunc.backward: sq=%d, b=%d, np=%d, "
            "TopK=%d, d_kv=%d, bwd_path=fused_dq_dkv",
            sq, b, np_, TopK, d_kv,
        )

        # --- Optimized path: bf16 BMM scores + Triton fused dQ/dKV ---
        # Uses fused_dq and fused_dkv kernels that keep P, dov, dS in registers,
        # eliminating ~768MB of intermediate GMEM allocations.
        # Key savings vs previous torch.bmm path:
        #   1. No P (S,H,TopK) materialization in GMEM
        #   2. No dov (S,H,TopK) materialization in GMEM
        #   3. No dS (S,H,TopK) materialization in GMEM
        #   4. Fused mask+scatter (saves 384MB read pass)
        prof.start("bwd_prepare_gather")
        dO_flat = grad_output.reshape(total_Sq, np_, d_v)

        # Shared indices: (total_Sq, 1, TopK) -> squeeze to (total_Sq, TopK)
        idxs_shared = global_idxs.squeeze(1)  # (total_Sq, TopK)
        valid_shared = idxs_shared >= 0       # (total_Sq, TopK)
        safe_shared = idxs_shared.clamp(min=0).long()

        # Gather KV once in bf16 鈥?no f32 upcast! cuBLAS bf16 BMM does f32 accumulation.
        flat_idxs = safe_shared.reshape(-1)   # (total_Sq * TopK)
        kv_gathered = kv_flat[flat_idxs].reshape(total_Sq, TopK, d_kv)  # bf16, 384MB

        # Di = sum(dO * O) per (query, head) 鈥?needed for dS
        # Use bf16 inputs, f32 reduction (accurate enough for Di)
        Di = (dO_flat.float() * out_flat.float()).sum(dim=-1)  # (total_Sq, np)
        prof.stop()

        # --- Recompute scores via cuBLAS BMM in bf16 (matches HP WGMMA forward) ---
        # Forward uses tl.dot(Q_bf16, K_bf16^T) 鈫?f32 accumulator.
        # cuBLAS bf16 BMM also does bf16脳bf16鈫抐32 accumulation internally,
        # ensuring exp(scores_bwd - lse_fwd) is numerically consistent.
        prof.start("bwd_scores")
        scores = torch.bmm(
            q_flat.reshape(total_Sq, np_, d),
            kv_gathered[:, :, :d].transpose(1, 2)
        ).float() * ctx.softmax_scale  # (S, H, TopK) f32
        prof.stop()

        # --- Fused dQ: eliminates P, dov, dS materialization ---
        prof.start("bwd_dQ")
        dq = fused_dq(
            scores, lse, Di, dO_flat, kv_gathered[:, :, :d],
            valid_shared, ctx.softmax_scale
        )
        prof.stop()

        # --- Fused dKV: eliminates P, dov, dS materialization ---
        prof.start("bwd_dKV")
        dkv_gathered = fused_dkv(
            scores, lse, Di, dO_flat, q_flat.reshape(total_Sq, np_, d),
            kv_gathered[:, :, :d], valid_shared, ctx.softmax_scale
        )
        del scores
        prof.stop()

        prof.start("bwd_scatter")
        # Sorted scatter with local reduction (less atomic contention)
        valid_flat = valid_shared.reshape(-1)
        dkv = torch.zeros(skv * b, d_kv, dtype=torch.float32, device=q_flat.device)
        sorted_scatter_add(dkv_gathered, flat_idxs, valid_flat, dkv)

        grad_query = dq.to(q_flat.dtype).reshape(sq, b, np_, d)
        grad_kv_full = dkv.to(kv_flat.dtype).reshape(skv, b, -1)
        prof.stop()

        # d_sink
        d_sink = None
        if attn_sink is not None:
            p_sink = torch.exp(attn_sink.unsqueeze(0) - lse)  # (total_Sq, np)
            ds_sink = -p_sink * Di  # (total_Sq, np)
            d_sink = ds_sink.sum(0)  # (np,)

        prof.start("bwd_indexer_grads")
        # Scale pre-computed indexer grads by actual grad_loss
        grad_q_indexer = precomputed_grad_q_indexer * grad_loss
        grad_k_indexer = precomputed_grad_k_indexer * grad_loss
        grad_weights = precomputed_grad_weights * grad_loss
        prof.stop()

        prof.report(f"  [sq={sq}, b={b}, np={np_}, topk={TopK}] ")

        return (
            grad_query,
            grad_kv_full,
            d_sink,
            None,  # window_idxs
            grad_q_indexer,
            grad_k_indexer,
            grad_weights,
            None, None, None, None, None, None, None, None,  # scalar args
            None,  # tp_group  ##### FlagScale Add #####
            None,  # valid_comp_mask
            None,  # loss_q_mask
        )


class FusedIndexerSparseAttnFromTopkFunc(torch.autograd.Function):
    """CP/THD sparse attention with caller-supplied compressed top-k.

    Top-k selection and CP index lowering stay with the caller. This function
    supplies the Triton sparse-attention forward/backward and sparse indexer
    loss backward needed by the existing CP training path.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        kv_full: Tensor,
        attn_sink: Tensor,
        topk_idxs: Tensor,
        q_indexer: Tensor,
        k_indexer: Tensor,
        weights: Tensor,
        indexer_topk_idxs: Tensor,
        compressed_kv: Tensor,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        loss_divisor: float,
        sparse_loss: bool,
        ratio: int,
        max_seqlen_q: int,
        indexer_layout,
        q_padding_mask: Optional[Tensor] = None,
        tp_group=None,
    ) -> Tuple[Tensor, Tensor]:
        if not sparse_loss:
            raise NotImplementedError(
                "The Triton CP from-topk path currently supports sparse indexer loss only"
            )

        total_q, np_, d = query.shape
        indexer_topk = indexer_topk_idxs.shape[-1]
        shared_idxs = topk_idxs.unsqueeze(1) if topk_idxs.ndim == 2 else topk_idxs
        expanded_idxs = shared_idxs.expand(-1, np_, -1)
        out, lse, lse_indexer = triton_sparse_attn_forward(
            query,
            kv_full,
            expanded_idxs,
            softmax_scale,
            d,
            attn_sink,
            indexer_topk=indexer_topk,
        )

        loss_topk = indexer_topk_idxs
        if q_padding_mask is not None:
            loss_topk = loss_topk.masked_fill(q_padding_mask.unsqueeze(-1), -1)

        q_idx_bshd = q_indexer.unsqueeze(0)
        k_idx_bsd = k_indexer.unsqueeze(0)
        w_bsh_scaled = (weights.float() * indexer_softmax_scale).unsqueeze(0)
        topk_bst = loss_topk.unsqueeze(0)
        local_head_sum = compute_sparse_local_target_head_sum(
            query.unsqueeze(0),
            compressed_kv.unsqueeze(0),
            lse_indexer.unsqueeze(0),
            topk_bst,
            softmax_scale=softmax_scale,
        ).contiguous()
        if tp_group is not None and tp_group.size() > 1:
            torch.distributed.all_reduce(
                local_head_sum,
                op=torch.distributed.ReduceOp.SUM,
                group=tp_group,
            )

        predict_state = compute_sparse_indexer_predict_state(
            q_idx_bshd, k_idx_bsd, w_bsh_scaled, topk_bst
        )
        effective_loss_coeff = loss_coeff / loss_divisor
        indexer_loss, grad_q_indexer, grad_k_indexer, grad_weights = (
            sparse_indexer_kl_and_backward(
                local_head_sum,
                predict_state,
                q_idx_bshd,
                k_idx_bsd,
                w_bsh_scaled,
                loss_coeff=effective_loss_coeff,
                calculate_per_token_loss=True,
            )
        )
        grad_q_indexer = grad_q_indexer.squeeze(0)
        grad_k_indexer = grad_k_indexer.squeeze(0)
        grad_weights = grad_weights.squeeze(0) * indexer_softmax_scale
        if q_padding_mask is not None:
            grad_q_indexer = grad_q_indexer.masked_fill(
                q_padding_mask[:, None, None], 0
            )
            grad_weights = grad_weights.masked_fill(q_padding_mask[:, None], 0)

        out_for_backward = out.clone()
        ctx.save_for_backward(
            query,
            kv_full,
            attn_sink,
            shared_idxs,
            out_for_backward,
            lse,
            grad_q_indexer,
            grad_k_indexer,
            grad_weights,
        )
        ctx.softmax_scale = softmax_scale
        ctx.d_v = out.shape[-1]
        ctx.used_hp_fwd = (
            expanded_idxs.stride(1) == 0
            and np_ >= 16
            and np_ % 16 == 0
            and d % 16 == 0
            and ctx.d_v % 16 == 0
        )
        return out.reshape(total_q, np_ * ctx.d_v), indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss):
        (
            query,
            kv_full,
            attn_sink,
            shared_idxs,
            out,
            lse,
            grad_q_indexer,
            grad_k_indexer,
            grad_weights,
        ) = ctx.saved_tensors
        dO = grad_output.reshape(query.shape[0], query.shape[1], ctx.d_v)
        expanded_idxs = shared_idxs.expand(-1, query.shape[1], -1)
        if ctx.used_hp_fwd:
            dq, dkv, d_sink = _DSASparseAttnFunc._hp_bmm_backward(
                dO,
                query,
                kv_full,
                expanded_idxs,
                out,
                lse,
                attn_sink,
                ctx.softmax_scale,
                ctx.d_v,
            )
        else:
            result = triton_sparse_attn_backward(
                dO,
                query,
                kv_full,
                out,
                lse,
                expanded_idxs,
                ctx.softmax_scale,
                ctx.d_v,
                attn_sink,
            )
            dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]

        return (
            dq,
            dkv,
            d_sink,
            None,
            grad_q_indexer * grad_loss,
            grad_k_indexer * grad_loss,
            grad_weights * grad_loss,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_indexer_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    window_idxs: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk: int,
    ratio: int,
    softmax_scale: float,
    indexer_softmax_scale: float = 1.0,
    loss_coeff: float = 0.0,
    sparse_loss: bool = False,
    kv_offset: int = 0,
    calculate_per_token_loss: bool = False,
    tp_group=None,  ##### FlagScale Add #####
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    cu_seqlens_kv_full: Optional[Tensor] = None,
    cu_seqlens_compressed_idx: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_compressed_idx: Optional[int] = None,
    compressed_kv: Optional[Tensor] = None,
    cu_seqlens_q_unpadded: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Fused indexer loss + sparse attention for SBHD or packed THD.

    The Triton kernels retain one BSHD implementation. THD inputs are padded
    only at this boundary, then gathered back to packed order. Per-sequence Q/K
    masks ensure the padding is invisible to top-K, loss, and backward.

    ##### FlagScale Add #####
    Args:
        tp_group: TP process group. When provided with size > 1, the sparse
            indexer target is all-reduced across TP ranks before normalization.
        cu_seqlens_q: Supplying this selects THD mode.

    ##### FlagScale End #####
    Returns:
        ``(output, indexer_loss)`` where output is ``(sq, b, np * d_v)`` bf16
        and indexer_loss is a scalar f32.
    """
    if cu_seqlens_q is None:
        return FusedIndexerSparseAttnFunc.apply(
            query,
            kv_full,
            attn_sink,
            window_idxs,
            q_indexer,
            k_indexer,
            weights,
            indexer_topk,
            ratio,
            softmax_scale,
            indexer_softmax_scale,
            loss_coeff,
            sparse_loss,
            kv_offset,
            calculate_per_token_loss,
            tp_group,
            None,
            None,
        )

    required = {
        "cu_seqlens_kv": cu_seqlens_kv,
        "cu_seqlens_kv_full": cu_seqlens_kv_full,
        "cu_seqlens_compressed_idx": cu_seqlens_compressed_idx,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_compressed_idx": max_seqlen_compressed_idx,
        "compressed_kv": compressed_kv,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"fused_indexer_sparse_attn THD mode requires {missing}")
    if query.ndim != 3 or kv_full.ndim != 2:
        raise ValueError("THD fused DSA expects query [T,H,D] and kv_full [Tkv,D]")
    if q_indexer.ndim != 3 or k_indexer.ndim != 2 or weights.ndim != 2:
        raise ValueError("THD fused DSA indexer inputs must be [T,H,D], [Tk,D], and [T,H]")

    max_q = int(max_seqlen_q)
    max_comp = int(max_seqlen_compressed_idx)
    batch = cu_seqlens_q.shape[0] - 1
    total_q = query.shape[0]

    query_sbhd = _packed_to_padded_sb(query, cu_seqlens_q, max_q)
    q_indexer_sbhd = _packed_to_padded_sb(q_indexer, cu_seqlens_q, max_q)
    weights_sbh = _packed_to_padded_sb(weights, cu_seqlens_q, max_q)
    k_indexer_sbd = _packed_to_padded_sb(
        k_indexer, cu_seqlens_compressed_idx, max_comp
    )
    kv_full_sbd = _packed_full_kv_to_padded(
        kv_full,
        cu_seqlens_kv,
        cu_seqlens_kv_full,
        max_q,
        max_comp,
    )
    window_bsq = _packed_to_padded_sb(window_idxs, cu_seqlens_q, max_q).permute(1, 0, 2)

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    q_positions = torch.arange(max_q, device=query.device)[None, :]
    physical_q_mask = q_positions < q_lens[:, None]
    window_bsq = window_bsq.masked_fill(~physical_q_mask[:, :, None], -1)

    comp_lens = cu_seqlens_compressed_idx[1:] - cu_seqlens_compressed_idx[:-1]
    valid_comp_mask = (
        torch.arange(max_comp, device=query.device)[None, :] < comp_lens[:, None]
    )
    loss_q_lens = (
        cu_seqlens_q_unpadded[1:] - cu_seqlens_q_unpadded[:-1]
        if cu_seqlens_q_unpadded is not None
        else q_lens
    )
    loss_q_mask = q_positions < loss_q_lens[:, None]

    # The shared BSHD loss averages over B*max_q rows. THD's contract averages
    # over physical packed rows, so compensate only for mean reduction.
    padded_loss_coeff = loss_coeff
    if not calculate_per_token_loss and total_q:
        padded_loss_coeff *= (batch * max_q) / total_q

    output_sbhd, indexer_loss = FusedIndexerSparseAttnFunc.apply(
        query_sbhd,
        kv_full_sbd,
        attn_sink,
        window_bsq,
        q_indexer_sbhd,
        k_indexer_sbd,
        weights_sbh,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale,
        padded_loss_coeff,
        sparse_loss,
        max_q,
        calculate_per_token_loss,
        tp_group,
        valid_comp_mask,
        loss_q_mask,
    )
    output_width = output_sbhd.shape[-1]
    output_thd = _padded_sb_to_packed(
        output_sbhd.reshape(max_q, batch, output_width), cu_seqlens_q, total_q
    )
    return output_thd, indexer_loss


__all__ = [
    "build_flat_topk_idxs",
    "local_to_global_flat",
    "dsa_sparse_attn",
    "dsa_sparse_attn_sbhd",
    "indexer_topk",
    "fused_indexer_sparse_attn",
    "FusedIndexerSparseAttnFromTopkFunc",
]
