# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Triton kernels for the total-sequence sparse indexer training path.

The compute kernels deliberately use a layout-neutral flattened contract:

* query tensors are ``(total_q, heads, dim)``;
* key tensors are ``(total_k, dim)``;
* LSE is ``(total_q, attention_heads)``;
* selected indices are absolute rows in the corresponding flattened key.

SBHD uses :func:`pack_sbhd_sparse_indices` as its layout adapter.  THD can
feed the same compute kernels directly once its sequence-local indices have
been lowered to absolute flat rows.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl


_BLOCK_HEAD = 16
_BLOCK_TOPK = 32


@triton.jit
def _pack_sbhd_sparse_indices_kernel(
    CMP_ptr,
    WIN_ptr,
    IDX_OUT_ptr,
    ATTN_OUT_ptr,
    batch_size,
    kv_offset,
    stride_cb,
    stride_cq,
    stride_ct,
    stride_wb,
    stride_wq,
    stride_wt,
    CMP_K: tl.constexpr,
    WIN_K: tl.constexpr,
    TOTAL_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Lower batch-local SBHD indices to total-sequence absolute indices."""
    row = tl.program_id(0)
    batch = row % batch_size
    query = row // batch_size
    lane = tl.arange(0, BLOCK)

    cmp_mask = lane < CMP_K
    cmp_idx = tl.load(
        CMP_ptr
        + batch * stride_cb
        + query * stride_cq
        + lane * stride_ct,
        mask=cmp_mask,
        other=-1,
    ).to(tl.int32)
    cmp_flat = tl.where(cmp_idx >= 0, cmp_idx * batch_size + batch, -1)
    tl.store(IDX_OUT_ptr + row * CMP_K + lane, cmp_flat, mask=cmp_mask)

    win_lane = lane - CMP_K
    win_mask = (lane >= CMP_K) & (lane < TOTAL_K)
    win_idx = tl.load(
        WIN_ptr
        + batch * stride_wb
        + query * stride_wq
        + win_lane * stride_wt,
        mask=win_mask,
        other=-1,
    ).to(tl.int32)
    cmp_attn = tl.where(
        cmp_idx >= 0,
        (cmp_idx + kv_offset) * batch_size + batch,
        -1,
    )
    win_attn = tl.where(win_idx >= 0, win_idx * batch_size + batch, -1)
    attn_idx = tl.where(lane < CMP_K, cmp_attn, win_attn)
    tl.store(ATTN_OUT_ptr + row * TOTAL_K + lane, attn_idx, mask=lane < TOTAL_K)


@triton.jit
def _sparse_teacher_total_seq_kernel(
    Q_ptr,
    K_ptr,
    LSE_ptr,
    IDX_ptr,
    OUT_ptr,
    softmax_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kd,
    stride_lt,
    stride_lh,
    stride_it,
    stride_ik,
    stride_ot,
    stride_ok,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute selected compressed attention mass summed over local heads."""
    row = tl.program_id(0)
    k_block = tl.program_id(1)
    k_lane = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    d_lane = tl.arange(0, BLOCK_D)
    k_mask = k_lane < TOPK
    d_mask = d_lane < DIM

    selected = tl.load(
        IDX_ptr + row * stride_it + k_lane * stride_ik,
        mask=k_mask,
        other=-1,
    )
    valid_k = k_mask & (selected >= 0)
    safe_selected = tl.where(valid_k, selected, 0)
    k_tile = tl.load(
        K_ptr + safe_selected[:, None] * stride_kt + d_lane[None, :] * stride_kd,
        mask=valid_k[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    head_sum = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for head_start in tl.static_range(0, HEADS, 16):
        head_lane = head_start + tl.arange(0, 16)
        head_mask = head_lane < HEADS
        q_tile = tl.load(
            Q_ptr
            + row * stride_qt
            + head_lane[:, None] * stride_qh
            + d_lane[None, :] * stride_qd,
            mask=head_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q_tile, tl.trans(k_tile)) * softmax_scale
        lse = tl.load(
            LSE_ptr + row * stride_lt + head_lane * stride_lh,
            mask=head_mask,
            other=float("inf"),
        )
        probs = tl.exp(scores - lse[:, None])
        probs = tl.where(head_mask[:, None] & valid_k[None, :], probs, 0.0)
        head_sum += tl.sum(probs, axis=0)

    tl.store(
        OUT_ptr + row * stride_ot + k_lane * stride_ok,
        head_sum,
        mask=k_mask,
    )


@triton.jit
def _non_compressed_lse_total_seq_kernel(
    Q_ptr,
    K_ptr,
    IDX_ptr,
    SINK_ptr,
    OUT_ptr,
    softmax_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kd,
    stride_it,
    stride_ik,
    stride_ot,
    stride_oh,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute window-plus-sink LSE in absolute total-sequence layout."""
    row = tl.program_id(0)
    head_start = tl.program_id(1) * 16
    head_lane = head_start + tl.arange(0, 16)
    d_lane = tl.arange(0, BLOCK_D)
    head_mask = head_lane < HEADS
    d_mask = d_lane < DIM
    q_tile = tl.load(
        Q_ptr
        + row * stride_qt
        + head_lane[:, None] * stride_qh
        + d_lane[None, :] * stride_qd,
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    running_max = tl.load(SINK_ptr + head_lane, mask=head_mask, other=float("-inf"))
    running_sum = tl.where(head_mask, 1.0, 0.0)
    for k_start in tl.static_range(0, WINDOW, BLOCK_K):
        k_lane = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_lane < WINDOW
        selected = tl.load(
            IDX_ptr + row * stride_it + k_lane * stride_ik,
            mask=k_mask,
            other=-1,
        )
        valid = k_mask & (selected >= 0)
        safe_selected = tl.where(valid, selected, 0)
        k_tile = tl.load(
            K_ptr
            + safe_selected[:, None] * stride_kt
            + d_lane[None, :] * stride_kd,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q_tile, tl.trans(k_tile)) * softmax_scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, float("-inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max)
        running_sum += tl.sum(tl.exp(scores - new_max[:, None]), axis=1)
        running_max = new_max

    lse = running_max + tl.log(running_sum)
    tl.store(
        OUT_ptr + row * stride_ot + head_lane * stride_oh,
        lse,
        mask=head_mask,
    )


@triton.jit
def _student_logits_tile(
    Q_ptr,
    K_ptr,
    W_ptr,
    IDX_ptr,
    row,
    k_start: tl.constexpr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kd,
    stride_wt,
    stride_wh,
    stride_it,
    stride_ik,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k_lane = k_start + tl.arange(0, BLOCK_K)
    d_lane = tl.arange(0, BLOCK_D)
    k_mask = k_lane < TOPK
    d_mask = d_lane < DIM
    selected = tl.load(
        IDX_ptr + row * stride_it + k_lane * stride_ik,
        mask=k_mask,
        other=-1,
    )
    valid_k = k_mask & (selected >= 0)
    safe_selected = tl.where(valid_k, selected, 0)
    k_tile = tl.load(
        K_ptr + safe_selected[:, None] * stride_kt + d_lane[None, :] * stride_kd,
        mask=valid_k[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    combined = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for head_start in tl.static_range(0, HEADS, 16):
        head_lane = head_start + tl.arange(0, 16)
        head_mask = head_lane < HEADS
        q_tile = tl.load(
            Q_ptr
            + row * stride_qt
            + head_lane[:, None] * stride_qh
            + d_lane[None, :] * stride_qd,
            mask=head_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q_tile, tl.trans(k_tile))
        scores = tl.maximum(scores, 0.0)
        weights = tl.load(
            W_ptr + row * stride_wt + head_lane * stride_wh,
            mask=head_mask,
            other=0.0,
        ).to(tl.float32)
        combined += tl.sum(scores * weights[:, None], axis=0)

    return tl.where(valid_k, combined, float("-inf")), valid_k


@triton.jit
def _sparse_student_total_seq_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    IDX_ptr,
    LOGITS_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kd,
    stride_wt,
    stride_wh,
    stride_it,
    stride_ik,
    stride_ot,
    stride_ok,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute selected-token student logits without materialized gathers."""
    row = tl.program_id(0)
    k_block = tl.program_id(1)
    k_start = k_block * BLOCK_K
    logits, _ = _student_logits_tile(
        Q_ptr,
        K_ptr,
        W_ptr,
        IDX_ptr,
        row,
        k_start,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_wt,
        stride_wh,
        stride_it,
        stride_ik,
        HEADS=HEADS,
        DIM=DIM,
        TOPK=TOPK,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )
    k_lane = k_start + tl.arange(0, BLOCK_K)
    tl.store(
        LOGITS_ptr + row * stride_ot + k_lane * stride_ok,
        logits,
        mask=k_lane < TOPK,
    )


@triton.jit
def _sparse_student_softmax_total_seq_kernel(
    LOGITS_ptr,
    IDX_ptr,
    PRED_ptr,
    stride_lt,
    stride_lk,
    stride_it,
    stride_ik,
    stride_pt,
    stride_pk,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Normalize one selected-token row after its tiled logits are ready."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    lane_mask = lane < TOPK
    selected = tl.load(
        IDX_ptr + row * stride_it + lane * stride_ik,
        mask=lane_mask,
        other=-1,
    )
    valid = lane_mask & (selected >= 0)
    logits = tl.load(
        LOGITS_ptr + row * stride_lt + lane * stride_lk,
        mask=valid,
        other=float("-inf"),
    )
    row_max = tl.max(logits, axis=0)
    numerator = tl.where(valid, tl.exp(logits - row_max), 0.0)
    denominator = tl.sum(numerator, axis=0)
    predict = tl.where(valid, numerator / denominator, 0.0)
    tl.store(
        PRED_ptr + row * stride_pt + lane * stride_pk,
        predict,
        mask=lane_mask,
    )


@triton.jit
def _sparse_kl_total_seq_kernel(
    HEAD_ptr,
    PRED_ptr,
    IDX_ptr,
    KL_ptr,
    grad_scale,
    stride_ht,
    stride_hk,
    stride_pt,
    stride_pk,
    stride_it,
    stride_ik,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse target normalization, epsilon KL, and exact student logit grad."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    lane_mask = lane < TOPK
    selected = tl.load(
        IDX_ptr + row * stride_it + lane * stride_ik,
        mask=lane_mask,
        other=-1,
    )
    valid = lane_mask & (selected >= 0)
    head_sum = tl.load(
        HEAD_ptr + row * stride_ht + lane * stride_hk,
        mask=lane_mask,
        other=0.0,
    )
    predict = tl.load(
        PRED_ptr + row * stride_pt + lane * stride_pk,
        mask=lane_mask,
        other=0.0,
    )
    head_sum = tl.where(valid, head_sum, 0.0)
    predict = tl.where(valid, predict, 0.0)

    denom = tl.maximum(tl.sum(head_sum, axis=0), 1e-12)
    target = head_sum / denom
    eps = 1e-10
    kl = tl.sum(
        target * (tl.log(target + eps) - tl.log(predict + eps)), axis=0
    )
    row_valid = tl.sum(valid.to(tl.int32), axis=0) > 0
    tl.store(KL_ptr + row, tl.where(row_valid, kl, 0.0))

    scaled_target = target / (predict + eps)
    correction = tl.sum(scaled_target * predict, axis=0)
    grad_logits = predict * (correction - scaled_target) * grad_scale
    grad_logits = tl.where(valid, grad_logits, 0.0)
    tl.store(
        PRED_ptr + row * stride_pt + lane * stride_pk,
        grad_logits,
        mask=lane_mask,
    )


def pack_sbhd_sparse_indices(
    topk_indices_cmp: Tensor,
    window_indices: Tensor,
    kv_offset: int,
) -> Tuple[Tensor, Tensor]:
    """Return absolute total-sequence indexer and attention indices.

    The first result is ``(Sq*B, K)`` and indexes flattened compressed
    indexer keys.  The second is ``(Sq*B, K+W)`` and indexes flattened full
    attention KV, with compressed candidates first and window candidates next.
    """
    batch, seqlen_q, cmp_topk = topk_indices_cmp.shape
    win_topk = window_indices.shape[-1]
    total_topk = cmp_topk + win_topk
    rows = batch * seqlen_q
    if not topk_indices_cmp.is_cuda:
        compressed = topk_indices_cmp.permute(1, 0, 2).reshape(rows, cmp_topk)
        window = window_indices.permute(1, 0, 2).reshape(rows, win_topk)
        batch_ids = torch.arange(rows, device=compressed.device) % batch
        indexer_indices = torch.where(
            compressed >= 0,
            compressed * batch + batch_ids[:, None],
            -1,
        ).int()
        attention_compressed = torch.where(
            compressed >= 0,
            (compressed + kv_offset) * batch + batch_ids[:, None],
            -1,
        ).int()
        attention_window = torch.where(
            window >= 0,
            window * batch + batch_ids[:, None],
            -1,
        ).int()
        return indexer_indices, torch.cat(
            (attention_compressed, attention_window), dim=-1
        )

    indexer_indices = torch.empty(
        rows, cmp_topk, dtype=torch.int32, device=topk_indices_cmp.device
    )
    attention_indices = torch.empty(
        rows, total_topk, dtype=torch.int32, device=topk_indices_cmp.device
    )
    block = triton.next_power_of_2(total_topk)
    _pack_sbhd_sparse_indices_kernel[(rows,)](
        topk_indices_cmp,
        window_indices,
        indexer_indices,
        attention_indices,
        batch,
        kv_offset,
        topk_indices_cmp.stride(0),
        topk_indices_cmp.stride(1),
        topk_indices_cmp.stride(2),
        window_indices.stride(0),
        window_indices.stride(1),
        window_indices.stride(2),
        CMP_K=cmp_topk,
        WIN_K=win_topk,
        TOTAL_K=total_topk,
        BLOCK=block,
        num_warps=4,
    )
    return indexer_indices, attention_indices


def sparse_teacher_total_seq(
    q_attn: Tensor,
    k_attn: Tensor,
    lse: Tensor,
    attention_topk_indices: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Compute local teacher head sums using total-sequence absolute indices."""
    total_q, heads, dim = q_attn.shape
    topk = attention_topk_indices.shape[-1]
    head_sum = torch.empty(
        total_q, topk, dtype=torch.float32, device=q_attn.device
    )
    block_d = triton.next_power_of_2(dim)
    grid = (total_q, triton.cdiv(topk, _BLOCK_TOPK))
    _sparse_teacher_total_seq_kernel[grid](
        q_attn,
        k_attn,
        lse,
        attention_topk_indices,
        head_sum,
        softmax_scale,
        q_attn.stride(0),
        q_attn.stride(1),
        q_attn.stride(2),
        k_attn.stride(0),
        k_attn.stride(1),
        lse.stride(0),
        lse.stride(1),
        attention_topk_indices.stride(0),
        attention_topk_indices.stride(1),
        head_sum.stride(0),
        head_sum.stride(1),
        HEADS=heads,
        DIM=dim,
        TOPK=topk,
        BLOCK_D=block_d,
        BLOCK_K=_BLOCK_TOPK,
        num_warps=4,
        num_stages=2,
    )
    return head_sum


def non_compressed_lse_total_seq(
    q_attn: Tensor,
    k_attn: Tensor,
    window_indices: Tensor,
    attn_sink: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Return window-plus-sink LSE using absolute flattened KV indices."""
    total_q, heads, dim = q_attn.shape
    window = window_indices.shape[-1]
    output = torch.empty(total_q, heads, dtype=torch.float32, device=q_attn.device)
    grid = (total_q, triton.cdiv(heads, 16))
    _non_compressed_lse_total_seq_kernel[grid](
        q_attn,
        k_attn,
        window_indices,
        attn_sink,
        output,
        softmax_scale,
        q_attn.stride(0),
        q_attn.stride(1),
        q_attn.stride(2),
        k_attn.stride(0),
        k_attn.stride(1),
        window_indices.stride(0),
        window_indices.stride(1),
        output.stride(0),
        output.stride(1),
        HEADS=heads,
        DIM=dim,
        WINDOW=window,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_K=_BLOCK_TOPK,
        num_warps=4,
        num_stages=2,
    )
    return output


def sparse_student_total_seq(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk_indices: Tensor,
) -> Tensor:
    """Compute the selected-token student distribution in total-sequence layout."""
    total_q, heads, dim = q_indexer.shape
    topk = indexer_topk_indices.shape[-1]
    logits = torch.empty(
        total_q, topk, dtype=torch.float32, device=q_indexer.device
    )
    grid = (total_q, triton.cdiv(topk, _BLOCK_TOPK))
    _sparse_student_total_seq_kernel[grid](
        q_indexer,
        k_indexer,
        weights,
        indexer_topk_indices,
        logits,
        q_indexer.stride(0),
        q_indexer.stride(1),
        q_indexer.stride(2),
        k_indexer.stride(0),
        k_indexer.stride(1),
        weights.stride(0),
        weights.stride(1),
        indexer_topk_indices.stride(0),
        indexer_topk_indices.stride(1),
        logits.stride(0),
        logits.stride(1),
        HEADS=heads,
        DIM=dim,
        TOPK=topk,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_K=_BLOCK_TOPK,
        num_warps=4,
        num_stages=2,
    )
    predict = torch.empty_like(logits)
    _sparse_student_softmax_total_seq_kernel[(total_q,)](
        logits,
        indexer_topk_indices,
        predict,
        logits.stride(0),
        logits.stride(1),
        indexer_topk_indices.stride(0),
        indexer_topk_indices.stride(1),
        predict.stride(0),
        predict.stride(1),
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    return predict


def sparse_kl_total_seq(
    head_sum: Tensor,
    predict: Tensor,
    indexer_topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool,
) -> Tuple[Tensor, Tensor]:
    """Return scalar KL and exact scaled grad logits.

    ``predict`` is overwritten in place with grad-logits after the KL row has
    consumed it, avoiding a second ``(total_q, topk)`` allocation.
    """
    total_q, topk = head_sum.shape
    kl_rows = torch.empty(total_q, dtype=torch.float32, device=head_sum.device)
    grad_scale = loss_coeff if calculate_per_token_loss else loss_coeff / total_q
    _sparse_kl_total_seq_kernel[(total_q,)](
        head_sum,
        predict,
        indexer_topk_indices,
        kl_rows,
        grad_scale,
        head_sum.stride(0),
        head_sum.stride(1),
        predict.stride(0),
        predict.stride(1),
        indexer_topk_indices.stride(0),
        indexer_topk_indices.stride(1),
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    reduction = kl_rows.sum() if calculate_per_token_loss else kl_rows.mean()
    return loss_coeff * reduction, predict


def sparse_total_seq_eligible(
    q_indexer: Tensor,
    q_attn: Tensor,
) -> bool:
    """Route only the large-head BF16 contract to these Tensor-Core kernels."""
    return (
        q_indexer.is_cuda
        and q_attn.is_cuda
        and q_indexer.dtype == torch.bfloat16
        and q_attn.dtype == torch.bfloat16
        and q_indexer.shape[1] >= _BLOCK_HEAD
        and q_attn.shape[1] >= _BLOCK_HEAD
        and q_indexer.shape[2] % 16 == 0
        and q_attn.shape[2] % 16 == 0
    )
