# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""CPU reference gates for the sparse DSA indexer teacher contract.

These tests deliberately avoid the Hopper-only attention kernels.  They freeze
the loss semantics that every fused implementation must preserve:

* selected compressed logits are the teacher numerator;
* compressed TopK, local-window logits, and the sink form the per-head LSE;
* TP reduces the unnormalised local head sums before TopK L1 normalisation;
* the student softmax is defined only over its selected compressed candidates.
"""

from __future__ import annotations

import torch
import pytest

from megatron.plugin.dsa_kernel.backends.triton.indexer import (
    _SPARSE_KL_EPS,
    compute_sparse_indexer_predict_state,
    compute_sparse_local_target_head_sum,
    sparse_indexer_kl_and_backward,
)


def _teacher_case():
    torch.manual_seed(17)
    batch, seqlen_q, heads, dim = 2, 3, 4, 5
    compressed_topk, window_topk = 3, 2

    query = torch.randn(batch, seqlen_q, heads, dim, dtype=torch.float32)
    compressed_key = torch.randn(batch, compressed_topk, dim, dtype=torch.float32)
    window_key = torch.randn(batch, window_topk, dim, dtype=torch.float32)
    full_key = torch.cat((compressed_key, window_key), dim=1)
    sink = torch.tensor([-1.5, -0.25, 0.75, 1.5], dtype=torch.float32)
    scale = dim**-0.5

    compressed_logits = torch.einsum("bqhd,bkd->bqhk", query, compressed_key) * scale
    window_logits = torch.einsum("bqhd,bkd->bqhk", query, window_key) * scale
    sink_logits = sink.view(1, 1, heads, 1).expand(batch, seqlen_q, -1, -1)
    full_lse = torch.logsumexp(
        torch.cat((compressed_logits, window_logits, sink_logits), dim=-1), dim=-1
    )
    compressed_indices = torch.arange(compressed_topk, dtype=torch.int32).view(1, 1, -1)
    compressed_indices = compressed_indices.expand(batch, seqlen_q, -1).clone()
    return query, full_key, full_lse, compressed_indices, compressed_logits, scale


def _l1_normalize(head_sum: torch.Tensor) -> torch.Tensor:
    return head_sum / head_sum.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def test_sparse_teacher_uses_full_lse_but_compressed_numerator_only():
    query, full_key, full_lse, indices, compressed_logits, scale = _teacher_case()

    actual = compute_sparse_local_target_head_sum(
        query, full_key, full_lse, indices, softmax_scale=scale, kv_offset=0
    )
    expected = torch.exp(compressed_logits - full_lse.unsqueeze(-1)).sum(dim=2)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_window_and_sink_change_teacher_through_denominator():
    query, full_key, full_lse, indices, compressed_logits, scale = _teacher_case()
    full_head_sum = compute_sparse_local_target_head_sum(
        query, full_key, full_lse, indices, softmax_scale=scale, kv_offset=0
    )

    compressed_only_lse = torch.logsumexp(compressed_logits, dim=-1)
    compressed_only_head_sum = compute_sparse_local_target_head_sum(
        query, full_key, compressed_only_lse, indices, softmax_scale=scale, kv_offset=0
    )

    # Window and sink never add candidate columns, but their per-head mass
    # changes how the heads are mixed before the final TopK L1 normalisation.
    assert full_head_sum.shape == compressed_only_head_sum.shape == indices.shape
    assert not torch.allclose(_l1_normalize(full_head_sum), _l1_normalize(compressed_only_head_sum))


def test_tp_reduce_precedes_topk_l1_normalization():
    query, full_key, full_lse, indices, _, scale = _teacher_case()
    full = compute_sparse_local_target_head_sum(
        query, full_key, full_lse, indices, softmax_scale=scale, kv_offset=0
    )

    local = []
    for head_slice in (slice(0, 2), slice(2, 4)):
        local.append(
            compute_sparse_local_target_head_sum(
                query[:, :, head_slice],
                full_key,
                full_lse[:, :, head_slice],
                indices,
                softmax_scale=scale,
                kv_offset=0,
            )
        )

    reduced_then_normalized = _l1_normalize(local[0] + local[1])
    normalized_locally = (_l1_normalize(local[0]) + _l1_normalize(local[1])) / 2
    torch.testing.assert_close(reduced_then_normalized, _l1_normalize(full))
    assert not torch.allclose(reduced_then_normalized, normalized_locally)


def test_invalid_compressed_candidates_contribute_zero_mass():
    query, full_key, full_lse, indices, _, scale = _teacher_case()
    indices[:, 0, :] = -1
    indices[0, 1, -1] = -1
    head_sum = compute_sparse_local_target_head_sum(
        query, full_key, full_lse, indices, softmax_scale=scale, kv_offset=0
    )

    assert torch.isfinite(head_sum).all()
    assert torch.count_nonzero(head_sum[:, 0]) == 0
    assert head_sum[0, 1, -1] == 0


def test_sparse_student_manual_backward_matches_autograd_reference():
    torch.manual_seed(23)
    batch, seqlen_q, heads, dim, seqlen_k, topk = 2, 3, 3, 4, 5, 3
    loss_coeff = 0.17

    q = torch.randn(batch, seqlen_q, heads, dim, dtype=torch.float32)
    k = torch.randn(batch, seqlen_k, dim, dtype=torch.float32)
    weights = torch.randn(batch, seqlen_q, heads, dtype=torch.float32)
    indices = torch.tensor(
        [[[0, 2, 4], [1, 3, -1], [0, 1, 2]], [[4, 3, 1], [2, -1, -1], [1, 0, 3]]], dtype=torch.int32
    )
    head_sum = torch.rand(batch, seqlen_q, topk, dtype=torch.float32)
    head_sum = head_sum.masked_fill(indices < 0, 0.0)

    state = compute_sparse_indexer_predict_state(q, k, weights, indices)
    loss, grad_q, grad_k, grad_w = sparse_indexer_kl_and_backward(
        head_sum, state, q, k, weights, loss_coeff=loss_coeff, calculate_per_token_loss=False
    )

    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    w_ref = weights.clone().requires_grad_(True)
    safe_indices = indices.long().clamp(min=0)
    batch_index = torch.arange(batch)[:, None, None]
    gathered = k_ref[batch_index, safe_indices]
    per_head = torch.relu(torch.einsum("bqhd,bqtd->bqht", q_ref, gathered))
    logits = (per_head * w_ref.unsqueeze(-1)).sum(dim=2)
    logits = logits.masked_fill(indices < 0, float("-inf"))
    predict = torch.softmax(logits, dim=-1).masked_fill(indices < 0, 0.0)
    target = _l1_normalize(head_sum)
    reference_loss = (
        loss_coeff
        * (target * (torch.log(target + _SPARSE_KL_EPS) - torch.log(predict + _SPARSE_KL_EPS)))
        .sum(dim=-1)
        .mean()
    )
    reference_loss.backward()

    torch.testing.assert_close(loss, reference_loss.detach(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(grad_q, q_ref.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(grad_k, k_ref.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(grad_w, w_ref.grad, rtol=2e-5, atol=2e-6)


def test_dkv_scatter_defaults_to_sorted_and_atomic_is_opt_in(monkeypatch):
    from unittest.mock import patch

    from megatron.plugin.dsa_kernel.backends.triton import sparse_attention_backward as bwd

    dkv_gathered = torch.empty(1, 1, 1)
    flat_indices = torch.zeros(1, dtype=torch.int64)
    valid = torch.ones(1, dtype=torch.bool)
    output = torch.zeros(1, 1)

    monkeypatch.delenv("MEGATRON_DSA_DKV_SCATTER", raising=False)
    with (
        patch.object(bwd, "sorted_scatter_add") as sorted_scatter,
        patch.object(bwd, "fused_mask_scatter_add") as direct_atomic,
    ):
        bwd.scatter_dkv(dkv_gathered, flat_indices, valid, output)
        sorted_scatter.assert_called_once()
        direct_atomic.assert_not_called()

    monkeypatch.setenv("MEGATRON_DSA_DKV_SCATTER", "atomic")
    with (
        patch.object(bwd, "sorted_scatter_add") as sorted_scatter,
        patch.object(bwd, "fused_mask_scatter_add") as direct_atomic,
    ):
        bwd.scatter_dkv(dkv_gathered, flat_indices, valid, output)
        direct_atomic.assert_called_once()
        sorted_scatter.assert_not_called()


def test_dkv_scatter_rejects_unknown_backend():
    from megatron.plugin.dsa_kernel.backends.triton.sparse_attention_backward import scatter_dkv

    with pytest.raises(ValueError, match="must be 'sorted' or 'atomic'"):
        scatter_dkv(
            torch.empty(1, 1, 1),
            torch.zeros(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, 1),
            backend="surprise",
        )


def test_fused_sparse_loss_wires_attention_full_lse_into_teacher():
    from unittest.mock import patch

    import megatron.plugin.dsa_kernel.backends.triton.fused_ops as fused_ops

    torch.manual_seed(31)
    sq, batch, heads, dim = 4, 1, 2, 3
    compressed, indexer_dim = 2, 2
    query = torch.randn(sq, batch, heads, dim)
    kv_full = torch.randn(sq + compressed, batch, dim)
    sink = torch.randn(heads)
    window = torch.arange(sq, dtype=torch.int32).view(1, sq, 1)
    q_indexer = torch.randn(sq, batch, 1, indexer_dim)
    k_indexer = torch.randn(compressed, batch, indexer_dim)
    weights = torch.randn(sq, batch, 1)
    fake_output = torch.zeros(sq * batch, heads, dim, dtype=torch.bfloat16)
    full_lse_flat = torch.randn(sq * batch, heads) + 5.0
    captured = {}
    original_target = fused_ops.compute_sparse_local_target_head_sum

    def fake_attention(*args, **kwargs):
        return fake_output, full_lse_flat, None

    def capture_target(q_attn, k_attn, lse, *args, **kwargs):
        captured["lse"] = lse
        return original_target(q_attn, k_attn, lse, *args, **kwargs)

    with (
        patch.object(fused_ops, "triton_sparse_attn_forward", fake_attention),
        patch.object(fused_ops, "compute_sparse_local_target_head_sum", capture_target),
    ):
        _, loss = fused_ops.fused_indexer_sparse_attn(
            query,
            kv_full,
            sink,
            window,
            q_indexer,
            k_indexer,
            weights,
            indexer_topk=1,
            ratio=1,
            softmax_scale=dim**-0.5,
            indexer_softmax_scale=indexer_dim**-0.5,
            loss_coeff=0.1,
            sparse_loss=True,
            kv_offset=sq,
            calculate_per_token_loss=False,
            tp_group=None,
        )

    expected_lse = full_lse_flat.reshape(sq, batch, heads).permute(1, 0, 2)
    torch.testing.assert_close(captured["lse"], expected_lse)
    assert torch.isfinite(loss)
