# Copyright (c) 2026, FlagOS Contributors. All rights reserved.

"""Regression gates for CSA window-plus-sink teacher normalization."""

import torch
import pytest

from megatron.core.transformer.experimental_attention_variant.csa import (
    _compute_unfused_csa_non_compressed_lse,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    FusedDSAIndexerLoss,
    compute_dsa_indexer_loss,
)
from megatron.plugin.dsa_kernel.backends.triton.indexer_sparse_kernels import (
    non_compressed_lse_total_seq,
)
from megatron.plugin.dsa_kernel.backends.triton.indexer import (
    fused_dense_indexer_loss_and_backward,
)


class _SingleRankTP:
    @staticmethod
    def size():
        return 1


class _SingleRankPG:
    tp = _SingleRankTP()


def _manual_loss(index_scores, query, key, non_compressed_lse, loss_coeff):
    attention_logits = torch.einsum(
        "sbhd,tbhd->bhst", query.float(), key.float()
    )
    compressed_lse = torch.logsumexp(attention_logits, dim=-1)
    full_lse = torch.logaddexp(non_compressed_lse, compressed_lse)
    target = torch.exp(attention_logits - full_lse.unsqueeze(-1)).sum(dim=1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    predict = torch.softmax(index_scores, dim=-1)
    return (
        target
        * (torch.log(target + 1e-10) - torch.log(predict + 1e-10))
    ).sum(dim=-1).mean() * loss_coeff


def test_unfused_non_compressed_lse_matches_window_and_sink_oracle():
    torch.manual_seed(71)
    seqlen_q, batch, heads, dim = 4, 2, 3, 5
    query = torch.randn(seqlen_q, batch, heads, dim)
    kv = torch.randn(seqlen_q, batch, dim)
    sink = torch.randn(heads)
    indices = torch.tensor(
        [
            [[0, -1], [0, 1], [1, 2], [2, 3]],
            [[0, -1], [0, 1], [1, 2], [2, 3]],
        ],
        dtype=torch.int32,
    )

    actual = _compute_unfused_csa_non_compressed_lse(
        query, kv, sink, indices, softmax_scale=0.5, chunk_size=1
    )
    expected = torch.empty(batch, heads, seqlen_q)
    for b in range(batch):
        for q in range(seqlen_q):
            for h in range(heads):
                logits = [sink[h]]
                for k in indices[b, q]:
                    if k >= 0:
                        logits.append(torch.dot(query[q, b, h], kv[k, b]) * 0.5)
                expected[b, h, q] = torch.logsumexp(torch.stack(logits), dim=0)

    assert actual.shape == (batch, heads, seqlen_q)
    assert actual.dtype == torch.float32
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("sparse_loss", [False, True])
def test_unfused_indexer_loss_uses_non_compressed_teacher_mass(sparse_loss):
    torch.manual_seed(73)
    seqlen_q, batch, heads, dim, compressed = 3, 1, 2, 4, 3
    loss_coeff = 0.37
    query = torch.randn(seqlen_q, batch, heads, dim)
    key = torch.randn(compressed, batch, heads, dim)
    non_compressed_lse = torch.randn(batch, heads, seqlen_q) + 2.0
    index_scores_leaf = torch.randn(batch, seqlen_q, compressed, requires_grad=True)
    index_scores = index_scores_leaf * 1.0
    topk = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
    mask = torch.zeros(batch, seqlen_q, compressed)

    actual = compute_dsa_indexer_loss(
        index_scores,
        topk,
        query,
        key,
        1.0,
        loss_coeff,
        sparse_loss,
        _SingleRankPG(),
        causal_mask_override=mask,
        non_compressed_lse=non_compressed_lse,
    )
    expected_scores = index_scores_leaf.detach().clone().requires_grad_(True)
    expected = _manual_loss(
        expected_scores, query, key, non_compressed_lse, loss_coeff
    )
    actual_grad = torch.autograd.grad(actual, index_scores_leaf)[0]
    expected_grad = torch.autograd.grad(expected, expected_scores)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


def test_fused_dsa_indexer_loss_optional_lse_backward_matches_reference():
    torch.manual_seed(79)
    seqlen_q, batch, idx_heads, idx_dim = 3, 1, 2, 4
    attn_heads, attn_dim, compressed = 2, 4, 3
    q = torch.randn(
        seqlen_q, batch, idx_heads, idx_dim, requires_grad=True
    )
    weights = torch.rand(seqlen_q, batch, idx_heads, requires_grad=True)
    k = torch.randn(compressed, batch, idx_dim, requires_grad=True)
    query = torch.randn(seqlen_q, batch, attn_heads, attn_dim)
    key = torch.randn(compressed, batch, attn_heads, attn_dim)
    mask = torch.zeros(batch, seqlen_q, compressed)
    non_compressed_lse = torch.randn(batch, attn_heads, seqlen_q) + 2.0

    _, loss = FusedDSAIndexerLoss.apply(
        q,
        weights,
        k,
        query,
        key,
        1.0,
        compressed,
        0.2,
        mask,
        False,
        _SingleRankPG(),
        False,
        non_compressed_lse,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_total_seq_non_compressed_lse_matches_materialized_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton compile/launch parity")
    torch.manual_seed(83)
    device = torch.device("cuda")
    total_q, total_k, heads, dim = 4, 9, 16, 16
    query = torch.randn(total_q, heads, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(total_k, dim, device=device, dtype=torch.bfloat16)
    sink = torch.randn(heads, device=device)
    indices = torch.tensor(
        [[0, 2, 4, 6, 8], [1, 3, 5, 7, -1],
         [8, 6, 4, -1, -1], [0, 1, 2, 3, 4]],
        device=device,
        dtype=torch.int32,
    )
    scale = dim**-0.5

    actual = non_compressed_lse_total_seq(query, key, indices, sink, scale)
    gathered = key.float()[indices.long().clamp_min(0)]
    logits = torch.einsum("thd,tkd->thk", query.float(), gathered) * scale
    logits = logits.masked_fill(indices[:, None, :] < 0, float("-inf"))
    expected = torch.logaddexp(torch.logsumexp(logits, dim=-1), sink.view(1, -1))
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)


def test_dense_fused_loss_and_grads_include_non_compressed_lse():
    torch.manual_seed(89)
    batch, seqlen_q, idx_heads, idx_dim = 1, 4, 2, 4
    attn_heads, attn_dim, compressed = 3, 4, 4
    q_idx = torch.randn(batch, seqlen_q, idx_heads, idx_dim)
    k_idx = torch.randn(batch, compressed, idx_dim)
    weights = torch.rand(batch, seqlen_q, idx_heads)
    q_attn = torch.randn(batch, seqlen_q, attn_heads, attn_dim)
    k_attn = torch.randn(batch, compressed, attn_dim)
    non_compressed_lse = torch.randn(batch, seqlen_q, attn_heads) + 2.0
    topk = torch.zeros(batch, seqlen_q, 1, dtype=torch.int32)
    loss_coeff = 0.23

    actual = fused_dense_indexer_loss_and_backward(
        q_idx,
        k_idx,
        weights,
        topk,
        q_attn,
        k_attn,
        non_compressed_lse,
        indexer_softmax_scale=1.0,
        softmax_scale=1.0,
        loss_coeff=loss_coeff,
        ratio=1,
    )

    q_ref = q_idx.permute(1, 0, 2, 3).detach().clone().requires_grad_(True)
    k_ref = k_idx.permute(1, 0, 2).detach().clone().requires_grad_(True)
    w_ref = weights.permute(1, 0, 2).detach().clone().requires_grad_(True)
    q_attn_ref = q_attn.permute(1, 0, 2, 3)
    key_ref = k_attn.permute(1, 0, 2).unsqueeze(2).expand(-1, -1, attn_heads, -1)
    causal = torch.triu(
        torch.full((seqlen_q, compressed), float("-inf")), diagonal=1
    ).unsqueeze(0)
    _, expected_loss = FusedDSAIndexerLoss.apply(
        q_ref,
        w_ref,
        k_ref,
        q_attn_ref,
        key_ref,
        1.0,
        1,
        loss_coeff,
        causal,
        False,
        _SingleRankPG(),
        False,
        non_compressed_lse.permute(0, 2, 1),
    )
    expected_loss.backward()

    torch.testing.assert_close(actual[0], expected_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[1], q_ref.grad.permute(1, 0, 2, 3), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[2], k_ref.grad.permute(1, 0, 2), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[3], w_ref.grad.permute(1, 0, 2), rtol=2e-5, atol=2e-6)
